"""Tests for the per-model half of case summarization.

These cover the only things that legitimately differ between the two models:
how a document gets attached to a request, how the schema is declared, and
which exception means what. Everything above that is shared and tested in
test_case_summarization_service.py.

The schema conversion gets the most attention here because it is the one place
the two providers could silently drift apart. There is a single source schema;
Gemini's dialect is derived from it, so a field added for one model cannot go
missing for the other. Nothing below makes a network call.
"""

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from src.core.exceptions import (
    BadGatewayAPIException,
    GatewayTimeoutAPIException,
    ServiceUnavailableAPIException,
    TooManyRequestsAPIException,
)
from src.prompts.case_summary_prompt import (
    GEMINI_RESPONSE_SCHEMA,
    RESPONSE_SCHEMA,
    _to_gemini_node,
    build_gemini_response_schema,
)
from src.services import summary_providers
from src.services.summary_providers import (
    MODEL_GEMINI,
    MODEL_GPT,
    GeminiSummaryProvider,
    OpenAISummaryProvider,
    PromptPayload,
    build_providers,
)

PDF_PAYLOAD = PromptPayload(
    instruction="Summarise the attached legal document.",
    inline_data=b"%PDF-1.4 scanned",
    media_type="application/pdf",
)
IMAGE_PAYLOAD = PromptPayload(
    instruction="Summarise the attached legal document.",
    inline_data=b"\x89PNG\r\n\x1a\n",
    media_type="image/png",
)
TEXT_PAYLOAD = PromptPayload(instruction="<<<BEGIN DOCUMENT>>> ... <<<END DOCUMENT>>>")


@pytest.fixture
def openai_provider():
    return OpenAISummaryProvider(client=object())


@pytest.fixture
def gemini_provider():
    return GeminiSummaryProvider(client=object())


# --- Schema: one source, two dialects ---------------------------------------


def walk(node, path="root"):
    """Every dict node in a schema, with a readable path."""
    if isinstance(node, dict):
        yield path, node
        for key, value in node.items():
            yield from walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from walk(value, f"{path}[{index}]")


def test_gemini_schema_has_no_type_arrays():
    """A JSON Schema type array is not in the subset Gemini's
    response_json_schema supports; it must be an anyOf union instead."""
    offenders = [
        path for path, node in walk(GEMINI_RESPONSE_SCHEMA)
        if isinstance(node.get("type"), list)
    ]
    assert offenders == []


def test_gemini_schema_has_no_null_inside_an_enum():
    """Gemini takes enums for strings and numbers only - a null member is
    rejected, so nullability has to move out to the union."""
    offenders = [
        path for path, node in walk(GEMINI_RESPONSE_SCHEMA)
        if isinstance(node.get("enum"), list) and None in node["enum"]
    ]
    assert offenders == []


def test_nullable_field_becomes_an_explicit_union():
    assert GEMINI_RESPONSE_SCHEMA["properties"]["facts_summary"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]


def test_a_plain_enum_passes_through_untouched():
    """statement_type is not nullable, so there is nothing to split - the enum
    must survive as it is, or Gemini could return a third statement type the
    response model rejects."""
    statement_type = GEMINI_RESPONSE_SCHEMA["properties"]["assertions"]["items"][
        "properties"
    ]["statement_type"]

    assert statement_type["type"] == "string"
    assert statement_type["enum"] == ["fact", "allegation"]


def test_a_nullable_enum_keeps_its_values_on_the_non_null_branch():
    """No field in the current schema is a nullable enum, but the converter
    still has to handle one: the enum belongs on the string branch, because
    Gemini rejects a null member inside an enum and dropping the enum outright
    would silently let the model answer with anything at all."""
    union = _to_gemini_node(
        {"type": ["string", "null"], "enum": ["plaint", "petition", None]}
    )
    string_branch, null_branch = union["anyOf"]

    assert null_branch == {"type": "null"}
    assert string_branch["type"] == "string"
    assert string_branch["enum"] == ["plaint", "petition"]


def test_both_dialects_describe_the_same_fields():
    """The point of deriving one from the other: a field added for one model
    cannot go missing for the other."""
    assert GEMINI_RESPONSE_SCHEMA["required"] == RESPONSE_SCHEMA["required"]
    assert set(GEMINI_RESPONSE_SCHEMA["properties"]) == set(
        RESPONSE_SCHEMA["properties"]
    )


def test_nested_objects_are_converted_too():
    """Conversion has to recurse - every grounded item schema is nested two
    levels down and carries nullable fields of its own."""
    snippet = GEMINI_RESPONSE_SCHEMA["properties"]["chronology"]["items"]["properties"][
        "source_snippet"
    ]
    assert snippet["anyOf"] == [{"type": "string"}, {"type": "null"}]


def test_the_source_schema_is_left_untouched():
    """Conversion must not mutate the OpenAI schema in place - both are
    module-level singletons shared by every request."""
    build_gemini_response_schema()
    assert RESPONSE_SCHEMA["properties"]["facts_summary"]["type"] == ["string", "null"]


# --- OpenAI: content translation --------------------------------------------


def test_openai_text_payload_is_a_single_text_part(openai_provider):
    parts = openai_provider._content(TEXT_PAYLOAD)

    assert [part["type"] for part in parts] == ["input_text"]
    assert parts[0]["text"] == TEXT_PAYLOAD.instruction


def test_openai_pdf_becomes_an_input_file_with_a_generic_filename(openai_provider):
    """The client's own filename is never sent upstream - in this domain it
    routinely carries a client's name."""
    parts = openai_provider._content(PDF_PAYLOAD)

    assert [part["type"] for part in parts] == ["input_text", "input_file"]
    assert parts[1]["filename"] == "document.pdf"
    assert parts[1]["file_data"].startswith("data:application/pdf;base64,")


def test_openai_image_becomes_an_input_image(openai_provider):
    parts = openai_provider._content(IMAGE_PAYLOAD)

    assert [part["type"] for part in parts] == ["input_text", "input_image"]
    assert parts[1]["image_url"].startswith("data:image/png;base64,")


def test_openai_inlines_the_bytes_rather_than_uploading_them(openai_provider):
    """Inline base64, not the Files API: no round trip, and no client filing
    left sitting in provider-side storage after the call."""
    import base64

    parts = openai_provider._content(PDF_PAYLOAD)
    encoded = parts[1]["file_data"].split(",", 1)[1]

    assert base64.b64decode(encoded) == PDF_PAYLOAD.inline_data


# --- Gemini: content translation --------------------------------------------


def test_gemini_text_payload_is_a_single_text_part(gemini_provider):
    contents = gemini_provider._contents(TEXT_PAYLOAD)

    assert len(contents) == 1
    assert contents[0].role == "user"
    assert [part.text for part in contents[0].parts] == [TEXT_PAYLOAD.instruction]


@pytest.mark.parametrize(
    "payload, mime",
    [(PDF_PAYLOAD, "application/pdf"), (IMAGE_PAYLOAD, "image/png")],
    ids=["pdf", "image"],
)
def test_gemini_attaches_pdfs_and_images_the_same_way(gemini_provider, payload, mime):
    """Unlike the OpenAI content types, Gemini does not distinguish them - one
    inline part carries either."""
    parts = gemini_provider._contents(payload)[0].parts

    assert parts[0].text == payload.instruction
    assert parts[1].inline_data.mime_type == mime
    assert parts[1].inline_data.data == payload.inline_data


# --- Gemini: error translation ----------------------------------------------


def api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"message": "boom"}})


@pytest.mark.parametrize(
    "code, expected",
    [
        (429, TooManyRequestsAPIException),
        (401, ServiceUnavailableAPIException),
        (403, ServiceUnavailableAPIException),
        (400, BadGatewayAPIException),
        (500, BadGatewayAPIException),
        (503, BadGatewayAPIException),
    ],
)
def test_gemini_status_codes_map_onto_our_exceptions(gemini_provider, code, expected):
    """Gemini reports everything as one exception family carrying an HTTP
    status, so the status is what has to distinguish the cases that the OpenAI
    SDK gives separate classes to."""
    assert isinstance(gemini_provider._translate(api_error(code)), expected)


def test_a_gemini_400_is_logged_as_our_fault(gemini_provider, caplog):
    """A 400 means our request was malformed - most likely a schema Gemini
    won't accept. That is the failure a schema change would introduce, so it
    must be loud in the logs even though the caller sees a generic 502."""
    with caplog.at_level("ERROR"):
        gemini_provider._translate(api_error(400))

    assert "rejected our request" in caplog.text


def test_rejected_credentials_are_logged_not_leaked(gemini_provider, caplog):
    with caplog.at_level("ERROR"):
        result = gemini_provider._translate(api_error(401))

    assert "credentials" in caplog.text
    assert "not configured correctly" in result.detail["error_message"]


# --- Shared response handling -----------------------------------------------


@pytest.mark.parametrize("text", ["", None], ids=["empty", "none"])
def test_an_empty_response_is_not_blamed_on_length(text):
    """Truncation is refused by each provider before `parse` runs, so an empty
    answer here is a refusal or a safety block. Telling the caller to send a
    shorter document would point them at the wrong fix."""
    with pytest.raises(BadGatewayAPIException) as excinfo:
        summary_providers.parse(text, MODEL_GPT)

    message = excinfo.value.detail["error_message"]
    assert "no result" in message
    assert "cut off" not in message
    assert "shorter" not in message


def test_unparseable_json_is_rejected():
    with pytest.raises(BadGatewayAPIException, match="unreadable"):
        summary_providers.parse("not json at all", MODEL_GEMINI)


def test_a_json_array_is_rejected():
    """Valid JSON, wrong shape - normalisation downstream assumes a dict."""
    with pytest.raises(BadGatewayAPIException, match="unexpected"):
        summary_providers.parse('["a", "b"]', MODEL_GEMINI)


def test_a_json_object_is_returned():
    assert summary_providers.parse('{"is_valid": true}', MODEL_GPT) == {
        "is_valid": True
    }


# --- Provider selection -----------------------------------------------------


def test_gemini_is_omitted_when_no_key_is_configured(monkeypatch, caplog):
    """Rather than constructed and left to fail on first use: the service turns
    a missing provider into a 503 that names it, which beats an auth error
    surfacing three layers down."""
    monkeypatch.setattr(summary_providers.settings, "GEMINI_API_KEY", None)
    with caplog.at_level("WARNING"):
        providers = build_providers()

    assert set(providers) == {MODEL_GPT}
    assert "GEMINI_API_KEY not set" in caplog.text


def test_gemini_is_built_when_a_key_is_configured(monkeypatch):
    monkeypatch.setattr(summary_providers.settings, "GEMINI_API_KEY", "test-key")
    providers = build_providers()

    assert set(providers) == {MODEL_GPT, MODEL_GEMINI}
    assert providers[MODEL_GEMINI].model == summary_providers.settings.GEMINI_MODEL


def test_both_providers_are_built_when_both_keys_are_set(monkeypatch):
    """What makes the per-request override possible at all: an override has
    nothing to switch to unless both adapters exist."""
    monkeypatch.setattr(summary_providers.settings, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(summary_providers.settings, "OPENAI_API_KEY", "sk-test")

    assert set(build_providers()) == {MODEL_GPT, MODEL_GEMINI}


def test_a_provider_whose_key_is_missing_is_omitted_not_built_broken(monkeypatch):
    """Left out entirely rather than constructed and allowed to fail on first
    use: the service turns a missing provider into a 503 naming what IS
    available, which beats an auth error surfacing three layers down."""
    monkeypatch.setattr(summary_providers.settings, "OPENAI_API_KEY", None)
    monkeypatch.setattr(summary_providers.settings, "GEMINI_API_KEY", "test-key")

    assert set(build_providers()) == {MODEL_GEMINI}


def test_the_configured_default_is_logged_with_what_was_built(monkeypatch, caplog):
    """One line at startup answering the question every misconfiguration
    report starts with: which providers exist here, and which one runs by
    default."""
    monkeypatch.setattr(summary_providers.settings, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(summary_providers.settings, "SUMMARY_PROVIDER", MODEL_GEMINI)

    with caplog.at_level("INFO"):
        build_providers()

    assert "default gemini" in caplog.text


def test_the_provider_names_have_one_definition(monkeypatch):
    """Re-exported from settings rather than redeclared here - settings has to
    validate SUMMARY_PROVIDER at startup and cannot import this module to get
    the list, since this module imports settings."""
    from src.config import settings as settings_module

    assert MODEL_GPT is settings_module.PROVIDER_GPT
    assert MODEL_GEMINI is settings_module.PROVIDER_GEMINI
    assert summary_providers.SUPPORTED_MODELS is (
        settings_module.SUPPORTED_SUMMARY_PROVIDERS
    )


def test_the_gemini_timeout_is_expressed_in_milliseconds(monkeypatch):
    """The genai SDK takes milliseconds where the OpenAI SDK takes seconds.
    Passing 180 here would be a 0.18 second timeout and every call would fail."""
    monkeypatch.setattr(summary_providers.settings, "GEMINI_API_KEY", "test-key")
    provider = GeminiSummaryProvider()

    options = provider.client._api_client._http_options
    assert options.timeout == summary_providers.REQUEST_TIMEOUT_SECONDS * 1000
    assert options.timeout >= 60_000


# --- Error mapping through the real generate() path -------------------------


class FakeGeminiClient:
    """Enough of the genai client surface for generate() to run against."""

    def __init__(self, raises=None, text=None, finish_reason=None):
        self._raises = raises
        self._text = text
        self._finish_reason = finish_reason
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._call))

    async def _call(self, **kwargs):
        self.kwargs = kwargs
        if self._raises is not None:
            raise self._raises
        candidates = (
            [SimpleNamespace(finish_reason=self._finish_reason)]
            if self._finish_reason is not None
            else []
        )
        return SimpleNamespace(
            text=self._text, usage_metadata=None, candidates=candidates
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "raises, expected",
    [
        (httpx.TimeoutException("slow"), GatewayTimeoutAPIException),
        (httpx.ConnectError("no route"), BadGatewayAPIException),
        (genai_errors.APIError(429, {}), TooManyRequestsAPIException),
        (genai_errors.APIError(401, {}), ServiceUnavailableAPIException),
    ],
    ids=["timeout", "connect", "rate_limit", "auth"],
)
async def test_gemini_failures_map_through_generate(raises, expected):
    """httpx raises the transport errors straight through the genai SDK rather
    than wrapping them in an APIError, so each needs its own except clause -
    and only driving generate() proves they are all actually there."""
    provider = GeminiSummaryProvider(client=FakeGeminiClient(raises=raises))

    with pytest.raises(expected):
        await provider.generate(TEXT_PAYLOAD)


@pytest.mark.anyio
async def test_gemini_returns_the_parsed_object_on_success():
    client = FakeGeminiClient(text='{"is_valid": true, "court": null}')
    provider = GeminiSummaryProvider(client=client)

    assert await provider.generate(TEXT_PAYLOAD) == {"is_valid": True, "court": None}
    # The converted schema, not the OpenAI one, and under the JSON-Schema key.
    config = client.kwargs["config"]
    assert config.response_json_schema == GEMINI_RESPONSE_SCHEMA
    assert config.response_schema is None
    assert config.response_mime_type == "application/json"
    assert config.temperature == 0


@pytest.mark.anyio
async def test_both_providers_send_the_same_system_prompt():
    """Same prompt, same schema, different model - otherwise comparing the two
    measures the prompt rather than the model."""
    from src.prompts.case_summary_prompt import SYSTEM_PROMPT

    client = FakeGeminiClient(text="{}")
    await GeminiSummaryProvider(client=client).generate(TEXT_PAYLOAD)

    assert client.kwargs["config"].system_instruction == SYSTEM_PROMPT


# --- The output-token budget is one budget ----------------------------------
#
# Gemini 2.5 charges reasoning tokens against max_output_tokens. Both of these
# exist because of one real failure: a 13-page scanned writ petition spent
# 7,498 tokens thinking and 8,871 answering, hit the 16,384 ceiling, and came
# back as truncated JSON that the caller was told was "unreadable".


@pytest.mark.anyio
async def test_gemini_does_not_spend_the_answer_budget_on_thinking():
    """Load-bearing, not a tuning preference. With thinking on, the shared
    MAX_OUTPUT_TOKENS means "the whole answer" for OpenAI and "whatever
    reasoning left over" for Gemini - so the two are no longer held to one
    ceiling, which is the basis for comparing them at all."""
    client = FakeGeminiClient(text="{}")
    await GeminiSummaryProvider(client=client).generate(TEXT_PAYLOAD)

    config = client.kwargs["config"]
    assert config.thinking_config.thinking_budget == 0
    assert config.max_output_tokens == summary_providers.MAX_OUTPUT_TOKENS


def test_the_ceiling_fits_both_providers():
    """One number for both, so it has to stay inside the smaller of the two
    limits - gpt-4.1-mini accepts 32768, Gemini would take more."""
    assert summary_providers.MAX_OUTPUT_TOKENS <= 32768


@pytest.mark.anyio
async def test_a_truncated_gemini_answer_says_it_was_cut_off():
    """Not "unreadable". A truncated object is unparseable, so without this
    check it reached `parse` and came back blaming the model for malformed
    JSON - pointing away from the real cause and inviting a retry that is
    guaranteed to fail identically. The OpenAI adapter has always checked its
    own `status == "incomplete"`; this is the missing half."""
    client = FakeGeminiClient(
        text='{"is_valid": true, "chronology": [{"event": "half an ob',
        finish_reason=genai_types.FinishReason.MAX_TOKENS,
    )

    with pytest.raises(BadGatewayAPIException) as excinfo:
        await GeminiSummaryProvider(client=client).generate(TEXT_PAYLOAD)

    message = excinfo.value.detail["error_message"]
    assert "cut off" in message
    assert "shorter document" in message


@pytest.mark.anyio
async def test_a_complete_gemini_answer_is_not_treated_as_truncated():
    client = FakeGeminiClient(
        text='{"is_valid": true}', finish_reason=genai_types.FinishReason.STOP
    )

    result = await GeminiSummaryProvider(client=client).generate(TEXT_PAYLOAD)

    assert result == {"is_valid": True}
