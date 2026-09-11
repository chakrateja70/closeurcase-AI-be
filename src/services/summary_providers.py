"""The two models case summarization can run on, behind one interface.

A caller picks "gpt" or "gemini" per request. Everything that decides *what* the
summary says - the prompt, the JSON schema, the document framing, the
normalisation of the result - is shared and lives outside this module, in the
prompt module and the service. What lives here is only the difference between
the two SDKs: how a document is attached to a request, how a structured-output
schema is declared, and which exception means what.

That split is the point. If the prompt or the schema could vary per provider,
"which model summarises better" would stop being an answerable question. Both
providers are handed the identical `PromptPayload` and both must return a dict
matching the same schema; the only asymmetry is the schema *dialect*, and that
is a mechanical conversion of one source (see `build_gemini_response_schema`).

Both are held to the same output-token ceiling and the same one-retry policy,
for the same reason - a three-minute call retried twice is worse than a failure
the caller can act on.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Protocol

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from src.config.settings import (
    PROVIDER_GEMINI,
    PROVIDER_GPT,
    SUPPORTED_SUMMARY_PROVIDERS,
    settings,
)
from src.core.exceptions import (
    BadGatewayAPIException,
    GatewayTimeoutAPIException,
    ServiceUnavailableAPIException,
    TooManyRequestsAPIException,
)
from src.prompts.case_summary_prompt import (
    GEMINI_RESPONSE_SCHEMA,
    RESPONSE_SCHEMA,
    SCHEMA_NAME,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Defined in `config.settings` so that startup validation of SUMMARY_PROVIDER
# has a list to check against without importing this module (which imports
# settings). Re-exported here under the names the rest of the tree already
# uses, so there is one definition and no caller has to know where it lives.
MODEL_GPT = PROVIDER_GPT
MODEL_GEMINI = PROVIDER_GEMINI
SUPPORTED_MODELS = SUPPORTED_SUMMARY_PROVIDERS

# A 30-page filing is a genuinely slow call, so the ceiling is far above the
# 30s case detection uses. Retries are capped at one deliberately.
REQUEST_TIMEOUT_SECONDS = 180
MAX_RETRIES = 1
# Raised from 8192 when the citable fields started carrying a quotation each.
# Running out mid-object does not degrade gracefully: the JSON is truncated, so
# the whole call is lost and reported as cut off after it has been paid for.
# Output is billed per token generated, so headroom that goes unused is free.
MAX_OUTPUT_TOKENS = 16384


@dataclass(frozen=True)
class PromptPayload:
    """One provider-neutral request.

    `instruction` always carries the framing the model is asked to follow. On
    the text branch it also carries the document itself, delimited; on the file
    and image branches the document travels as `inline_data` instead and the
    instruction is just the framing sentence.

    `media_type` is the discriminator both providers switch on - a PDF and a
    JPEG are attached differently by the OpenAI SDK, though not by Gemini's.
    """

    instruction: str
    inline_data: bytes | None = None
    media_type: str | None = None


class SummaryProvider(Protocol):
    """What the service needs from a model, and nothing more."""

    name: str
    model: str

    async def generate(self, payload: PromptPayload) -> dict: ...

    async def aclose(self) -> None: ...


def _parse(text: str | None, provider: str) -> dict:
    """Shared result handling, so a malformed answer fails the same way on both
    providers rather than in two subtly different ways."""
    if not text:
        raise BadGatewayAPIException(
            "The summary was cut off before it was complete. Please try a "
            "shorter document."
        )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("%s: unparseable response", provider)
        raise BadGatewayAPIException(
            "Case summarization returned an unreadable result. Please try again."
        ) from exc

    if not isinstance(parsed, dict):
        raise BadGatewayAPIException(
            "Case summarization returned an unexpected result. Please try again."
        )
    return parsed


# --- OpenAI -----------------------------------------------------------------


class OpenAISummaryProvider:
    name = MODEL_GPT

    def __init__(self, client: AsyncOpenAI | None = None):
        # An injected client belongs to the caller (tests, mostly), so only a
        # client built here is ours to close.
        self._owns_client = client is None
        self.model = settings.OPENAI_SUMMARY_MODEL
        self.client = client or AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.close()

    def _content(self, payload: PromptPayload) -> list[dict]:
        parts: list[dict] = [{"type": "input_text", "text": payload.instruction}]
        if payload.inline_data is None:
            return parts

        # Inline base64 rather than an upload through the Files API: it saves a
        # round trip, and a client's court filing is never left sitting in
        # provider-side storage after the call.
        encoded = base64.b64encode(payload.inline_data).decode("ascii")
        data_url = f"data:{payload.media_type};base64,{encoded}"
        if payload.media_type == "application/pdf":
            parts.append(
                {
                    "type": "input_file",
                    # Generic on purpose - the client's own filename is never
                    # sent upstream; in this domain it carries a client's name.
                    "filename": "document.pdf",
                    "file_data": data_url,
                }
            )
        else:
            parts.append({"type": "input_image", "image_url": data_url})
        return parts

    async def generate(self, payload: PromptPayload) -> dict:
        logger.debug("calling %s", self.model)
        try:
            response = await self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=[{"role": "user", "content": self._content(payload)}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": SCHEMA_NAME,
                        "schema": RESPONSE_SCHEMA,
                        "strict": True,
                    }
                },
                temperature=0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                store=False,
            )
        except APITimeoutError as exc:
            raise GatewayTimeoutAPIException(
                "Summarising the document timed out. It may be too long - try a "
                "shorter document."
            ) from exc
        except RateLimitError as exc:
            raise TooManyRequestsAPIException(
                "The summarization model is busy right now. Please retry in a moment."
            ) from exc
        except AuthenticationError as exc:
            raise ServiceUnavailableAPIException(
                "Case summarization is not configured correctly on the server."
            ) from exc
        except (APIConnectionError, APIStatusError) as exc:
            raise BadGatewayAPIException(
                "Case summarization is unavailable. Please try again."
            ) from exc

        if response.usage:
            details = response.usage.input_tokens_details
            logger.info(
                "gpt usage input=%s output=%s cached=%s total=%s",
                response.usage.input_tokens,
                response.usage.output_tokens,
                details.cached_tokens if details else 0,
                response.usage.total_tokens,
            )

        if response.status == "incomplete":
            # Almost always the output-token cap on a dense document, which a
            # retry would hit again - so don't suggest one.
            raise BadGatewayAPIException(
                "The summary was cut off before it was complete. Please try a "
                "shorter document."
            )
        return _parse(response.output_text, self.name)


# --- Gemini -----------------------------------------------------------------


class GeminiSummaryProvider:
    name = MODEL_GEMINI

    def __init__(self, client: genai.Client | None = None):
        self._owns_client = client is None
        self.model = settings.GEMINI_MODEL
        self.client = client or genai.Client(
            api_key=settings.GEMINI_API_KEY,
            # MILLISECONDS here, unlike the OpenAI SDK's seconds. Passing 180
            # would be a 0.18s timeout and every call would fail.
            http_options=genai_types.HttpOptions(
                timeout=REQUEST_TIMEOUT_SECONDS * 1000,
                retry_options=genai_types.HttpRetryOptions(attempts=MAX_RETRIES + 1),
            ),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aio.aclose()

    def _contents(self, payload: PromptPayload) -> list:
        parts = [genai_types.Part(text=payload.instruction)]
        if payload.inline_data is not None:
            # One call for both PDFs and images - Gemini does not distinguish
            # them the way the OpenAI content types do.
            parts.append(
                genai_types.Part.from_bytes(
                    data=payload.inline_data, mime_type=payload.media_type
                )
            )
        return [genai_types.Content(role="user", parts=parts)]

    async def generate(self, payload: PromptPayload) -> dict:
        logger.debug("calling %s", self.model)
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=self._contents(payload),
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    # `response_json_schema`, not `response_schema`: the latter
                    # takes an OpenAPI-flavoured subset, this one takes the JSON
                    # Schema we already build for OpenAI.
                    response_json_schema=GEMINI_RESPONSE_SCHEMA,
                    temperature=0,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    # We pass no tools and never want the SDK calling Python
                    # functions on our behalf. Left at its default the client
                    # logs a warning recommending AsyncChat on *every* call,
                    # which buries the real log lines; turning the feature off
                    # explicitly says what we mean and silences it.
                    automatic_function_calling=(
                        genai_types.AutomaticFunctionCallingConfig(disable=True)
                    ),
                ),
            )
        except httpx.TimeoutException as exc:
            raise GatewayTimeoutAPIException(
                "Summarising the document timed out. It may be too long - try a "
                "shorter document."
            ) from exc
        except genai_errors.APIError as exc:
            raise self._translate(exc) from exc
        except httpx.HTTPError as exc:
            raise BadGatewayAPIException(
                "Case summarization is unavailable. Please try again."
            ) from exc

        usage = response.usage_metadata
        if usage:
            logger.info(
                "gemini usage input=%s output=%s cached=%s total=%s",
                usage.prompt_token_count,
                usage.candidates_token_count,
                usage.cached_content_token_count or 0,
                usage.total_token_count,
            )

        return _parse(response.text, self.name)

    def _translate(self, exc: genai_errors.APIError) -> Exception:
        """Gemini reports everything as one exception family carrying an HTTP
        status, so the status is what distinguishes the cases the OpenAI SDK
        gives distinct classes to."""
        status = getattr(exc, "code", None)
        if status == 429:
            return TooManyRequestsAPIException(
                "The summarization model is busy right now. Please retry in a moment."
            )
        if status in (401, 403):
            logger.error("gemini: rejected our credentials (%s)", status)
            return ServiceUnavailableAPIException(
                "Case summarization is not configured correctly on the server."
            )
        if status == 400:
            # A 400 is our request's fault, not the caller's - a schema Gemini
            # will not accept, or a media type it cannot read. Log loudly: this
            # is the failure mode a schema change would introduce.
            logger.error("gemini: rejected our request - %s", exc)
            return BadGatewayAPIException(
                "Case summarization could not process this document. Please try "
                "again or use a different file."
            )
        logger.warning("gemini: api error %s - %s", status, exc)
        return BadGatewayAPIException(
            "Case summarization is unavailable. Please try again."
        )


# --- Selection --------------------------------------------------------------


# Which key each provider needs to be constructible. A provider whose key is
# absent is left out entirely rather than built and allowed to fail on first
# use: the service turns a missing provider into a 503 naming what *is*
# available, which is a far better signal than an auth error surfacing three
# layers down, and it costs nothing to discover at startup.
_PROVIDER_BUILDERS = {
    MODEL_GPT: ("OPENAI_API_KEY", OpenAISummaryProvider),
    MODEL_GEMINI: ("GEMINI_API_KEY", GeminiSummaryProvider),
}


def build_providers() -> dict[str, SummaryProvider]:
    """Every provider that is actually usable on this deployment.

    Both adapters are built whenever both keys are present - that is what makes
    the per-request `model` override possible. With one key, there is one
    provider and the override has nothing to switch to, which the service
    reports as a 503 naming what is available.

    `settings.SUMMARY_PROVIDER` is guaranteed to be among these: settings
    validates the name at startup and refuses to boot when the selected
    provider's key is missing. So the default is always constructible, and a
    request that names no model can always be served.
    """
    providers: dict[str, SummaryProvider] = {}
    for name, (key_setting, build) in _PROVIDER_BUILDERS.items():
        if getattr(settings, key_setting):
            providers[name] = build()
        else:
            logger.warning(
                "%s not set - the '%s' summarization provider is unavailable",
                key_setting,
                name,
            )

    logger.info(
        "summarization providers: %s (default %s)",
        ", ".join(sorted(providers)) or "none",
        settings.SUMMARY_PROVIDER,
    )
    return providers
