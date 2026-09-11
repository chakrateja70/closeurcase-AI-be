"""HTTP-level tests for /case_summarization/summarize.

The model call is stubbed out - what these cover is everything around it that
only exists at the transport layer: that the form fields reach the service as
given, that rejections come back in the shared flat error envelope rather than
FastAPI's nested {"detail": ...}, and that the rate limit actually applies.

The document now arrives as a URL rather than an upload, so the fetch is
stubbed here too; what the fetcher itself refuses is `test_url_fetch.py`.

The route is also the one place the slowapi `response` argument is load-bearing:
omitting it fails at request time, not import time, so only a real request
catches it.
"""

from typing import get_args

import pytest
from fastapi.testclient import TestClient
from test_document import PLEADING

import main
from src.api.case_summarization import SummaryModel
from src.api.deps import get_case_summarization_service
from src.core.rate_limit import limiter
from src.services.case_summarization_service import CaseSummarizationService
from src.services.summary_providers import MODEL_GEMINI, MODEL_GPT, SUPPORTED_MODELS
from src.utils.document import DocumentError

ENDPOINT = "/case_summarization/summarize"


def test_the_route_accepts_exactly_the_supported_models():
    """`SummaryModel` spells the names out because type checkers reject
    variables inside `Literal`. This keeps that copy honest: rename or add a
    provider in settings and the route would otherwise reject it silently."""
    assert set(get_args(SummaryModel)) == set(SUPPORTED_MODELS)


class StubService:
    """Stands in for the real service; records what it was handed."""

    def __init__(self, default_model: str = MODEL_GPT):
        self.received_url: str | None = None
        self.received_text: str | None = None
        self.model: str | None = None
        self.default_model = default_model

    @property
    def available_models(self) -> list[str]:
        return [MODEL_GEMINI, MODEL_GPT]

    async def summarize_document(
        self,
        data: bytes | None = None,
        *,
        url: str | None = None,
        text: str | None = None,
        model: str,
        client: str = "-",
    ) -> dict:
        self.received_url = url
        self.received_text = text
        self.model = model
        return {
            "is_valid": True,
            "parties": [{"name": "A Rao", "role": "plaintiff", "counsel": None}],
            "chronology": [
                {
                    "date": "12.03.2024",
                    "event": "Cheque dishonoured",
                    "source_snippet": "a cheque dated 12.03.2024",
                    "source_pages": [1],
                    "validation_status": "verified",
                    "page_status": "mapped",
                }
            ],
            "facts_summary": "A cheque was dishonoured.",
            "assertions": [],
            "confidence": 0.9,
            "grounding": {
                "quotes": {"verified": 1, "unverified": 0, "not_verifiable": 0},
                "pages": {"mapped": 1, "unmapped": 0, "unavailable": 0},
            },
            "model": model,
            "model_id": f"{model}-stub-1",
            "source_pages_available": True,
            "truncated": False,
            "page_count": 1,
            "fallback_response": None,
        }


@pytest.fixture
def stub():
    return StubService()


@pytest.fixture
def client(stub):
    """Rate limiting off by default - it is keyed on IP, and every test here
    shares one. The one test that needs it turns it back on."""
    main.app.dependency_overrides[get_case_summarization_service] = lambda: stub
    limiter.enabled = False
    with TestClient(main.app) as test_client:
        yield test_client
    limiter.enabled = True
    limiter.reset()
    main.app.dependency_overrides.clear()


DOC_URL = "https://example.org/petition.pdf"


class FakeFetcher:
    """Stands in for the network. Returns preset bytes, or raises the same
    `DocumentError` a real refusal would - what the real fetcher accepts and
    refuses is `test_url_fetch.py`."""

    def __init__(self, data: bytes = b"", error: str | None = None):
        self.data = data
        self.error = error
        self.requested: str | None = None

    async def fetch(self, url: str) -> bytes:
        self.requested = url
        if self.error:
            raise DocumentError(self.error)
        return self.data

    async def aclose(self) -> None:
        pass


def link(url: str = DOC_URL) -> dict:
    return {"url": url}


def form(model: str = MODEL_GPT) -> dict:
    """`model` is optional; this is for the tests that pin one."""
    return {"model": model}


def test_a_linked_document_is_summarised(client, stub):
    response = client.post(ENDPOINT, data={**form(), **link()})

    assert response.status_code == 200
    body = response.json()
    assert body["status_message"] == "success"
    assert body["is_valid"] is True
    assert body["facts_summary"] == "A cheque was dishonoured."
    assert body["chronology"][0]["event"] == "Cheque dishonoured"
    assert body["parties"][0]["name"] == "A Rao"
    # The route hands the URL through untouched; fetching is the service's job,
    # so that it happens after the provider has been resolved.
    assert stub.received_url == DOC_URL
    assert stub.received_text is None


def test_a_validation_error_uses_the_shared_envelope(client):
    """`model` is the one field the route itself validates, so an unknown value
    is the cheapest way to reach FastAPI's validation handler - which must
    flatten onto the same shape as every success."""
    response = client.post(ENDPOINT, data=form("gpt-5-turbo-ultra"))

    assert response.status_code == 400
    body = response.json()
    assert set(body) == {"status_code", "status_message", "error_message"}
    assert "detail" not in body


def test_a_refused_fetch_becomes_a_400_in_the_shared_envelope():
    """Whatever the fetcher refuses - too large, private address, 404 - arrives
    as a `DocumentError`, and every one of them has to land in the same flat
    envelope a bad upload used to."""
    response = _real_service_response(
        fetcher=FakeFetcher(error="That document is too large. The limit is 20 MB."),
        data=link(),
    )

    assert response.status_code == 400
    body = response.json()
    assert set(body) == {"status_code", "status_message", "error_message"}
    assert "too large" in body["error_message"]
    assert "detail" not in body


def test_the_url_does_not_decide_what_the_document_is():
    """The real service, so extraction actually runs. The link ends in .pdf and
    the bytes are not a PDF - what arrived is decided by the magic bytes, never
    by the URL, exactly as it was never decided by a filename."""
    response = _real_service_response(
        fetcher=FakeFetcher(data=b"just some text"), data=link()
    )

    assert response.status_code == 400
    body = response.json()
    assert set(body) == {"status_code", "status_message", "error_message"}
    assert "Unsupported file type" in body["error_message"]


def test_rate_limit_headers_and_429(stub):
    """slowapi writes X-RateLimit-* onto the `response` argument the handler
    declares; without that argument this request would raise instead."""
    main.app.dependency_overrides[get_case_summarization_service] = lambda: stub
    limiter.enabled = True
    limiter.reset()
    try:
        with TestClient(main.app) as test_client:
            first = test_client.post(ENDPOINT, data={**form(), **link()})
            assert first.status_code == 200
            assert "x-ratelimit-limit" in first.headers

            statuses = [
                test_client.post(
                    ENDPOINT, data={**form(), **link()}
                ).status_code
                for _ in range(5)
            ]
            assert 429 in statuses

            limited = test_client.post(ENDPOINT, data={**form(), **link()})
            assert limited.status_code == 429
            assert "Rate limit exceeded" in limited.json()["error_message"]
    finally:
        limiter.reset()
        main.app.dependency_overrides.clear()


def test_detect_endpoint_still_responds(client):
    """The summarization router is additive; wiring it must not disturb the
    existing one. /categories needs no model call, so it proves the routing."""
    response = client.get("/case_detection/categories")

    assert response.status_code == 200
    assert response.json()["categories"]


# --- Model selection --------------------------------------------------------


@pytest.mark.parametrize("model", [MODEL_GPT, MODEL_GEMINI])
def test_the_selected_model_reaches_the_service_and_comes_back(client, stub, model):
    response = client.post(ENDPOINT, data={**form(model), **link()})

    assert response.status_code == 200
    assert stub.model == model
    # The answer says which model produced it - with two selectable, that is
    # part of the result rather than metadata.
    assert response.json()["model"] == model
    assert response.json()["model_id"] == f"{model}-stub-1"


def test_omitting_the_model_leaves_the_choice_to_the_service(client, stub):
    """`model` is optional: a client that does not care which model runs sends
    no model, and the server's SUMMARY_PROVIDER decides. The route must pass
    the omission through as None rather than substituting a name of its own -
    the default belongs to configuration, not to this layer."""
    response = client.post(ENDPOINT, data=link())

    assert response.status_code == 200
    assert stub.model is None


def test_an_unknown_model_is_rejected_before_the_document_is_fetched(client, stub):
    """Validated by the route's Literal, so a typo costs nothing - it never
    reaches the service, so no outbound request is made on its behalf."""
    response = client.post(
        ENDPOINT, data={**form("gpt-5-turbo-ultra"), **link()}
    )

    assert response.status_code == 400
    assert stub.received_url is None


def test_models_endpoint_lists_what_is_available(client):
    """A frontend rendering a picker has to ask - Gemini is only present when a
    key is configured, so a hardcoded list would offer an option that 503s."""
    response = client.get("/case_summarization/models")

    assert response.status_code == 200
    body = response.json()
    assert body["models"] == [MODEL_GEMINI, MODEL_GPT]
    # A picker that assumed the default would preselect the wrong model on a
    # deployment configured the other way round.
    assert body["default"] == MODEL_GPT
    assert body["default"] in body["models"]


def test_the_grounding_fields_reach_the_client(client, stub):
    """The quote, the pages and the verdict are what let a lawyer check a line
    without reopening the file - so they have to survive serialisation, not
    just exist in the service."""
    response = client.post(ENDPOINT, data={**form(), **link()})

    assert response.status_code == 200
    event = response.json()["chronology"][0]
    assert event["source_snippet"] == "a cheque dated 12.03.2024"
    assert event["validation_status"] == "verified"
    assert event["page_status"] == "mapped"
    assert event["source_pages"] == [1]
    assert response.json()["grounding"] == {
        "quotes": {"verified": 1, "unverified": 0, "not_verifiable": 0},
        "pages": {"mapped": 1, "unmapped": 0, "unavailable": 0},
    }


# --- Text input --------------------------------------------------------------


def test_pasted_text_reaches_the_service(client, stub):
    response = client.post(ENDPOINT, data={**form(), "text": PLEADING})

    assert response.status_code == 200
    assert stub.received_text == PLEADING
    assert stub.received_url is None


def test_an_untouched_url_field_alongside_text_is_not_a_url(client, stub):
    """A form sends an empty string for a field the user never filled in.
    Reading that as "a URL was given" would make it impossible to paste text
    through any form carrying both fields."""
    response = client.post(ENDPOINT, data={**form(), "url": "", "text": PLEADING})

    assert response.status_code == 200
    assert stub.received_text == PLEADING
    assert stub.received_url is None


def _real_service_response(fetcher=None, **kwargs):
    """A real service behind the route, with a provider that would raise if
    reached - these rejections all happen before the model call - and a fetcher
    that never touches the network."""
    # `default_model` pinned rather than inherited: SUMMARY_PROVIDER comes from
    # the developer's own .env, and a service whose default is not among its
    # providers answers 503 before it ever reaches the rejection under test.
    service = CaseSummarizationService(
        providers={MODEL_GPT: object()},
        default_model=MODEL_GPT,
        fetcher=fetcher or FakeFetcher(),
    )
    main.app.dependency_overrides[get_case_summarization_service] = lambda: service
    limiter.enabled = False
    try:
        with TestClient(main.app) as test_client:
            return test_client.post(ENDPOINT, **kwargs)
    finally:
        limiter.enabled = True
        main.app.dependency_overrides.clear()


def test_sending_a_url_and_text_together_is_a_400():
    response = _real_service_response(data={**form(), **link(), "text": PLEADING})

    assert response.status_code == 400
    assert "not both" in response.json()["error_message"]


def test_sending_neither_a_url_nor_text_is_a_400_that_says_what_to_send():
    response = _real_service_response(data=form())

    assert response.status_code == 400
    message = response.json()["error_message"]
    assert "URL" in message and "paste" in message


def test_text_below_the_floor_is_rejected_before_the_model():
    response = _real_service_response(data={**form(), "text": "too short"})

    assert response.status_code == 400
    assert "too short" in response.json()["error_message"]
