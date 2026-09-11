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

MODEL_GPT = PROVIDER_GPT
MODEL_GEMINI = PROVIDER_GEMINI
SUPPORTED_MODELS = SUPPORTED_SUMMARY_PROVIDERS

REQUEST_TIMEOUT_SECONDS = 180
MAX_RETRIES = 1

MAX_OUTPUT_TOKENS = 32768


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


def parse(text: str | None, provider: str) -> dict:
    """Shared result handling, so a malformed answer fails the same way on both
    providers rather than in two subtly different ways."""
    if not text:
        # Not truncation - each provider refuses a cut-off answer before this.
        # An empty answer is a refusal or a safety block, and blaming the
        # document's length would send the caller after the wrong fix.
        logger.warning("%s: empty response", provider)
        raise BadGatewayAPIException(
            "Case summarization returned no result for this document. Please "
            "try again, or try a different document."
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
        return parse(response.output_text, self.name)


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
                    # `response_json_schema` takes JSON Schema; `response_schema`
                    # is a different, OpenAPI-flavoured format.
                    response_json_schema=GEMINI_RESPONSE_SCHEMA,
                    temperature=0,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    # Off on purpose: Gemini 2.5 counts thinking against
                    # max_output_tokens, which would leave it a smaller answer
                    # budget than OpenAI gets from the same constant.
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
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
                "gemini usage input=%s output=%s thoughts=%s cached=%s total=%s",
                usage.prompt_token_count,
                usage.candidates_token_count,
                getattr(usage, "thoughts_token_count", None) or 0,
                usage.cached_content_token_count or 0,
                usage.total_token_count,
            )

        self._check_complete(response)
        return parse(response.text, self.name)

    def _check_complete(self, response) -> None:
        """Refuse a cut-off answer before trying to parse it."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return
        finish_reason = candidates[0].finish_reason
        if finish_reason == genai_types.FinishReason.MAX_TOKENS:
            logger.error(
                "gemini: hit the %d output-token ceiling; the answer was "
                "truncated and is unusable",
                MAX_OUTPUT_TOKENS,
            )
            raise BadGatewayAPIException(
                "The summary was cut off before it was complete. Please try a "
                "shorter document."
            )
        if finish_reason not in (None, genai_types.FinishReason.STOP):
            # SAFETY, RECITATION and the like usually arrive with no text; this
            # log line is the only record of why.
            logger.warning("gemini: finish_reason=%s", finish_reason)

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
            logger.error("gemini: rejected our request - %s", exc)
            return BadGatewayAPIException(
                "Case summarization could not process this document. Please try "
                "again or use a different file."
            )
        logger.warning("gemini: api error %s - %s", status, exc)
        return BadGatewayAPIException(
            "Case summarization is unavailable. Please try again."
        )

_PROVIDER_BUILDERS = {
    MODEL_GPT: ("OPENAI_API_KEY", OpenAISummaryProvider),
    MODEL_GEMINI: ("GEMINI_API_KEY", GeminiSummaryProvider),
}


def build_providers() -> dict[str, SummaryProvider]:
    """Every provider that is actually usable on this deployment."""
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
