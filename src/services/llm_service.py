#Provider-agnostic structured-output LLM client.

from __future__ import annotations

import base64
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
from src.config.settings import settings
from src.core.exceptions import (
    LLMIncompleteResponseError,
    LLMMisconfiguredError,
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMUnexpectedResponseError,
    LLMUnreadableResponseError,
)

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 2


@dataclass(frozen=True)
class TextPart:
    text: str

@dataclass(frozen=True)
class DocumentPart:
    data: bytes
    filename: str
    mime_type: str = "application/pdf"

ContentPart = TextPart | DocumentPart

class LLMClient(ABC):
    """One structured-output call with instructions, content parts, and a JSON schema.
    Provider-specific transport and error handling live in subclasses.
    """

    @abstractmethod
    async def complete_json(
        self,
        *,
        instructions: str,
        parts: list[ContentPart],
        schema: dict,
        schema_name: str,
        max_output_tokens: int,
    ) -> dict:
        ...

    @abstractmethod
    async def aclose(self) -> None:
        ...


class OpenAILLMClient(LLMClient):
    def __init__(self, client: AsyncOpenAI | None = None):
        # An injected client belongs to the caller (tests, mostly), so only a
        # client we built here is ours to close.
        self._owns_client = client is None
        self.client = client or AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.close()

    async def complete_json(
        self,
        *,
        instructions: str,
        parts: list[ContentPart],
        schema: dict,
        schema_name: str,
        max_output_tokens: int,
    ) -> dict:
        logger.debug("calling %s", settings.OPENAI_MODEL)
        try:
            response = await self.client.responses.create(
                model=settings.OPENAI_MODEL,
                instructions=instructions,
                input=[{"role": "user", "content": [_openai_content(p) for p in parts]}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": _openai_strict_schema(schema),
                        "strict": True,
                    }
                },
                max_output_tokens=max_output_tokens,
                store=False,
            )
        except APITimeoutError as exc:
            raise LLMTimeoutError() from exc
        except RateLimitError as exc:
            raise LLMRateLimitedError() from exc
        except AuthenticationError as exc:
            raise LLMMisconfiguredError() from exc
        except (APIConnectionError, APIStatusError) as exc:
            raise LLMUnavailableError() from exc

        if response.usage:
            details = response.usage.input_tokens_details
            cached_tokens = details.cached_tokens if details else 0
            logger.info(
                "complete_json usage input=%s output=%s cached=%s total=%s",
                response.usage.input_tokens,
                response.usage.output_tokens,
                cached_tokens,
                response.usage.total_tokens,
            )

        if response.status == "incomplete" or not response.output_text:
            raise LLMIncompleteResponseError()

        return _parse_json(response.output_text)


class GeminiLLMClient(LLMClient):
    def __init__(self, client: genai.Client | None = None):
        self.client = client or genai.Client(api_key=settings.GEMINI_API_KEY)

    async def aclose(self) -> None:
        # google-genai has no explicit close - it wraps httpx clients it
        # owns internally and tears them down with the process, unlike
        # AsyncOpenAI's connection pool.
        return None

    async def complete_json(
        self,
        *,
        instructions: str,
        parts: list[ContentPart],
        schema: dict,
        schema_name: str,
        max_output_tokens: int,
    ) -> dict:
        logger.debug("calling %s", settings.GEMINI_MODEL)
        try:
            response = await self.client.aio.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=[_gemini_part(p) for p in parts],
                config=genai_types.GenerateContentConfig(
                    system_instruction=instructions,
                    response_mime_type="application/json",
                    response_schema=schema,
                    max_output_tokens=max_output_tokens,
                ),
            )
        except genai_errors.ClientError as exc:
            code = getattr(exc, "code", None)
            if code == 429:
                raise LLMRateLimitedError() from exc
            if code in (401, 403):
                raise LLMMisconfiguredError() from exc
            raise LLMUnavailableError() from exc
        except genai_errors.ServerError as exc:
            raise LLMUnavailableError() from exc
        except TimeoutError as exc:
            raise LLMTimeoutError() from exc

        if response.usage_metadata:
            logger.info(
                "complete_json usage input=%s output=%s total=%s",
                response.usage_metadata.prompt_token_count,
                response.usage_metadata.candidates_token_count,
                response.usage_metadata.total_token_count,
            )

        if not response.text:
            raise LLMIncompleteResponseError()

        return _parse_json(response.text)


def _openai_strict_schema(schema: dict) -> dict:
    """Inject `additionalProperties: false` recursively for OpenAI strict schema mode."""
    if schema.get("type") == "object":
        schema = {**schema, "additionalProperties": False}
        if "properties" in schema:
            schema["properties"] = {
                key: _openai_strict_schema(value)
                for key, value in schema["properties"].items()
            }
    elif schema.get("type") == "array" and "items" in schema:
        schema = {**schema, "items": _openai_strict_schema(schema["items"])}
    return schema


def _openai_content(part: ContentPart) -> dict:
    if isinstance(part, DocumentPart):
        encoded = base64.b64encode(part.data).decode("ascii")
        return {
            "type": "input_file",
            "filename": part.filename,
            "file_data": f"data:{part.mime_type};base64,{encoded}",
        }
    return {"type": "input_text", "text": part.text}


def _gemini_part(part: ContentPart) -> genai_types.Part:
    if isinstance(part, DocumentPart):
        return genai_types.Part.from_bytes(data=part.data, mime_type=part.mime_type)
    return genai_types.Part.from_text(text=part.text)


def _parse_json(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMUnreadableResponseError() from exc
    if not isinstance(parsed, dict):
        raise LLMUnexpectedResponseError()
    return parsed


def build_llm_client() -> LLMClient:
    """Picks the backend from `settings.SUMMARY_PROVIDER`. Settings validates
    that value at startup ("gpt" or "gemini" only), so anything else here
    would already have failed the app before this is ever called."""
    if settings.SUMMARY_PROVIDER == "gemini":
        return GeminiLLMClient()
    return OpenAILLMClient()
