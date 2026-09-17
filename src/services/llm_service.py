from __future__ import annotations
import base64
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
import httpx
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from src.config.settings import settings
from src.core.exceptions import (
    LLMMisconfiguredError,
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMUnexpectedResponseError,
    LLMUnreadableResponseError,
)
from src.core.tracing import callback_handler

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
        trace_label: str = "-",
    ) -> dict:
        ...

    @abstractmethod
    async def aclose(self) -> None:
        ...


def _content_blocks(parts: list[ContentPart]) -> list[dict]:
    """Build provider-neutral LangChain content blocks."""
    blocks = []
    for part in parts:
        if isinstance(part, DocumentPart):
            block_type = "image" if part.mime_type.startswith("image/") else "file"
            blocks.append(
                {
                    "type": block_type,
                    "mime_type": part.mime_type,
                    "base64": base64.b64encode(part.data).decode("ascii"),
                    "filename": part.filename,
                }
            )
        else:
            blocks.append({"type": "text", "text": part.text})
    return blocks


class _LangChainLLMClient(LLMClient):
    """Shared call, error, and logging path for LangChain chat models."""

    provider_name: str
    _structured_output_kwargs: dict = {}
    _max_tokens_kwarg: str

    def __init__(self, model, model_name: str):
        self._model = model
        self._model_name = model_name

    async def complete_json(
        self,
        *,
        instructions: str,
        parts: list[ContentPart],
        schema: dict,
        schema_name: str,
        max_output_tokens: int,
        trace_label: str = "-",
    ) -> dict:
        structured = self._model.with_structured_output(
            schema, method="json_schema", include_raw=True, **self._structured_output_kwargs
        )
        messages = [
            SystemMessage(content=instructions),
            HumanMessage(content=_content_blocks(parts)),
        ]

        handler = callback_handler()
        config = (
            {
                "callbacks": [handler],
                "metadata": {
                    "langfuse_trace_name": schema_name,
                    "langfuse_session_id": trace_label,
                    "langfuse_tags": [f"provider:{self.provider_name}"],
                },
            }
            if handler
            else {}
        )

        started = time.monotonic()
        try:
            output = await structured.ainvoke(
                messages, config=config, **{self._max_tokens_kwarg: max_output_tokens}
            )
        except (ModelAuthenticationError, ModelPermissionDeniedError) as exc:
            raise LLMMisconfiguredError() from exc
        except ModelRateLimitError as exc:
            raise LLMRateLimitedError() from exc
        except ModelTimeoutError as exc:
            raise LLMTimeoutError() from exc
        except ModelError as exc:
            raise LLMUnavailableError() from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError() from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError() from exc

        usage = output["raw"].usage_metadata or {}
        logger.info(
            "complete_json provider=%s model=%s elapsed_ms=%d "
            "input_tokens=%s output_tokens=%s total_tokens=%s",
            self.provider_name,
            self._model_name,
            int((time.monotonic() - started) * 1000),
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
        )

        if output["parsing_error"] is not None:
            raise LLMUnreadableResponseError() from output["parsing_error"]

        result = output["parsed"]
        if not isinstance(result, dict):
            raise LLMUnexpectedResponseError()
        return result


class OpenAILLMClient(_LangChainLLMClient):
    provider_name = "gpt"
    _structured_output_kwargs = {"strict": True}
    _max_tokens_kwarg = "max_tokens"

    def __init__(self, model: ChatOpenAI | None = None):
        # An injected model belongs to the caller (tests, mostly), so only a
        # model we built here is ours to close.
        self._owns_model = model is None
        super().__init__(
            model
            or ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            ),
            settings.OPENAI_MODEL,
        )

    async def aclose(self) -> None:
        if self._owns_model:
            await self._model.root_async_client.close()


class GeminiLLMClient(_LangChainLLMClient):
    provider_name = "gemini"
    _max_tokens_kwarg = "max_output_tokens"

    def __init__(self, model: ChatGoogleGenerativeAI | None = None):
        super().__init__(
            model
            or ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                api_key=settings.GEMINI_API_KEY,
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            ),
            settings.GEMINI_MODEL,
        )

    async def aclose(self) -> None:
        return None


def build_llm_client() -> LLMClient:
    """Build the client selected by ``settings.SUMMARY_PROVIDER``."""
    if settings.SUMMARY_PROVIDER == "gemini":
        return GeminiLLMClient()
    return OpenAILLMClient()