"""Case detection: classifies a user query using an OpenAI model constrained
to the case taxonomy (structured output, ids validated against it again here).

The model returns case-type ids only. Each id is then expanded from the
taxonomy in `_expand` into the case type, its parent category, and the legal
services offered under it - so none of those three can contradict each other,
and the model spends no tokens on the two it does not decide.
"""

from __future__ import annotations

import json
import logging

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from src.config.settings import settings
from src.core.case_categories import (
    FALLBACK_RESPONSE,
    expand_case_type,
    resolve_case_types,
)
from src.core.exceptions import (
    BadGatewayAPIException,
    GatewayTimeoutAPIException,
    ServiceUnavailableAPIException,
    TooManyRequestsAPIException,
)
from src.prompts.case_detection_prompt import RESPONSE_SCHEMA, SCHEMA_NAME, SYSTEM_PROMPT
from src.utils.helper import clean_text, find_security_issue, flatten

MIN_QUERY_LENGTH = 5
MAX_OUTPUT_TOKENS = 3072
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 2

logger = logging.getLogger(__name__)

# Queries run to 600 characters; log lines should stay one line.
LOG_QUERY_PREVIEW = 120


def _preview(text: str) -> str:
    return text if len(text) <= LOG_QUERY_PREVIEW else text[:LOG_QUERY_PREVIEW] + "..."

BLOCKED_QUERY_RESPONSE = (
    "Your query could not be processed. Please rephrase it as a plain "
    "description of your legal issue."
)


class CaseDetectionService:
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
        """Release the HTTP connection pool. Called from the app lifespan;
        without it every reload leaks the pool."""
        if self._owns_client:
            await self.client.close()

    async def detect_case(self, query: str, client: str = "-") -> dict:
        """`client` is a caller label used only for logging - the IP and how
        many requests it has made - so a line can be traced to who sent it."""
        text = clean_text(query)
        logger.info(
            "[%s] detect: query=%r (%d chars)", client, _preview(text), len(text)
        )

        if len(text) < MIN_QUERY_LENGTH:
            logger.info("[%s] detect: rejected, too short (%d chars)", client, len(text))
            return self._invalid(
                "The query is too short to classify. Please describe the issue "
                "in a sentence or two."
            )

        issue = find_security_issue(text)
        if issue:
            logger.warning(
                "[%s] detect: BLOCKED reason=%s query=%r",
                client,
                issue,
                _preview(text),
            )
            return self._invalid(BLOCKED_QUERY_RESPONSE)

        raw = await self._classify(flatten(text))
        result = self._normalise(raw)

        if result["is_valid"]:
            logger.info(
                "[%s] detect: primary=%s (%s) secondary=%s confidence=%s services=%d",
                client,
                result["primary_case_type_id"],
                result["primary_case_category_id"],
                result["secondary_case_type_id"],
                result["confidence"],
                len(result["primary_legal_services"]),
            )
        else:
            logger.info("[%s] detect: model returned not a legal issue", client)
        return result

    async def _classify(self, query: str) -> dict:
        logger.debug("calling %s", settings.OPENAI_MODEL)
        try:
            response = await self.client.responses.create(
                model=settings.OPENAI_MODEL,
                instructions=SYSTEM_PROMPT,
                input=[
                    {"role": "user", "content": [{"type": "input_text", "text": query}]}
                ],
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
                "Case detection timed out. Please try again."
            ) from exc
        except RateLimitError as exc:
            raise TooManyRequestsAPIException(
                "Case detection is busy right now. Please retry in a moment."
            ) from exc
        except AuthenticationError as exc:
            raise ServiceUnavailableAPIException(
                "Case detection is not configured correctly on the server."
            ) from exc
        except (APIConnectionError, APIStatusError) as exc:
            raise BadGatewayAPIException(
                "Case detection is unavailable. Please try again."
            ) from exc

        if response.usage:
            details = response.usage.input_tokens_details
            cached_tokens = details.cached_tokens if details else 0
            logger.info(
                "classify usage input=%s output=%s cached=%s total=%s",
                response.usage.input_tokens,
                response.usage.output_tokens,
                cached_tokens,
                response.usage.total_tokens,
            )

        if response.status == "incomplete" or not response.output_text:
            raise BadGatewayAPIException(
                "Case detection returned an incomplete result. Please try again."
            )

        try:
            parsed = json.loads(response.output_text)
        except json.JSONDecodeError as exc:
            raise BadGatewayAPIException(
                "Case detection returned an unreadable result. Please try again."
            ) from exc

        if not isinstance(parsed, dict):
            raise BadGatewayAPIException(
                "Case detection returned an unexpected result. Please try again."
            )
        return parsed

    def _normalise(self, raw: dict) -> dict:
        if not raw.get("is_valid"):
            return self._invalid(raw.get("fallback_response") or FALLBACK_RESPONSE)

        # Secondary is a second, distinct matter - never the same case type as
        # the primary, though it may sit under the same category (e.g. Divorce
        # alongside Child Custody). Only present when the model actually set one.
        primary, secondary = resolve_case_types(
            raw.get("primary_case_type_id"), raw.get("secondary_case_type_id")
        )

        return {
            "is_valid": True,
            **expand_case_type(primary, "primary"),
            **expand_case_type(secondary, "secondary"),
            "confidence": self._confidence(raw.get("confidence")),
            "summary": self._clean_text(raw.get("summary")),
            "fallback_response": None,
        }

    def _confidence(self, value) -> float:
        try:
            return round(min(max(float(value), 0.0), 1.0), 2)
        except (TypeError, ValueError):
            return 0.0

    def _clean_text(self, value) -> str | None:
        if not isinstance(value, str):
            return None
        return " ".join(value.split()) or None

    def _invalid(self, fallback_response: str) -> dict:
        return {
            "is_valid": False,
            **expand_case_type(None, "primary"),
            **expand_case_type(None, "secondary"),
            "confidence": 0.0,
            "summary": None,
            "fallback_response": fallback_response,
        }
