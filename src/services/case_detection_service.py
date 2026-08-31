"""Case detection: classifies a user query using an OpenAI model constrained
to the case taxonomy (structured output, ids validated against it again here).
"""

from __future__ import annotations

import json

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from src.config.settings import settings
from src.core.case_categories import FALLBACK_RESPONSE, OTHER_CATEGORY_ID, get_category
from src.core.exceptions import (
    BadGatewayAPIException,
    GatewayTimeoutAPIException,
    ServiceUnavailableAPIException,
    TooManyRequestsAPIException,
)
from src.prompts.case_detection_prompt import RESPONSE_SCHEMA, SCHEMA_NAME, SYSTEM_PROMPT
from src.utils.input_sanitizer import clean_text as sanitize_query, find_security_issue

MIN_QUERY_LENGTH = 10
MAX_OUTPUT_TOKENS = 3072
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 2

BLOCKED_QUERY_RESPONSE = (
    "Your query could not be processed. Please rephrase it as a plain "
    "description of your legal issue."
)


class CaseDetectionService:
    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
        )

    async def detect_case(self, query: str) -> dict:
        text = sanitize_query(query)
        if len(text) < MIN_QUERY_LENGTH:
            return self._invalid(
                "The query is too short to classify. Please describe the issue "
                "in a sentence or two."
            )

        issue = find_security_issue(text)
        if issue:
            print(f"[case_detection] blocked query reason={issue}")
            return self._invalid(BLOCKED_QUERY_RESPONSE)

        raw = await self._classify(text)
        return self._normalise(raw)

    async def _classify(self, query: str) -> dict:
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
            print(
                f"[case_detection] input_tokens={response.usage.input_tokens} "
                f"output_tokens={response.usage.output_tokens} "
                f"cached_tokens={cached_tokens} "
                f"total_tokens={response.usage.total_tokens}"
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

        # Schema-constrained, so a miss here means the model claimed valid
        # without picking a category. Treat it as the catch-all bucket.
        category = get_category(raw.get("primary_case_category_id")) or get_category(
            OTHER_CATEGORY_ID
        )

        # Secondary is a second, distinct top-level domain - never the same
        # as the primary, and only present when the model actually set one.
        secondary_id = raw.get("secondary_case_category_id")
        secondary = (
            get_category(secondary_id) if secondary_id != category["id"] else None
        )

        return {
            "is_valid": True,
            "primary_case_category": category["title"],
            "primary_case_category_id": category["id"],
            "secondary_case_category": secondary["title"] if secondary else None,
            "secondary_case_category_id": secondary["id"] if secondary else None,
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
            "primary_case_category": None,
            "primary_case_category_id": None,
            "secondary_case_category": None,
            "secondary_case_category_id": None,
            "confidence": 0.0,
            "summary": None,
            "fallback_response": fallback_response,
        }
