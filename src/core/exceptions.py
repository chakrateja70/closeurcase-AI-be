import logging

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_502_BAD_GATEWAY,
    HTTP_503_SERVICE_UNAVAILABLE,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_504_GATEWAY_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Centralized status messages
STATUS_MESSAGES = {
    HTTP_400_BAD_REQUEST: "Bad Request",
    HTTP_401_UNAUTHORIZED: "Unauthorized",
    HTTP_403_FORBIDDEN: "Forbidden",
    HTTP_404_NOT_FOUND: "Not Found",
    HTTP_409_CONFLICT: "Conflict",
    HTTP_422_UNPROCESSABLE_CONTENT: "Unprocessable Content",
    HTTP_429_TOO_MANY_REQUESTS: "Too Many Requests",
    HTTP_500_INTERNAL_SERVER_ERROR: "Internal Server Error",
    HTTP_502_BAD_GATEWAY: "Bad Gateway",
    HTTP_503_SERVICE_UNAVAILABLE: "Service Unavailable",
    HTTP_504_GATEWAY_TIMEOUT: "Gateway Timeout",
}


def error_envelope(
    status_code: int, error_message: str, status_message: str | None = None
) -> dict:
    """The single error body shape, matching the success envelope's flat
    snake_case keys so a client parses one shape rather than two."""
    return {
        "status_code": status_code,
        "status_message": status_message or STATUS_MESSAGES.get(status_code, "Error"),
        "error_message": error_message,
    }


class BaseAPIException(HTTPException):
    """Base exception class for all API exceptions."""

    def __init__(self, status_code: int, error_message: str):
        status_message = STATUS_MESSAGES.get(status_code, "Error")
        super().__init__(
            status_code=status_code,
            detail=error_envelope(status_code, error_message, status_message),
        )


class BadRequestAPIException(BaseAPIException):
    def __init__(self, error_message: str = "Bad request"):
        super().__init__(HTTP_400_BAD_REQUEST, error_message)


class UnauthorizedAPIException(BaseAPIException):
    def __init__(self, error_message: str = "Unauthorized"):
        super().__init__(HTTP_401_UNAUTHORIZED, error_message)


class TooManyRequestsAPIException(BaseAPIException):
    """Upstream provider rate limited us."""

    def __init__(self, error_message: str = "Too many requests, please retry shortly"):
        super().__init__(HTTP_429_TOO_MANY_REQUESTS, error_message)


class GatewayTimeoutAPIException(BaseAPIException):
    """Upstream provider did not answer in time."""

    def __init__(self, error_message: str = "Upstream service timed out"):
        super().__init__(HTTP_504_GATEWAY_TIMEOUT, error_message)


class BadGatewayAPIException(BaseAPIException):
    """Upstream provider answered, but not with something usable."""

    def __init__(self, error_message: str = "Invalid response from upstream service"):
        super().__init__(HTTP_502_BAD_GATEWAY, error_message)


class ServiceUnavailableAPIException(BaseAPIException):
    """Dependency is not configured or is down."""

    def __init__(self, error_message: str = "Service temporarily unavailable"):
        super().__init__(HTTP_503_SERVICE_UNAVAILABLE, error_message)


class InvalidCaseInputError(ValueError):
    """Base class for case-input validation failures."""


class MissingCaseInputError(InvalidCaseInputError):
    def __init__(self):
        super().__init__("Provide either document_urls or case_text.")


class ConflictingCaseInputError(InvalidCaseInputError):
    def __init__(self):
        super().__init__("Provide only one of document_urls or case_text, not both.")


class InvalidDocumentUrlSchemeError(InvalidCaseInputError):
    def __init__(self, url: str):
        super().__init__(f"document url must be http or https: {url}")


class InvalidDocumentUrlHostError(InvalidCaseInputError):
    def __init__(self, url: str):
        super().__init__(f"document url must include a host: {url}")


class InvalidDocumentUrlTypeError(InvalidCaseInputError):
    def __init__(self, url: str):
        super().__init__(f"document url must point to a .pdf file: {url}")


# --- Document fetching ---------------------------------------------------
#
# Raised by src/services/case_summarization_service.py while downloading a
# document_url. These are the caller's fault (a broken/oversized/blocked
# link), not ours, hence BadRequestAPIException rather than a 502/504 - the
# upstream that failed is a URL the client supplied, not our own dependency.


class DocumentUnreachableError(BadRequestAPIException):
    def __init__(self, url: str):
        super().__init__(f"Could not reach document url: {url}")


class DocumentFetchTimedOutError(BadRequestAPIException):
    def __init__(self, url: str):
        super().__init__(f"Timed out fetching document url: {url}")


class DocumentTooLargeError(BadRequestAPIException):
    def __init__(self, url: str, max_bytes: int):
        super().__init__(
            f"document at {url} exceeds the {max_bytes // (1024 * 1024)}MB limit"
        )


class DocumentNotPdfError(BadRequestAPIException):
    def __init__(self, url: str):
        super().__init__(f"document at {url} is not a PDF")


class DocumentHostNotAllowedError(BadRequestAPIException):
    def __init__(self, url: str):
        super().__init__(f"document url is not allowed: {url}")


# --- LLM pipeline -------------------------------------------------------
#
# Raised by src/services/llm_service.py regardless of which provider (OpenAI
# or Gemini) handled the call, so callers see one failure shape either way.


class LLMTimeoutError(GatewayTimeoutAPIException):
    def __init__(self):
        super().__init__("The AI service timed out. Please try again.")


class LLMRateLimitedError(TooManyRequestsAPIException):
    def __init__(self):
        super().__init__("The AI service is busy right now. Please retry in a moment.")


class LLMMisconfiguredError(ServiceUnavailableAPIException):
    def __init__(self):
        super().__init__("The AI service is not configured correctly on the server.")


class LLMUnavailableError(BadGatewayAPIException):
    def __init__(self):
        super().__init__("The AI service is unavailable. Please try again.")


class LLMIncompleteResponseError(BadGatewayAPIException):
    def __init__(self):
        super().__init__("The AI service returned an incomplete result. Please try again.")


class LLMUnreadableResponseError(BadGatewayAPIException):
    def __init__(self):
        super().__init__("The AI service returned an unreadable result. Please try again.")


class LLMUnexpectedResponseError(BadGatewayAPIException):
    def __init__(self):
        super().__init__("The AI service returned an unexpected result. Please try again.")


class UnavailableSummaryProviderError(ServiceUnavailableAPIException):
    """Requested (or default) provider has no API key configured."""

    def __init__(self, requested: str, available: list[str]):
        super().__init__(
            f"'{requested}' is not available - no key is configured for it. "
            f"Available: {', '.join(available)}."
        )


async def base_api_exception_handler(
    _: Request, exc: BaseAPIException
) -> JSONResponse:
    """Our own exceptions already carry a built envelope as their detail."""
    return JSONResponse(
        status_code=exc.status_code, content=exc.detail, headers=exc.headers
    )


async def http_exception_handler(
    _: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Everything raising a plain HTTPException - the docs auth 401, 404s on
    unknown paths, 405s. `headers` matters here: dropping it would strip the
    WWW-Authenticate challenge and break the docs login prompt."""
    detail = exc.detail
    message = detail if isinstance(detail, str) else str(detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_envelope(exc.status_code, message),
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    """Pydantic request-body rejections (an empty query, one over the length
    cap). The per-field errors are summarised into one message; the raw list
    is dropped because it echoes the submitted body back to the caller.

    A body that isn't valid JSON at all ("type": "json_invalid") reports its
    `loc` as the byte offset where parsing failed (e.g. `("body", 21)`) -
    meaningless to a caller, so it gets a fixed message instead of that
    offset threaded through the usual "loc: msg" format.
    """
    problems = "; ".join(
        "Invalid JSON format. Please check your syntax."
        if err.get("type") == "json_invalid"
        else f"{'.'.join(str(part) for part in err['loc'][1:]) or 'body'}: {err['msg']}"
        for err in exc.errors()
    )
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content=error_envelope(
            HTTP_400_BAD_REQUEST, problems or "Request validation failed"
        ),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort. Logs the traceback server-side and returns a generic body -
    an unexpected exception's message can carry internals worth not leaking."""
    logger.exception(
        "unhandled error on %s %s", request.method, request.url.path, exc_info=exc
    )
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_envelope(
            HTTP_500_INTERNAL_SERVER_ERROR, "An unexpected error occurred"
        ),
    )


async def rate_limit_exception_handler(_: Request, exc) -> JSONResponse:
    """slowapi's RateLimitExceeded, in our envelope instead of its own."""
    return JSONResponse(
        status_code=HTTP_429_TOO_MANY_REQUESTS,
        content=error_envelope(
            HTTP_429_TOO_MANY_REQUESTS,
            f"Rate limit exceeded: {exc.detail}. Please retry shortly.",
        ),
    )
