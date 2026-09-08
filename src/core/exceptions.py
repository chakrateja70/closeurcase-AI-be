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


# --- Handlers ---------------------------------------------------------------
#
# Registered in main.py. Without these, FastAPI wraps every error body in
# {"detail": ...} - so a client would parse a flat snake_case object on
# success and a nested one on failure. These unwrap all four error paths onto
# the same envelope the success responses use.


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
    is dropped because it echoes the submitted body back to the caller."""
    problems = "; ".join(
        f"{'.'.join(str(part) for part in err['loc'][1:]) or 'body'}: {err['msg']}"
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
