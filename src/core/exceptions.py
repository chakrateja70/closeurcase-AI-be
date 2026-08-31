from fastapi import HTTPException
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
    HTTP_504_GATEWAY_TIMEOUT,
)

# Centralized status messages
STATUS_MESSAGES = {
    HTTP_400_BAD_REQUEST: "Bad Request",
    HTTP_401_UNAUTHORIZED: "Unauthorized",
    HTTP_403_FORBIDDEN: "Forbidden",
    HTTP_404_NOT_FOUND: "Not Found",
    HTTP_409_CONFLICT: "Conflict",
    HTTP_429_TOO_MANY_REQUESTS: "Too Many Requests",
    HTTP_500_INTERNAL_SERVER_ERROR: "Internal Server Error",
    HTTP_502_BAD_GATEWAY: "Bad Gateway",
    HTTP_503_SERVICE_UNAVAILABLE: "Service Unavailable",
    HTTP_504_GATEWAY_TIMEOUT: "Gateway Timeout",
}


class BaseAPIException(HTTPException):
    """Base exception class for all API exceptions."""

    def __init__(self, status_code: int, error_message: str):
        status_message = STATUS_MESSAGES.get(status_code, "Error")
        super().__init__(
            status_code=status_code,
            detail={
                "statusCode": status_code,
                "statusMessage": status_message,
                "errorMessage": error_message,
            },
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
