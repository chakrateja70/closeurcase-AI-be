from fastapi import HTTPException
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

# Centralized status messages
STATUS_MESSAGES = {
    HTTP_400_BAD_REQUEST: "Bad Request",
    HTTP_401_UNAUTHORIZED: "Unauthorized",
    HTTP_403_FORBIDDEN: "Forbidden",
    HTTP_404_NOT_FOUND: "Not Found",
    HTTP_409_CONFLICT: "Conflict",
    HTTP_500_INTERNAL_SERVER_ERROR: "Internal Server Error",
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