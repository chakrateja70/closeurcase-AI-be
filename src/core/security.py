import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from src.config.settings import settings

docs_basic_auth = HTTPBasic()


def verify_docs_access(
    credentials: Annotated[HTTPBasicCredentials, Depends(docs_basic_auth)],
) -> str:
    """
    Gate the interactive docs behind HTTP Basic auth.

    Both comparisons always run so the response time does not leak which half
    of the credentials was wrong.
    """
    valid_username = secrets.compare_digest(
        credentials.username.encode("utf-8"), settings.DOCS_USERNAME.encode("utf-8")
    )
    valid_password = secrets.compare_digest(
        credentials.password.encode("utf-8"), settings.DOCS_PASSWORD.encode("utf-8")
    )
    if not (valid_username and valid_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid documentation credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
