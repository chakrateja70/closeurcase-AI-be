from __future__ import annotations
from dataclasses import dataclass, field
from urllib.parse import urlparse
from src.core.exceptions import (
    InvalidDocumentUrlHostError,
    InvalidDocumentUrlSchemeError,
    InvalidDocumentUrlTypeError,
)

# http(s) only - a file://, ftp://, or data: URL here would either read the
# server's own filesystem or embed a payload directly, and neither is a "case
# document to fetch".
_ALLOWED_URL_SCHEMES = {"http", "https"}
_ALLOWED_DOCUMENT_SUFFIXES = (".pdf", ".jpg", ".jpeg", ".png")

@dataclass(frozen=True)
class CaseInput:
    urls: list[str] = field(default_factory=list)
    text: str = ""

def resolve_case_input(document_urls, case_text: str | None) -> CaseInput:
    """Validate and classify the provided case input."""
    urls: list[str] = []
    if document_urls:
        urls = [str(url) for url in document_urls]
        for url in urls:
            validate_document_url(url)
    return CaseInput(urls=urls, text=(case_text or "").strip())

def validate_document_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        raise InvalidDocumentUrlSchemeError(url)
    if not parsed.netloc:
        raise InvalidDocumentUrlHostError(url)
    if not parsed.path.lower().endswith(_ALLOWED_DOCUMENT_SUFFIXES):
        raise InvalidDocumentUrlTypeError(url)