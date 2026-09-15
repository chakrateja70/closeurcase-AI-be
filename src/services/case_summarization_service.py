"""Case summarization: given either case document URLs (PDFs, handed to the model directly
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import httpx
from urllib.parse import unquote, urlparse
from src.core.case_input import CaseInput, CaseInputType
from src.core.exceptions import (
    DocumentFetchTimedOutError,
    DocumentHostNotAllowedError,
    DocumentNotPdfError,
    DocumentTooLargeError,
    DocumentUnreachableError,
)
from src.core.summary_provider import SummaryProvider, available_summary_providers
from src.prompts.case_summarization_prompt import (
    RESPONSE_SCHEMA,
    SCHEMA_NAME,
    SYSTEM_PROMPT,
)
from src.services.llm_service import (
    DocumentPart,
    GeminiLLMClient,
    LLMClient,
    OpenAILLMClient,
    TextPart,
)
from src.utils.helper import clean_text, find_security_issue, flatten

MAX_OUTPUT_TOKENS = 2048

DOCUMENT_FETCH_TIMEOUT_SECONDS = 30
MAX_DOCUMENT_BYTES = 15 * 1024 * 1024  # 15MB per document
_PDF_MAGIC = b"%PDF-"

logger = logging.getLogger(__name__)

BLOCKED_TEXT_BRIEF = (
    "This text could not be processed. Please provide a plain description of "
    "the case, or a document to summarize."
)


def _build_clients() -> dict[SummaryProvider, LLMClient]:
    clients: dict[SummaryProvider, LLMClient] = {SummaryProvider.GPT: OpenAILLMClient()}
    if SummaryProvider.GEMINI in available_summary_providers():
        clients[SummaryProvider.GEMINI] = GeminiLLMClient()
    return clients


def _filename_from_url(url: str) -> str:
    name = unquote(urlparse(url).path.rsplit("/", 1)[-1])
    return name or "document.pdf"


async def _fetch_pdf(url: str) -> bytes:
    _guard_against_private_host(url)

    try:
        async with httpx.AsyncClient(
            timeout=DOCUMENT_FETCH_TIMEOUT_SECONDS, follow_redirects=True
        ) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                _guard_declared_size(url, response.headers.get("content-length"))

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_DOCUMENT_BYTES:
                        raise DocumentTooLargeError(url, MAX_DOCUMENT_BYTES)
                    chunks.append(chunk)
    except httpx.TimeoutException as exc:
        raise DocumentFetchTimedOutError(url) from exc
    except httpx.HTTPError as exc:
        raise DocumentUnreachableError(url) from exc

    data = b"".join(chunks)
    if not data.startswith(_PDF_MAGIC):
        raise DocumentNotPdfError(url)
    return data


def _guard_declared_size(url: str, content_length: str | None) -> None:
    if content_length is None:
        return
    try:
        declared = int(content_length)
    except ValueError:
        return
    if declared > MAX_DOCUMENT_BYTES:
        raise DocumentTooLargeError(url, MAX_DOCUMENT_BYTES)


def _guard_against_private_host(url: str) -> None:
    """Resolve the host and reject anything not a public address, before
    connecting - otherwise a document_url could point at the server's own
    cloud metadata endpoint or an internal service. This is a best-effort,
    point-in-time check (the actual connection is a separate DNS lookup), not
    a substitute for network-level egress controls."""
    host = urlparse(url).hostname
    if not host:
        raise DocumentHostNotAllowedError(url)

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise DocumentUnreachableError(url) from exc

    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
        ):
            raise DocumentHostNotAllowedError(url)


class CaseSummarizationService:
    def __init__(self, clients: dict[SummaryProvider, LLMClient] | None = None):
        # Injected clients belong to the caller (tests, mostly), so only
        # clients we built here are ours to close.
        self._owns_clients = clients is None
        self._clients: dict[SummaryProvider, LLMClient] = clients or _build_clients()

    async def aclose(self) -> None:
        if self._owns_clients:
            for client in self._clients.values():
                await client.aclose()

    async def summarize(
        self, case_input: CaseInput, provider: SummaryProvider, client: str = "-"
    ) -> dict:
        """`client` is a caller label used only for logging."""
        if case_input.type is CaseInputType.TEXT:
            cleaned = clean_text(case_input.text)
            issue = find_security_issue(cleaned)
            if issue:
                logger.warning("[%s] summarize: BLOCKED reason=%s", client, issue)
                return {"brief": BLOCKED_TEXT_BRIEF, "key_points": []}
            parts = [TextPart(text=flatten(cleaned))]
        else:
            logger.info(
                "[%s] summarize: fetching %d document(s)", client, len(case_input.urls)
            )
            documents = await asyncio.gather(
                *(_fetch_pdf(url) for url in case_input.urls)
            )
            parts = [
                DocumentPart(data=data, filename=_filename_from_url(url))
                for url, data in zip(case_input.urls, documents)
            ]

        raw = await self._clients[provider].complete_json(
            instructions=SYSTEM_PROMPT,
            parts=parts,
            schema=RESPONSE_SCHEMA,
            schema_name=SCHEMA_NAME,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        return self._normalise(raw)

    def _normalise(self, raw: dict) -> dict:
        brief = raw.get("brief")
        key_points = raw.get("key_points")
        return {
            "brief": brief if isinstance(brief, str) else "",
            "key_points": (
                [point for point in key_points if isinstance(point, str)]
                if isinstance(key_points, list)
                else []
            ),
        }
