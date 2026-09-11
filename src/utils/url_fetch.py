"""Fetching a document the caller named by URL, safely."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from urllib.parse import urlparse, urlunparse

import httpx

from src.utils.document import ACCEPTED_DESCRIPTION, MAX_FILE_BYTES, DocumentError

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = ("http", "https")

CONNECT_TIMEOUT_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 20.0
TOTAL_TIMEOUT_SECONDS = 30.0

MAX_REDIRECTS = 3

_REDIRECT_STATUSES = (301, 302, 303, 307, 308)

# Sent so an operator reading their access log can tell who we are. No caller
# data goes in it.
USER_AGENT = "closeurcase-ai-backend/1.0 (+document-fetch)"


def _unwrap(address: ipaddress._BaseAddress) -> ipaddress._BaseAddress:
    """IPv4 hiding inside IPv6."""
    mapped = getattr(address, "ipv4_mapped", None)
    return mapped if mapped is not None else address


def is_blocked_address(ip: str) -> bool:
    """Whether this resolved address is one we refuse to dial."""
    try:
        address = _unwrap(ipaddress.ip_address(ip))
    except ValueError:
        # Not an address we can reason about, so not one we will connect to.
        return True
    # `is_global` catches ranges the named checks miss, such as 100.64.0.0/10
    # (shared CGNAT space, internal on many cloud networks). The named checks
    # catch ranges `is_global` passes, such as NAT64 and IPv4-compatible IPv6.
    return (
        not address.is_global
        or address.is_loopback
        or address.is_private
        or address.is_link_local  # includes 169.254.169.254, the metadata host
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def parse_document_url(raw: str) -> tuple[str, str, int]:
    """Validate the shape of a URL and return (url, host, port).

    Shape only - the address check needs DNS and so lives in `_resolve`.
    """
    candidate = (raw or "").strip()
    if not candidate:
        raise DocumentError("No document URL was provided.")

    try:
        parsed = urlparse(candidate)
    except ValueError as exc:
        raise DocumentError("That does not look like a valid URL.") from exc

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise DocumentError(
            "The document URL must start with http:// or https://."
        )
    if not parsed.hostname:
        raise DocumentError("That does not look like a valid URL.")

    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        # urlparse defers port parsing, so a garbage port surfaces here.
        raise DocumentError("That does not look like a valid URL.") from exc

    # Drop any fragment; it is never sent and keeps the logged URL tidy.
    cleaned = urlunparse(parsed._replace(fragment=""))
    return cleaned, parsed.hostname, port


async def _resolve(host: str, port: int) -> None:
    """Refuse the hostname unless every address it resolves to is public."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError) as exc:
        raise DocumentError(
            "That URL's host could not be found. Please check the link."
        ) from exc

    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise DocumentError("That URL's host could not be found. Please check the link.")

    blocked = sorted(ip for ip in addresses if is_blocked_address(ip))
    if blocked:
        logger.warning("fetch: refused %s - resolves to %s", host, ", ".join(blocked))
        raise DocumentError(
            "That URL points to an address this server will not fetch. "
            "Please provide a publicly reachable link."
        )


class DocumentFetcher:
    """Downloads the document at a caller-supplied URL, into memory."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                TOTAL_TIMEOUT_SECONDS,
                connect=CONNECT_TIMEOUT_SECONDS,
                read=READ_TIMEOUT_SECONDS,
            ),
            # Followed by hand in `fetch` so every hop is re-validated.
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def fetch(self, raw_url: str) -> bytes:
        """The bytes at `raw_url`, or `DocumentError` explaining why not."""
        # httpx has no overall timeout: its read timeout restarts on every
        # chunk, so a server trickling bytes could hold the request open
        # indefinitely. One deadline covers DNS, every redirect hop and the body.
        try:
            async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
                return await self._download(raw_url)
        except TimeoutError as exc:
            logger.info("fetch: exceeded the %ss total deadline", TOTAL_TIMEOUT_SECONDS)
            raise DocumentError(
                "The document took too long to download. Please try a "
                "different link."
            ) from exc

    async def _download(self, raw_url: str) -> bytes:
        """Follow redirects by hand, re-checking the address at every hop."""
        url, host, port = parse_document_url(raw_url)

        for _ in range(MAX_REDIRECTS + 1):
            await _resolve(host, port)
            response = await self._open(url)

            try:
                if response.status_code in _REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location:
                        raise DocumentError(
                            "That link redirected somewhere this server could "
                            "not follow."
                        )
                    # Relative Locations are legal and common.
                    url, host, port = parse_document_url(
                        str(httpx.URL(url).join(location))
                    )
                    continue

                self._check_status(response)
                return await self._read_capped(response)
            finally:
                await response.aclose()

        raise DocumentError(
            "That link redirected too many times. Please provide a direct link "
            "to the document."
        )

    async def _open(self, url: str) -> httpx.Response:
        try:
            request = self.client.build_request("GET", url)
            return await self.client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise DocumentError(
                "The document took too long to download. Please try a "
                "different link."
            ) from exc
        except httpx.HTTPError as exc:
            logger.info("fetch: transport error - %s", exc)
            raise DocumentError(
                "The document could not be downloaded from that URL. Please "
                "check the link."
            ) from exc

    def _check_status(self, response: httpx.Response) -> None:
        """Upstream status, translated into something the caller can act on."""
        status = response.status_code
        if status in (401, 403):
            raise DocumentError(
                "That document is not publicly accessible. Please provide a "
                "link that does not require signing in."
            )
        if status == 404:
            raise DocumentError("No document was found at that URL.")
        if status >= 400:
            logger.info("fetch: upstream returned %s", status)
            raise DocumentError(
                f"The document could not be downloaded (the server answered "
                f"{status}). Please check the link."
            )

    async def _read_capped(self, response: httpx.Response) -> bytes:
        """Stream the body, refusing anything past the file-size ceiling."""
        limit_mb = MAX_FILE_BYTES // 1_048_576
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_FILE_BYTES:
            raise DocumentError(
                f"That document is too large ({int(declared) / 1_048_576:.1f} MB). "
                f"The limit is {limit_mb} MB."
            )

        chunks: list[bytes] = []
        total = 0
        try:
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_FILE_BYTES:
                    raise DocumentError(
                        f"That document is too large. The limit is {limit_mb} MB."
                    )
                chunks.append(chunk)
        except httpx.TimeoutException as exc:
            raise DocumentError(
                "The document took too long to download. Please try a "
                "different link."
            ) from exc
        except httpx.HTTPError as exc:
            logger.info("fetch: transport error mid-body - %s", exc)
            raise DocumentError(
                "The document download was interrupted. Please try again."
            ) from exc

        if not total:
            raise DocumentError(
                f"That URL returned an empty document. Please link to a "
                f"{ACCEPTED_DESCRIPTION} file."
            )
        return b"".join(chunks)
