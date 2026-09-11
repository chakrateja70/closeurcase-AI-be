"""The document fetcher, and what it refuses to dial.

This module exists because the summarization endpoint takes a link, which makes
this server issue outbound requests to addresses a stranger picked. Most of
what follows is not about downloading a PDF - it is about the requests that
must never leave the process.

Nothing here touches the network. Address checks are pure functions over
resolved IPs, DNS is stubbed, and the transport is an `httpx.MockTransport`, so
a regression that would have made a real outbound request instead fails here.
"""

import asyncio
import time

import httpx
import pytest

from src.utils import url_fetch
from src.utils.document import MAX_FILE_BYTES, DocumentError
from src.utils.url_fetch import (
    MAX_REDIRECTS,
    DocumentFetcher,
    is_blocked_address,
    parse_document_url,
)

PDF = b"%PDF-1.4 a real enough document"


# --- Addresses we refuse to dial ---------------------------------------------


@pytest.mark.parametrize(
    "ip, why",
    [
        ("127.0.0.1", "loopback - reaches this very API, which would recurse"),
        ("127.1.1.1", "the whole 127/8 block is loopback, not just .0.1"),
        ("0.0.0.0", "unspecified"),
        ("10.0.0.5", "private"),
        ("172.16.0.1", "private"),
        ("192.168.1.1", "private"),
        ("169.254.169.254", "cloud instance credentials - the prize target"),
        ("169.254.1.1", "link-local"),
        ("224.0.0.1", "multicast"),
        ("240.0.0.1", "reserved"),
        ("::1", "IPv6 loopback"),
        ("fd00::1", "IPv6 private"),
        ("fe80::1", "IPv6 link-local"),
        ("::ffff:127.0.0.1", "IPv4 loopback wearing an IPv6 costume"),
        ("::ffff:169.254.169.254", "the metadata host, same costume"),
        ("100.64.0.1", "shared CGNAT space - internal on many cloud networks"),
        ("64:ff9b::a9fe:a9fe", "the metadata host through a NAT64 prefix"),
        ("::127.0.0.1", "loopback in deprecated IPv4-compatible form"),
        ("not-an-address", "unparseable, so not something we will connect to"),
    ],
)
def test_blocked_addresses(ip, why):
    assert is_blocked_address(ip) is True, why


@pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700::1"])
def test_public_addresses_are_allowed(ip):
    """Only non-public space is refused - blocking real hosts would make the
    feature useless."""
    assert is_blocked_address(ip) is False


# --- URL shape ---------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "file://C:/Windows/win.ini",
        "gopher://127.0.0.1:11211/_stats",
        "ftp://example.org/doc.pdf",
        "data:application/pdf;base64,AAAA",
        "javascript:alert(1)",
        "//example.org/doc.pdf",
        "example.org/doc.pdf",
    ],
)
def test_only_http_and_https_are_dialled(url):
    """`file://` reads local disk and `gopher://` smuggles arbitrary bytes into
    whatever is listening. Neither is a document host."""
    with pytest.raises(DocumentError):
        parse_document_url(url)


@pytest.mark.parametrize("url", ["", "   ", None])
def test_an_empty_url_is_refused(url):
    with pytest.raises(DocumentError, match="No document URL"):
        parse_document_url(url)


def test_a_fragment_is_dropped():
    """Never sent on the wire anyway; dropping it keeps the logged URL honest
    about what was actually requested."""
    url, host, port = parse_document_url("https://example.org/a.pdf#page=3")

    assert url == "https://example.org/a.pdf"
    assert (host, port) == ("example.org", 443)


@pytest.mark.parametrize(
    "url, expected_port",
    [
        ("https://example.org/a.pdf", 443),
        ("http://example.org/a.pdf", 80),
        ("https://example.org:8443/a.pdf", 8443),
    ],
)
def test_the_port_is_derived_from_the_scheme(url, expected_port):
    """The port matters because it is what DNS is resolved for - and so what
    the address check is performed against."""
    assert parse_document_url(url)[2] == expected_port


def test_a_malformed_port_is_refused():
    with pytest.raises(DocumentError):
        parse_document_url("https://example.org:notaport/a.pdf")


# --- Fetching, with DNS and the transport stubbed ----------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


def resolve_to(monkeypatch, *addresses):
    """Pin what every hostname resolves to, so no test needs real DNS."""

    async def fake_getaddrinfo(host, port, **kwargs):
        return [(None, None, None, "", (ip, port)) for ip in addresses]

    import asyncio

    class Loop:
        getaddrinfo = staticmethod(fake_getaddrinfo)

    monkeypatch.setattr(
        asyncio, "get_running_loop", lambda: Loop(), raising=True
    )


def fetcher_returning(*responses):
    """A fetcher whose transport replays the given responses in order."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    )
    return DocumentFetcher(client=client), calls


@pytest.mark.anyio
async def test_a_document_is_downloaded(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    fetcher, calls = fetcher_returning(httpx.Response(200, content=PDF))

    assert await fetcher.fetch("https://example.org/a.pdf") == PDF
    assert str(calls[0].url) == "https://example.org/a.pdf"


@pytest.mark.anyio
async def test_a_private_address_is_never_dialled(monkeypatch):
    """The check must happen before the request, not after: a connection to
    127.0.0.1 that is then discarded has already done the damage."""
    resolve_to(monkeypatch, "127.0.0.1")
    fetcher, calls = fetcher_returning(httpx.Response(200, content=PDF))

    with pytest.raises(DocumentError, match="will not fetch"):
        await fetcher.fetch("http://sneaky.example/a.pdf")

    assert calls == []


@pytest.mark.anyio
async def test_the_metadata_host_is_refused(monkeypatch):
    """169.254.169.254 hands out this instance's cloud credentials to anything
    that asks. It is the single highest-value target of an SSRF bug."""
    resolve_to(monkeypatch, "169.254.169.254")
    fetcher, calls = fetcher_returning(httpx.Response(200, content=PDF))

    with pytest.raises(DocumentError, match="will not fetch"):
        await fetcher.fetch("http://169.254.169.254/latest/meta-data/")

    assert calls == []


@pytest.mark.anyio
async def test_one_private_address_poisons_the_whole_hostname(monkeypatch):
    """A name answering with one public and one private address is refused
    outright. Allowing it would let an attacker choose which we connect to."""
    resolve_to(monkeypatch, "93.184.216.34", "10.0.0.5")
    fetcher, calls = fetcher_returning(httpx.Response(200, content=PDF))

    with pytest.raises(DocumentError, match="will not fetch"):
        await fetcher.fetch("https://split-horizon.example/a.pdf")

    assert calls == []


@pytest.mark.anyio
async def test_the_refusal_never_names_the_internal_address(monkeypatch):
    """The caller learns their link is not allowed, never which internal host
    they nearly reached - that answer is itself a scan result."""
    resolve_to(monkeypatch, "10.1.2.3")
    fetcher, _ = fetcher_returning(httpx.Response(200, content=PDF))

    with pytest.raises(DocumentError) as excinfo:
        await fetcher.fetch("https://internal.example/a.pdf")

    assert "10.1.2.3" not in str(excinfo.value)


# --- Redirects ---------------------------------------------------------------


@pytest.mark.anyio
async def test_a_redirect_is_followed(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    fetcher, calls = fetcher_returning(
        httpx.Response(302, headers={"location": "https://cdn.example/real.pdf"}),
        httpx.Response(200, content=PDF),
    )

    assert await fetcher.fetch("https://example.org/a.pdf") == PDF
    assert str(calls[1].url) == "https://cdn.example/real.pdf"


@pytest.mark.anyio
async def test_a_redirect_to_a_private_address_is_refused(monkeypatch):
    """The attack a check on the typed URL alone would miss entirely: a public
    link that answers 302 -> the metadata host. Every hop is re-resolved and
    re-checked, which is why redirects are followed by hand.
    """
    addresses = iter([("93.184.216.34",), ("169.254.169.254",)])
    current = {"ips": next(addresses)}

    async def fake_getaddrinfo(host, port, **kwargs):
        return [(None, None, None, "", (ip, port)) for ip in current["ips"]]

    import asyncio

    class Loop:
        getaddrinfo = staticmethod(fake_getaddrinfo)

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: Loop())

    def handler(request: httpx.Request) -> httpx.Response:
        current["ips"] = next(addresses)
        return httpx.Response(
            302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    )

    with pytest.raises(DocumentError, match="will not fetch"):
        await DocumentFetcher(client=client).fetch("https://example.org/a.pdf")


@pytest.mark.anyio
async def test_a_redirect_to_a_forbidden_scheme_is_refused(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    fetcher, _ = fetcher_returning(
        httpx.Response(302, headers={"location": "file:///etc/passwd"})
    )

    with pytest.raises(DocumentError, match="http"):
        await fetcher.fetch("https://example.org/a.pdf")


@pytest.mark.anyio
async def test_a_redirect_loop_gives_up(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    fetcher, calls = fetcher_returning(
        httpx.Response(302, headers={"location": "https://example.org/a.pdf"})
    )

    with pytest.raises(DocumentError, match="redirected too many times"):
        await fetcher.fetch("https://example.org/a.pdf")

    assert len(calls) == MAX_REDIRECTS + 1


@pytest.mark.anyio
async def test_a_relative_redirect_resolves_against_the_current_url(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    fetcher, calls = fetcher_returning(
        httpx.Response(302, headers={"location": "/documents/real.pdf"}),
        httpx.Response(200, content=PDF),
    )

    assert await fetcher.fetch("https://example.org/a.pdf") == PDF
    assert str(calls[1].url) == "https://example.org/documents/real.pdf"


# --- The total deadline ------------------------------------------------------


@pytest.mark.anyio
async def test_a_trickling_download_hits_the_total_deadline(monkeypatch):
    """httpx's read timeout restarts on every chunk, so without one overall
    deadline a server sending a byte just inside it holds the request open
    for as long as it likes."""
    monkeypatch.setattr(url_fetch, "TOTAL_TIMEOUT_SECONDS", 0.3)
    resolve_to(monkeypatch, "93.184.216.34")

    async def trickle():
        for _ in range(20):
            await asyncio.sleep(0.1)
            yield b"x"

    fetcher, _ = fetcher_returning(httpx.Response(200, content=trickle()))
    started = time.monotonic()

    with pytest.raises(DocumentError, match="took too long"):
        await fetcher.fetch("https://example.org/a.pdf")

    assert time.monotonic() - started < 1.5


@pytest.mark.anyio
async def test_a_hanging_dns_lookup_hits_the_total_deadline(monkeypatch):
    """DNS has no timeout of its own, so the deadline has to cover it too."""
    monkeypatch.setattr(url_fetch, "TOTAL_TIMEOUT_SECONDS", 0.2)

    async def hang(host, port, **kwargs):
        await asyncio.sleep(5)

    class Loop:
        getaddrinfo = staticmethod(hang)

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: Loop())
    fetcher, calls = fetcher_returning(httpx.Response(200, content=PDF))

    with pytest.raises(DocumentError, match="took too long"):
        await fetcher.fetch("https://example.org/a.pdf")

    assert calls == []


# --- Size, status and content ------------------------------------------------


@pytest.mark.anyio
async def test_an_honestly_declared_oversize_body_fails_before_download(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    fetcher, _ = fetcher_returning(
        httpx.Response(
            200,
            headers={"content-length": str(MAX_FILE_BYTES + 1)},
            content=b"x",
        )
    )

    with pytest.raises(DocumentError, match="too large"):
        await fetcher.fetch("https://example.org/big.pdf")


@pytest.mark.anyio
async def test_a_lying_content_length_does_not_get_past_the_cap(monkeypatch):
    """Content-Length is a claim. The running total is what enforces the cap,
    so a server understating its size is still stopped."""
    resolve_to(monkeypatch, "93.184.216.34")
    fetcher, _ = fetcher_returning(
        httpx.Response(
            200, headers={"content-length": "10"}, content=b"x" * (MAX_FILE_BYTES + 1)
        )
    )

    with pytest.raises(DocumentError, match="too large"):
        await fetcher.fetch("https://example.org/big.pdf")


@pytest.mark.anyio
async def test_an_empty_body_is_refused(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    fetcher, _ = fetcher_returning(httpx.Response(200, content=b""))

    with pytest.raises(DocumentError, match="empty"):
        await fetcher.fetch("https://example.org/a.pdf")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status, expected",
    [
        (401, "not publicly accessible"),
        (403, "not publicly accessible"),
        (404, "No document was found"),
        (500, "could not be downloaded"),
        (503, "could not be downloaded"),
    ],
)
async def test_upstream_failures_are_the_callers_problem(
    monkeypatch, status, expected
):
    """Their 404 is the caller's typo and their 500 is nothing we can fix, so
    both come back as a 400 about the link rather than as our own error."""
    resolve_to(monkeypatch, "93.184.216.34")
    fetcher, _ = fetcher_returning(httpx.Response(status))

    with pytest.raises(DocumentError, match=expected):
        await fetcher.fetch("https://example.org/a.pdf")


@pytest.mark.anyio
async def test_the_content_type_is_not_consulted(monkeypatch):
    """A server can claim anything. What arrived is `document.sniff`'s
    decision, made from the magic bytes, exactly as for any other document - so a
    PDF served as text/html still downloads fine and is judged on its bytes."""
    resolve_to(monkeypatch, "93.184.216.34")
    fetcher, _ = fetcher_returning(
        httpx.Response(200, headers={"content-type": "text/html"}, content=PDF)
    )

    assert await fetcher.fetch("https://example.org/a") == PDF


@pytest.mark.anyio
async def test_nothing_is_written_to_disk(monkeypatch, tmp_path):
    """The whole 'stores temporarily, deletes after' requirement, met by never
    storing: the bytes live in memory and are released with the request, so
    there is no cleanup that can fail and nothing left behind if the process
    is killed mid-request."""
    resolve_to(monkeypatch, "93.184.216.34")
    monkeypatch.chdir(tmp_path)
    fetcher, _ = fetcher_returning(httpx.Response(200, content=PDF))

    await fetcher.fetch("https://example.org/a.pdf")

    assert list(tmp_path.iterdir()) == []


@pytest.mark.anyio
async def test_an_injected_client_is_not_closed(monkeypatch):
    """Same ownership rule as the model providers: what the caller passed in is
    the caller's to close."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    fetcher = DocumentFetcher(client=client)

    await fetcher.aclose()

    assert client.is_closed is False
    await client.aclose()
