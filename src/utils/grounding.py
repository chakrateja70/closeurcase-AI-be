from __future__ import annotations

import re

# Were these words found in the document?
VERIFIED = "verified"
UNVERIFIED = "unverified"
NOT_VERIFIABLE = "not_verifiable"

STATUSES = (VERIFIED, UNVERIFIED, NOT_VERIFIABLE)

# Which page are they on?
PAGES_MAPPED = "mapped"
PAGES_UNMAPPED = "unmapped"
PAGES_UNAVAILABLE = "unavailable"

PAGE_STATUSES = (PAGES_MAPPED, PAGES_UNMAPPED, PAGES_UNAVAILABLE)

MIN_SNIPPET_CHARS = 12

MATCH_PREFIX_CHARS = 300

MAX_OCCURRENCES = 8

_PAGE_MARKER = re.compile(r"^--- Page (\d+) ---$", re.MULTILINE)
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")

# Characters that survive a PDF-to-text-to-model round trip in altered form
# without any word having changed.
_LOOKALIKES = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
        "ﬁ": "fi",
        "ﬂ": "fl",
    }
)


def reduce_text(text: str) -> str:
    """Letters and digits only, lowercased."""
    return _NOT_ALNUM.sub("", text.translate(_LOOKALIKES).lower())


def _split_pages(text: str) -> list[tuple[int, str]]:
    """Reduced text per page, from the '--- Page N ---' markers that
    `document._join_pages` writes. Empty for anything with no markers - a DOCX
    reflows and has no pages, so there is nothing to attribute a quote to."""
    matches = list(_PAGE_MARKER.finditer(text))
    if not matches:
        return []
    pages = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        pages.append((int(match.group(1)), reduce_text(text[match.end() : end])))
    return pages


class DocumentIndex:
    """The extracted text in the one form quotes are compared against, plus a
    map from a position in it back to a page number.
    """

    def __init__(self, text: str | None, page_count: int | None = None) -> None:
        self.available = bool(text)
        self.page_count = page_count

        pages = _split_pages(text or "")
        self.has_pages = bool(pages)

        bounds: list[tuple[int, int, int]] = []
        offset = 0
        for number, body in pages:
            bounds.append((number, offset, offset + len(body)))
            offset += len(body)
        self._bounds = bounds
        self._whole = (
            "".join(body for _, body in pages) if pages else reduce_text(text or "")
        )

    def locate(self, snippet: str | None) -> tuple[bool, list[int]]:
        """Whether the quote is in the document, and every page it sits on.

        The page list can be empty for a quote that was found: a DOCX has no
        pages to report, and a phrase repeated throughout a filing places
        nothing. Found-but-unplaced is a real answer, not a failure.
        """
        if not self.available or not isinstance(snippet, str):
            return False, []
        needle = reduce_text(snippet[:MATCH_PREFIX_CHARS])
        if len(needle) < MIN_SNIPPET_CHARS:
            return False, []

        starts = []
        position = self._whole.find(needle)
        while position != -1 and len(starts) <= MAX_OCCURRENCES:
            starts.append(position)
            position = self._whole.find(needle, position + 1)
        if not starts:
            return False, []
        if len(starts) > MAX_OCCURRENCES:
            # Found, but boilerplate - true everywhere in the filing, and so
            # evidence of nothing about where this particular entry came from.
            return True, []

        pages = {
            number
            for start in starts
            for number, low, high in self._bounds
            if low < start + len(needle) and start < high
        }
        return True, sorted(pages)

    def check(self, snippet: str | None) -> tuple[str, list[int], str]:
        """Both verdicts and the pages, for one entry.

        Takes the quote and nothing else. There is deliberately no parameter
        for the pages the model claimed: they are not used, not asked for in
        the schema, and so cannot be published by accident.
        """
        if not self.available:
            return NOT_VERIFIABLE, [], PAGES_UNAVAILABLE

        found, pages = self.locate(snippet)
        status = VERIFIED if found else UNVERIFIED

        if not self.has_pages:
            return status, [], PAGES_UNAVAILABLE
        if pages:
            return status, pages, PAGES_MAPPED
        return status, [], PAGES_UNMAPPED
