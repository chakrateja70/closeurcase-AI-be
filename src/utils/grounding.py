"""Checking extracted items back against the document they came from, and
working out which page of it each one is on.

Everything else in this feature asks the model to be truthful. This module is
the only part that *checks*. The prompt's grounding rule - "if the document
does not state it, return null" - is unfalsifiable on its own, because a
fabricated chronology reads exactly like a correct one. So the model is also
asked to quote the words that support each entry, and those quotes are looked
for in the text we extracted ourselves.

PAGE NUMBERS ARE NEVER TAKEN FROM THE MODEL.

This is the rule the module is arranged around, and it is structural rather
than a matter of care: `check()` has no parameter through which a
model-supplied page could arrive, and `source_pages` is absent from the
response schema the model answers against. A page number is published only when
this module has found the quoted words inside that page's own text. There is no
path by which a plausible-looking number the model produced can reach a reader,
so the failure that matters here - a citation that looks checkable and is not -
cannot happen.

That leaves two independent questions about every entry, reported separately
because the answers genuinely differ:

* Are these words in the document?  -> `verified` / `unverified` /
  `not_verifiable`.
* Which page are they on?           -> `mapped` / `unmapped` / `unavailable`.

An entry can be verified but unmapped: the words are in the document, but the
file has no page structure, or the phrase recurs on so many pages that finding
it locates nothing. Collapsing the two verdicts would mean either discarding a
confirmed quote or claiming a page we did not find - so nothing is marked
page-mapped without pages to show, and `source_pages` is non-empty if and only
if `page_status` is `mapped`.

The match is deliberately forgiving about *characters* and strict about
*words*. A PDF text layer and a model transcription of it disagree constantly
about hyphenation, curly quotes, ligatures and line-wrap spacing, and none of
those differences mean the quote was invented. So both sides are reduced to
their letters and digits before comparing. What survives that reduction is the
actual sequence of words, which is the thing worth being strict about.

Nothing is verifiable on the FILE and IMAGE branches - a scan has no text layer
for us to hold, which is the whole reason it went to the model as a file. Those
entries are `not_verifiable` and `unavailable`, which are honest third answers
rather than synonyms for either of the others.

One caveat worth passing on to whoever reads a page number: these are positions
in the file, counted from one, not the numbers printed on the page. A filing
whose first two leaves are a cover sheet and an index reports the third leaf as
page 3 even where it is printed "1".

WHY UNVERIFIED ENTRIES ARE KEPT. Dropping them would be a silent edit of a
lawyer's brief on the strength of a string comparison, and the false-positive
rate against real filings is not yet known. Label first, measure on real
documents, and only then decide whether anything should be removed. This is the
same call `case_summarization_service` makes about running the injection
scanner in log-only mode, for the same reason.
"""

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

# Below this many letters and digits a quote proves nothing: "the plaintiff"
# occurs in every filing ever drafted, so finding it is neither evidence that
# the entry is grounded nor a usable way to place it on a page.
MIN_SNIPPET_CHARS = 12

# Only the head of a long quote is matched. A model told to quote 25 words
# sometimes returns a paragraph, and the longer the string the likelier some
# transcription difference breaks an exact comparison - while the first 300
# characters are already far past the point of proving where it came from.
MATCH_PREFIX_CHARS = 300

# A boilerplate phrase can recur on every page of a filing. Past this many
# occurrences the quote is not distinctive enough to place anything, and
# listing thirty pages would be worse than admitting we cannot locate it.
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
    """Letters and digits only, lowercased.

    Punctuation, case and every kind of whitespace are dropped rather than
    normalised, because that is exactly the set of things that differs between
    a text layer and a faithful quotation of it. Two strings that reduce to the
    same value contain the same words in the same order.
    """
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

    The page bodies are concatenated WITHOUT their markers, and each page's
    span in that concatenation is recorded. Searching the concatenation rather
    than page by page is what lets a quote spanning a page break be found at
    all; the span table is what turns the position it was found at back into
    page numbers. Reducing the raw text instead would leave "page1", "page2"
    embedded at exactly the point such a quote has to match across.

    Built once per request and reused across every entry: reducing a
    250 000-character document is not free and there can be a hundred entries.
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
