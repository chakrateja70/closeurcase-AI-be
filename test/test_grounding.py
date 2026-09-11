"""Tests for the only part of summarization that checks the model's work.

Two failure directions matter here and they pull against each other. Being too
strict marks honest quotations unverified, which trains a reader to ignore the
label; being too loose stamps `verified` on text that is not in the document,
which is worse than having no label at all. So the tests come in pairs: a
difference that should survive matching, and one that should not.

The page half has a third rule of its own, and it is the one worth breaking a
build over: a page number is published only when the quote was found on that
page. There is no test that a bad page number is filtered out, because there is
no path for one to arrive - `check` takes the quote and nothing else. What is
tested instead is that the invariant holds: pages are non-empty exactly when
the status says `mapped`.
"""

import inspect

import pytest

from src.utils.grounding import (
    MAX_OCCURRENCES,
    MIN_SNIPPET_CHARS,
    NOT_VERIFIABLE,
    PAGES_MAPPED,
    PAGES_UNAVAILABLE,
    PAGES_UNMAPPED,
    UNVERIFIED,
    VERIFIED,
    DocumentIndex,
    reduce_text,
)

PAGE_ONE = (
    "IN THE COURT OF THE CIVIL JUDGE AT VISAKHAPATNAM. Suit No. 412 of 2024. "
    "The plaintiff states that a cheque dated 12.03.2024 for Rs. 4,50,000 was "
    "dishonoured for insufficiency of funds."
)
PAGE_TWO = (
    "A statutory notice was issued on 20.03.2024 under Section 138 of the "
    "Negotiable Instruments Act, 1881, to which no reply has been received."
)

DOCUMENT = f"--- Page 1 ---\n{PAGE_ONE}\n\n--- Page 2 ---\n{PAGE_TWO}"


@pytest.fixture
def index():
    return DocumentIndex(DOCUMENT, page_count=2)


# --- What counts as the same words ------------------------------------------


def test_reduce_keeps_only_letters_and_digits():
    assert reduce_text("Rs. 4,50,000 was  DIS-honoured!") == "rs450000wasdishonoured"


@pytest.mark.parametrize(
    "snippet",
    [
        "a cheque dated 12.03.2024 for Rs. 4,50,000",  # verbatim
        "A CHEQUE DATED 12.03.2024 FOR RS. 4,50,000",  # case
        "a cheque   dated 12.03.2024\nfor Rs. 4,50,000",  # re-wrapped
        "a cheque dated 12.03.2024 for Rs 4 50 000",  # punctuation lost
        "dis-honoured for insufficiency of funds",  # hyphenated at a line break
        "“dishonoured for insufficiency of funds”",  # smart quotes
    ],
)
def test_transcription_differences_still_verify(index, snippet):
    """None of these mean the quote was invented - they are what a PDF text
    layer and a faithful quotation of it routinely disagree about."""
    status, pages, page_status = index.check(snippet)

    assert status == VERIFIED
    # And the entry is still placed: a transcription difference must not cost
    # the reader their page reference either.
    assert page_status == PAGES_MAPPED
    assert pages == [1]


@pytest.mark.parametrize(
    "snippet",
    [
        "a cheque dated 12.03.2024 for Rs. 5,00,000",  # one digit changed
        "the cheque was returned unpaid by the bank",  # a paraphrase
        "the plaintiff issued a legal notice on 01.01.2024",  # invented
    ],
)
def test_a_changed_or_invented_quote_does_not_verify(index, snippet):
    """The point of the whole module: rewording is exactly the failure this is
    meant to catch, so it must not be forgiven along with the spacing."""
    status, pages, page_status = index.check(snippet)

    assert status == UNVERIFIED
    assert (pages, page_status) == ([], PAGES_UNMAPPED)


def test_a_quote_too_short_to_prove_anything_is_not_verified(index):
    """'the plaintiff' is in every filing ever drafted; matching it is neither
    evidence that the entry is grounded nor a way to place it on a page."""
    short = "the pl"
    assert len(reduce_text(short)) < MIN_SNIPPET_CHARS
    assert index.check(short)[0] == UNVERIFIED


@pytest.mark.parametrize("snippet", [None, "", "   ", 42, ["not a string"]])
def test_a_missing_quote_is_unverified_not_verified(index, snippet):
    """A null snippet is the model saying it could not quote the passage. That
    is honest, and it is still an entry nothing supports."""
    assert index.check(snippet)[0] == UNVERIFIED


# --- Which page it is on -----------------------------------------------------


def test_the_page_comes_from_where_the_quote_is(index):
    status, pages, page_status = index.check("under Section 138 of the")

    assert (status, pages, page_status) == (VERIFIED, [2], PAGES_MAPPED)


def test_a_quote_spanning_a_page_break_is_found_and_placed_on_both(index):
    """Pages are searched as one concatenated stream and mapped back by
    position, so a quote running across the break belongs to both - not to
    neither, which is what a page-by-page search would have concluded."""
    status, pages, page_status = index.check(
        "insufficiency of funds A statutory notice was issued"
    )

    assert (status, pages, page_status) == (VERIFIED, [1, 2], PAGES_MAPPED)


def test_a_quote_repeated_on_several_pages_reports_all_of_them():
    recital = "the same recital of facts appears here"
    both = f"--- Page 1 ---\n{recital}\n\n--- Page 2 ---\n{recital}"

    assert DocumentIndex(both, page_count=2).check(recital)[1] == [1, 2]


def test_boilerplate_on_every_page_places_nothing():
    """A phrase that recurs throughout a filing is true everywhere and so
    evidence of nothing about where this entry came from. Listing thirty pages
    would be worse than admitting we cannot place it."""
    line = "in the court of the civil judge at visakhapatnam"
    pages = "\n\n".join(
        f"--- Page {n} ---\n{line}" for n in range(1, MAX_OCCURRENCES + 3)
    )
    status, found, page_status = DocumentIndex(pages).check(line)

    assert status == VERIFIED
    assert (found, page_status) == ([], PAGES_UNMAPPED)


def test_a_docx_verifies_but_has_no_pages_to_map_to():
    """No page markers, because a DOCX reflows - so a quote can be confirmed
    but not placed, and that is 'unavailable' rather than a failure to find."""
    index = DocumentIndex("The lease was terminated on 01.04.2024 by notice.")
    status, pages, page_status = index.check("terminated on 01.04.2024 by notice")

    assert status == VERIFIED
    assert (pages, page_status) == ([], PAGES_UNAVAILABLE)


@pytest.mark.parametrize("text", [None, ""])
def test_a_scan_is_not_verifiable_rather_than_unverified(text):
    """There is no text layer to look in - that is why the file went to the
    model as a file. Reporting those entries as unverified would blame the
    model for our own lack of evidence."""
    status, pages, page_status = DocumentIndex(text, page_count=9).check(
        "a cheque dated 12.03.2024"
    )

    assert status == NOT_VERIFIABLE
    assert (pages, page_status) == ([], PAGES_UNAVAILABLE)


# --- The invariant that makes a page number worth trusting -------------------


def test_check_cannot_be_given_a_page_number():
    """Structural, not a matter of discipline: there is no parameter through
    which a model-supplied page could arrive, so none can be published. If a
    future change adds one, this fails and the reviewer has to argue for it."""
    parameters = list(inspect.signature(DocumentIndex.check).parameters)

    assert parameters == ["self", "snippet"]


@pytest.mark.parametrize(
    "snippet",
    [
        "under Section 138 of the",  # verified and placed
        "insufficiency of funds A statutory notice was issued",  # spans a break
        "a paraphrase that is nowhere in this document at all",  # not found
        None,  # never quoted
        "the pl",  # too short to place
    ],
)
def test_pages_are_present_exactly_when_the_status_says_mapped(index, snippet):
    """The contract a reader relies on: a page number appears only alongside
    'mapped', and 'mapped' never appears without one. Anything else would let
    an entry look page-verified when its page is unknown."""
    _, pages, page_status = index.check(snippet)

    assert bool(pages) is (page_status == PAGES_MAPPED)


@pytest.mark.parametrize("snippet", ["under Section 138 of the", "not in here at all"])
def test_reported_pages_exist_in_the_document(index, snippet):
    _, pages, _ = index.check(snippet)

    assert all(1 <= page <= 2 for page in pages)
