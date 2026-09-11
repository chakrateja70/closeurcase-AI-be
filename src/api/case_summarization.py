from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile, status
from pydantic import BaseModel, Field

from src.api.deps import get_case_summarization_service
from src.prompts.case_summary_prompt import PARTY_ROLES, STATEMENT_TYPES
from src.core.exceptions import BadRequestAPIException
from src.core.rate_limit import SUMMARIZE_RATE_LIMIT, limiter
from src.core.request_context import client_ip, counter
from src.services.case_summarization_service import CaseSummarizationService
from src.services.summary_providers import MODEL_GEMINI, MODEL_GPT
from src.utils.document import (
    ACCEPTED_DESCRIPTION,
    MAX_FILE_BYTES,
    MIN_TEXT_CHARS,
)
from src.utils.grounding import (
    NOT_VERIFIABLE,
    PAGE_STATUSES,
    PAGES_UNAVAILABLE,
    STATUSES,
)

router = APIRouter(prefix="/case_summarization", tags=["case_summarization"])

SummarizationService = Annotated[
    CaseSummarizationService, Depends(get_case_summarization_service)
]

# Read the upload in chunks rather than in one go, so an oversized file is
# rejected after a megabyte instead of being buffered whole first.
_READ_CHUNK_BYTES = 1024 * 1024


class SourceRef(BaseModel):
    """Where an entry came from, and whether that survived checking.

    Shared by every list whose entries are worth acting on unread. The three
    fields are only meaningful together: the quote is the claim, the pages say
    where it sits, and the status says whether we found it there.
    """

    source_snippet: Optional[str] = Field(
        default=None,
        description=(
            "The words from the document that establish this entry, quoted by "
            "the model. Null when it could not quote them exactly."
        ),
    )
    source_pages: list[int] = Field(
        default=[],
        description=(
            "The pages of the uploaded file this entry is on, counted from "
            "one. Never supplied by the model: these are the pages the quote "
            "above was actually found on, so the list is empty unless "
            "page_status is 'mapped'. They are positions in the file, not the "
            "numbers printed on the page."
        ),
    )
    validation_status: str = Field(
        default=NOT_VERIFIABLE,
        description=(
            "Whether the quote is in the document. "
            "'verified' - it was found in the extracted text. "
            "'unverified' - it was not, so nothing here supports this entry; "
            "it may still be right, but check it. "
            "'not_verifiable' - a scan or an image, with no text layer to "
            "check against."
        ),
        examples=list(STATUSES),
    )
    page_status: str = Field(
        default=PAGES_UNAVAILABLE,
        description=(
            "Whether this entry could be placed on a page. "
            "'mapped' - source_pages says where it is. "
            "'unmapped' - the document has pages, but this entry could not be "
            "put on one; source_pages is empty. "
            "'unavailable' - the document has no page-level text to map "
            "against at all: a scan, an image, or a DOCX, which reflows and "
            "has no fixed pages."
        ),
        examples=list(PAGE_STATUSES),
    )


class Party(BaseModel):
    """Deliberately not a `SourceRef`. Every other list here quotes the
    document; a party is a name in a cause title, checked by reading it rather
    than by turning to a page, so it carries no source fields at all instead of
    carrying empty ones."""

    name: str
    role: str = Field(
        ...,
        description="Role the document gives this party; 'other' when it does not say.",
        examples=PARTY_ROLES,
    )
    counsel: Optional[str] = None


class ChronologyEvent(SourceRef):
    date: Optional[str] = None
    event: str


class Assertion(SourceRef):
    statement: str
    statement_type: str = Field(
        ...,
        description=(
            "'fact' where the document treats this as established, "
            "'allegation' where a party asserts it against another. The "
            "difference is legal, not stylistic: an allegation is something "
            "the other side is expected to answer."
        ),
        examples=STATEMENT_TYPES,
    )


class QuoteCounts(BaseModel):
    verified: int = 0
    unverified: int = 0
    not_verifiable: int = 0


class PageCounts(BaseModel):
    mapped: int = 0
    unmapped: int = 0
    unavailable: int = 0


class Grounding(BaseModel):
    """Totals across every list that quotes the document.

    Two tallies, not one, because they answer different questions: `quotes`
    says how much of the summary is in the document, `pages` says how much of
    it can be pointed to. A high `unverified` on a text document means read the
    summary against the original before relying on it.
    """

    quotes: QuoteCounts = QuoteCounts()
    pages: PageCounts = PageCounts()


class SummarizeDocumentResponse(BaseModel):
    """Three questions, answered from the document and nothing else: what
    happened, what the dispute is, and what the parties are arguing.

    `parties` rides along because the other three read as anonymous without it -
    "the applicant seeks an injunction" is not much use when the reader cannot
    see who that is.

    Everything else a reader can take off the face of their own filing - the
    court, the case number, the statutes, precedents and exhibits cited, the
    hearing dates - is deliberately absent. Restating it made the response long
    without making it more useful, and buried the fields that say something the
    document does not say plainly.

    Every field is optional or empty by design: the model is instructed to
    return null for anything the document does not state, so a sparse response
    means a sparse document, not a failure."""

    status_code: int = status.HTTP_200_OK
    status_message: str = "success"
    is_valid: bool

    parties: list[Party] = Field(
        default=[], description="Who the parties are, and who represents them."
    )
    chronology: list[ChronologyEvent] = Field(
        default=[], description="What happened, earliest first."
    )
    facts_summary: Optional[str] = Field(
        default=None,
        description="What the dispute is, in 2-4 plain-English paragraphs.",
    )
    assertions: list[Assertion] = Field(
        default=[],
        description=(
            "What the parties are arguing: the document's discrete factual "
            "assertions and allegations. One list rather than two, "
            "discriminated by statement_type - whether a sentence in a "
            "pleading is a fact or an allegation is a judgement call, and two "
            "lists would let the same sentence land in both or in neither."
        ),
    )
    confidence: float = 0.0
    grounding: Grounding = Grounding()
    model: Optional[str] = Field(
        default=None, description="Which model produced this summary."
    )
    model_id: Optional[str] = Field(
        default=None,
        description="The exact model version, e.g. 'gemini-2.5-flash'.",
    )
    page_count: Optional[int] = Field(
        default=None, description="Pages in the PDF; null for DOCX and images."
    )
    source_pages_available: bool = Field(
        default=False,
        description=(
            "Whether chronology entries can carry page numbers - false for "
            "scanned PDFs, DOCX and images."
        ),
    )
    truncated: bool = Field(
        default=False,
        description="True when the document exceeded the extraction limit.",
    )
    fallback_response: Optional[str] = None


async def _read_capped(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(_READ_CHUNK_BYTES):
        total += len(chunk)
        if total > MAX_FILE_BYTES:
            raise BadRequestAPIException(
                f"File is too large. The limit is {MAX_FILE_BYTES // 1_048_576} MB."
            )
        chunks.append(chunk)
    return b"".join(chunks)


class AvailableModelsResponse(BaseModel):
    status_code: int = status.HTTP_200_OK
    status_message: str = "success"
    models: list[str] = Field(
        ...,
        description="Model names accepted by /summarize on this deployment.",
        examples=[[MODEL_GPT, MODEL_GEMINI]],
    )
    default: str = Field(
        ...,
        description=(
            "The one used when /summarize is called without `model` - this "
            "server's SUMMARY_PROVIDER. Always a member of `models`. A picker "
            "should preselect it rather than assuming which it is."
        ),
        examples=[MODEL_GPT],
    )


@router.get("/models", response_model=AvailableModelsResponse)
async def get_available_models(service: SummarizationService) -> AvailableModelsResponse:
    """Which models `/summarize` will accept here, and which one it uses by
    default.

    Neither is a constant: a provider appears only when its key is configured,
    and the default is whatever `SUMMARY_PROVIDER` names on this deployment. A
    frontend that renders a model picker has to ask rather than assume, or it
    will offer an option that answers 503 - or preselect the wrong one and
    quietly send every document to the model nobody chose.
    """
    return AvailableModelsResponse(
        status_code=status.HTTP_200_OK,
        status_message="success",
        models=service.available_models,
        default=service.default_model,
    )


@router.post("/summarize", response_model=SummarizeDocumentResponse)
@limiter.limit(SUMMARIZE_RATE_LIMIT)
async def summarize_document(
    request: Request,
    response: Response,
    service: SummarizationService,
    file: Optional[UploadFile] = File(
        default=None,
        description=(
            f"The document to summarise ({ACCEPTED_DESCRIPTION}). "
            f"Send this or `text`, not both."
        ),
    ),
    text: Optional[str] = Form(
        default=None,
        description=(
            "The document's text, pasted rather than uploaded. Send this or "
            f"`file`, not both. At least {MIN_TEXT_CHARS} characters."
        ),
    ),
    model: Optional[Literal[MODEL_GPT, MODEL_GEMINI]] = Form(
        default=None,
        description=(
            "Optional override of which model summarises this document. "
            "Omit it to use the server's configured provider "
            "(`SUMMARY_PROVIDER`), which `GET /case_summarization/models` "
            "reports as `default`. Naming the other one works only where its "
            "key is configured; both are given the same prompt and held to the "
            "same schema, so the choice changes cost and quality, not shape."
        ),
        examples=[MODEL_GPT],
    ),
) -> SummarizeDocumentResponse:
    """Summarise a legal document into what happened, what the dispute is,
    and what the parties are arguing.

    Takes the document **either** as `file` or as `text` - exactly one.
    Sending both is a 400 rather than a silent preference: a client with a
    stale form field would otherwise get a summary of the wrong one with no
    indication which.

    `file` accepts a PDF, DOCX or image. The file type is detected from its
    contents, not from the filename or the declared content type. A PDF with no
    text layer (a scan or photocopy) is sent to the model as the file itself,
    so scanned filings work without OCR infrastructure.

    `text` is for a document pasted into a box. It takes exactly the same path
    as an extracted PDF - same length cap, same untrusted-data framing, same
    grounding check - with one difference the caller should expect: pasted text
    has no pages, so every entry comes back `page_status: "unavailable"` and
    `source_pages` empty, as a DOCX does. Quotes are still verified.

    `model` is **optional**. Omitted, the server's configured provider
    (`SUMMARY_PROVIDER`) runs the call, so which model summarises is a
    deployment decision and a client need not know or care. Naming one
    overrides that for this request, and only works where that provider's key
    is configured - otherwise it is a 503 saying what is available.

    Whichever provider runs it, the request is identical: the same prompt, the
    same JSON schema, the same document framing, the same grounding check. The
    adapters differ only in how the SDK is called. That is deliberate - if the
    framing varied per provider, "which model summarises better" would stop
    being an answerable question.

    `GET /case_summarization/models` lists what is available here and which one
    is the default, so a picker need not hardcode either.

    Rate limited per client IP, and much more tightly than `/detect`: the
    endpoint is unauthenticated and one call sends a whole document upstream.
    Neither `request` nor `response` is used in the body, but both are required
    by slowapi - it reads the client address off the request and writes the
    X-RateLimit-* headers onto the response.
    """
    caller = client_ip(request)
    # An empty file part is what a browser sends for an untouched file input, so
    # it means "no file" rather than "an empty file" - otherwise a paste made
    # through a form carrying both fields could never get past the both-given
    # check below.
    data = await _read_capped(file) if file is not None else None
    # The filename is deliberately not logged and not sent upstream: it is
    # client-supplied, and in this domain it routinely carries a client's name.
    result = await service.summarize_document(
        data or None,
        text=text,
        model=model,
        client=f"{caller} #{counter.get(caller)}",
    )
    return SummarizeDocumentResponse(
        status_code=status.HTTP_200_OK,
        status_message="success",
        **result,
    )
