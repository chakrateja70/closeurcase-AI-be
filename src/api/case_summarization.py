from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Depends, Form, Request, Response, status
from pydantic import BaseModel, Field

from src.api.deps import get_case_summarization_service
from src.core.rate_limit import SUMMARIZE_RATE_LIMIT, limiter
from src.core.request_context import client_ip, counter
from src.prompts.case_summary_prompt import PARTY_ROLES, STATEMENT_TYPES
from src.services.case_summarization_service import CaseSummarizationService
from src.services.summary_providers import MODEL_GEMINI, MODEL_GPT
from src.utils.document import ACCEPTED_DESCRIPTION, MIN_TEXT_CHARS
from src.utils.grounding import (
    NOT_VERIFIABLE,
    PAGE_STATUSES,
    PAGES_UNAVAILABLE,
    STATUSES,
)

# --- Response models ---------------------------------------------------------

class SourceRef(BaseModel):
    """Quoted evidence and its page status."""

    source_snippet: Optional[str] = Field(
        default=None,
        description="Quoted text from the document. Null when the model could not quote it exactly.",
    )
    source_pages: list[int] = Field(
        default_factory=list,
        description="Pages where the quote was found. Empty unless page_status is mapped.",
    )
    validation_status: str = Field(
        default=NOT_VERIFIABLE,
        description="Whether the quote appears in the document.",
        examples=list(STATUSES),
    )
    page_status: str = Field(
        default=PAGES_UNAVAILABLE,
        description="Whether the quote can be mapped to a page.",
        examples=list(PAGE_STATUSES),
    )


class Party(BaseModel):
    """A named party from the document title."""

    name: str
    role: str = Field(
        ...,
        description="Role stated in the document; 'other' when unspecified.",
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
        description="Whether this is a fact or an allegation.",
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
    """Counts for verified quotes and mapped pages."""

    quotes: QuoteCounts = Field(default_factory=QuoteCounts)
    pages: PageCounts = Field(default_factory=PageCounts)


class SummarizeDocumentResponse(BaseModel):
    """Summary of the document's timeline, dispute, and claims."""

    status_code: int = status.HTTP_200_OK
    status_message: str = "success"
    is_valid: bool

    parties: list[Party] = Field(
        default_factory=list,
        description="Named parties and counsel.",
    )
    chronology: list[ChronologyEvent] = Field(
        default_factory=list, description="Key events in time order."
    )
    facts_summary: Optional[str] = Field(
        default=None,
        description="Short overview of the dispute, at most about 120 words.",
    )
    assertions: list[Assertion] = Field(
        default_factory=list,
        description="The document's factual assertions and allegations.",
    )
    confidence: float = 0.0
    grounding: Grounding = Field(default_factory=Grounding)
    model: Optional[str] = Field(
        default=None, description="Model used for this summary."
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Exact model version.",
    )
    page_count: Optional[int] = Field(
        default=None, description="PDF page count; null for DOCX and images."
    )
    source_pages_available: bool = Field(
        default=False,
        description="Whether page numbers are available for this document.",
    )
    truncated: bool = Field(
        default=False,
        description="True if the document exceeded the extraction limit.",
    )
    fallback_response: Optional[str] = None


class AvailableModelsResponse(BaseModel):
    status_code: int = status.HTTP_200_OK
    status_message: str = "success"
    models: list[str] = Field(
        ...,
        description="Models accepted by /summarize on this deployment.",
        examples=[[MODEL_GPT, MODEL_GEMINI]],
    )
    default: str = Field(
        ...,
        description="Default model when /summarize is called without a model override.",
        examples=[MODEL_GPT],
    )


class ErrorResponse(BaseModel):
    """Standard API error payload."""

    status_code: int
    status_message: str
    error_message: str


# --- Error responses per route -----------------------------------------------


def _error(description: str) -> dict[str, Any]:
    return {"model": ErrorResponse, "description": description}


SUMMARIZE_DOCUMENT_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_400_BAD_REQUEST: _error("Bad request or invalid document input."),
    status.HTTP_429_TOO_MANY_REQUESTS: _error("Rate limit exceeded."),
    status.HTTP_500_INTERNAL_SERVER_ERROR: _error("Unhandled server error."),
    status.HTTP_502_BAD_GATEWAY: _error("Model call failed or returned unreadable output."),
    status.HTTP_503_SERVICE_UNAVAILABLE: _error("Requested or default model is unavailable."),
    status.HTTP_504_GATEWAY_TIMEOUT: _error("Model call timed out."),
}

GET_AVAILABLE_MODELS_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_500_INTERNAL_SERVER_ERROR: _error("Unhandled server error."),
}


# --- Routes ------------------------------------------------------------------

router = APIRouter(prefix="/case_summarization", tags=["case_summarization"])

# Spelled out because type checkers reject variables inside `Literal`. A test
# pins these to `SUPPORTED_MODELS`, so the two cannot drift apart.
SummaryModel = Literal["gpt", "gemini"]

SummarizationService = Annotated[
    CaseSummarizationService, Depends(get_case_summarization_service)
]


@router.get(
    "/models",
    response_model=AvailableModelsResponse,
    responses=GET_AVAILABLE_MODELS_ERRORS,
)
async def get_available_models(service: SummarizationService) -> AvailableModelsResponse:
    """Return the models available on this deployment and the default model."""
    return AvailableModelsResponse(
        models=service.available_models,
        default=service.default_model,
    )


@router.post(
    "/summarize",
    response_model=SummarizeDocumentResponse,
    responses=SUMMARIZE_DOCUMENT_ERRORS,
)
@limiter.limit(SUMMARIZE_RATE_LIMIT)
async def summarize_document(
    request: Request,
    response: Response,
    service: SummarizationService,
    url: Optional[str] = Form(
        default=None,
        description=(
            f"Public document URL ({ACCEPTED_DESCRIPTION}). Use this or `text`, not both."
        ),
        examples=["https://example.org/petition.pdf"],
    ),
    text: Optional[str] = Form(
        default=None,
        description=f"Pasted document text. Use this or `url`, not both. Minimum {MIN_TEXT_CHARS} chars.",
    ),
    model: Optional[SummaryModel] = Form(
        default=None,
        description="Optional model override. Omit to use the server default.",
        examples=[MODEL_GPT],
    ),
) -> SummarizeDocumentResponse:
    """Summarise a legal document into its events, dispute, and assertions."""
    caller = client_ip(request)
    result = await service.summarize_document(
        url=url or None,
        text=text or None,
        model=model,
        client=f"{caller} #{counter.get(caller)}",
    )
    return SummarizeDocumentResponse(**result)
