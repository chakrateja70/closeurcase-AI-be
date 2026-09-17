from __future__ import annotations
from typing import Annotated, List, Literal, Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from fastapi import APIRouter, Depends, Request, Response, status
from src.api.deps import get_case_summarization_service
from src.core.case_input import resolve_case_input
from src.core.exceptions import MissingCaseInputError
from src.core.rate_limit import SUMMARIZE_RATE_LIMIT, limiter
from src.core.request_context import client_ip, counter
from src.core.summary_provider import available_summary_providers, resolve_summary_provider
from src.services.case_summarization_service import CaseSummarizationService

MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 25000
MAX_DOCUMENT_URLS = 5

router = APIRouter(prefix="/case_summarization", tags=["case_summarization"])

class SummarizeCaseRequest(BaseModel):
    """Provide document URLs, case text, or both."""
    document_urls: Optional[List[HttpUrl]] = Field(
        default=None,
        min_length=1,
        max_length=MAX_DOCUMENT_URLS,
        description="URLs to case documents (PDF, JPG, or PNG) to summarize.",
        examples=["https://example.com/case1.pdf", "https://example.com/case2.pdf"],
    )
    case_text: Optional[str] = Field(
        default=None,
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description=f"Case text to summarize directly (max {MAX_TEXT_LENGTH} characters).",
        examples=["This is a sample case text that is long enough to be summarized."],
    )
    model: Optional[Literal["gpt", "gemini"]] = Field(
        default=None,
        description=(
            "Override the server's configured provider for this request only. "
            "Omit to use the default from GET /case_summarization/models."
        ),
    )

    @field_validator("document_urls", mode="before")
    @classmethod
    def _coerce_document_urls(cls, value):
        """Accept a single URL string as well as a list, since a caller with
        one document has no obvious reason to know this field is a list."""
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def validate_input(self) -> "SummarizeCaseRequest":
        if not self.document_urls and not self.case_text:
            raise MissingCaseInputError()
        resolve_case_input(self.document_urls, self.case_text)
        return self

class SummaryModelsResponse(BaseModel):
    status_code: int = status.HTTP_200_OK
    status_message: str = "success"
    models: list[str]

@router.get("/models", response_model=SummaryModelsResponse)
async def get_summary_models() -> SummaryModelsResponse:
    """Providers `/case_summarization/summarize` can use right now - only the
    ones with a key configured"""
    return SummaryModelsResponse(
        status_code=status.HTTP_200_OK,
        status_message="success",
        models=[p.value for p in available_summary_providers()],
    )

class SummarizeCaseResponse(BaseModel):
    status_code: int = status.HTTP_200_OK
    status_message: str = "success"
    brief: str
    key_points: list[str]


SummarizationService = Annotated[
    CaseSummarizationService, Depends(get_case_summarization_service)
]

@router.post("/summarize", response_model=SummarizeCaseResponse)
@limiter.limit(SUMMARIZE_RATE_LIMIT)
async def summarize_case(
    request: Request,
    response: Response,
    payload: SummarizeCaseRequest,
    service: SummarizationService,
) -> SummarizeCaseResponse:
    """Summarize a case from one or more document_urls (PDFs or images,
    handed to the model directly), case_text, or both together.
    """
    caller = client_ip(request)
    provider = resolve_summary_provider(payload.model)
    case_input = resolve_case_input(payload.document_urls, payload.case_text)
    result = await service.summarize(
        case_input, provider, client=f"{caller} #{counter.get(caller)}"
    )
    return SummarizeCaseResponse(
        status_code=status.HTTP_200_OK,
        status_message="success",
        **result,
    )
