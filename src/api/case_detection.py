from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel, Field

from src.api.deps import get_case_detection_service
from src.core.case_categories import list_categories
from src.core.rate_limit import DETECT_RATE_LIMIT, limiter
from src.core.request_context import client_ip, counter
from src.services.case_detection_service import CaseDetectionService

router = APIRouter(prefix="/case_detection", tags=["case_detection"])

DetectionService = Annotated[
    CaseDetectionService, Depends(get_case_detection_service)
]
MAX_QUERY_LENGTH = 600


class DetectCaseRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=MAX_QUERY_LENGTH,
        description=f"User's description of the issue (max {MAX_QUERY_LENGTH} characters).",
        examples=["My landlord is refusing to return my security deposit after eviction."],
    )

class LegalService(BaseModel):
    id: str = Field(
        ...,
        description="Namespaced under its case type, e.g. 'cheque_bounce.cheque_bounce_appeal'.",
    )
    title: str

class DetectCaseResponse(BaseModel):
    status_code: int = status.HTTP_200_OK
    status_message: str = "success"
    is_valid: bool
    primary_case_category: Optional[str] = None
    primary_case_category_id: Optional[str] = None
    primary_case_type: Optional[str] = None
    primary_case_type_id: Optional[str] = None
    primary_legal_services: list[LegalService] = []
    secondary_case_category: Optional[str] = None
    secondary_case_category_id: Optional[str] = None
    secondary_case_type: Optional[str] = None
    secondary_case_type_id: Optional[str] = None
    secondary_legal_services: list[LegalService] = []
    confidence: float = 0.0
    summary: Optional[str] = None
    fallback_response: Optional[str] = None

class CaseType(BaseModel):
    id: str
    title: str
    legal_services: list[LegalService]

class CaseCategory(BaseModel):
    id: str
    title: str
    case_types: list[CaseType]


class CaseCategoriesResponse(BaseModel):
    status_code: int = status.HTTP_200_OK
    status_message: str = "success"
    categories: list[CaseCategory]


@router.post("/detect", response_model=DetectCaseResponse)
@limiter.limit(DETECT_RATE_LIMIT)
async def detect_case(
    request: Request,
    response: Response,
    payload: DetectCaseRequest,
    service: DetectionService,
) -> DetectCaseResponse:
    """Classify a user query into a case type.

    Classification is done by an OpenAI model constrained to the case taxonomy.

    Rate limited per client IP: the endpoint is unauthenticated and every call
    costs money upstream. Neither `request` nor `response` is used in the body,
    but both are required by slowapi - it reads the client address off the
    request and writes the X-RateLimit-* headers onto the response.
    """
    caller = client_ip(request)
    result = await service.detect_case(
        payload.query, client=f"{caller} #{counter.get(caller)}"
    )
    return DetectCaseResponse(
        status_code=status.HTTP_200_OK,
        status_message="success",
        **result,
    )


@router.get("/categories", response_model=CaseCategoriesResponse)
async def get_case_categories() -> CaseCategoriesResponse:
    """The static case taxonomy the detector classifies against: categories,
    the case types under each, and the legal services offered per case type."""
    return CaseCategoriesResponse(
        status_code=status.HTTP_200_OK,
        status_message="success",
        categories=list_categories(),
    )
