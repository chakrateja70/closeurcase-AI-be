from typing import Optional

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from src.core.case_categories import list_categories
from src.services.case_detection_service import CaseDetectionService

router = APIRouter(prefix="/case_detection", tags=["case_detection"])

service = CaseDetectionService()
MAX_QUERY_LENGTH = 600


class DetectCaseRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
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
    """Detection decides a case type and nothing else. The category each case
    type sits under, and the legal services offered under it, are mapped from
    the taxonomy server-side and returned here, so a caller needs no second
    request to /categories to act on the result.

    `*_legal_services` is every service available for that case type, in
    taxonomy order - it is a lookup, not a ranking, so it is not filtered by
    how well each service fits the query. It is an empty list whenever the
    matching case type is absent.
    """

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
async def detect_case(payload: DetectCaseRequest) -> DetectCaseResponse:
    """Classify a user query into a case type.

    Classification is done by an OpenAI model constrained to the case taxonomy.
    """
    result = await service.detect_case(payload.query)
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
