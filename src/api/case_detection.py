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

class DetectCaseResponse(BaseModel):
    status_code: int = status.HTTP_200_OK
    status_message: str = "success"
    is_valid: bool
    primary_case_category: Optional[str] = None
    primary_case_category_id: Optional[str] = None
    secondary_case_category: Optional[str] = None
    secondary_case_category_id: Optional[str] = None
    confidence: float = 0.0
    summary: Optional[str] = None
    fallback_response: Optional[str] = None


@router.post("/detect", response_model=DetectCaseResponse)
async def detect_case(payload: DetectCaseRequest) -> DetectCaseResponse:
    """Classify a user query into a case category.

    Classification is done by an OpenAI model constrained to the case taxonomy.
    """
    result = await service.detect_case(payload.query)
    return DetectCaseResponse(
        status_code=status.HTTP_200_OK,
        status_message="success",
        **result,
    )


@router.get("/categories")
async def get_case_categories() -> dict:
    """The static case taxonomy the detector classifies against."""
    return {
        "status_code": status.HTTP_200_OK,
        "status_message": "success",
        "categories": list_categories(),
    }
