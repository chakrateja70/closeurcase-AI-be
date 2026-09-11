from __future__ import annotations

from fastapi import Request

from src.services.case_detection_service import CaseDetectionService
from src.services.case_summarization_service import CaseSummarizationService


def get_case_detection_service(request: Request) -> CaseDetectionService:
    return request.app.state.case_detection_service


def get_case_summarization_service(request: Request) -> CaseSummarizationService:
    return request.app.state.case_summarization_service
