"""Shared FastAPI dependencies.

Services are built once in the app lifespan and stashed on `app.state`; these
resolvers hand them to route handlers. Importing an API module therefore no
longer constructs an OpenAI client as a side effect, which is what previously
made `OPENAI_API_KEY` a hard requirement just to import the routes - and what
left the connection pool with no owner to close it.
"""

from __future__ import annotations

from fastapi import Request

from src.services.case_detection_service import CaseDetectionService
from src.services.case_summarization_service import CaseSummarizationService


def get_case_detection_service(request: Request) -> CaseDetectionService:
    return request.app.state.case_detection_service


def get_case_summarization_service(request: Request) -> CaseSummarizationService:
    return request.app.state.case_summarization_service
