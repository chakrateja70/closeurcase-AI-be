from fastapi import APIRouter

from src.api import case_detection, case_summarization

api_router = APIRouter()

api_router.include_router(case_detection.router)
api_router.include_router(case_summarization.router)
