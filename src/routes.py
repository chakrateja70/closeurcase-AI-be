from fastapi import APIRouter, Depends
from src.api import case_detection

api_router = APIRouter()

api_router.include_router(case_detection.router)