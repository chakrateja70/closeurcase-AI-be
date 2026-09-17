from __future__ import annotations
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config.settings import settings

# Requests per window, slowapi syntax ("20/minute", "5/second", ...).
DETECT_RATE_LIMIT = settings.DETECT_RATE_LIMIT
SUMMARIZE_RATE_LIMIT = settings.SUMMARIZE_RATE_LIMIT

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.RATE_LIMIT_STORAGE,
    headers_enabled=True,
)
