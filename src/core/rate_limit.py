"""Per-client rate limiting.

`/case_detection/detect` is unauthenticated and every call costs money at the
model provider, so it needs a ceiling that does not depend on callers behaving.
The limit is keyed on the client IP and configured by env so it can be tuned
without a deploy.

Storage is in-process, which means the limit is per worker: running N uvicorn
workers allows roughly N times the configured rate. Point `RATE_LIMIT_STORAGE`
at a redis:// URL to share one budget across workers and hosts.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config.settings import settings

# Requests per window, slowapi syntax ("20/minute", "5/second", ...).
DETECT_RATE_LIMIT = settings.DETECT_RATE_LIMIT

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.RATE_LIMIT_STORAGE,
    headers_enabled=True,
)
