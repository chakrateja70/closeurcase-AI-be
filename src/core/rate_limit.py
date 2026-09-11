"""Per-client rate limiting.

`/case_detection/detect` and `/case_summarization/summarize` are both
unauthenticated and every call costs money at the model provider, so each needs
a ceiling that does not depend on callers behaving. The limits are keyed on the
client IP and configured by env so they can be tuned without a deploy.

They are separate settings because the two endpoints are not comparable in
cost: a classification sends at most 600 characters, while a summarization
sends an entire filing - and, for a scanned one, every page as an image.

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
SUMMARIZE_RATE_LIMIT = settings.SUMMARIZE_RATE_LIMIT

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.RATE_LIMIT_STORAGE,
    headers_enabled=True,
)
