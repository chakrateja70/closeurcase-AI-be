from __future__ import annotations

import logging
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from src.config.settings import settings

logger = logging.getLogger(__name__)

REDACTED = "[case content redacted]"
TRACING_ENABLED = bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY)

_client: Langfuse | None = None


def _mask(*, data: object, **_: object) -> str:
    return REDACTED

def init_tracing() -> None:
    """Register Langfuse once with masking configured."""
    global _client
    if not TRACING_ENABLED:
        return
    _client = Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
        mask=_mask,
    )
    logger.info("langfuse tracing enabled host=%s", settings.LANGFUSE_HOST)

def shutdown_tracing() -> None:
    if _client is not None:
        _client.shutdown()

def callback_handler() -> CallbackHandler | None:
    return CallbackHandler() if TRACING_ENABLED else None
