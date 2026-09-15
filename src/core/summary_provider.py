"""Resolves which LLM provider backs a case-summarization request."""

from enum import Enum
from src.config.settings import settings
from src.core.exceptions import UnavailableSummaryProviderError

class SummaryProvider(str, Enum):
    GPT = "gpt"
    GEMINI = "gemini"

def available_summary_providers() -> list[SummaryProvider]:
    providers = [SummaryProvider.GPT]
    if settings.GEMINI_API_KEY:
        providers.append(SummaryProvider.GEMINI)
    return providers

def resolve_summary_provider(requested: str | None) -> SummaryProvider:
    """Resolve the provider from an override or the configured default."""
    provider = SummaryProvider(requested) if requested else SummaryProvider(settings.SUMMARY_PROVIDER)
    available = available_summary_providers()
    if provider not in available:
        raise UnavailableSummaryProviderError(
            provider.value, [p.value for p in available]
        )
    return provider
