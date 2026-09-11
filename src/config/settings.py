from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# The summarization providers this build knows how to construct.

PROVIDER_GPT = "gpt"
PROVIDER_GEMINI = "gemini"
SUPPORTED_SUMMARY_PROVIDERS = (PROVIDER_GPT, PROVIDER_GEMINI)

DEFAULT_SUMMARY_PROVIDER = PROVIDER_GPT


class ConfigurationError(ValueError):
    """The deployment is misconfigured and cannot start.

    Distinct from the API exceptions in `core.exceptions`: nothing here is a
    response to a request. It is raised at import, so the process fails on
    startup with a message naming the variable rather than serving traffic that
    500s on first use. A plain ValueError subclass so this module stays free of
    FastAPI - everything imports settings, including code that must be usable
    without a request.
    """


class Settings:
    """A centralized class to hold all application settings loaded from environment variables."""
    def __init__(self):
        self.DOCS_USERNAME = self._get_required("DOCS_USERNAME")
        self.DOCS_PASSWORD = self._get_required("DOCS_PASSWORD")

        # Which model summarization uses when a request does not name one.
        # Validated here rather than at first use so a typo is a startup
        # failure naming the valid values, not a 503 the first time somebody
        # submits a document.
        self.SUMMARY_PROVIDER = self._get_summary_provider()

        self.OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"
        # Summarization reads whole court filings, including scanned pages as
        # images, so it is kept separate from the classifier's model - the two
        # can be tuned independently without touching the other feature.
        self.OPENAI_SUMMARY_MODEL = "gpt-4.1-mini-2025-04-14"
        self.OPENAI_API_KEY = self._get_required("OPENAI_API_KEY")

        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.GEMINI_MODEL = "gemini-2.5-flash"
        if self.SUMMARY_PROVIDER == PROVIDER_GEMINI and not self.GEMINI_API_KEY:
            raise ConfigurationError(
                f"SUMMARY_PROVIDER is '{PROVIDER_GEMINI}' but GEMINI_API_KEY is "
                f"not set. Set GEMINI_API_KEY, or change SUMMARY_PROVIDER to "
                f"'{PROVIDER_GPT}'."
            )

        # Rate limiting. The storage backend is optional: "memory://" keeps
        # counters in-process, so the limit is per worker. Point it at a
        # redis:// URL to share one budget across workers and hosts.
        self.DETECT_RATE_LIMIT = self._get_required("DETECT_RATE_LIMIT")
        self.SUMMARIZE_RATE_LIMIT = self._get_required("SUMMARIZE_RATE_LIMIT")
        self.RATE_LIMIT_STORAGE = os.getenv("RATE_LIMIT_STORAGE", "memory://")

        # Logging verbosity for the app's own loggers.
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @staticmethod
    def _get_required(key: str) -> str:
        """Get a required environment variable or raise ValueError."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"{key} not found in environment variables. Please check your .env file.")
        return value

    @staticmethod
    def _get_summary_provider() -> str:
        """The default summarization provider, validated against what exists."""
        raw = os.getenv("SUMMARY_PROVIDER")
        if raw is None or not raw.strip():
            return DEFAULT_SUMMARY_PROVIDER

        value = raw.strip().lower()
        if value not in SUPPORTED_SUMMARY_PROVIDERS:
            raise ConfigurationError(
                f"SUMMARY_PROVIDER must be one of "
                f"{', '.join(SUPPORTED_SUMMARY_PROVIDERS)}; got {raw!r}."
            )
        return value


# Create a single instance of the settings to be imported across the application
settings = Settings()
