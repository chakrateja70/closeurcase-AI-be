from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv(override=True)

class Settings:
    """
    A centralized class to hold all application settings loaded from environment variables.
    """
    def __init__(self):
        self.DOCS_USERNAME = self._get_required("DOCS_USERNAME")
        self.DOCS_PASSWORD = self._get_required("DOCS_PASSWORD")

        self.OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"
        self.OPENAI_API_KEY = self._get_required("OPENAI_API_KEY")

        # Rate limiting. The storage backend is optional: "memory://" keeps
        # counters in-process, so the limit is per worker. Point it at a
        # redis:// URL to share one budget across workers and hosts.
        self.DETECT_RATE_LIMIT = self._get_required("DETECT_RATE_LIMIT")
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


# Create a single instance of the settings to be imported across the application
settings = Settings()
