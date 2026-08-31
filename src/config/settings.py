from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv(override=True)

class Settings:
    """
    A centralized class to hold all application settings loaded from environment variables.
    """
    def __init__(self):
        self.DOCS_USERNAME = "admin"
        self.DOCS_PASSWORD = "1234"

        # OpenAI / case detection LLM
        self.OPENAI_MODEL = "gpt-4.1-nano-2025-04-14"
        self.OPENAI_API_KEY = self._get_required("OPENAI_API_KEY")

    @staticmethod
    def _get_required(key: str) -> str:
        """Get a required environment variable or raise ValueError."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"{key} not found in environment variables. Please check your .env file.")
        return value


# Create a single instance of the settings to be imported across the application
settings = Settings()
