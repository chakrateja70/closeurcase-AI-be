"""Test-run configuration.

`src/config/settings.py` raises at import for a missing required variable, so
any test touching a service - which imports settings transitively - needs them
present. These are placeholders: nothing in the suite makes a network call, and
the key is never used.

`setdefault`, not assignment: settings calls `load_dotenv(override=True)`, so a
real `.env` wins anyway, and a developer running the suite locally gets their
own values rather than these. This only fills the gap where no `.env` exists,
which is CI and a fresh clone.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
os.environ.setdefault("DOCS_USERNAME", "test")
os.environ.setdefault("DOCS_PASSWORD", "test")
os.environ.setdefault("DETECT_RATE_LIMIT", "10/minute")
os.environ.setdefault("SUMMARIZE_RATE_LIMIT", "3/minute")


import pytest


@pytest.fixture
def anyio_backend():
    """Async tests run under `@pytest.mark.anyio`, whose plugin ships with the
    anyio that starlette already pulls in - so no extra test dependency. It
    needs this fixture to know which backend to run on, and the app runs on
    asyncio.
    """
    return "asyncio"
