"""Startup configuration: which summarization provider, and which keys it needs.

`settings` is a module-level singleton built at import, so these tests build
their own `Settings()` against a patched environment rather than reading the
one the rest of the suite shares. That is the only way to exercise a
misconfiguration: by the time a bad value could be observed on the singleton,
the process would already have failed to start - which is the whole point.

The rule being pinned is narrow and easy to get backwards. SUMMARY_PROVIDER
decides which key *summarization* requires. It does not decide whether
OPENAI_API_KEY is required at all, because case detection runs on OpenAI and
has no alternative provider - an app without that key could not serve
/case_detection/detect whichever summarizer is selected.
"""

import pytest

from src.config.settings import (
    DEFAULT_SUMMARY_PROVIDER,
    PROVIDER_GEMINI,
    PROVIDER_GPT,
    SUPPORTED_SUMMARY_PROVIDERS,
    ConfigurationError,
    Settings,
)

# Everything Settings requires regardless of provider. Each test starts from a
# valid deployment and breaks exactly one thing, so a failure names the rule it
# broke rather than an unrelated missing variable.
BASE_ENV = {
    "DOCS_USERNAME": "u",
    "DOCS_PASSWORD": "p",
    "OPENAI_API_KEY": "sk-test",
    "DETECT_RATE_LIMIT": "10/minute",
    "SUMMARIZE_RATE_LIMIT": "3/minute",
}


@pytest.fixture
def env(monkeypatch):
    """A valid deployment, which each test then edits."""

    def apply(**overrides):
        for key, value in {**BASE_ENV, **overrides}.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        return Settings

    monkeypatch.delenv("SUMMARY_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return apply


# --- Choosing the provider --------------------------------------------------


def test_the_provider_defaults_to_gpt_when_unset(env):
    """Unset is not a misconfiguration - it is a deployment that predates this
    variable, and it must keep working exactly as it did."""
    env()

    assert Settings().SUMMARY_PROVIDER == DEFAULT_SUMMARY_PROVIDER == PROVIDER_GPT


@pytest.mark.parametrize("provider", SUPPORTED_SUMMARY_PROVIDERS)
def test_every_supported_provider_is_accepted(env, provider):
    env(SUMMARY_PROVIDER=provider, GEMINI_API_KEY="gm-test")

    assert Settings().SUMMARY_PROVIDER == provider


@pytest.mark.parametrize(
    "raw", ["  gemini  ", "GEMINI", "Gemini"], ids=["padded", "upper", "title"]
)
def test_case_and_whitespace_are_forgiven(env, raw):
    """A trailing space in a .env line is a copy-paste artefact, not a typo
    worth refusing to start over."""
    env(SUMMARY_PROVIDER=raw, GEMINI_API_KEY="gm-test")

    assert Settings().SUMMARY_PROVIDER == PROVIDER_GEMINI


def test_blank_is_treated_as_unset(env):
    """`SUMMARY_PROVIDER=` in a .env file is a variable someone commented out
    by deleting the value, not a request for a provider named ''."""
    env(SUMMARY_PROVIDER="   ")

    assert Settings().SUMMARY_PROVIDER == DEFAULT_SUMMARY_PROVIDER


@pytest.mark.parametrize("raw", ["gtp", "openai", "claude", "gpt5", "none"])
def test_an_unrecognised_provider_stops_startup(env, raw):
    """The failure that matters. Falling back to the default would leave
    someone convinced they were running on the other model - so a value that is
    set but unknown is fatal, and the message names what is valid."""
    env(SUMMARY_PROVIDER=raw)

    with pytest.raises(ConfigurationError) as excinfo:
        Settings()

    message = str(excinfo.value)
    assert "SUMMARY_PROVIDER" in message
    assert raw in message
    for supported in SUPPORTED_SUMMARY_PROVIDERS:
        assert supported in message


# --- Which key each provider requires ---------------------------------------


def test_gemini_key_is_not_required_when_running_on_gpt(env):
    """The ordinary single-provider deployment: one key, one provider, and the
    app starts."""
    env(SUMMARY_PROVIDER=PROVIDER_GPT, GEMINI_API_KEY=None)

    settings = Settings()

    assert settings.SUMMARY_PROVIDER == PROVIDER_GPT
    assert settings.GEMINI_API_KEY is None


def test_selecting_gemini_without_its_key_stops_startup(env):
    """Every summarization request would 503, so this fails at startup rather
    than at the first upload - and the message says both ways out."""
    env(SUMMARY_PROVIDER=PROVIDER_GEMINI, GEMINI_API_KEY=None)

    with pytest.raises(ConfigurationError) as excinfo:
        Settings()

    message = str(excinfo.value)
    assert "GEMINI_API_KEY" in message
    assert PROVIDER_GEMINI in message and PROVIDER_GPT in message


def test_selecting_gemini_with_its_key_starts(env):
    env(SUMMARY_PROVIDER=PROVIDER_GEMINI, GEMINI_API_KEY="gm-test")

    settings = Settings()

    assert settings.SUMMARY_PROVIDER == PROVIDER_GEMINI
    assert settings.GEMINI_API_KEY == "gm-test"


def test_openai_key_stays_required_even_on_gemini(env):
    """Deliberate, and the one place this rule is asymmetric: case detection
    runs on OpenAI and has no alternative provider, so the app cannot serve
    /case_detection/detect without this key whichever summarizer is selected.
    SUMMARY_PROVIDER governs summarization's requirements, not the whole app's.
    """
    env(
        SUMMARY_PROVIDER=PROVIDER_GEMINI,
        GEMINI_API_KEY="gm-test",
        OPENAI_API_KEY=None,
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings()
