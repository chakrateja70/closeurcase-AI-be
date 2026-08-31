# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Fresh scaffold, no commits yet. `main.py`, one placeholder endpoint, and a directory skeleton are all that exist — most of `src/` is empty dirs waiting to be filled. Expect to be creating structure, not navigating it.

## Environment & commands

Package manager is **uv** (`uv.lock` is authoritative; `requirements.txt` is a stale one-line stub — don't treat it as the dependency list). Python 3.12 per `.python-version`.

```bash
uv sync                              # install from uv.lock into .venv
uv add <pkg>                         # add a dep (updates pyproject.toml + uv.lock)
uv run python main.py                # run the API (uvicorn on 0.0.0.0:8000, reload=True)
uv run uvicorn main:app --reload     # equivalent
```

No test runner, linter, or formatter is configured yet. If tests are needed, add pytest via `uv add --dev pytest` and run `uv run pytest -k <name>` for a single test.

**Known gap:** `main.py` imports `uvicorn`, but it is neither in `pyproject.toml` nor installed in `.venv`. `uv add uvicorn` (or `uvicorn[standard]`) before trying to start the server.

## Architecture

Layered FastAPI app rooted at `main.py`, which builds the `FastAPI` instance, mounts CORS (currently fully open: `allow_origins=["*"]` with credentials), and includes a single aggregate router.

Router wiring is two-level and deliberate — follow it when adding endpoints:

1. `src/api/<feature>.py` defines a module-local `APIRouter` with its own `prefix` and `tags` (see `src/api/case_detection.py`, prefix `/case_detection`).
2. `src/routes.py` owns `api_router` and calls `api_router.include_router(...)` for each feature module.
3. `main.py` includes only `api_router`.

So a new endpoint group means: new file under `src/api/`, then one `include_router` line in `src/routes.py`. Nothing else in `main.py` changes.

The empty directories encode the intended separation, and new code should land accordingly rather than accumulating in the route handlers:

- `src/api/` — HTTP layer only: request/response shapes, dependency injection, delegation.
- `src/services/` — business logic, called from `src/api/`.
- `src/core/` — cross-cutting app internals. Holds `security.py` (HTTP Basic dependency guarding the docs); exceptions and shared base classes belong here too.
- `src/db/` — persistence; nothing chosen yet, so a DB decision here is a real architectural choice, not a fill-in.
- `src/prompts/` — LLM prompt templates (the "AI" half of the product; no LLM client is wired up yet).
- `src/utils/` — generic helpers with no domain knowledge.
- `src/config/settings.py` — plain `os.getenv` config with defaults; currently just the docs credentials. If it grows, pydantic-settings is the natural fit (pydantic is already a transitive dep).

There are no `__init__.py` files anywhere; imports work via implicit namespace packages, and absolute imports rooted at `src.` (e.g. `from src.routes import api_router`) are the convention. Keep it that way or convert the whole tree at once — don't mix styles.

`/health` is defined directly on `app` in `main.py`, outside the router tree, and returns a bare string.

## Protected API docs

The built-in docs are disabled (`docs_url`/`redoc_url`/`openapi_url` are all `None`) and re-served from `main.py` as three routes that each depend on `verify_docs_access`: `/docs`, `/redoc`, and `/openapi.json`. All three must stay gated — Swagger UI fetches the schema from the browser, so leaving `/openapi.json` open would publish the whole API surface regardless of the page being protected.

Credentials come from `DOCS_USERNAME` / `DOCS_PASSWORD` (defaults `admin` / `1234`). Because `openapi_url=None` on the app, `get_openapi(...)` is called explicitly in the route rather than reading `app.openapi()`.
