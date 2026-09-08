# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & commands

Package manager is **uv** (`uv.lock` is authoritative). Python 3.12 per `.python-version`.

**Always run through `uv run`.** A bare `python` on this machine resolves to a
different interpreter (system 3.11) that does not have the project's
dependencies installed, so `python foo.py` fails or silently tests the wrong
environment.

```bash
uv sync                              # install from uv.lock into .venv
uv run uvicorn main:app --reload     # run the API on :8000
uv run python main.py                # equivalent
uv run pytest -q                     # all tests
uv run pytest -k newline             # a single test by name
```

`requirements.txt` is stale and incomplete (missing `openai`, `uvicorn`) — it
would produce a broken install. Treat `pyproject.toml` + `uv.lock` as the only
dependency source.

## Configuration

All config goes through `src/config/settings.py`; there are no `os.getenv`
calls elsewhere in the tree, and new settings should keep it that way.

`load_dotenv(override=True)` means **`.env` beats real environment variables** —
the reverse of the usual convention. `FOO=bar uv run ...` is silently ignored
if `FOO` is also in `.env`; edit `.env` instead. This will matter when
deploying to a platform that injects config as env vars.

`_get_required` raises at import if a var is missing, so `OPENAI_API_KEY`,
`DOCS_USERNAME`, `DOCS_PASSWORD` and `DETECT_RATE_LIMIT` must all be present to
start. `OPENAI_MODEL` is still hardcoded despite `.env.example` advertising it,
as are the `OPENAI_MAX_OUTPUT_TOKENS` / `TIMEOUT` / `MAX_RETRIES` constants in
the service.

## Architecture

Layered FastAPI app. Router wiring is two-level: `src/api/<feature>.py` owns a
prefixed `APIRouter`, `src/routes.py` aggregates them into `api_router`, and
`main.py` includes only that. A new endpoint group means a new file under
`src/api/` plus one `include_router` line — nothing in `main.py` changes.

There are no `__init__.py` files anywhere; imports work via implicit namespace
packages, and absolute imports rooted at `src.` are the convention. Keep it
that way or convert the whole tree at once.

### Case detection is the whole product

One feature spans five files, and the design intent is not obvious from any one
of them:

- `src/data/case.json` — the taxonomy, **the only place to edit it**. Three
  levels: category → case_type → legal_services.
- `src/core/case_categories.py` — parses that JSON at import and derives ids as
  slugs from titles. Adding a case type to the JSON automatically updates both
  the prompt and the LLM's enum. Duplicate ids raise at import.
- `src/prompts/case_detection_prompt.py` — prompt text plus the structured-output
  schema, both rendered from the taxonomy.
- `src/services/case_detection_service.py` — the model call and normalisation.
- `src/api/case_detection.py` — request/response shapes only.

**The model chooses a case type and nothing else.** The parent category and the
legal services are looked up server-side in `_expand`, so they cannot
contradict the case type, and the model spends no tokens deciding them.
Categories are not even shown to it. A `secondary_case_type_id` is set only
when the facts carry a genuinely different *remedy* (prosecution vs.
compensation vs. recovery) — that rule is load-bearing and easy to break by
rewording.

Detection is verified against live API calls, not unit tests. When changing the
prompt or taxonomy, re-run a spread of queries and check the primary ids, since
prompt edits regress classification silently.

### Input sanitization runs in a fixed order

`clean_text` → `find_security_issue` → `flatten` (see the header comment in
`src/utils/helper.py`). Cleaning deliberately **preserves line breaks** because
the injection patterns anchor to a sentence or line start; flattening before
scanning would erase that boundary and let payloads through. This ordering is
pinned by tests — don't collapse the two steps.

The scanner is intentionally the weaker of two layers, leaning on the prompt's
own "ignore embedded instructions" rule, because a false positive silently
rejects a real client describing a real dispute.

### Cross-cutting behaviour in `main.py`

- **Lifespan owns services.** `CaseDetectionService` is built in the lifespan
  and stashed on `app.state`, resolved per-request via `src/api/deps.py`.
  Constructing it at module import instead would make `OPENAI_API_KEY` a
  requirement just to import the routes and leave the HTTP pool with no owner
  to close.
- **One error envelope.** Handlers in `src/core/exceptions.py` flatten every
  error path — our exceptions, plain `HTTPException`, validation failures, rate
  limits, unhandled crashes — onto the same flat snake_case shape the success
  responses use, instead of FastAPI's nested `{"detail": ...}`.
- **Rate limiting** via slowapi on `/case_detection/detect`, keyed on client IP.
  A rate-limited route handler must take both `request: Request` and
  `response: Response` — slowapi reads the address off one and writes
  `X-RateLimit-*` headers onto the other, and omitting `response` raises at
  request time, not import time.
- **Request logging** middleware (`src/core/request_context.py`) logs one line
  per request with the caller IP and a running per-IP count. `client_ip()` is
  the single place the address is resolved — behind a proxy, run uvicorn with
  `--proxy-headers --forwarded-allow-ips=<proxy>` rather than reading
  `X-Forwarded-For` in code, which is spoofable.

Rate-limit counters and request counts are in-process (`memory://`), so both
are **per worker** and reset on restart. Running multiple workers multiplies
the effective limit; point `RATE_LIMIT_STORAGE` at a `redis://` URL to share
one budget.

## Protected API docs

The built-in docs are disabled (`docs_url`/`redoc_url`/`openapi_url` are
`None`) and re-served from `main.py` as three routes that each depend on
`verify_docs_access`. All three must stay gated — Swagger UI fetches the schema
from the browser, so leaving `/openapi.json` open would publish the whole API
surface regardless of the page being protected. Because `openapi_url=None`,
`get_openapi(...)` is called explicitly rather than reading `app.openapi()`.

## Known gaps

- CORS is `allow_origins=["*"]` with `allow_credentials=True`, which lets any
  origin make credentialed requests. Needs a real allowlist before launch.
- `/detect` is unauthenticated and costs money per call; the rate limit is the
  only thing standing in front of the OpenAI bill.
- Logs pair a caller's IP with the text of their legal problem — sensitive in
  this domain. Worth demoting the full-query lines to `DEBUG` for production.
- `src/api/counter_generation.py` is empty and unreferenced.
