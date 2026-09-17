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

`requirements.txt` is stale and incomplete — it would produce a broken install.
Treat `pyproject.toml` + `uv.lock` as the only dependency source.
`[tool.pytest.ini_options] pythonpath = ["."]` is what makes `src.`-rooted
imports resolve under pytest; there are no `__init__.py` files to infer it from.

## Configuration

All config goes through `src/config/settings.py`; there are no `os.getenv`
calls elsewhere in the tree, and new settings should keep it that way.

`load_dotenv(override=True)` means **`.env` beats real environment variables** —
the reverse of the usual convention. `FOO=bar uv run ...` is silently ignored
if `FOO` is also in `.env`; edit `.env` instead. This will matter when
deploying to a platform that injects config as env vars.

`_get_required` raises at import if a var is missing, so `OPENAI_API_KEY`,
`DOCS_USERNAME`, `DOCS_PASSWORD`, `DETECT_RATE_LIMIT` and `SUMMARIZE_RATE_LIMIT`
must all be present to start. `SUMMARY_PROVIDER` is validated at startup
(`gpt` | `gemini`) so a typo fails the app rather than the first request, and
`GEMINI_API_KEY` becomes required only when it is `gemini`. Model ids
(`OPENAI_MODEL`, `GEMINI_MODEL`) are hardcoded in settings, and the token /
timeout / retry constants live as module-level constants in the services.

`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` are the one
exception to the "required" pattern above: they gate optional tracing (see
below) and a missing pair does not fail startup — `src/core/tracing.py`
just skips it.

## Architecture

Layered FastAPI app. Router wiring is two-level: `src/api/<feature>.py` owns a
prefixed `APIRouter`, `src/routes.py` aggregates them into `api_router`, and
`main.py` includes only that. A new endpoint group means a new file under
`src/api/` plus one `include_router` line — nothing in `main.py` changes.

There are no `__init__.py` files anywhere; imports work via implicit namespace
packages, and absolute imports rooted at `src.` are the convention. Keep it
that way or convert the whole tree at once.

Two features: **case detection** (`/case_detection/*`) and **case
summarization** (`/case_summarization/*`). They are deliberately asymmetric —
detection is OpenAI-only and taxonomy-driven; summarization is
provider-agnostic and free-form. Don't "unify" them without reading both
sections below.

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

Note this service still builds its own `AsyncOpenAI` client and maps OpenAI
errors inline, rather than going through `llm_service`. That duplication is
current state, not a rule — but it means an error-mapping or timeout change
made in `llm_service` does *not* reach detection.

### Summarization: one call shape, two providers

`src/services/llm_service.py` is the provider abstraction, built on LangChain's
provider integrations (`langchain-openai`, `langchain-google-genai`) rather
than the raw `openai`/`google-genai` SDKs directly. `LLMClient` exposes a
single method — `complete_json(instructions, parts, schema, schema_name,
max_output_tokens, trace_label)` — implemented by `OpenAILLMClient` and
`GeminiLLMClient`, both thin subclasses of `_LangChainLLMClient` that differ
only in which `ChatOpenAI`/`ChatGoogleGenerativeAI` instance they wrap and
which `with_structured_output` kwargs they pass (`strict=True` for OpenAI
only). Content is passed as provider-neutral `TextPart` / `DocumentPart`
dataclasses and translated once, in `_content_blocks`, into LangChain's
standard v1 message content blocks (`{"type": "text", ...}` /
`{"type": "file", "mime_type": ..., "base64": ...}`) — both chat models
translate those to their own wire format internally, so there is no
per-provider branching left in this file. Adding a third summarization
provider means adding one more `_LangChainLLMClient` subclass here, wiring
its API key through `settings.py` and `summary_provider.py`, and nothing
else.

Every provider error surfaces through LangChain's own provider-agnostic
hierarchy (`langchain_core.exceptions.Model*Error` — `ModelAuthenticationError`,
`ModelRateLimitError`, `ModelTimeoutError`, etc.), which `ChatOpenAI` and
`ChatGoogleGenerativeAI` both populate under the hood by wrapping the same
`openai`/`google-genai` exceptions the old raw-SDK version caught directly.
`_LangChainLLMClient.complete_json` catches that one shared hierarchy and
re-raises the `LLM*Error` classes in `src/core/exceptions.py`, so callers see
one failure shape whichever backend ran — the mapping used to be duplicated
per provider; now it isn't.

Schemas no longer need `additionalProperties: false` hand-injected — LangChain's
`with_structured_output(..., method="json_schema")` does that recursively for
OpenAI strict mode on its own. The one thing schemas must still carry
themselves is a top-level `"title"` (see `case_summarization_prompt.py`),
which LangChain uses as the schema/function name; without it,
`with_structured_output` raises at call time, not at import time.

The two providers don't even agree on the invocation kwarg for
`max_output_tokens`: `ChatOpenAI` forwards straight to the Chat
Completions/Responses API, which calls it `max_tokens`; `ChatGoogleGenerativeAI`'s
config object only accepts `max_output_tokens` and raises a pydantic
`ValidationError` (not one of the `Model*Error` types) if you pass `max_tokens`
instead. Each subclass names its own kwarg via `_max_tokens_kwarg` for exactly
this reason — collapsing it back to one shared name will break Gemini silently
until the first real call.

`GeminiLLMClient.aclose()` is intentionally a no-op — google-genai (still the
library `ChatGoogleGenerativeAI` wraps internally) owns its httpx clients
internally, unlike `AsyncOpenAI`'s pool, which `OpenAILLMClient.aclose()`
reaches via `ChatOpenAI.root_async_client`.

### Optional Langfuse tracing

`src/core/tracing.py` wires a Langfuse callback handler into every
summarization LLM call via LangChain's callback system
(`langfuse.langchain.CallbackHandler`), for both providers, at no cost to
callers who haven't configured it. `TRACING_ENABLED` is computed once from
`settings.LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`; when either is empty,
`callback_handler()` returns `None` and `_LangChainLLMClient.complete_json`
passes an empty `RunnableConfig` — tracing is best-effort, never a hard
dependency for summarization to work.

`init_tracing()` must run in `main.py`'s lifespan *before* any request can
reach `callback_handler()`: Langfuse's `get_client()` lazily creates a
default client from raw env vars on first use, and a client created that way
would not carry the `mask` function configured below. Constructing the
`Langfuse(...)` singleton explicitly at startup is what guarantees every
later `CallbackHandler()` call reuses it.

Case text and document content are sensitive (see "Known gaps" below), so the
client is configured with `mask` set to a function that unconditionally
redacts every span's `input`/`output`/`metadata` before export — Langfuse
still receives `model`, `usage_details`, `tags`, and latency (these are
separate, non-maskable span fields), but never the case content itself. If a
future change needs partial visibility into trace content, it has to happen
in that `_mask` function, not by disabling masking outright.

Provider selection is three-layered: `settings.SUMMARY_PROVIDER` is the default,
`available_summary_providers()` (`src/core/summary_provider.py`) filters to the
ones with a key actually configured, and a per-request `model` field overrides
for that call. Asking for an unconfigured provider returns 503 naming what *is*
available, and `GET /case_summarization/models` advertises the same list.
`CaseSummarizationService` holds a dict of clients — one per available provider,
built once — rather than one client.

**Input is either documents or text, never both.** `src/core/case_input.py`
validates and classifies into a `CaseInput`; the API model rejects zero or two
inputs before that. PDFs are fetched by URL and handed to the model as bytes,
never parsed here. Text input goes through the sanitization pipeline below and
returns a canned brief when blocked; document input does not (the prompt's
"ignore embedded instructions" rule is the only layer there).

Document fetching in `case_summarization_service.py` has four guards that exist
for a reason: `_guard_against_private_host` resolves the host and rejects
private/loopback/link-local/reserved/multicast addresses — this is the SSRF
check, since a `document_url` could otherwise point at cloud metadata or an
internal service — plus a declared-`content-length` check, a streaming byte cap
(15MB), and a `%PDF-` magic-bytes check after download. The URL-side checks in
`case_input.py` (http/https only, host present, `.pdf` suffix) are separate and
run first. Fetch failures raise `BadRequest`, not 502/504, because the upstream
that failed is a URL the *client* supplied.

### Input sanitization runs in a fixed order

`clean_text` → `find_security_issue` → `flatten` (see the header comment in
`src/utils/helper.py`). Cleaning deliberately **preserves line breaks** because
the injection patterns anchor to a sentence or line start; flattening before
scanning would erase that boundary and let payloads through. This ordering is
pinned by `test/test_helper.py` — don't collapse the two steps.

The scanner is intentionally the weaker of two layers, leaning on the prompt's
own "ignore embedded instructions" rule, because a false positive silently
rejects a real client describing a real dispute. The tests carry a
`BYPASS_CORPUS` (payloads that must be caught) and an `INNOCENT_CORPUS`
(ordinary legal text that must not be) — extend both when touching patterns.

### Cross-cutting behaviour in `main.py`

- **Lifespan owns services.** Both services are built in the lifespan and
  stashed on `app.state`, resolved per-request via `src/api/deps.py`, and
  `aclose()`d on shutdown. Constructing them at module import instead would make
  the API keys a requirement just to import the routes and leave the HTTP pools
  with no owner to close.
- **One error envelope.** Handlers in `src/core/exceptions.py` flatten every
  error path — our exceptions, plain `HTTPException`, validation failures, rate
  limits, unhandled crashes — onto the same flat snake_case shape the success
  responses use, instead of FastAPI's nested `{"detail": ...}`. Exceptions are
  organised by *response*, not by source: subclass the right `*APIException`
  (e.g. `BadRequestAPIException` for a bad `document_url`) and the status code
  and message come with it.
- **Rate limiting** via slowapi on `/case_detection/detect` and
  `/case_summarization/summarize`, keyed on client IP, with separate limits —
  summarization should stay well below detection since a call uploads a whole
  document. A rate-limited route handler must take both `request: Request` and
  `response: Response` — slowapi reads the address off one and writes
  `X-RateLimit-*` headers onto the other, and omitting `response` raises at
  request time, not import time.
- **Middleware order is load-bearing.** `RequestContextMiddleware`
  (`src/core/request_context.py`) is added last so it is outermost and still
  logs the requests `SlowAPIMiddleware` rejects. It logs one line per request
  with the caller IP and a running per-IP count. `client_ip()` is the single
  place the address is resolved — behind a proxy, run uvicorn with
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
- Both LLM endpoints are unauthenticated and cost money per call; the rate
  limit is the only thing standing in front of the bill.
- Logs pair a caller's IP with the text of their legal problem — sensitive in
  this domain. Worth demoting the full-query lines to `DEBUG` for production.
- `src/api/counter_generation.py` is empty and unreferenced.
- Only `src/utils/helper.py` has tests; the document-fetch guards, provider
  resolution, and error handlers are untested despite being pure and easy to
  test. `llm_service.py`'s LangChain-based clients and `tracing.py`'s masking
  are in the same untested category.
- Langfuse tracing is a new outbound network dependency of the summarization
  path (batched/async span export, not on the request's critical path), so a
  Langfuse outage should not fail requests — but this isn't covered by a
  test, only by `TRACING_ENABLED`'s best-effort design.
