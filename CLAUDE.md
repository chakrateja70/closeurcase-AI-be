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
`DOCS_USERNAME`, `DOCS_PASSWORD`, `DETECT_RATE_LIMIT` and `SUMMARIZE_RATE_LIMIT`
must all be present to start. `OPENAI_MODEL`, `OPENAI_SUMMARY_MODEL` and
`GEMINI_MODEL` are hardcoded, as are the output-token / timeout / retry
constants in `summary_providers`.

`SUMMARY_PROVIDER` (`gpt` | `gemini`) picks which model summarization uses when
a request names none. Three rules, all enforced in `settings.py` at import:

- **Unset means `gpt`.** A deployment that predates the variable keeps working.
- **A value that is set but unrecognised is fatal** — `ConfigurationError`
  naming the valid options. Falling back to the default would leave someone
  convinced they were running on the other model. Case and surrounding
  whitespace are forgiven; spelling is not.
- **The selected provider's key is required; the other's is not.**
  `SUMMARY_PROVIDER=gemini` without `GEMINI_API_KEY` fails at startup, because
  every summarization request would otherwise 503.

**`OPENAI_API_KEY` is required regardless, and that is deliberate** — the one
place the rule above is asymmetric. Case detection runs on OpenAI and has no
alternative provider, so an app without that key could not serve
`/case_detection/detect` whichever summarizer is selected. `SUMMARY_PROVIDER`
governs what *summarization* demands, not what the app demands. A test pins
this so it does not get "fixed" into a Gemini-only boot that silently breaks
detection.

`ConfigurationError` lives in `settings.py`, not `core/exceptions.py`: nothing
about it is a response to a request, and settings must stay importable without
FastAPI.

`test/conftest.py` fills those in with placeholders via `setdefault`, so the
suite runs on a fresh clone with no `.env`. A real `.env` still wins, because
of the `override=True` above.

## Architecture

Layered FastAPI app. Router wiring is two-level: `src/api/<feature>.py` owns a
prefixed `APIRouter`, `src/routes.py` aggregates them into `api_router`, and
`main.py` includes only that. A new endpoint group means a new file under
`src/api/` plus one `include_router` line — nothing in `main.py` changes.

There are no `__init__.py` files anywhere; imports work via implicit namespace
packages, and absolute imports rooted at `src.` are the convention. Keep it
that way or convert the whole tree at once.

### Two features, one taxonomy — no longer

Case detection (typed query) and case summarization (a linked or pasted
document) are
separate stacks. They used to meet at `src/core/case_categories.py`: both ended
with a model-chosen case-type id and both called `expand_case_type` to turn it
into the same primary/secondary block. **Summarization no longer classifies at
all** — it answers three questions about the document and nothing else, so the
taxonomy, `expand_case_type` and `resolve_case_types` are now case detection's
alone. If summarization ever needs to route a document to services again, call
those same two helpers rather than growing a second expansion.

### Case detection

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

### The summary answers three questions and stops

`SummarizeDocumentResponse` carries what happened (`chronology`), what the
dispute is (`facts_summary`), and what the parties are arguing (`assertions`) —
plus `parties`, naming who is arguing, and grounding, confidence and provenance.
Nothing else.

The court, case number, title, filing date, jurisdiction, document type, legal
issues, monetary claims, statutes, precedents, exhibits, key dates, open
questions and the whole case-type block were all extracted once and have been
**deliberately removed**. Whoever sent the filing can read that off its
first page; restating it spent output tokens, lengthened the response, and
buried the fields that actually say something. Adding a field back means
arguing that a lawyer holding the document cannot already see it — `parties`
earns its place on exactly that test, because the other three fields read as
anonymous without it.

**The response is JSON so that each line can be checked, not because JSON is
the only option.** A prose summary is a legitimate shape for this feature and
would be cheaper to render — but there would be nothing to attach a
`validation_status` to, and the grounding layer below is the whole reason to
trust the output. If prose is ever wanted from the API, render it server-side
from these fields rather than asking the model for it: a second generation
would be unverifiable and could disagree with the structured answer beside it.

The cut is enforced in one place per layer and they must stay in agreement:
`build_response_schema` in `case_summary_prompt.py` (what the model may
return), `_LIST_FIELDS` / `_SCALAR_FIELDS` in the service (what survives
normalisation), and `SummarizeDocumentResponse` in the API (what ships). The
prompt also tells the model outright not to report the omitted things, so it
does not spend tokens producing values the schema will drop.

### Summarization runs on either of two models

One pipeline, two adapters. `POST /case_summarization/summarize` takes an
**optional** `model` form field, `gpt` or `gemini`; omitted, the request runs on
`settings.SUMMARY_PROVIDER`. The split is:

- `case_summarization_service.py` owns everything provider-neutral —
  extraction, sanitization, building the `PromptPayload`, grounding,
  normalising the result, the log line. It resolves the provider *before*
  parsing the document, so an unusable provider costs nothing.
- `summary_providers.py` owns only the SDK differences: how bytes are attached,
  how the schema is declared, which exception means what, and which model id
  and timeout each SDK wants. Nothing that decides *what the summary says*
  belongs here.

**Provider selection is configuration, not a caller's problem.** `model` is an
override for the caller who wants the other provider specifically — a cost or
quality comparison on the same document — and it only resolves where that
provider's key is set. `build_providers()` builds every adapter whose key is
present, which is what makes an override possible at all; with one key there is
one provider and the override 503s naming what is available.

`_provider()` distinguishes two failures that look alike and are not. A **named**
model that is absent is the caller asking for something this server does not
offer — recoverable by naming the other one, logged at WARNING. The **default**
being absent is our misconfiguration, which no caller can work around — logged
at ERROR, with a message that says so. Settings makes the second unreachable in
a normally-built app by refusing to boot; it survives because providers can be
injected.

The provider names live in `config/settings.py` and are re-exported by
`summary_providers` as `MODEL_GPT` / `MODEL_GEMINI`. That inversion is not
stylistic: settings must validate `SUMMARY_PROVIDER` at import and cannot
import the provider module to get the list, because that module imports
settings. One definition, no cycle, and callers keep the name they always used.

`GET /case_summarization/models` returns both the available list **and**
`default`. A picker that hardcoded either would offer an option that 503s, or
preselect the wrong model and quietly send every document to the one nobody
chose.

**Both providers get the identical `PromptPayload` and the same schema.** If
the prompt or framing could vary per provider, "which model summarises better"
would stop being answerable — that is the whole reason for the split, and it is
pinned by a test. Two things follow that are easy to break:

- There is **one source schema** (`RESPONSE_SCHEMA`). Gemini's is *derived*
  from it by `build_gemini_response_schema`, never maintained separately.
  Gemini's `response_json_schema` accepts JSON Schema but not type arrays
  (`["string", "null"]`) or a null inside an `enum`, so the converter rewrites
  both as an explicit `anyOf` union. Add a field in the one place and both
  models get it.
- The genai SDK takes its **timeout in milliseconds**, where the OpenAI SDK
  takes seconds. Passing the seconds value would be a sub-second timeout and
  every Gemini call would fail.

Gemini reports every failure as one `APIError` family carrying an HTTP status,
so `_translate` maps on the status where the OpenAI SDK gives distinct classes.
A Gemini **400 is our fault, not the caller's** — usually a schema it won't
accept — so it is logged at ERROR even though the caller sees a generic 502.

`build_providers()` omits any provider whose key is unset rather than building
it and letting it fail on first use, and the service turns a missing provider
into a 503 naming what *is* available — a far better signal than an auth error
surfacing three layers down.

### Grounding is verified, not trusted

The prompt tells the model to return null rather than infer, and that rule is
unfalsifiable from the answer alone. So the two fields worth acting on
unread — `chronology` and `assertions`, listed in
`_GROUNDED_FIELDS` — each carry a `source_snippet`, and `src/utils/grounding.py`
looks for that quotation in the text we extracted ourselves.

`parties` is the one list field that is **not** grounded, and a test pins that
it is the only one. A party is a name in a cause title: checked by reading it,
not by turning to a page. So a `Party` carries no source fields at all rather
than empty ones a reader would mistake for a failed check.

Each entry comes back with **two** verdicts, and they are separate on purpose:
`validation_status` (`verified` / `unverified` / `not_verifiable`) says whether
the words are in the document, `page_status` (`mapped` / `unmapped` /
`unavailable`) says whether we could place them on a page. An entry can be
verified and unmapped — a DOCX has no pages, and a phrase recurring throughout a
filing locates nothing. Collapsing them would mean discarding a confirmed quote
or claiming a page we did not find. The response carries a `grounding` block
with a count of each.

Five things here are load-bearing:

- **Page numbers never come from the model.** This is structural, not a matter
  of care: there is no page field in `RESPONSE_SCHEMA`, and `DocumentIndex.check`
  takes the quote and nothing else, so there is no parameter through which one
  could arrive. A page is published only where that page's own text contained
  the quote. A test asserts the signature, so adding a parameter fails the build
  rather than quietly reintroducing guessed citations.
- **`source_pages` is non-empty if and only if `page_status` is `mapped`.**
  Pinned by tests at both the module and the service level. Break it and an
  entry can look page-verified when its page is unknown, which is worse than
  showing no page at all.
- **The model never grades itself.** Neither status is in the schema; both are
  computed after the call. A model rating its own output would be as confident
  about an invented case number as a real one — which is also why per-field
  `extraction_confidence` was considered and rejected.
- **Matching ignores characters and respects words.** Both sides are reduced to
  letters and digits before comparing, because a PDF text layer and a faithful
  quotation of it disagree constantly about hyphenation, curly quotes,
  ligatures and line-wrap spacing, and none of that means the quote was
  invented. Loosening it further — fuzzy or token-overlap matching —
  would start verifying paraphrases, which is the one thing this exists to
  catch.
- **Failed entries are labelled, never dropped.** The false-positive rate
  against real filings is unknown, and deleting a line from a lawyer's brief
  on a string comparison is the worse failure. The counts are logged on every
  call precisely so that rate can be measured before anyone changes this.

Pages are searched as one concatenated stream with a span table mapping
position back to page, not page by page. That is what lets a quote straddling a
page break be found at all, and it reports both pages rather than neither. The
markers themselves are excluded from the stream — reducing the raw text instead
would leave `page1`, `page2` embedded at exactly the point such a quote has to
match across.

`assertions` is one list with a `statement_type` discriminator (`fact` /
`allegation`) rather than two lists. Whether a sentence in a pleading is a fact
or an allegation is a judgement call; two lists would make the model take it
twice and let the same sentence land in both or in neither. `attributed_to` —
which party says it — is the natural next field here and is not implemented yet.

Because every citable entry carries a quotation, `MAX_OUTPUT_TOKENS` in
`summary_providers` is 32768 — raised from 8192, then from 16384 after a real
13-page scanned petition overran it. Running out does not degrade gracefully:
the JSON is truncated mid-object and the whole call is lost after being paid
for. Output is billed per token generated, so unused headroom is free; 32768 is
gpt-4.1-mini's ceiling, and the two providers are deliberately held to one
number.

**Gemini runs with thinking disabled (`thinking_budget=0`), and that is
load-bearing.** Gemini 2.5 charges reasoning tokens against
`max_output_tokens`, so with it on the shared constant means "the whole answer"
for OpenAI and "whatever reasoning left over" for Gemini — the two stop being
held to one ceiling, which is the entire basis for comparing them. On the
petition above that was 7,498 tokens of thinking against 8,871 of answer. The
log line reports `thoughts=` on every call: a non-zero value there is the first
sign the answer is sharing its budget again.

Both adapters must refuse a cut-off answer *before* parsing it — OpenAI on
`status == "incomplete"`, Gemini on `finish_reason == MAX_TOKENS`. Gemini's
check was missing, so a truncated answer reached `parse` and was reported to
the caller as "returned an unreadable result", which blames the model for
malformed JSON, points away from the real cause, and invites a retry certain to
fail identically.

`_items` drops a bare string arriving in a list of objects. That matters more
than it looks: Gemini enforces its schema less rigidly than OpenAI strict mode,
and a string reaching `_ground` would be handed to `.get`.

### Case summarization takes one of three branches

The document arrives as a **URL**, not an upload. `src/utils/url_fetch.py`
downloads it; `src/utils/document.py` then decides what it is and how to send
it — and every rejection (empty, oversized, unsupported, encrypted, over the
page cap) happens there, so a bad document never costs a model call.

### Fetching a linked document is SSRF surface

`url_fetch.py` exists because the endpoint makes an outbound request to an
address a stranger chose. Read this before touching it:

- **http(s) only.** `file://` reads local disk, `gopher://` smuggles bytes into
  whatever is listening.
- **Every resolved address must be public.** Anything outside global address
  space is refused — that is what catches 100.64.0.0/10 (CGNAT), which Python
  does not count as private — and so are loopback, private, link-local,
  multicast, reserved and unspecified, which catch the NAT64 and
  IPv4-compatible forms `is_global` passes. Neither half alone is enough.
  Link-local is where
  `169.254.169.254` lives — the cloud instance-credentials endpoint, the single
  highest-value target of an SSRF bug.
- **Every address, not the first.** A hostname answering with one public and
  one private address is refused outright, or an attacker picks which we dial.
- **Every redirect hop, not just the typed URL.** Redirects are followed by
  hand precisely so each `Location` is re-parsed and re-resolved. A public URL
  answering `302 -> 169.254.169.254` is the attack a front-door-only check
  misses entirely.
- **IPv4-in-IPv6 is unwrapped first.** `IPv6Address('::ffff:127.0.0.1')
  .is_loopback` is False; without `_unwrap` that notation walks past every
  check.
- **`Content-Type` is ignored**, like any other declared file type. Magic
  bytes decide.
- **One deadline for the whole fetch.** httpx has no total timeout — its read
  timeout restarts on every chunk — so a server trickling bytes would hold the
  request open indefinitely. `fetch` wraps DNS, every redirect hop and the body
  in a single `asyncio.timeout(TOTAL_TIMEOUT_SECONDS)`. Tests pin both the slow
  body and a hanging DNS lookup.
- **Nothing touches disk.** The body streams into memory under the same 20 MB
  ceiling, released with the request — so "store temporarily then delete" is
  met by never storing, and there is no cleanup that can fail.
- The refusal message never names the internal address that was resolved; that
  answer is itself a scan result. It is logged, not returned.

Known gap, written down rather than left to be found: **DNS rebinding**. Our
resolution and httpx's are separate lookups, so a hostile nameserver can answer
differently. Closing it means dialling the vetted IP with the hostname in the
`Host` header.

The fetch happens **in the service, after `_provider()` resolves** — not in the
route. An unconfigured model must not first cost an outbound request, on the
same principle that already stops it costing a 20 MB PDF parse.

`document.py` has **two entry points**, and they converge immediately.
`extract(bytes)` takes whatever the fetch returned; `from_text(str)` is a
document pasted into a box. `from_text` skips the two rules that have no meaning for
characters — sniffing and page counting — and shares every other one: the same
`MAX_TEXT_CHARS` cap, the same TEXT branch, the same `ExtractedDocument`.
Nothing downstream can tell them apart, which is the point. It adds one rule of
its own, `MIN_TEXT_CHARS`: a file at least had to be a real PDF to get that
far, whereas a text box will submit `hi` and the call is billed anyway.

Pasted text has no `--- Page N ---` markers, so every entry comes back
`page_status: unavailable` exactly as a DOCX does. That is correct, not a gap —
synthesising markers to fill `source_pages` would publish page numbers
referring to nothing.

Which is why **`source_pages_available` cannot be derived from the branch
alone** — testing only `branch == BRANCH_TEXT` was a bug, since a DOCX and a
paste both pass it while having no pages, and the flag then promised page
references that every entry reported as `unavailable`. It is
`branch == BRANCH_TEXT and page_count is not None`: markers come from
`_join_pages`, only a PDF gets them, and `page_count` being set is exactly that
condition. The flag exists so a client can decide once, up front, whether to
render a page affordance — so `False` has to mean *nothing* here can ever be
paged. A parametrized test checks it against what the entries actually report,
for all five input kinds.

The route takes `url` **or** `text`, and sending both is a 400 rather than a
silent preference: a client with a stale form field would otherwise get a
summary of the wrong one with no way to tell. Note the route reads an empty
string as *not given* — that is what a form sends for a field the user left
alone, and reading it as "a URL was given" would make pasting impossible from
any form carrying both fields.

- **TEXT** — a digital PDF or a DOCX. Extracted locally, joined with
  `--- Page N ---` markers, sent as `input_text`. Those markers are the only
  reason `source_pages` exists: `grounding._split_pages` parses them back out
  to build the position-to-page table. Change their format and every entry
  silently becomes `unmapped`, since a DOCX (which has no markers) is a valid
  input and produces exactly that. Note they are never shown to the model as
  something to cite — it is not asked for pages at all.
- **FILE** — a PDF whose text yield averages under `MIN_CHARS_PER_PAGE`, i.e. a
  scan. Sent as `input_file` with an inline base64 data URL, so the model reads
  the rendered pages. Inline rather than the Files API so a client's filing is
  never left in provider-side storage.
- **IMAGE** — a JPG or PNG, sent as `input_image`.

The branch is provider-neutral: `document.py` says "here are bytes and this is
their media type", and each provider decides how to attach them (OpenAI splits
`input_file` from `input_image`; Gemini uses one inline part for both). Base64
encoding lives with the OpenAI provider, not on `ExtractedDocument`, because
Gemini takes raw bytes.

Type detection is by **magic bytes only** — never `content_type` or the
filename, both of which the client sets. The encrypted-PDF check must stay
*before* anything touches `reader.pages`: pypdf raises there on an encrypted
file, and were the check dropped the file would extract as empty pages,
look exactly like a scan, and silently take the expensive branch.

PDF extraction is **pypdf (BSD)**. Do not swap in PyMuPDF/`fitz` — it is AGPL.

`_LIST_FIELDS` in the service maps each list field to the keys an entry must
actually carry. That exists because the JSON schema constrains shape but not
emptiness: a party named `""` passes the model's schema and then fails the
Pydantic response model, after the call has been paid for.

**`_LIST_FIELDS` and the response model are two statements of the same
contract, and they have drifted apart once already.** `parties` required only
`name` while `Party.role` was a required string, so a role-less party validated
in the service and raised in the route — a 500 on a call already billed.
Whenever a required field is added to a response model, the matching entry here
has to say so too. A test now constructs the response model from normalised
output for every list, so a future drift fails the build instead of a request.

The fix for `role` was **not** to require it. `_settle_roles` turns an absent
or unrecognised role into `other` rather than dropping the party, because a
party is worth keeping for its name alone — losing a party who is plainly in
the document is the worse error — and `other` is already what the prompt tells
the model to use when the document does not say. It doubles as the only guard
on the role vocabulary: `Party.role` is a plain `str`, so without it an
invented role would ship to a frontend with no rendering for it. OpenAI strict
mode makes both cases impossible; Gemini enforces less rigidly and is
selectable, so neither can be assumed away.

Like detection, summary quality is verified against live API calls, not unit
tests — and now on **both models**, since a prompt edit can regress one and not
the other. The failure mode to check for is **grounding**, and the statuses now
do most of that work: the numbers to watch across a run of real filings are the
`unverified` and `unmapped` rates. A high `unverified` on a digital PDF means
the matching is too strict, not that the model is lying — and a label nobody
trusts is worse than no label. Still spot-check a few `verified` entries, since
nothing here catches a real quote attached to a conclusion it does not support.

### Input sanitization runs in a fixed order

`clean_text` → `find_security_issue` → `flatten` (see the header comment in
`src/utils/helper.py`). Cleaning deliberately **preserves line breaks** because
the injection patterns anchor to a sentence or line start; flattening before
scanning would erase that boundary and let payloads through. This ordering is
pinned by tests — don't collapse the two steps.

The scanner is intentionally the weaker of two layers, leaning on the prompt's
own "ignore embedded instructions" rule, because a false positive silently
rejects a real client describing a real dispute.

**Documents deliberately break this order: they run `clean_text` only.**
`flatten` would erase the page and paragraph boundaries the summary needs, and
`find_security_issue` would reject real filings — its scope patterns block
"summarize the following document", and pleadings genuinely say things like
"directed to disregard the earlier instructions of the Board". On that path the
scanner runs in log-only mode (`_log_only_scan`) and the prompt's untrusted-data
rule is the primary defence rather than the second layer. Don't re-enable the
gate without testing against a corpus of real filings first.

### Cross-cutting behaviour in `main.py`

- **Lifespan owns services.** `CaseDetectionService` and
  `CaseSummarizationService` are built in the lifespan and stashed on
  `app.state`, resolved per-request via `src/api/deps.py`. Constructing them at
  module import instead would make `OPENAI_API_KEY` a requirement just to
  import the routes and leave the HTTP pools with no owner to close. Each
  `aclose()` is wrapped separately on shutdown so one failing does not strand
  the other's pool — and `CaseSummarizationService` does the same across its
  providers, since it now owns two clients rather than one.
- **One error envelope.** Handlers in `src/core/exceptions.py` flatten every
  error path — our exceptions, plain `HTTPException`, validation failures, rate
  limits, unhandled crashes — onto the same flat snake_case shape the success
  responses use, instead of FastAPI's nested `{"detail": ...}`.
- **Rate limiting** via slowapi on `/case_detection/detect` and
  `/case_summarization/summarize`, keyed on client IP, with a separate env var
  each — summarization's is far stricter because one call sends a whole
  document. A rate-limited route handler must take both `request: Request` and
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
- `/detect` and `/summarize` are both unauthenticated and cost money per call;
  the rate limits are the only thing standing in front of the OpenAI bill.
  `/summarize` is much the more expensive: a scanned filing is billed per
  rendered page.
- **`/summarize` now makes an outbound request to a caller-supplied address.**
  `url_fetch` closes the standard SSRF paths, but two things remain: DNS
  rebinding (described above), and the fact that an unauthenticated endpoint
  can be pointed at *any* public host, making this server a small, rate-limited
  request amplifier someone else's logs will see as us. If that matters,
  `SUMMARY_URL_ALLOWED_HOSTS`-style allowlisting is the next step, not more
  denylisting.
- `/summarize` is synchronous, and a long document can take minutes — past the
  idle timeout of most proxies. The fetch adds at most 30s to that budget.
  The 20 MB / 30-page caps in `src/utils/document.py` keep it inside that;
  lifting them means moving the
  call behind a job queue, which also needs shared storage rather than the
  in-process kind noted above.
- Logs pair a caller's IP with the text of their legal problem — sensitive in
  this domain. Worth demoting the full-query lines to `DEBUG` for production.
- `src/api/counter_generation.py` is empty and unreferenced. The summarization
  response is designed to be its input — structured rather than prose, so a
  reply can be drafted from the fields without re-reading the document.
