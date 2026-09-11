# Closeurcase AI Backend

A FastAPI service that reads a legal problem — as plain, often messy English, or
as an uploaded court document — and works out what it is.

Give it *"someone hit and ran away, the person sitting behind me died"* and it
comes back with **Motor Accident** — and, because a hit-and-run is two legal
matters rather than one, a secondary of **Criminal**. Along with each it
returns the legal services that apply, so the frontend can route the person to
the right kind of help without a second request.

Hand it a petition instead and it returns the same classification wrapped
around a structured brief: the parties, the court, a dated chronology, the
statutes relied on, what the filing asks for, and a plain-English account of
the dispute.

## What it does

Two features, four endpoints.

**Case detection.** `POST /case_detection/detect` classifies a typed query.
`GET /case_detection/categories` returns the full taxonomy — 10 categories, 50
case types, 247 legal services — which is what the frontend renders as menus.

**Case summarization.** `POST /case_summarization/summarize` takes an uploaded
document (PDF, DOCX, JPG or PNG) and returns a structured brief plus a
narrative summary — and the same case type and services a typed query would
get, so a summarized document routes exactly like a described one. It runs on
**GPT or Gemini, chosen per request**; `GET /case_summarization/models` says
which are available. Both are given the same prompt and held to the same
schema, so the only difference in the output is the model.

Everything runs through a model constrained by a JSON schema, so none of them
can answer with a case type that doesn't exist. Everything else in the
response — the parent category, the list of services — is looked up here from
that one answer. The model never gets to disagree with itself.

## Getting started

You'll need [uv](https://docs.astral.sh/uv/) and an OpenAI API key.

```bash
uv sync                      # installs into .venv from the lockfile
cp .env.example .env         # then put your real OPENAI_API_KEY in it
uv run uvicorn main:app --reload
```

The API comes up on `http://localhost:8000`. Try it:

```bash
curl -X POST http://localhost:8000/case_detection/detect \
  -H "Content-Type: application/json" \
  -d '{"query": "My landlord is refusing to return my security deposit."}'
```

```json
{
  "status_code": 200,
  "is_valid": true,
  "primary_case_category": "Property Law",
  "primary_case_type": "Landlord / Tenant",
  "primary_case_type_id": "landlord_tenant",
  "primary_legal_services": [
    { "id": "landlord_tenant.rent_recovery", "title": "Rent Recovery" }
  ],
  "secondary_case_type_id": null,
  "confidence": 0.95,
  "summary": "The query concerns a landlord-tenant dispute regarding the return of a security deposit."
}
```

And with a document. Send it **either** as `file` or as `text`, never both:

```bash
# no model named: runs on whatever SUMMARY_PROVIDER says
curl -F "file=@petition.pdf" \
  http://localhost:8000/case_summarization/summarize

# or paste the text instead of uploading anything
curl -F "text=$(cat petition.txt)" \
  http://localhost:8000/case_summarization/summarize

# `model` overrides the configured provider for one request
curl -F "file=@petition.pdf" -F "model=gemini" \
  http://localhost:8000/case_summarization/summarize
```

**`model` is optional.** Which model summarises is a deployment decision —
`SUMMARY_PROVIDER` in `.env` — so an ordinary client doesn't name one and
doesn't need to know. Naming one overrides that for a single request, and works
only where that provider's key is configured; otherwise it's a 503 saying what
is available. `GET /case_summarization/models` reports both the available list
and which one is `default`, so a picker need not hardcode either.

Whichever provider runs it, the request is identical — same prompt, same JSON
schema, same document framing, same grounding check. The two adapters differ
only in how their SDK is called. That's deliberate: if the framing varied per
provider, "which model summarises better" would stop being an answerable
question.

Pasted text takes exactly the same path as an extracted PDF — same length cap,
same untrusted-data framing, same grounding check — with one difference to
expect: it has no pages, so every entry comes back `page_status:
"unavailable"` and `source_pages` empty, just as a DOCX does. The quotes are
still verified. There's a 200-character floor, since a text box will happily
submit `hi` and the call gets billed either way.

```json
{
  "status_code": 200,
  "is_valid": true,
  "parties": [
    { "name": "A Rao", "role": "plaintiff", "counsel": "M Sharma" },
    { "name": "B Naidu", "role": "defendant", "counsel": null }
  ],
  "chronology": [
    {
      "date": "12.03.2024",
      "event": "Cheque dishonoured for insufficient funds",
      "source_snippet": "the said cheque was returned unpaid for insufficiency of funds",
      "source_pages": [2],
      "validation_status": "verified",
      "page_status": "mapped"
    }
  ],
  "facts_summary": "The plaintiff received a cheque that was returned unpaid...",
  "assertions": [
    {
      "statement": "The defendant issued the cheque towards a legally enforceable debt.",
      "statement_type": "allegation",
      "source_snippet": "issued in discharge of a legally enforceable debt",
      "source_pages": [1],
      "validation_status": "verified",
      "page_status": "mapped"
    }
  ],
  "grounding": {
    "quotes": { "verified": 6, "unverified": 1, "not_verifiable": 0 },
    "pages": { "mapped": 6, "unmapped": 1, "unavailable": 0 }
  },
  "confidence": 0.91,
  "model": "gpt",
  "model_id": "gpt-4.1-mini-2025-04-14"
}
```

**The summary answers three questions and stops.** What happened
(`chronology`), what the dispute is (`facts_summary`), and what the parties are
arguing (`assertions`) — plus `parties`, naming who is arguing, because the
other three read as anonymous
without it. The court, the case number, the statutes, the precedents, the
exhibits and the hearing dates are deliberately not extracted: whoever uploaded
the filing can read those off its first page, and restating them made the
response long without making it more useful.

**The output is JSON so that each line can be checked.** Prose would be shorter
to render, but there would be nothing to hang a `validation_status` on — the
per-entry quote is what makes the grounding below possible at all. Render the
bullets and paragraphs from these fields on the client, and the verified badge
stays attached to each one.

Every field is nullable. The model is told to return `null` for anything the
document doesn't state, so a sparse response means a sparse document — not a
failure. The response says which model produced it, so two runs of the same
document can actually be compared.

**Every part a lawyer would act on says where in the document it came from.**
Each entry in the chronology, the assertions and the prayer carries the words
from the document that establish it — and each is answered twice, because "are
these the document's words" and "which page are they on" are different
questions. `parties` is the one list that carries no quote: a name is checked
by reading it, not by turning to a page.

Is it really in the document?

- `verified` — the quote was found in the extracted text.
- `unverified` — it was not. The entry may still be right, but nothing here
  supports it, so read that one against the original.
- `not_verifiable` — a scan or a photo, which has no text layer to check
  against. An honest third answer, not a synonym for either of the others.

Where is it?

- `mapped` — `source_pages` lists the pages the quote was found on.
- `unmapped` — the document has pages, but this entry couldn't be put on one.
  `source_pages` is empty.
- `unavailable` — nothing to map against: a scan, an image, or a DOCX, which
  reflows and has no fixed pages.

**Page numbers are never taken from the model.** There is no page field in the
schema it answers, so it is never given the chance to guess one; a page is
published only where we found the quoted words on that page ourselves. Which
means `source_pages` is non-empty if and only if `page_status` is `mapped` —
nothing is ever shown as page-verified when its page is unknown. (They're
positions in the file, counted from one, not the numbers printed on the page.)

`grounding` totals both tallies across the response. Nothing is ever deleted on
the strength of these checks — silently dropping a line from a lawyer's brief
because a string comparison failed would be the worse mistake.

**You only need the key for the provider you selected.** `SUMMARY_PROVIDER=gpt`
(the default when it's unset) runs fine with no `GEMINI_API_KEY` — `gemini` just
doesn't appear in `GET /case_summarization/models`, and asking for it returns a
503 naming what is available. Set both keys and the per-request `model`
override becomes usable.

Selecting a provider whose key is missing is a **startup** failure, not a
runtime one: `SUMMARY_PROVIDER=gemini` without `GEMINI_API_KEY` refuses to boot,
because every summarization request would otherwise 503. A `SUMMARY_PROVIDER`
value that isn't `gpt` or `gemini` fails the same way, naming the valid options
— falling back silently would leave you convinced you were running on the other
model.

One asymmetry to know: `OPENAI_API_KEY` is required whichever summarizer you
choose, because `/case_detection/detect` runs on OpenAI and has no alternative.
`SUMMARY_PROVIDER` governs what summarization demands, not what the app demands.

Interactive docs live at `/docs` and `/redoc`, behind HTTP Basic auth using the
`DOCS_USERNAME` / `DOCS_PASSWORD` from your `.env`.

> **One gotcha worth knowing up front.** Config is loaded with
> `load_dotenv(override=True)`, which means `.env` wins over real environment
> variables — the opposite of what you'd expect. Setting `FOO=bar uv run ...`
> does nothing if `FOO` is also in `.env`. Edit the file instead.

## Running the tests

```bash
uv run pytest -q             # everything
uv run pytest -k newline     # one test by name
```

The suite covers the parts where a subtle mistake stops protecting anything or
costs money quietly: input sanitization, file identification and the upload
rejections, and the normalisation layer where the model's schema and the API's
response model can disagree.

Neither classification nor summarization *quality* is unit-tested — both are
checked by running real inputs against the API, because prompt changes regress
silently and no assertion catches that. For summaries the check that matters is
grounding, and the per-entry statuses now do most of that work for you: read
the `unverified` ones against the document first, then turn to the pages the
`mapped` ones name. What no test can cover is an entry that is verified and
still wrong — a real quote attached to a conclusion it does not support — so
the chronology and the citations are
still worth a human pass.

Always go through `uv run`. A bare `python` may pick up a different interpreter
without the project's dependencies.

## How it fits together

```
main.py                              app, middleware, error handlers, lifespan
└── src/
    ├── api/          HTTP shapes only — request/response models, no logic
    ├── services/     result normalisation, and one file per model provider
    ├── prompts/      prompt text + the JSON schema, both built from the taxonomy
    ├── core/         security, exceptions, rate limiting, logging, taxonomy
    ├── utils/        input sanitization, uploaded-document handling
    └── data/case.json    ← the taxonomy itself
```

**To change what the service can classify, edit `src/data/case.json`.** Nothing
else. Ids are derived from the titles, and the prompt and the model's schema are
both generated from that file at startup, so adding a case type there makes it
immediately selectable. There is no list of ids to keep in sync — that was the
point.

## A few deliberate decisions

**The model picks a case type and nothing else.** Categories aren't even shown
to it. Asking it for the category too would mean it could return a case type
and a category that don't belong together, and it would cost tokens to get a
worse answer than a dictionary lookup gives for free.

**A secondary case type means a second *remedy*, not a second topic.** A road
death is one incident but two matters: prosecuting the driver and claiming
compensation are separate proceedings in front of different authorities. A
dispute that's merely *about* two things isn't.

**Queries are sanitized before they cost anything.** Anything that looks like a
prompt injection is rejected without an API call. The scanner is deliberately
the weaker of two layers — the prompt itself also refuses embedded
instructions — because blocking too eagerly means silently turning away someone
with a genuine legal problem.

**Two models, one prompt and one schema.** GPT and Gemini are picked per
request, but everything that decides *what* the summary says is shared — the
prompt, the framing of the document, the normalisation of the result. Only the
SDK translation differs, in `src/services/summary_providers.py`. There's one
source schema; Gemini's dialect is *derived* from it rather than maintained
alongside it, because two hand-written copies would drift and then "which model
is better" would stop being an answerable question.

**The model is asked to quote, and the quotes are checked.** Telling a model to
stay grounded is unfalsifiable — a fabricated chronology reads exactly like a
correct one. Asking it to quote the passage each entry rests on, and then
looking for that passage in the text we extracted ourselves, is the only part of
this feature that can actually catch an invention. The comparison is
deliberately forgiving about characters and strict about words: a PDF text layer
and a faithful quotation of it disagree constantly about hyphenation, curly
quotes and line-wrap spacing, and none of that means the quote was made up.

**The safest way to stop a page number being guessed is not to ask for one.**
The model has no page field to fill in. Pages come out of locating its quote in
our own extracted text, which makes every citation in the response something we
observed rather than something we passed along — and it means an entry can be
`verified` but `unmapped`, which is the honest answer where a phrase recurs on
every page of a filing and locating it proves nothing about this entry.

**Scanned filings don't need OCR.** Court documents are very often photocopies
with no text layer at all. If a PDF yields too little text per page, it's sent
to the model as the file itself and read as rendered pages. One code path, two
branches, no OCR infrastructure to run.

**Uploads are sanitized differently from typed queries, on purpose.** A query
runs `clean_text → find_security_issue → flatten`; a document runs `clean_text`
only. Flattening would destroy the page and paragraph boundaries the model
needs to build a chronology, and the injection scanner would reject real
filings — its rules block phrases like *"summarize the following document"*,
and genuine pleadings say things like *"directed to disregard the earlier
instructions of the Board"*. On uploads the scanner runs in log-only mode and
the prompt's own "this is data, not instructions" rule does the work.

**The two endpoints that spend money are rate limited, separately.** Health
checks and the taxonomy stay unlimited so uptime probes don't get throttled.
`SUMMARIZE_RATE_LIMIT` is set far below `DETECT_RATE_LIMIT` because one call
sends a whole document rather than 600 characters.

## Before you deploy this

Some things are fine for development and are not fine in production:

- **CORS allows every origin with credentials enabled.** Any website can call
  this API. Needs a real allowlist.
- **Neither `/detect` nor `/summarize` has authentication.** The rate limit is
  the only thing between a script and your OpenAI bill — and `/summarize` is
  the expensive one, since a scanned 30-page filing is billed per rendered
  page.
- **`/summarize` is synchronous.** A long document can take a couple of
  minutes, which will hit the idle timeout on most proxies and load balancers.
  The caps in `src/utils/document.py` (20 MB, 30 pages) keep it inside that
  today; lifting them means moving the call onto a job queue first.
- **Rate limit counters live in memory**, so they're per worker and reset on
  restart. Run more than one worker and the effective limit multiplies. Point
  `RATE_LIMIT_STORAGE` at a `redis://` URL to fix that.
- **Logs record the caller's IP next to the text of their legal problem.** In
  this domain that's divorces, criminal charges, domestic violence. Decide on
  retention, and consider dropping the full-query lines to `DEBUG`. Uploads
  already log only size, page count and branch — never the document text or
  the filename, which routinely carries a client's name.
- **Behind a proxy, run with `--proxy-headers --forwarded-allow-ips=<proxy>`.**
  Otherwise every request looks like it came from the proxy and shares one rate
  limit bucket.
