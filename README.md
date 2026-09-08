# Closeurcase AI Backend

A FastAPI service that reads someone's description of their legal problem — in
plain, often messy English — and works out what kind of case it is.

Give it *"someone hit and ran away, the person sitting behind me died"* and it
comes back with **Motor Accident** — and, because a hit-and-run is two legal
matters rather than one, a secondary of **Criminal**. Along with each it
returns the legal services that apply, so the frontend can route the person to
the right kind of help without a second request.

## What it does

The service exposes two endpoints. `POST /case_detection/detect` classifies a
query. `GET /case_detection/categories` returns the full taxonomy — 10
categories, 50 case types, 247 legal services — which is what the frontend
renders as menus.

Classification runs through an OpenAI model constrained by a JSON schema, so it
can only ever answer with a case type that actually exists. Everything else in
the response — the parent category, the list of services — is looked up here
from that one answer. The model never gets to disagree with itself.

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

The suite covers input sanitization, which is the part where a subtle mistake
quietly stops protecting anything. Classification quality isn't unit-tested —
it's checked by running real queries against the API, because prompt changes
regress silently and no assertion catches that.

Always go through `uv run`. A bare `python` may pick up a different interpreter
without the project's dependencies.

## How it fits together

```
main.py                              app, middleware, error handlers, lifespan
└── src/
    ├── api/          HTTP shapes only — request/response models, no logic
    ├── services/     the model call and result normalisation
    ├── prompts/      prompt text + the JSON schema, both built from the taxonomy
    ├── core/         security, exceptions, rate limiting, logging
    ├── utils/        input sanitization
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

**`/detect` is rate limited, the rest aren't.** It's the only endpoint that
spends money upstream. Health checks and the taxonomy stay unlimited so uptime
probes don't get throttled.

## Before you deploy this

Some things are fine for development and are not fine in production:

- **CORS allows every origin with credentials enabled.** Any website can call
  this API. Needs a real allowlist.
- **`/detect` has no authentication.** The rate limit is the only thing between
  a script and your OpenAI bill.
- **Rate limit counters live in memory**, so they're per worker and reset on
  restart. Run more than one worker and the effective limit multiplies. Point
  `RATE_LIMIT_STORAGE` at a `redis://` URL to fix that.
- **Logs record the caller's IP next to the text of their legal problem.** In
  this domain that's divorces, criminal charges, domestic violence. Decide on
  retention, and consider dropping the full-query lines to `DEBUG`.
- **Behind a proxy, run with `--proxy-headers --forwarded-allow-ips=<proxy>`.**
  Otherwise every request looks like it came from the proxy and shares one rate
  limit bucket.
