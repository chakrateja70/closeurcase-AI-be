# Project Analysis: Closeurcase-AI Backend

**Date**: 2026-09-02  
**Status**: ✅ Generally Good Architecture with **⚠️ One Critical Bug Found**

---

## Overview

This is a FastAPI-based legal case detection system that classifies user queries into Indian legal domains using OpenAI's structured output feature, with security protections applied via LLM Guard.

---

## ✅ What's Working Well

### 1. **LLM Guard Protection - IMPLEMENTED & ACTIVE**
- **Location**: [src/utils/helper.py](src/utils/helper.py)
- **Implementation**: 
  - Prompt injection detection using `llm_guard` library (v0.3.17+)
  - `PromptInjection` scanner with `MatchType.SIMILARITY` and 0.75 threshold
  - Handles both old (v0.3.x) and new (v0.4.0+) llm-guard API versions
  - Input length validation (10,000 char max by default)
  - Text normalization (Unicode NFKC + whitespace collapse)

- **Protection Scope**:
  - Detects and blocks prompt injection attempts
  - Prevents excessively long inputs
  - Runs before sending query to OpenAI

### 2. **Clean Architecture**
- **Router structure** (2-level hierarchy):
  - Feature modules in `src/api/` (e.g., `case_detection.py`)
  - Aggregated in `src/routes.py`
  - Single inclusion in `main.py`
  - Clean separation of concerns

- **Layering**:
  - **API Layer** (`src/api/`): Request/response, validation
  - **Service Layer** (`src/services/`): Business logic
  - **Core** (`src/core/`): Shared utilities, exceptions, security
  - **Config** (`src/config/`): Environment-based settings
  - **Utils** (`src/utils/`): Generic helpers
  - **Prompts** (`src/prompts/`): LLM templates

### 3. **Security & Authentication**
- **Docs Protection**: HTTP Basic Auth on `/docs`, `/redoc`, `/openapi.json`
- **Credential Comparison**: Uses `secrets.compare_digest()` to prevent timing attacks
- **CORS**: Properly configured middleware

### 4. **API Documentation Protection**
- Docs disabled by default (`docs_url=None`, `redoc_url=None`, `openapi_url=None`)
- Re-served behind auth gates
- All three endpoints (`/docs`, `/redoc`, `/openapi.json`) protected
- Credentials from env: `DOCS_USERNAME` / `DOCS_PASSWORD` (defaults: `admin`/`1234`)

### 5. **Exception Handling**
- Centralized exception classes in [src/core/exceptions.py](src/core/exceptions.py)
- Consistent error response format
- Proper HTTP status code mapping
- Specific OpenAI API error handling (timeouts, rate limits, auth failures)

### 6. **Dependency Management**
- Uses `uv` package manager with locked `uv.lock`
- Clean `pyproject.toml` with all dependencies declared
- Python 3.12 pinned in `.python-version`

### 7. **Case Taxonomy**
- Comprehensive 35-category legal domain taxonomy
- Proper categorization with primary + optional secondary categories
- Handles complex cases spanning multiple legal domains

### 8. **OpenAI Integration**
- Uses structured output with JSON Schema enforcement
- Token usage tracking (input, output, cached)
- Error handling for timeouts, rate limits, auth issues
- Async API calls

---

## ⚠️ CRITICAL BUG FOUND

### **Bug: Incorrect LLM Guard Return Value Handling**

**Location**: [src/services/case_detection_service.py](src/services/case_detection_service.py#L57)

**Issue**:
```python
# Current (WRONG) - Line 57
issue = find_security_issue(text)
if issue:
    print(f"[case_detection] blocked query reason={issue}")
    return self._invalid(BLOCKED_QUERY_RESPONSE)
```

**Problem**:
- `find_security_issue()` returns a **tuple**: `(is_safe, issue_type)` → e.g., `(False, "prompt_injection")`
- The check `if issue:` will **ALWAYS be truthy** because a non-empty tuple is truthy
- This means **EVERY SINGLE QUERY gets blocked** with the current code
- The actual security verdict is being ignored

**Correct Implementation**:
```python
# Correct (FIXED)
is_safe, issue = find_security_issue(text)
if not is_safe:
    print(f"[case_detection] blocked query reason={issue}")
    return self._invalid(BLOCKED_QUERY_RESPONSE)
```

**Impact**: 
- 🔴 **HIGH SEVERITY** - LLM Guard is technically applied but non-functional
- Safe queries are being rejected
- Malicious queries could slip through if the logic worked correctly
- API is essentially broken for all requests

---

## 📋 Code Quality Observations

### Functions Removed/Simplified:
1. ✅ `sanitize_prompt()` in helper.py - exists but not used in main flow (good for future use)
2. ✅ Complex response formatting - delegated to `_normalise()` method
3. ✅ Direct API call details - abstracted into `_classify()` method

### Clean Code Practices:
- ✅ Type hints throughout
- ✅ Docstrings on key functions
- ✅ Constants defined at module level
- ✅ Private methods prefixed with `_`
- ✅ Async/await properly used
- ✅ Error messages user-friendly

### Areas Needing Attention:
- ⚠️ `counter_generation.py` - exists but is empty (intentional scaffolding?)
- ⚠️ `requirements.txt` - marked as stale; should remove or remove from tracking
- ⚠️ Print statements used for logging - consider `logging` module for production

---

## 🔧 Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | >=0.141.1 | Web framework |
| openai | >=3.6.0 | LLM client |
| python-dotenv | >=1.2.3 | Environment config |
| uvicorn[standard] | >=0.52.4 | ASGI server |
| **llm-guard** | **>=0.3.17** | **🔒 Security scanning** |

---

## 🔐 Security Posture

| Aspect | Status | Notes |
|--------|--------|-------|
| LLM Guard installed | ✅ Yes | Dependency declared |
| LLM Guard initialized | ✅ Yes | Singleton `_PROMPT_INJECTION_SCANNER` |
| LLM Guard called | ✅ Yes | Called in `detect_case()` |
| LLM Guard output handled | 🔴 **BUG** | Return value misinterpreted |
| Docs auth | ✅ Yes | HTTP Basic on all doc routes |
| CORS | ✅ Yes | Properly configured |
| Input validation | ✅ Yes | Length, format, content |

---

## 📁 Project Structure - Current State

```
closeurcase-AI-be/
├── main.py                           # FastAPI app + docs routes
├── pyproject.toml                    # Dependencies (uv managed)
├── .env.example                      # Environment template
├── CLAUDE.md                         # Guidance document ✅
├── src/
│   ├── routes.py                     # Router aggregation
│   ├── api/
│   │   ├── case_detection.py         # /case_detection endpoints
│   │   └── counter_generation.py     # Empty (scaffolding)
│   ├── services/
│   │   └── case_detection_service.py # Business logic ⚠️ BUG HERE
│   ├── core/
│   │   ├── security.py               # Auth guards
│   │   ├── case_categories.py        # Legal taxonomy
│   │   └── exceptions.py             # Centralized errors
│   ├── config/
│   │   └── settings.py               # Env-based config
│   ├── prompts/
│   │   └── case_detection_prompt.py  # LLM system prompt + schema
│   └── utils/
│       └── helper.py                 # LLM Guard + text utilities ✅
└── .python-version                   # Python 3.12
```

---

## ✅ Removed Functions (Good Cleanup)

The codebase shows intentional simplification:
- Old response formatting methods removed → consolidated into `_normalise()`
- Separated concerns: `_classify()`, `_normalise()`, `_invalid()` are focused
- Helper functions in `utils/` are reusable and generic

---

## 🎯 Recommendation

**FIX THE BUG IMMEDIATELY** - Line 57 in [src/services/case_detection_service.py](src/services/case_detection_service.py)

Change:
```python
issue = find_security_issue(text)
if issue:
```

To:
```python
is_safe, issue = find_security_issue(text)
if not is_safe:
```

This single-line fix will make LLM Guard actually functional.

---

## Summary

| Metric | Score | Notes |
|--------|-------|-------|
| Architecture | 9/10 | Clean layering, good separation |
| Security Implementation | 5/10 | Guard in place but **broken by bug** |
| Code Quality | 8/10 | Type hints, docstrings, async |
| Dependency Management | 9/10 | `uv` + locked versions |
| Documentation | 7/10 | CLAUDE.md is good, needs inline comments |
| **Overall** | **6/10** | **Good structure, CRITICAL BUG breaks security** |

