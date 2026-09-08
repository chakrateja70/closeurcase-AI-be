"""Case taxonomy - the contract shared with the frontend and the LLM.

Three levels, loaded from `src/data/case.json` at import time:

    category  ->  case_type  ->  legal_service

`case.json` carries titles only; the ids used by the API and the LLM enum are
slugs derived from those titles (`_slug`), so editing the JSON is enough to
change the taxonomy - no code change, no hand-maintained id list. Case-type
ids are globally unique (enforced on load); legal-service ids are namespaced
under their case type ("cheque_bounce.file_cheque_bounce_case") because a few
service titles repeat across case types.

Classification targets the CASE TYPE, not the category: the model picks one
case-type id and the parent category is derived from it here, which makes a
category/case-type mismatch structurally impossible. A second case type may be
set when the facts genuinely span two distinct matters - it is drawn from the
same flat pool of case-type ids and may or may not share the primary's
category (e.g. Divorce + Child Custody are both Family Law).

The catch-all bucket ("other") is defined in code rather than in the JSON: it
is where a legal-but-unplaceable query lands, not a service the product sells.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "data" / "case.json"

# Bucket used when a query is legal but fits no case type in the taxonomy.
OTHER_CATEGORY_ID = "other"
OTHER_CASE_TYPE_ID = "other"

FALLBACK_RESPONSE = (
    "We could not identify a legal issue in your query. Please describe your "
    "situation with more detail - what happened, who is involved, and what "
    "outcome you are looking for."
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _slug(title: str) -> str:
    """Title -> stable id. "Trademark & Copyright" -> "trademark_and_copyright"."""
    return _NON_ALNUM_RE.sub("_", title.lower().replace("&", " and ")).strip("_")


def _build_case_type(title: str, services: list[str]) -> dict:
    case_type_id = _slug(title)
    return {
        "id": case_type_id,
        "title": title,
        "legal_services": [
            {"id": f"{case_type_id}.{_slug(service)}", "title": service}
            for service in services
        ],
    }


def _other_category() -> dict:
    """Synthetic catch-all, appended after the JSON-defined categories. Its ids
    are pinned to the OTHER_* constants rather than slugged from the titles,
    because the service and the prompt both refer to them by name."""
    return {
        "id": OTHER_CATEGORY_ID,
        "title": "Other Legal Matters",
        "case_types": [
            {
                "id": OTHER_CASE_TYPE_ID,
                "title": "Other Legal Matter",
                "legal_services": [
                    {
                        "id": f"{OTHER_CASE_TYPE_ID}.general_legal_consultation",
                        "title": "General Legal Consultation",
                    }
                ],
            }
        ],
    }


def _load_categories() -> list[dict]:
    with TAXONOMY_PATH.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    categories = [
        {
            "id": _slug(entry["category"]),
            "title": entry["category"],
            "case_types": [
                _build_case_type(case_type["case_type"], case_type["legal_services"])
                for case_type in entry["case_types"]
            ],
        }
        for entry in raw
    ]
    categories.append(_other_category())
    return categories


CASE_CATEGORIES: list[dict] = _load_categories()

_CATEGORIES_BY_ID: dict[str, dict] = {c["id"]: c for c in CASE_CATEGORIES}

# case_type_id -> {"case_type": ..., "category": ...}. Case-type ids are the
# LLM's enum, so a collision would silently make one of them unreachable.
_CASE_TYPES_BY_ID: dict[str, dict] = {}
for _category in CASE_CATEGORIES:
    for _case_type in _category["case_types"]:
        if _case_type["id"] in _CASE_TYPES_BY_ID:
            raise ValueError(
                f"Duplicate case_type id {_case_type['id']!r} in {TAXONOMY_PATH.name}"
            )
        _CASE_TYPES_BY_ID[_case_type["id"]] = {
            "case_type": _case_type,
            "category": _category,
        }

if len(_CATEGORIES_BY_ID) != len(CASE_CATEGORIES):
    raise ValueError(f"Duplicate category id in {TAXONOMY_PATH.name}")


def get_category(category_id: str | None) -> dict | None:
    return _CATEGORIES_BY_ID.get(category_id) if category_id else None


def get_case_type(case_type_id: str | None) -> dict | None:
    """Resolved case type, flattened with its parent category. None if unknown."""
    entry = _CASE_TYPES_BY_ID.get(case_type_id) if case_type_id else None
    if entry is None:
        return None
    case_type, category = entry["case_type"], entry["category"]
    return {
        "id": case_type["id"],
        "title": case_type["title"],
        "category_id": category["id"],
        "category_title": category["title"],
        "legal_services": case_type["legal_services"],
    }


def category_ids() -> list[str]:
    return [category["id"] for category in CASE_CATEGORIES]


def case_type_ids() -> list[str]:
    return list(_CASE_TYPES_BY_ID)


def list_categories() -> list[dict]:
    """The full three-level tree, as served to the frontend."""
    return [
        {
            "id": category["id"],
            "title": category["title"],
            "case_types": [
                {
                    "id": case_type["id"],
                    "title": case_type["title"],
                    "legal_services": [dict(s) for s in case_type["legal_services"]],
                }
                for case_type in category["case_types"]
            ],
        }
        for category in CASE_CATEGORIES
    ]
