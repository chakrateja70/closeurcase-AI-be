"""Case taxonomy - the contract shared with the frontend and the LLM.

Top-level legal domains, all flat (no sub-categories). A case is classified
into one primary domain, and optionally a second domain when the facts
genuinely span two distinct areas of law (e.g. a dispute that is both a
criminal offence and a civil claim for damages). The secondary category is
drawn from this SAME list - it is not a sub-category of the primary, and most
cases will have none (null).

Some entries are intentionally more specific siblings of a broader one (e.g.
"divorce" alongside "family", "cheque_bounce" alongside "banking") rather than
nested under it - see the "more specific wins" rule in the system prompt.
"""

CASE_CATEGORIES: list[dict] = [
    {"id": "criminal", "title": "Criminal Law"},
    {"id": "civil", "title": "Civil Law"},
    {"id": "family", "title": "Family Law"},
    {"id": "property", "title": "Land & Property"},
    {"id": "consumer", "title": "Consumer Law"},
    {"id": "corporate", "title": "Corporate & Commercial"},
    {"id": "labour", "title": "Labour & Employment"},
    {"id": "cyber", "title": "Cyber Crime & IT"},
    {"id": "tax", "title": "Tax Law"},
    {"id": "constitutional", "title": "Constitutional Law"},
    {"id": "service", "title": "Service Law"},
    {"id": "banking", "title": "Banking & Finance"},
    {"id": "insurance", "title": "Insurance Law"},
    {"id": "ipr", "title": "Intellectual Property"},
    {"id": "environment", "title": "Environmental Law"},
    {"id": "motor_accident", "title": "Motor Accident Claims"},
    {"id": "arbitration", "title": "Arbitration & Mediation"},
    {"id": "divorce", "title": "Divorce"},
    {"id": "family_dispute", "title": "Family Dispute"},
    {"id": "child_custody", "title": "Child Custody"},
    {"id": "muslim_law", "title": "Muslim Law"},
    {"id": "medical_negligence", "title": "Medical Negligence"},
    {"id": "landlord_tenant", "title": "Landlord / Tenant"},
    {"id": "wills_trusts", "title": "Wills / Trusts"},
    {"id": "documentation", "title": "Documentation"},
    {"id": "cheque_bounce", "title": "Cheque Bounce"},
    {"id": "recovery", "title": "Recovery"},
    {"id": "customs_excise", "title": "Customs & Central Excise"},
    {"id": "startup", "title": "Startup"},
    {"id": "gst", "title": "GST"},
    {"id": "armed_forces_tribunal", "title": "Armed Forces Tribunal"},
    {"id": "supreme_court", "title": "Supreme Court"},
    {"id": "immigration", "title": "Immigration"},
    {"id": "international_law", "title": "International Law"},
    {"id": "other", "title": "Other Legal Matters"},
]

# Bucket used when a query is legal but fits no specific category.
OTHER_CATEGORY_ID = "other"

FALLBACK_RESPONSE = (
    "We could not identify a legal issue in your query. Please describe your "
    "situation with more detail - what happened, who is involved, and what "
    "outcome you are looking for."
)

_CATEGORIES_BY_ID: dict[str, dict] = {c["id"]: c for c in CASE_CATEGORIES}


def get_category(category_id: str | None) -> dict | None:
    return _CATEGORIES_BY_ID.get(category_id) if category_id else None


def category_ids() -> list[str]:
    return [category["id"] for category in CASE_CATEGORIES]


def list_categories() -> list[dict]:
    return [
        {"id": category["id"], "title": category["title"]}
        for category in CASE_CATEGORIES
    ]
