"""Tests for the shared taxonomy helpers in src/core/case_categories.py.

`expand_case_type` and `resolve_case_types` are the two pieces case detection
and case summarization have in common: both features take a model-chosen
case-type id and turn it into the same primary/secondary response shape. They
live in one place precisely so the two cannot drift, and these pin the
behaviour that makes that safe - notably that the category and services are
always derived from the case type rather than supplied alongside it.
"""

from src.core.case_categories import (
    OTHER_CASE_TYPE_ID,
    expand_case_type,
    get_case_type,
    resolve_case_types,
)

SAMPLE_ID = "anticipatory_bail"


def test_expansion_derives_the_category_from_the_case_type():
    """The whole point of expanding server-side: a category that disagrees with
    its case type is not merely unlikely, it is unrepresentable."""
    case_type = get_case_type(SAMPLE_ID)
    expanded = expand_case_type(case_type, "primary")

    assert expanded["primary_case_type_id"] == SAMPLE_ID
    assert expanded["primary_case_category_id"] == case_type["category_id"]
    assert expanded["primary_case_category"] == case_type["category_title"]
    assert expanded["primary_legal_services"] == case_type["legal_services"]


def test_expansion_copies_services_rather_than_sharing_them():
    """The taxonomy dicts are module-level and shared across every request, so
    a caller mutating a response must not corrupt them."""
    expanded = expand_case_type(get_case_type(SAMPLE_ID), "primary")
    expanded["primary_legal_services"][0]["title"] = "mutated"

    assert get_case_type(SAMPLE_ID)["legal_services"][0]["title"] != "mutated"


def test_empty_slot_keeps_the_same_keys():
    """An absent secondary must produce nulls, not missing keys - the response
    model has a fixed shape either way."""
    filled = expand_case_type(get_case_type(SAMPLE_ID), "secondary")
    empty = expand_case_type(None, "secondary")

    assert empty.keys() == filled.keys()
    assert empty["secondary_case_type_id"] is None
    assert empty["secondary_legal_services"] == []


def test_unknown_primary_falls_back_to_the_catch_all():
    primary, _ = resolve_case_types("not_a_real_id", None)
    assert primary["id"] == OTHER_CASE_TYPE_ID


def test_secondary_repeating_the_primary_is_dropped():
    """Secondary exists to carry a genuinely distinct second matter; echoing
    the primary back would render as a duplicate recommendation."""
    _, secondary = resolve_case_types(SAMPLE_ID, SAMPLE_ID)
    assert secondary is None


def test_unknown_secondary_is_dropped_without_failing_the_primary():
    primary, secondary = resolve_case_types(SAMPLE_ID, "not_a_real_id")

    assert primary["id"] == SAMPLE_ID
    assert secondary is None


def test_distinct_secondary_is_kept():
    primary, secondary = resolve_case_types(SAMPLE_ID, "cyber_crime")

    assert primary["id"] == SAMPLE_ID
    assert secondary["id"] == "cyber_crime"
