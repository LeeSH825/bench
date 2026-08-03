from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from tools.research.validate_phase0_1_evidence_index import (
    INDEX_RELATIVE_PATH,
    MASTER_EXPECTED_INPUT_SHA256,
    MASTER_RELATIVE_PATH,
    load_and_validate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def index() -> dict:
    return json.loads((REPO_ROOT / INDEX_RELATIVE_PATH).read_text(encoding="utf-8"))


def test_validator_accepts_canonical_index() -> None:
    counts = load_and_validate(REPO_ROOT)
    assert counts["representative_lookups"] == 21
    assert counts["questions"] >= 80
    assert counts["numeric_facts"] >= 30


def test_representative_lookup_sources_resolve(index: dict) -> None:
    lookups = index["representative_lookups"]
    assert len(lookups) == 21
    for lookup in lookups:
        source = lookup["first_source"]
        assert (REPO_ROOT / source["path"]).is_file()
        assert source["locator"]


def test_current_and_historical_exit_decisions_are_distinct(index: dict) -> None:
    decisions = {item["id"]: item for item in index["decisions"]}
    current = decisions["D-P1-EXIT-UPDATED"]
    original = decisions["D-P1-EXIT-ORIGINAL"]
    assert current["status"] == "current"
    assert current["decision"] == "CONDITIONAL_GO"
    assert current["supersedes"] == original["id"]
    assert original["status"] == "historical"


def test_scoped_nees_values_are_not_conflated(index: dict) -> None:
    facts = {
        fact["id"]: fact
        for topic in index["topics"]
        for fact in topic.get("numeric_facts", [])
    }
    assert facts["N-MAIN-ORIGINAL-SETTLED-NEES"]["value"] == 1.8730178719854724
    assert facts["N-CLOSURE-VALIDATION-SETTLED-NEES"]["value"] == 1.9062451467732702
    assert facts["N-CLOSURE-FBASE-CONFIRM-NEES"]["value"] == 1.4180268635870965
    assert facts["N-CLOSURE-FCAL-CONFIRM-NEES"]["value"] == 1.0206761630935368
    assert len({facts[key]["scope"] for key in (
        "N-MAIN-ORIGINAL-SETTLED-NEES",
        "N-CLOSURE-VALIDATION-SETTLED-NEES",
        "N-CLOSURE-FBASE-CONFIRM-NEES",
        "N-CLOSURE-FCAL-CONFIRM-NEES",
    )}) == 4


def test_master_summary_is_present_audited_and_not_numeric_authority(index: dict) -> None:
    master_path = REPO_ROOT / MASTER_RELATIVE_PATH
    assert master_path.is_file()
    assert index["master_summary"]["expected_input_sha256"] == MASTER_EXPECTED_INPUT_SHA256
    assert index["master_summary"]["final_sha256"] == hashlib.sha256(master_path.read_bytes()).hexdigest()
    assert index["master_summary"]["canonical_numeric_authority"] is False
    assert all(
        fact["source"]["path"] != str(MASTER_RELATIVE_PATH)
        for topic in index["topics"]
        for fact in topic.get("numeric_facts", [])
    )
    unresolved = {item["id"] for item in index["unresolved_ambiguities"]}
    resolved = {item["id"] for item in index["resolved_ambiguities"]}
    assert "A-MISSING-MASTER-SUMMARY" not in unresolved
    assert "A-MISSING-MASTER-SUMMARY" in resolved
    assert index["current_phase_status"]["phase2_authorized"] is False
    assert index["current_phase_status"]["phase2_design_review_status"] == "requires_explicit_user_request"
    assert index["current_phase_status"]["phase2_implementation_status"] == "not_started_not_authorized"
