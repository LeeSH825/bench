#!/usr/bin/env python3
"""Validate the portable Phase 0--1 repository evidence index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


INDEX_RELATIVE_PATH = Path(
    "docs/research/index/phase0_1/phase0_1_evidence_index.json"
)
MASTER_RELATIVE_PATH = Path(
    "docs/research/phase1b/AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md"
)
HANDOFF_RELATIVE_PATH = Path(
    "docs/research/index/phase0_1/PHASE2_REPOSITORY_LOOKUP_HANDOFF.md"
)
MASTER_EXPECTED_INPUT_SHA256 = (
    "657b956362457472c25ba03177e521114d1d92082cc30e10cf5f4170f52b96a2"
)

MANDATORY_TOPIC_IDS = {
    "NAV-MASTER-SUMMARY",
    "P0-OBJECTIVE",
    "P0-STATE-CONVENTION",
    "P0-TRUTH-BOUNDARY",
    "P0-SENSOR-ROLES",
    "P0-CONTEXT-CONTRACT",
    "P1A-GATE-A-CORE",
    "P1A-EXACT-PI-IMMUTABILITY",
    "P1A-TYPED-EVENTS",
    "P1A-BASILISK-FRAME",
    "P1A-CANONICAL-METRICS",
    "P1A-ADAPTER-RUNNER",
    "P1B-STEP1-FROZEN-BASELINES",
    "P1B-C1-STATIONARY",
    "P1B-C2-PROCESS",
    "P1B-C3-MEASUREMENT",
    "P1B-C5-IDENTIFIABILITY",
    "P1B-LONG-HORIZON",
    "P1B-STEP2-FUSION-SCHEMA",
    "P1B-MAIN-FUSION",
    "P1B-STRESS-MAG",
    "P1B-C4-COMBINED",
    "P1B-INITIAL-EXIT",
    "P1B-CLOSURE-DIAGNOSTICS",
    "P1B-F-CALIBRATED",
    "P1B-CLOSURE-CONFIRMATION",
    "P1B-CURRENT-EXIT",
    "PROVENANCE-TESTS",
    "PROVENANCE-COMMANDS",
    "PROVENANCE-DIRTY-TREE",
}


class IndexValidationError(AssertionError):
    """Raised when the evidence index violates its checked contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IndexValidationError(message)


def _unique_ids(records: Iterable[dict[str, Any]], label: str) -> list[str]:
    ids = [str(record.get("id", "")) for record in records]
    _require(all(ids), f"{label}: every record must have a non-empty id")
    _require(len(ids) == len(set(ids)), f"{label}: duplicate id")
    return ids


def _walk_strings(value: Any, key: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _walk_strings(child, child_key)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child, key)
    elif isinstance(value, str):
        yield key, value


def _check_portable_paths(data: dict[str, Any]) -> None:
    for key, value in _walk_strings(data):
        if key == "repository_root":
            continue
        if key in {"path", "expected_path"}:
            _require(
                not value.startswith("/"),
                f"absolute path is forbidden outside repository_root metadata: {value}",
            )


def _check_source_ref(repo_root: Path, source: dict[str, Any], context: str) -> None:
    path = source.get("path")
    locator = source.get("locator")
    _require(isinstance(path, str) and path, f"{context}: missing source path")
    _require(isinstance(locator, str) and locator, f"{context}: missing exact locator")
    _require((repo_root / path).is_file(), f"{context}: source path does not exist: {path}")


def load_and_validate(repo_root: Path) -> dict[str, Any]:
    """Load the canonical JSON index, validate it, and return count metadata."""

    repo_root = repo_root.resolve()
    index_path = repo_root / INDEX_RELATIVE_PATH
    _require(index_path.is_file(), f"missing index: {INDEX_RELATIVE_PATH}")
    raw = index_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    canonical = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _require(raw == canonical, "index JSON is not canonical sorted-key JSON")
    _check_portable_paths(data)

    _require(data.get("schema_version") == "phase0-1-evidence-index-v1", "schema version")
    metadata = data.get("metadata", {})
    _require(metadata.get("repository_root") == str(repo_root), "repository_root metadata")
    _require(
        metadata.get("validation_disposition")
        == "PASS_PHASE0_1_REPOSITORY_EVIDENCE_INDEX",
        "final validation disposition",
    )

    master_path = repo_root / MASTER_RELATIVE_PATH
    _require(master_path.is_file(), f"mandatory master missing: {MASTER_RELATIVE_PATH}")
    master_bytes = master_path.read_bytes()
    master_sha256 = hashlib.sha256(master_bytes).hexdigest()
    master_metadata = data.get("master_summary", {})
    _require(master_metadata.get("path") == str(MASTER_RELATIVE_PATH), "master metadata path")
    _require(master_metadata.get("authority") == "navigation_handoff_only", "master authority")
    _require(master_metadata.get("canonical_numeric_authority") is False, "master numeric authority")
    _require(master_metadata.get("claim_audit_status") == "complete", "master claim audit")
    _require(
        master_metadata.get("expected_input_sha256") == MASTER_EXPECTED_INPUT_SHA256,
        "master expected input SHA-256",
    )
    _require(master_metadata.get("final_sha256") == master_sha256, "master final SHA-256")
    _require(
        master_metadata.get("phase2_design_review") == "requires_explicit_user_request",
        "Phase 2 Design Review boundary",
    )
    _require(
        master_metadata.get("phase2_implementation") == "not_started_not_authorized",
        "Phase 2 implementation boundary",
    )

    topics = data.get("topics", [])
    files = data.get("files", [])
    decisions = data.get("decisions", [])
    ambiguities = data.get("unresolved_ambiguities", [])
    resolved_ambiguities = data.get("resolved_ambiguities", [])
    lookups = data.get("representative_lookups", [])
    topic_ids = set(_unique_ids(topics, "topics"))
    _unique_ids(files, "files")
    decision_ids = _unique_ids(decisions, "decisions")
    _unique_ids(ambiguities, "unresolved_ambiguities")
    _unique_ids(resolved_ambiguities, "resolved_ambiguities")
    _unique_ids(lookups, "representative_lookups")

    missing_topics = sorted(MANDATORY_TOPIC_IDS - topic_ids)
    _require(not missing_topics, f"missing mandatory topics: {missing_topics}")

    question_count = 0
    numeric_fact_count = 0
    for topic in topics:
        questions = topic.get("question_patterns", [])
        _require(isinstance(questions, list) and questions, f"{topic['id']}: questions")
        question_count += len(questions)
        sources = topic.get("canonical_sources", [])
        _require(sources, f"{topic['id']}: canonical source required")
        for source in sources:
            _check_source_ref(repo_root, source, topic["id"])
            if source.get("path") == str(MASTER_RELATIVE_PATH):
                _require(
                    topic["id"] == "NAV-MASTER-SUMMARY",
                    f"{topic['id']}: master may be canonical only for its navigation role",
                )
        for source in topic.get("navigation_sources", []):
            _check_source_ref(repo_root, source, f"{topic['id']} navigation")
        for field in ("implementation_paths", "test_paths", "config_paths", "result_paths"):
            for path in topic.get(field, []):
                _require((repo_root / path).exists(), f"{topic['id']}: missing {field}: {path}")
        for fact in topic.get("numeric_facts", []):
            numeric_fact_count += 1
            _require(
                isinstance(fact.get("value"), (int, float))
                and not isinstance(fact.get("value"), bool),
                f"{topic['id']}/{fact.get('id')}: numeric value required",
            )
            _require(fact.get("unit"), f"{topic['id']}/{fact.get('id')}: unit required")
            _require(fact.get("scope"), f"{topic['id']}/{fact.get('id')}: scope required")
            _check_source_ref(repo_root, fact.get("source", {}), f"numeric {fact.get('id')}")
            _require(
                fact.get("source", {}).get("path") != str(MASTER_RELATIVE_PATH),
                f"{topic['id']}/{fact.get('id')}: master cannot be a numeric source",
            )

    _require(question_count >= 80, f"only {question_count} indexed questions; require >= 80")
    _require(numeric_fact_count >= 30, "numeric catalog is too small")

    for record in files:
        path = record.get("path")
        _require(isinstance(path, str) and path, f"{record['id']}: file path")
        _require((repo_root / path).exists(), f"{record['id']}: indexed path missing: {path}")
        _require(record.get("role"), f"{record['id']}: role")
        _require(record.get("status") in {"current", "historical", "superseded"}, f"{record['id']}: status")

    master_files = [record for record in files if record.get("path") == str(MASTER_RELATIVE_PATH)]
    _require(len(master_files) == 1, "exactly one master file entry required")
    _require(master_files[0].get("status") == "current", "master file entry status")
    _require("navigation" in master_files[0].get("role", ""), "master file role")

    _require(len(lookups) >= 21, "at least 21 representative lookups are required")
    for lookup in lookups:
        _require(lookup.get("topic_id") in topic_ids, f"{lookup['id']}: unknown topic")
        _check_source_ref(repo_root, lookup.get("first_source", {}), lookup["id"])
    _require(any(lookup.get("id") == "L21" for lookup in lookups), "master lookup L21")

    current_status = data.get("current_phase_status", {})
    _require(current_status.get("phase") == "Phase 1 exit", "current phase")
    _require(current_status.get("decision") == "CONDITIONAL_GO", "current decision")
    _require(current_status.get("phase2_authorized") is False, "Phase 2 must not be authorized")
    _require(
        current_status.get("phase2_implementation_status") == "not_started_not_authorized",
        "Phase 2 implementation status",
    )
    _require(
        current_status.get("phase2_design_review_status") == "requires_explicit_user_request",
        "Phase 2 Design Review status",
    )
    _check_source_ref(repo_root, current_status.get("source", {}), "current_phase_status")

    current_exit = [
        d for d in decisions
        if d.get("domain") == "P1_EXIT" and d.get("status") == "current"
    ]
    historical_exit = [
        d for d in decisions
        if d.get("domain") == "P1_EXIT" and d.get("status") in {"historical", "superseded"}
    ]
    _require(len(current_exit) == 1, "exactly one current P1_EXIT decision required")
    _require(historical_exit, "historical P1_EXIT predecessor required")
    _require(
        current_exit[0].get("supersedes") in decision_ids,
        "current P1_EXIT decision must name its predecessor",
    )
    for decision in decisions:
        _check_source_ref(repo_root, decision.get("source", {}), decision["id"])

    ambiguity_ids = {item["id"] for item in ambiguities}
    _require("A-MISSING-MASTER-SUMMARY" not in ambiguity_ids, "missing-master ambiguity must be resolved")
    _require("A-PHASE2-ENTRY-CONDITION" in ambiguity_ids, "Phase 2 boundary ambiguity must remain")
    resolved_ids = {item["id"] for item in resolved_ambiguities}
    _require("A-MISSING-MASTER-SUMMARY" in resolved_ids, "missing-master resolution ledger")

    handoff = (repo_root / HANDOFF_RELATIVE_PATH).read_text(encoding="utf-8")
    ordered_paths = [
        str(MASTER_RELATIVE_PATH),
        "experiments/phase1b/reports/P1_EXIT_REVIEW_UPDATED.md",
        "docs/research/index/phase0_1/DECISION_AND_STATUS_LEDGER.md",
        "docs/research/index/phase0_1/SOURCE_OF_TRUTH_INDEX.md",
        "docs/research/index/phase0_1/NUMERIC_EVIDENCE_CATALOG.md",
    ]
    positions = [handoff.find(path) for path in ordered_paths]
    _require(all(position >= 0 for position in positions), "master-first handoff paths")
    _require(positions == sorted(positions), "master-first handoff order")

    return {
        "ambiguities": len(ambiguities),
        "decisions": len(decisions),
        "files": len(files),
        "numeric_facts": numeric_fact_count,
        "questions": question_count,
        "representative_lookups": len(lookups),
        "resolved_ambiguities": len(resolved_ambiguities),
        "topics": len(topics),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    try:
        counts = load_and_validate(args.repo_root)
    except (IndexValidationError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "PASS", **counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
