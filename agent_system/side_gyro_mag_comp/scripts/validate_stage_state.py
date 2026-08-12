#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

STAGES = [
    "DR0_DESIGN_REVIEW", "DR0_INDEPENDENT_AUDIT", "DR0_REPAIR",
    "IMPLEMENTATION", "IMPLEMENTATION_AUDIT", "IMPLEMENTATION_REPAIR",
    "ORACLE_HEADROOM", "LEARNED_COMPENSATION", "FEATURE_INCREMENT",
    "EVIDENCE_AUDIT", "EVIDENCE_REPAIR", "FINAL_SYNTHESIS", "FINAL_AUDIT",
    "COMPLETE", "COMPLETE_BLOCKED", "COMPLETE_REJECTED",
]
STATUSES = {"READY", "RUNNING", "WAITING_FOR_PEER", "BLOCKED", "PASS", "FAIL", "TERMINAL"}
ACTORS = {"CODEX", "CLAUDE", "NONE"}


def fail(msg: str) -> None:
    raise SystemExit(f"INVALID_STAGE_STATE: {msg}")


def main() -> int:
    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "schema_version", "study_id", "automation_mode", "human_review_mode",
        "current_stage", "stage_status", "source_commit", "repair_round_by_stage",
        "next_actor", "next_allowed_stage", "method_decision", "frozen_scope",
        "final_result_paths",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        fail(f"missing keys {missing}")
    if data["automation_mode"] != "AUTONOMOUS_UNTIL_FINAL":
        fail("automation_mode must be AUTONOMOUS_UNTIL_FINAL")
    if data["human_review_mode"] != "FINAL_ONLY":
        fail("human_review_mode must be FINAL_ONLY")
    if data["current_stage"] not in STAGES:
        fail(f"unknown current_stage {data['current_stage']}")
    if data["stage_status"] not in STATUSES:
        fail(f"unknown stage_status {data['stage_status']}")
    if data["next_actor"] not in ACTORS:
        fail(f"unknown next_actor {data['next_actor']}")
    repairs = data["repair_round_by_stage"]
    if not isinstance(repairs, dict) or any(v not in (0, 1) for v in repairs.values()):
        fail("each stage repair round must be 0 or 1")
    scope = data["frozen_scope"]
    if scope.get("runtime_sensors") != ["gyro", "magnetometer"]:
        fail("runtime_sensors must be exactly gyro and magnetometer")
    if scope.get("feature_dim_per_sensor") != 8:
        fail("feature dimension must remain 8 for this pilot")
    if scope.get("conditioning") != "branch_specific_film":
        fail("conditioning must remain branch_specific_film")
    if data["current_stage"].startswith("COMPLETE"):
        if data["stage_status"] != "TERMINAL" or data["next_actor"] != "NONE":
            fail("complete state must be TERMINAL with next_actor NONE")
    print("PASS_STAGE_STATE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
