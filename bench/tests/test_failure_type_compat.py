from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from bench.reports.aggregate import aggregate_by_seed, scan_runs


@dataclass
class FailureCompatResult:
    ok: bool
    note: str


def run_failure_type_compat() -> FailureCompatResult:
    with tempfile.TemporaryDirectory(prefix="bench_failure_compat_") as tmp:
        root = Path(tmp) / "runs"
        run_dir = root / "compat_suite" / "task_legacy" / "model_legacy" / "budgeted" / "seed_0" / "scenario_abcd1234"
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "config_snapshot.yaml").write_text(
            "\n".join(
                [
                    "suite:",
                    "  name: compat_suite",
                    "task:",
                    "  task_id: task_legacy",
                    "model:",
                    "  model_id: model_legacy",
                    "scenario_id: abcd1234",
                    "seed: 0",
                    "track_id: budgeted",
                    "init_id: trained",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        failure_obj = {
            "status": "failed",
            "category": "budget_overflow",  # legacy key to map
            "phase": "adapt",
            "message": "legacy overflow path",
            "context": {
                "suite_name": "compat_suite",
                "task_id": "task_legacy",
                "model_id": "model_legacy",
                "track_id": "budgeted",
                "init_id": "trained",
                "scenario_id": "abcd1234",
                "seed": 0,
            },
        }
        (run_dir / "failure.json").write_text(json.dumps(failure_obj, indent=2), encoding="utf-8")

        records = scan_runs(runs_root=root, suite_name="compat_suite")
        if len(records) != 1:
            return FailureCompatResult(ok=False, note=f"expected 1 record, got {len(records)}")
        rec = records[0]
        if rec.failure_type != "budget_overflow":
            return FailureCompatResult(
                ok=False,
                note=f"legacy category->failure_type mapping failed: got {rec.failure_type}",
            )
        if rec.failure_stage != "adapt":
            return FailureCompatResult(ok=False, note=f"expected failure_stage=adapt, got {rec.failure_stage}")

        agg = aggregate_by_seed(records)
        if len(agg) != 1:
            return FailureCompatResult(ok=False, note=f"expected one aggregate row, got {len(agg)}")
        row = agg[0]
        if str(row.get("failure_type", "")) != "budget_overflow":
            return FailureCompatResult(
                ok=False,
                note=f"aggregate failure_type mismatch: {row.get('failure_type')}",
            )
        if int(row.get("fail_count", 0)) != 1:
            return FailureCompatResult(ok=False, note=f"aggregate fail_count mismatch: {row.get('fail_count')}")

    return FailureCompatResult(ok=True, note="legacy category mapping to failure_type works in report ingestion")
