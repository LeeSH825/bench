from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from bench.reports.aggregate import (
    RunRecord,
    build_scenario_cfg_basis as report_build_scenario_cfg_basis,
    canonicalize_scenario_id as report_canonicalize_scenario_id,
    scan_runs,
)
from bench.reports.plots import _build_severity_sweep_series
from bench.runners.run_suite import (
    _build_scenario_cfg_basis as runner_build_scenario_cfg_basis,
    _canonicalize_scenario_id as runner_canonicalize_scenario_id,
)


@dataclass
class ReportingReproFixesResult:
    ok: bool
    note: str


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def run_reporting_repro_fixes() -> ReportingReproFixesResult:
    with tempfile.TemporaryDirectory(prefix="bench_reporting_fixes_") as tmp:
        root = Path(tmp)
        runs_root = root / "runs"
        suite_name = "shift"

        run_new = runs_root / suite_name / "taskA" / "modelA" / "frozen" / "seed_0" / "scenario_new"
        run_old = runs_root / suite_name / "taskA" / "modelA" / "frozen" / "seed_0" / "scenario_old"
        for rd, scen in ((run_new, "new"), (run_old, "old")):
            _write_json(
                rd / "metrics.json",
                {
                    "status": "ok",
                    "suite_name": suite_name,
                    "task_id": "taskA",
                    "model_id": "modelA",
                    "track_id": "frozen",
                    "seed": 0,
                    "scenario_id": scen,
                    "accuracy": {"mse": 1.0, "rmse": 1.0, "mse_db": 0.0},
                    "run_plan": {
                        "suite_name": suite_name,
                        "task_id": "taskA",
                        "model_id": "modelA",
                        "track_id": "frozen",
                        "init_id": "trained",
                        "seed": 0,
                        "scenario_id": scen,
                    },
                    "budgets": {
                        "train_updates_used": 1,
                        "train_outer_updates_used": 1,
                        "train_inner_updates_used": 3,
                        "adapt_updates_used": 0,
                    },
                },
            )

        manifest_dir = runs_root / suite_name / "_manifests"
        _write_json(
            manifest_dir / "20260227-000000_a.json",
            {
                "suite_name": suite_name,
                "run_dirs": [str(run_new)],
            },
        )

        rec_latest = scan_runs(runs_root=runs_root, suite_name=suite_name, input_scope="latest_manifest")
        if len(rec_latest) != 1:
            return ReportingReproFixesResult(
                ok=False,
                note=f"latest_manifest scope expected 1 run, got {len(rec_latest)}",
            )

        rec_all = scan_runs(runs_root=runs_root, suite_name=suite_name, input_scope="all_runs")
        if len(rec_all) != 2:
            return ReportingReproFixesResult(
                ok=False,
                note=f"all_runs scope expected 2 runs, got {len(rec_all)}",
            )

        _write_json(
            run_new / "failure.json",
            {
                "status": "failed",
                "failure_type": "budget_overflow",
                "phase": "adapt",
                "message": "stale failure artifact",
            },
        )
        rec_after_stale_failure = scan_runs(
            runs_root=runs_root,
            suite_name=suite_name,
            input_scope="latest_manifest",
        )
        if len(rec_after_stale_failure) != 1:
            return ReportingReproFixesResult(ok=False, note="stale-failure check missing latest run record")
        rr = rec_after_stale_failure[0]
        if str(rr.status).lower() != "ok":
            return ReportingReproFixesResult(
                ok=False,
                note=f"metrics success should win over stale failure.json, got status={rr.status}",
            )
        if rr.failure_type not in (None, "", "None"):
            return ReportingReproFixesResult(
                ok=False,
                note=f"stale failure metadata should be ignored for successful metrics, got failure_type={rr.failure_type}",
            )

        task = {
            "task_id": "C_shift_basis_check_v0",
            "noise": {
                "pre_shift": {"Q": {"q2": 1.0e-3}, "R": {"r2": 1.0e-3}},
                "shift": {"t0": 20, "post_shift": {"R_scale": 10.0}},
            },
        }
        scenario_settings = {"shift.post_shift.R_scale": 30.0}
        runner_basis = runner_build_scenario_cfg_basis(task, scenario_settings)
        report_basis = report_build_scenario_cfg_basis(task, scenario_settings)
        if runner_basis != report_basis:
            return ReportingReproFixesResult(
                ok=False,
                note=f"runner/report scenario_cfg_basis mismatch: runner={runner_basis} report={report_basis}",
            )
        runner_sid = runner_canonicalize_scenario_id(str(task["task_id"]), runner_basis)
        report_sid = report_canonicalize_scenario_id(str(task["task_id"]), report_basis)
        if str(runner_sid) != str(report_sid):
            return ReportingReproFixesResult(
                ok=False,
                note=f"runner/report scenario_id mismatch: runner={runner_sid} report={report_sid}",
            )

        inv_settings = {"inv_r2_db": 20.0}
        runner_basis_inv = runner_build_scenario_cfg_basis(task, inv_settings)
        report_basis_inv = report_build_scenario_cfg_basis(task, inv_settings)
        if runner_basis_inv != report_basis_inv:
            return ReportingReproFixesResult(
                ok=False,
                note=f"runner/report inv_r2_db basis mismatch: runner={runner_basis_inv} report={report_basis_inv}",
            )
        try:
            inv_r2 = float(runner_basis_inv["pre_shift"]["R"]["r2"])
        except Exception:
            return ReportingReproFixesResult(
                ok=False,
                note=f"inv_r2_db basis did not map to pre_shift.R.r2: basis={runner_basis_inv}",
            )
        if abs(inv_r2 - 0.01) > 1e-12:
            return ReportingReproFixesResult(
                ok=False,
                note=f"inv_r2_db=20 should map r2=0.01, got r2={inv_r2}",
            )
        runner_sid_inv = runner_canonicalize_scenario_id(str(task["task_id"]), runner_basis_inv)
        report_sid_inv = report_canonicalize_scenario_id(str(task["task_id"]), report_basis_inv)
        if str(runner_sid_inv) != str(report_sid_inv):
            return ReportingReproFixesResult(
                ok=False,
                note=f"runner/report inv_r2_db scenario_id mismatch: runner={runner_sid_inv} report={report_sid_inv}",
            )

        rec_base = {
            "suite": "shift",
            "task_id": "C_shift_basis_check_v0",
            "scenario_id": "scenario_a",
            "seed": 0,
            "model_id": "adaptive_knet",
            "track_id": "frozen",
            "status": "ok",
            "run_dir": Path("."),
            "severity_r_scale": 10.0,
        }
        severity_records = [
            RunRecord(init_id="trained", mse=1.0, mse_db=0.0, rmse=1.0, **rec_base),
            RunRecord(init_id="untrained", mse=2.0, mse_db=3.0, rmse=1.4142, **rec_base),
        ]
        series = _build_severity_sweep_series(
            records=severity_records,
            severity_key="shift.post_shift.R_scale",
            x_field="severity_key",
            y_field="mse_mean",
        )
        if len(series) != 2:
            return ReportingReproFixesResult(
                ok=False,
                note=f"severity grouping must separate init_id; expected 2 lines, got {len(series)}",
            )
        if ("adaptive_knet", "frozen", "trained") not in series:
            return ReportingReproFixesResult(ok=False, note="missing trained severity series key")
        if ("adaptive_knet", "frozen", "untrained") not in series:
            return ReportingReproFixesResult(ok=False, note="missing untrained severity series key")

    return ReportingReproFixesResult(
        ok=True,
        note="manifest scope, stale-failure precedence, scenario-id consistency, and init-aware severity grouping passed",
    )
