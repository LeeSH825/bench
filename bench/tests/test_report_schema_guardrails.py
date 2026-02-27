from __future__ import annotations

import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .test_plan_matrix_minimal import run_plan_matrix_minimal


@dataclass
class ReportSchemaGuardrailsResult:
    ok: bool
    out_dir: Path
    note: str


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    cp = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    out = (cp.stdout or "") + (cp.stderr or "")
    return cp.returncode, out


def _csv_columns(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or [])


def _has_rows(path: Path) -> bool:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for _ in r:
            return True
    return False


def run_report_schema_guardrails(suite_yaml: Path) -> ReportSchemaGuardrailsResult:
    bench_root = _bench_root()
    suite_yaml = suite_yaml.expanduser().resolve()
    suite_name = "plan_matrix_smoke"
    task_id = "C_shift_plan_matrix_smoke_v0"

    # Ensure plan-matrix runs exist.
    run_root = bench_root / "runs" / suite_name
    if not run_root.exists():
        pm = run_plan_matrix_minimal(suite_yaml=suite_yaml)
        if not pm.ok:
            return ReportSchemaGuardrailsResult(ok=False, out_dir=bench_root / "reports", note=pm.note)

    env = os.environ.copy()
    env["BENCH_DATA_CACHE"] = str((bench_root / "bench_data_cache").resolve())

    cmd_report = [
        sys.executable,
        "-m",
        "bench.reports.make_report",
        "--suite-yaml",
        str(suite_yaml),
        "--runs-root",
        str(bench_root / "runs"),
        "--out-dir",
        str(bench_root / "reports"),
        "--input-scope",
        "all_runs",
        "--group-by",
        "init_id,track_id",
        "--plan-views",
        "--include-ops",
        "--budget-curves",
    ]
    rc_rep, out_rep = _run(cmd_report, bench_root, env)
    if rc_rep != 0:
        return ReportSchemaGuardrailsResult(
            ok=False,
            out_dir=bench_root / "reports",
            note=f"make_report failed:\n{out_rep}",
        )

    out_dir = bench_root / "reports"
    summary_csv = out_dir / f"summary_{suite_name}.csv"
    aggregate_csv = out_dir / f"aggregate_{suite_name}.csv"
    plan_compare_csv = out_dir / f"plan_compare_{suite_name}.csv"
    failure_csv = out_dir / f"failure_by_plan_{suite_name}.csv"
    ops_csv = out_dir / f"ops_by_plan_{suite_name}.csv"
    ops_png = out_dir / f"ops_tradeoff_{suite_name}.png"
    required_files = [summary_csv, aggregate_csv, plan_compare_csv, failure_csv, ops_csv, ops_png]
    missing = [str(p.name) for p in required_files if not p.exists()]
    if missing:
        return ReportSchemaGuardrailsResult(
            ok=False,
            out_dir=out_dir,
            note=f"required report artifacts missing: {missing}",
        )

    # run_dir/report invariants: required plots
    if not list(out_dir.glob(f"track_compare_{task_id}_*.png")):
        return ReportSchemaGuardrailsResult(
            ok=False,
            out_dir=out_dir,
            note=f"missing track comparison plot for task={task_id}",
        )
    if not list(out_dir.glob(f"budget_curve_{task_id}_*.png")):
        return ReportSchemaGuardrailsResult(
            ok=False,
            out_dir=out_dir,
            note=f"missing budget curve plot for task={task_id}",
        )

    # schema guardrails
    req_summary = {
        "model_id",
        "task_id",
        "scenario_id",
        "seed",
        "init_id",
        "track_id",
        "status",
        "failure_type",
        "failure_stage",
        "mse_db",
        "recovery_k",
        "train_updates_used",
        "adapt_updates_used",
        "cache_hit",
        "total_time_s",
    }
    req_agg = {
        "model_id",
        "task_id",
        "scenario_id",
        "init_id",
        "track_id",
        "fail_rate",
        "failure_type",
        "mse_db_mean",
        "recovery_k_mean",
        "cache_hit_rate",
        "train_skipped_rate",
    }
    req_plan = {
        "suite",
        "task_id",
        "scenario_id",
        "seed",
        "model_id",
        "status__trained__frozen",
        "status__trained__budgeted",
        "mse_db__trained__frozen",
        "mse_db__trained__budgeted",
        "delta_mse_db_trained_budgeted_minus_frozen",
        "delta_recovery_k_trained_budgeted_minus_frozen",
    }
    req_failure = {
        "suite",
        "task_id",
        "model_id",
        "init_id",
        "track_id",
        "failure_type",
        "fail_count",
        "n_total",
        "fail_rate",
    }
    req_ops = {
        "suite",
        "task_id",
        "model_id",
        "init_id",
        "track_id",
        "n_total",
        "n_success",
        "total_time_s_mean",
        "train_updates_used_mean",
        "adapt_updates_used_mean",
        "cache_hit_rate",
        "train_skipped_rate",
    }

    cols_summary = set(_csv_columns(summary_csv))
    cols_agg = set(_csv_columns(aggregate_csv))
    cols_plan = set(_csv_columns(plan_compare_csv))
    cols_failure = set(_csv_columns(failure_csv))
    cols_ops = set(_csv_columns(ops_csv))

    for name, have, need in (
        ("summary", cols_summary, req_summary),
        ("aggregate", cols_agg, req_agg),
        ("plan_compare", cols_plan, req_plan),
        ("failure_by_plan", cols_failure, req_failure),
        ("ops_by_plan", cols_ops, req_ops),
    ):
        missing_cols = sorted(need - have)
        if missing_cols:
            return ReportSchemaGuardrailsResult(
                ok=False,
                out_dir=out_dir,
                note=f"{name} missing required columns: {missing_cols}",
            )

    # Ensure new S6 tables are not empty for smoke suite.
    for p in (plan_compare_csv, failure_csv, ops_csv):
        if not _has_rows(p):
            return ReportSchemaGuardrailsResult(
                ok=False,
                out_dir=out_dir,
                note=f"{p.name} must contain at least one row",
            )

    return ReportSchemaGuardrailsResult(
        ok=True,
        out_dir=out_dir,
        note="report schema guardrails passed (required tables/columns/plots present)",
    )
