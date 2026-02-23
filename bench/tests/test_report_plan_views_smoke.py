from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ReportPlanViewsResult:
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


def run_report_plan_views_smoke(suite_yaml: Path) -> ReportPlanViewsResult:
    bench_root = _bench_root()
    suite_yaml = suite_yaml.expanduser().resolve()
    suite_name = "plan_matrix_smoke"
    task_id = "C_shift_plan_matrix_smoke_v0"
    model_id = "adaptive_knet"

    shutil.rmtree(bench_root / "runs" / suite_name, ignore_errors=True)
    for p in [
        bench_root / "reports" / f"summary_{suite_name}.csv",
        bench_root / "reports" / f"aggregate_{suite_name}.csv",
        bench_root / "reports" / f"plan_compare_{suite_name}.csv",
        bench_root / "reports" / f"failure_by_plan_{suite_name}.csv",
        bench_root / "reports" / f"ops_by_plan_{suite_name}.csv",
        bench_root / "reports" / f"ops_tradeoff_{suite_name}.png",
    ]:
        if p.exists():
            p.unlink()
    for pat in ("track_compare_*.png", "budget_curve_*.png"):
        for p in (bench_root / "reports").glob(pat):
            if suite_name in p.name or task_id in p.name:
                p.unlink()

    env = os.environ.copy()
    env["BENCH_DATA_CACHE"] = str((bench_root / "bench_data_cache").resolve())

    cmd_data = [
        sys.executable,
        "-m",
        "bench.tasks.smoke_data",
        "--suite-yaml",
        str(suite_yaml),
        "--task",
        task_id,
        "--seed",
        "0",
    ]
    rc_data, out_data = _run(cmd_data, bench_root, env)
    if rc_data != 0:
        return ReportPlanViewsResult(ok=False, out_dir=bench_root / "reports", note=f"smoke_data failed:\n{out_data}")

    cmd_run = [
        sys.executable,
        "-m",
        "bench.runners.run_suite",
        "--suite-yaml",
        str(suite_yaml),
        "--tasks",
        task_id,
        "--models",
        model_id,
        "--seeds",
        "0",
        "--plans",
        "trained:frozen",
        "trained:budgeted",
        "--device",
        "cpu",
    ]
    rc_run, out_run = _run(cmd_run, bench_root, env)
    if rc_run != 0:
        return ReportPlanViewsResult(ok=False, out_dir=bench_root / "reports", note=f"run_suite failed:\n{out_run}")

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
        "--group-by",
        "init_id,track_id",
        "--plan-views",
        "--include-ops",
        "--budget-curves",
    ]
    rc_rep, out_rep = _run(cmd_report, bench_root, env)
    if rc_rep != 0:
        return ReportPlanViewsResult(ok=False, out_dir=bench_root / "reports", note=f"make_report failed:\n{out_rep}")

    out_dir = bench_root / "reports"
    required = [
        out_dir / f"summary_{suite_name}.csv",
        out_dir / f"aggregate_{suite_name}.csv",
        out_dir / f"plan_compare_{suite_name}.csv",
        out_dir / f"failure_by_plan_{suite_name}.csv",
        out_dir / f"ops_by_plan_{suite_name}.csv",
        out_dir / f"ops_tradeoff_{suite_name}.png",
    ]
    missing = [str(p.name) for p in required if not p.exists()]
    if missing:
        return ReportPlanViewsResult(
            ok=False,
            out_dir=out_dir,
            note=f"missing expected S6 report artifacts: {missing}\n{out_rep}",
        )

    if not list(out_dir.glob(f"track_compare_{task_id}_*.png")):
        return ReportPlanViewsResult(
            ok=False,
            out_dir=out_dir,
            note=f"missing track comparison plot for task={task_id}",
        )
    if not list(out_dir.glob(f"budget_curve_{task_id}_*.png")):
        return ReportPlanViewsResult(
            ok=False,
            out_dir=out_dir,
            note=f"missing budget curve plot for task={task_id}",
        )

    summary_cols = set(_csv_columns(out_dir / f"summary_{suite_name}.csv"))
    required_summary_cols = {
        "model_id",
        "task_id",
        "scenario_id",
        "seed",
        "init_id",
        "track_id",
        "status",
        "failure_type",
        "failure_stage",
        "mse",
        "rmse",
        "mse_db",
        "recovery_k",
        "t0",
        "train_updates_used",
        "adapt_updates_used",
        "adapt_updates_per_step_max",
        "cache_enabled",
        "cache_hit",
        "train_skipped",
        "train_time_s",
        "eval_time_s",
        "adapt_time_s",
        "total_time_s",
    }
    if not required_summary_cols.issubset(summary_cols):
        return ReportPlanViewsResult(
            ok=False,
            out_dir=out_dir,
            note=f"summary columns missing required fields: {sorted(required_summary_cols - summary_cols)}",
        )

    agg_cols = set(_csv_columns(out_dir / f"aggregate_{suite_name}.csv"))
    for k in ("init_id", "track_id", "cache_hit_rate", "train_skipped_rate"):
        if k not in agg_cols:
            return ReportPlanViewsResult(ok=False, out_dir=out_dir, note=f"aggregate missing column: {k}")

    return ReportPlanViewsResult(
        ok=True,
        out_dir=out_dir,
        note="report plan views smoke passed (tables + S6 plots + required columns)",
    )
