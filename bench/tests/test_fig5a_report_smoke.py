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
class Fig5aReportSmokeResult:
    ok: bool
    out_dir: Path
    note: str


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    cp = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    out = (cp.stdout or "") + (cp.stderr or "")
    return cp.returncode, out


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return list(r.fieldnames or []), rows


def run_fig5a_report_smoke(suite_yaml: Path) -> Fig5aReportSmokeResult:
    bench_root = _bench_root()
    suite_yaml = suite_yaml.expanduser().resolve()
    suite_name = "fig5a_overlay_smoke"
    tasks = [
        "F5aS_m2n2_T50_invR2db_0",
        "F5aS_m2n2_T50_invR2db_20",
    ]

    shutil.rmtree(bench_root / "runs" / suite_name, ignore_errors=True)

    report_paths = [
        bench_root / "reports" / f"summary_{suite_name}.csv",
        bench_root / "reports" / f"aggregate_{suite_name}.csv",
        bench_root / "reports" / f"fig5a_points_{suite_name}.csv",
        bench_root / "reports" / f"fig5a_overlay_mse_db_vs_inv_r2_db_{suite_name}.png",
        bench_root / "reports" / f"fig5a_mse_db_vs_inv_r2_db_{suite_name}.png",
    ]
    for p in report_paths:
        if p.exists():
            p.unlink()

    env = os.environ.copy()
    env["BENCH_DATA_CACHE"] = str((bench_root / "bench_data_cache").resolve())

    for task_id in tasks:
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
            return Fig5aReportSmokeResult(ok=False, out_dir=bench_root / "reports", note=f"smoke_data failed:\n{out_data}")

    cmd_run_knet = [
        sys.executable,
        "-m",
        "bench.runners.run_suite",
        "--suite-yaml",
        str(suite_yaml),
        "--tasks",
        *tasks,
        "--models",
        "kalmannet_tsp",
        "--seeds",
        "0",
        "--plans",
        "trained:frozen",
        "--device",
        "cpu",
    ]
    rc_kn, out_kn = _run(cmd_run_knet, bench_root, env)
    if rc_kn != 0:
        return Fig5aReportSmokeResult(ok=False, out_dir=bench_root / "reports", note=f"run_suite knet failed:\n{out_kn}")

    cmd_run_kf = [
        sys.executable,
        "-m",
        "bench.runners.run_suite",
        "--suite-yaml",
        str(suite_yaml),
        "--tasks",
        *tasks,
        "--models",
        "oracle_kf",
        "--seeds",
        "0",
        "--plans",
        "pretrained:frozen",
        "--device",
        "cpu",
    ]
    rc_kf, out_kf = _run(cmd_run_kf, bench_root, env)
    if rc_kf != 0:
        return Fig5aReportSmokeResult(ok=False, out_dir=bench_root / "reports", note=f"run_suite oracle_kf failed:\n{out_kf}")

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
        "--fig5a-plot",
    ]
    rc_rep, out_rep = _run(cmd_report, bench_root, env)
    if rc_rep != 0:
        return Fig5aReportSmokeResult(ok=False, out_dir=bench_root / "reports", note=f"make_report failed:\n{out_rep}")

    required = [
        bench_root / "reports" / f"summary_{suite_name}.csv",
        bench_root / "reports" / f"aggregate_{suite_name}.csv",
        bench_root / "reports" / f"fig5a_points_{suite_name}.csv",
        bench_root / "reports" / f"fig5a_overlay_mse_db_vs_inv_r2_db_{suite_name}.png",
    ]
    missing = [str(p.name) for p in required if not p.exists()]
    if missing:
        return Fig5aReportSmokeResult(
            ok=False,
            out_dir=bench_root / "reports",
            note=f"missing fig5a report outputs: {missing}\n{out_rep}",
        )

    cols, rows = _read_csv(bench_root / "reports" / f"fig5a_points_{suite_name}.csv")
    required_cols = {
        "model_id",
        "init_id",
        "track_id",
        "task_id",
        "scenario_id",
        "seed",
        "x_dim",
        "y_dim",
        "T",
        "q2",
        "r2",
        "inv_r2_db",
        "mse_db",
    }
    if not required_cols.issubset(set(cols)):
        return Fig5aReportSmokeResult(
            ok=False,
            out_dir=bench_root / "reports",
            note=f"fig5a_points missing required columns: {sorted(required_cols - set(cols))}",
        )

    model_ids = {str(r.get("model_id", "")) for r in rows}
    if not {"kalmannet_tsp", "oracle_kf"}.issubset(model_ids):
        return Fig5aReportSmokeResult(
            ok=False,
            out_dir=bench_root / "reports",
            note=f"fig5a_points missing expected model rows; found={sorted(model_ids)}",
        )

    plan_pairs = {(str(r.get("model_id", "")), str(r.get("init_id", "")), str(r.get("track_id", ""))) for r in rows}
    expected_pairs = {
        ("kalmannet_tsp", "trained", "frozen"),
        ("oracle_kf", "pretrained", "frozen"),
    }
    if not expected_pairs.issubset(plan_pairs):
        return Fig5aReportSmokeResult(
            ok=False,
            out_dir=bench_root / "reports",
            note=f"fig5a_points missing expected plan pairs; found={sorted(plan_pairs)}",
        )

    return Fig5aReportSmokeResult(
        ok=True,
        out_dir=bench_root / "reports",
        note="fig5a report smoke passed (points csv + overlay png + model/plan coverage)",
    )
