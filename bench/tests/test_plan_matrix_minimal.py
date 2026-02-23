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
class PlanMatrixResult:
    ok: bool
    run_dir: Path
    note: str


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    cp = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    out = (cp.stdout or "") + (cp.stderr or "")
    return cp.returncode, out


def _find_plan_run_dir(bench_root: Path, track: str, init_id: str) -> Path:
    root = bench_root / "runs" / "plan_matrix_smoke" / "C_shift_plan_matrix_smoke_v0" / "adaptive_knet" / track / "seed_0"
    scen_dirs = sorted(root.glob("scenario_*"))
    for scen in scen_dirs:
        cand = scen / f"init_{init_id}"
        if cand.exists():
            return cand
    return Path("")


def run_plan_matrix_minimal(suite_yaml: Path) -> PlanMatrixResult:
    bench_root = _bench_root()
    suite_yaml = suite_yaml.expanduser().resolve()
    task_id = "C_shift_plan_matrix_smoke_v0"
    model_id = "adaptive_knet"

    shutil.rmtree(bench_root / "runs" / "plan_matrix_smoke", ignore_errors=True)
    summary_csv = bench_root / "reports" / "summary_plan_matrix_smoke.csv"
    if summary_csv.exists():
        summary_csv.unlink()

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
        return PlanMatrixResult(ok=False, run_dir=Path(""), note=f"smoke_data failed:\n{out_data}")

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
        return PlanMatrixResult(ok=False, run_dir=Path(""), note=f"plan-matrix run failed:\n{out_run}")

    run_dir_frozen = _find_plan_run_dir(bench_root, track="frozen", init_id="trained")
    run_dir_budgeted = _find_plan_run_dir(bench_root, track="budgeted", init_id="trained")
    if not run_dir_frozen.exists() or not run_dir_budgeted.exists():
        return PlanMatrixResult(
            ok=False,
            run_dir=Path(""),
            note=(
                "expected both plan run dirs; "
                f"frozen_exists={run_dir_frozen.exists()} budgeted_exists={run_dir_budgeted.exists()}"
            ),
        )

    for run_dir in (run_dir_frozen, run_dir_budgeted):
        required = [
            "run_plan.json",
            "budget_ledger.json",
            "checkpoints/model.pt",
            "checkpoints/train_state.json",
            "metrics.json",
            "metrics_step.csv",
        ]
        missing = [p for p in required if not (run_dir / p).exists()]
        if missing:
            return PlanMatrixResult(ok=False, run_dir=run_dir, note=f"missing required artifacts: {missing}")

    if not summary_csv.exists():
        return PlanMatrixResult(ok=False, run_dir=run_dir_budgeted, note="summary_plan_matrix_smoke.csv missing")

    rows: List[Dict[str, str]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (
                row.get("suite") == "plan_matrix_smoke"
                and row.get("task_id") == task_id
                and row.get("model_id") == model_id
                and row.get("seed") == "0"
            ):
                rows.append(row)

    got = {(row.get("init_id", ""), row.get("track_id", "")) for row in rows}
    required_pairs = {("trained", "frozen"), ("trained", "budgeted")}
    if not required_pairs.issubset(got):
        return PlanMatrixResult(
            ok=False,
            run_dir=run_dir_budgeted,
            note=f"summary rows missing required (init_id,track_id) pairs. got={sorted(got)}",
        )

    return PlanMatrixResult(
        ok=True,
        run_dir=run_dir_budgeted,
        note="plan matrix run produced trained/frozen + trained/budgeted with distinct summary keys",
    )
