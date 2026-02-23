from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


CPU_MSE_DB_ABS_TOL = 1e-9
CPU_RECOVERY_K_ABS_TOL = 0.0


@dataclass
class DeterminismCpuResult:
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
    root = (
        bench_root
        / "runs"
        / "plan_matrix_smoke"
        / "C_shift_plan_matrix_smoke_v0"
        / "adaptive_knet"
        / track
        / "seed_0"
    )
    scen_dirs = sorted(root.glob("scenario_*"))
    for scen in scen_dirs:
        cand = scen / f"init_{init_id}"
        if cand.exists():
            return cand
    return Path("")


def _read_stability_metrics(run_dir: Path) -> Dict[str, float]:
    obj = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if str(obj.get("status", "")).lower() != "ok":
        raise RuntimeError(f"metrics status is not ok at {run_dir}")
    acc = obj.get("accuracy", {}) or {}
    rec = obj.get("shift_recovery", {}) or {}
    rk = rec.get("recovery_k")
    return {
        "mse_db": float(acc.get("mse_db", 0.0)),
        "recovery_k": float(rk if rk is not None else 0.0),
    }


def run_determinism_cpu_seed_stable(suite_yaml: Path) -> DeterminismCpuResult:
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
        return DeterminismCpuResult(ok=False, run_dir=Path(""), note=f"smoke_data failed:\n{out_data}")

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

    rc1, out1 = _run(cmd_run, bench_root, env)
    if rc1 != 0:
        return DeterminismCpuResult(ok=False, run_dir=Path(""), note=f"run_suite #1 failed:\n{out1}")

    run_dir_frozen = _find_plan_run_dir(bench_root, track="frozen", init_id="trained")
    run_dir_budgeted = _find_plan_run_dir(bench_root, track="budgeted", init_id="trained")
    if not run_dir_frozen.exists() or not run_dir_budgeted.exists():
        return DeterminismCpuResult(
            ok=False,
            run_dir=Path(""),
            note=(
                "expected both plan run dirs after run #1; "
                f"frozen={run_dir_frozen.exists()} budgeted={run_dir_budgeted.exists()}"
            ),
        )

    required = [
        "run_plan.json",
        "budget_ledger.json",
        "checkpoints/model.pt",
        "metrics.json",
        "metrics_step.csv",
        "timing.csv",
    ]
    for rd in (run_dir_frozen, run_dir_budgeted):
        missing = [p for p in required if not (rd / p).exists()]
        if missing:
            return DeterminismCpuResult(ok=False, run_dir=rd, note=f"run #1 missing artifacts: {missing}")

    m1f = _read_stability_metrics(run_dir_frozen)
    m1b = _read_stability_metrics(run_dir_budgeted)

    rc2, out2 = _run(cmd_run, bench_root, env)
    if rc2 != 0:
        return DeterminismCpuResult(ok=False, run_dir=run_dir_budgeted, note=f"run_suite #2 failed:\n{out2}")

    m2f = _read_stability_metrics(run_dir_frozen)
    m2b = _read_stability_metrics(run_dir_budgeted)

    for plan_id, a, b in (
        ("trained,frozen", m1f, m2f),
        ("trained,budgeted", m1b, m2b),
    ):
        if abs(float(a["mse_db"]) - float(b["mse_db"])) > CPU_MSE_DB_ABS_TOL:
            return DeterminismCpuResult(
                ok=False,
                run_dir=(run_dir_budgeted if "budgeted" in plan_id else run_dir_frozen),
                note=(
                    f"{plan_id} mse_db unstable: run1={a['mse_db']} run2={b['mse_db']} "
                    f"abs_tol={CPU_MSE_DB_ABS_TOL}"
                ),
            )
        if abs(float(a["recovery_k"]) - float(b["recovery_k"])) > CPU_RECOVERY_K_ABS_TOL:
            return DeterminismCpuResult(
                ok=False,
                run_dir=(run_dir_budgeted if "budgeted" in plan_id else run_dir_frozen),
                note=(
                    f"{plan_id} recovery_k unstable: run1={a['recovery_k']} run2={b['recovery_k']} "
                    f"abs_tol={CPU_RECOVERY_K_ABS_TOL}"
                ),
            )

    return DeterminismCpuResult(
        ok=True,
        run_dir=run_dir_budgeted,
        note=(
            "CPU determinism stable for trained,frozen + trained,budgeted "
            f"(mse_db_abs_tol={CPU_MSE_DB_ABS_TOL}, recovery_k_abs_tol={CPU_RECOVERY_K_ABS_TOL})"
        ),
    )
