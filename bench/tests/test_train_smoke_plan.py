from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class TrainSmokeResult:
    ok: bool
    run_dir: Path
    note: str


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    cp = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    out = (cp.stdout or "") + (cp.stderr or "")
    return cp.returncode, out


def _find_run_dir(bench_root: Path) -> Path:
    root = bench_root / "runs" / "train_smoke" / "A_linear_train_smoke_v0" / "kalmannet_tsp" / "frozen" / "seed_0"
    cands = sorted(root.glob("scenario_*"))
    if not cands:
        return Path("")
    return cands[0]


def _read_metrics(run_dir: Path) -> Dict[str, float]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    acc = metrics.get("accuracy", {}) or {}
    return {
        "mse": float(acc.get("mse", 0.0)),
        "rmse": float(acc.get("rmse", 0.0)),
        "mse_db": float(acc.get("mse_db", 0.0)),
    }


def run_train_smoke_route_b(suite_yaml: Path) -> TrainSmokeResult:
    bench_root = _bench_root()
    suite_yaml = suite_yaml.expanduser().resolve()
    task_id = "A_linear_train_smoke_v0"
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
        return TrainSmokeResult(ok=False, run_dir=Path(""), note=f"smoke_data failed:\n{out_data}")

    cmd_run = [
        sys.executable,
        "-m",
        "bench.runners.run_suite",
        "--suite-yaml",
        str(suite_yaml),
        "--tasks",
        task_id,
        "--models",
        "kalmannet_tsp",
        "--seeds",
        "0",
        "--track",
        "frozen",
        "--init-id",
        "trained",
        "--device",
        "cpu",
    ]

    rc1, out1 = _run(cmd_run, bench_root, env)
    if rc1 != 0:
        return TrainSmokeResult(ok=False, run_dir=Path(""), note=f"run_suite #1 failed:\n{out1}")

    run_dir = _find_run_dir(bench_root)
    if not run_dir.exists():
        return TrainSmokeResult(ok=False, run_dir=Path(""), note="run_dir not found after run_suite #1")

    required = [
        "run_plan.json",
        "budget_ledger.json",
        "checkpoints/model.pt",
        "checkpoints/train_state.json",
        "metrics.json",
        "metrics_step.csv",
        "timing.csv",
    ]
    missing = [p for p in required if not (run_dir / p).exists()]
    if missing:
        return TrainSmokeResult(
            ok=False,
            run_dir=run_dir,
            note=f"missing required artifacts: {missing}",
        )

    ledger = json.loads((run_dir / "budget_ledger.json").read_text(encoding="utf-8"))
    if int(ledger.get("adapt_updates_used", -1)) != 0:
        return TrainSmokeResult(
            ok=False,
            run_dir=run_dir,
            note=f"frozen track violation: adapt_updates_used={ledger.get('adapt_updates_used')}",
        )
    if int(ledger.get("train_updates_used", 0)) <= 0:
        return TrainSmokeResult(ok=False, run_dir=run_dir, note="train_updates_used must be > 0")
    if int(ledger.get("train_updates_used", 0)) > int(ledger.get("train_max_updates", 0)):
        return TrainSmokeResult(ok=False, run_dir=run_dir, note="train_updates_used exceeded train_max_updates")

    m1 = _read_metrics(run_dir)

    rc2, out2 = _run(cmd_run, bench_root, env)
    if rc2 != 0:
        return TrainSmokeResult(ok=False, run_dir=run_dir, note=f"run_suite #2 failed:\n{out2}")
    m2 = _read_metrics(run_dir)

    tol = 1e-9
    for k in ("mse", "rmse", "mse_db"):
        if abs(float(m1[k]) - float(m2[k])) > tol:
            return TrainSmokeResult(
                ok=False,
                run_dir=run_dir,
                note=f"reproducibility check failed for {k}: run1={m1[k]} run2={m2[k]}",
            )

    return TrainSmokeResult(
        ok=True,
        run_dir=run_dir,
        note="trained,frozen smoke passed with artifact checks + reproducibility",
    )
