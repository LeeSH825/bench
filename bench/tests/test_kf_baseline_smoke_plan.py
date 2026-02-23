from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class KFBaselineSmokeResult:
    ok: bool
    run_dir: Path
    note: str


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    cp = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    out = (cp.stdout or "") + (cp.stderr or "")
    return cp.returncode, out


def _find_run_dir(
    bench_root: Path,
    *,
    task_id: str,
    model_id: str,
    track: str,
) -> Path:
    root = bench_root / "runs" / "kf_baseline_smoke" / task_id / model_id / track / "seed_0"
    cands = sorted(root.glob("scenario_*"))
    if not cands:
        return Path("")
    return cands[0]


def run_kf_baseline_smoke_plan(suite_yaml: Path) -> KFBaselineSmokeResult:
    bench_root = _bench_root()
    suite_yaml = suite_yaml.expanduser().resolve()
    suite_name = "kf_baseline_smoke"

    tasks = [
        "A_linear_kf_baseline_smoke_v0",
        "C_shift_kf_baseline_smoke_v0",
    ]
    models = [
        "oracle_kf",
        "nominal_kf",
        "oracle_shift_kf",
    ]
    expected_mode = {
        "oracle_kf": "oracle",
        "nominal_kf": "nominal",
        "oracle_shift_kf": "oracle_shift",
    }

    shutil.rmtree(bench_root / "runs" / suite_name, ignore_errors=True)
    summary_csv = bench_root / "reports" / "summary_kf_baseline_smoke.csv"
    if summary_csv.exists():
        summary_csv.unlink()

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
            return KFBaselineSmokeResult(ok=False, run_dir=Path(""), note=f"smoke_data failed for {task_id}:\n{out_data}")

    cmd_run = [
        sys.executable,
        "-m",
        "bench.runners.run_suite",
        "--suite-yaml",
        str(suite_yaml),
        "--tasks",
        *tasks,
        "--models",
        *models,
        "--seeds",
        "0",
        "--plans",
        "pretrained:frozen",
        "--device",
        "cpu",
    ]
    rc_run, out_run = _run(cmd_run, bench_root, env)
    if rc_run != 0:
        return KFBaselineSmokeResult(ok=False, run_dir=Path(""), note=f"run_suite failed:\n{out_run}")

    last_run_dir = Path("")
    for task_id in tasks:
        for model_id in models:
            run_dir = _find_run_dir(
                bench_root=bench_root,
                task_id=task_id,
                model_id=model_id,
                track="frozen",
            )
            if not run_dir.exists():
                return KFBaselineSmokeResult(
                    ok=False,
                    run_dir=last_run_dir,
                    note=f"run_dir not found for task={task_id} model={model_id}",
                )
            last_run_dir = run_dir

            if (run_dir / "failure.json").exists():
                failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
                return KFBaselineSmokeResult(
                    ok=False,
                    run_dir=run_dir,
                    note=f"unexpected failure.json for task={task_id} model={model_id}: {failure}",
                )

            required = [
                "run_plan.json",
                "budget_ledger.json",
                "metrics.json",
                "metrics_step.csv",
                "timing.csv",
                "artifacts/preds_test.npz",
            ]
            missing = [p for p in required if not (run_dir / p).exists()]
            if missing:
                return KFBaselineSmokeResult(
                    ok=False,
                    run_dir=run_dir,
                    note=f"missing artifacts for task={task_id} model={model_id}: {missing}",
                )

            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            if str(metrics.get("status", "")).lower() != "ok":
                return KFBaselineSmokeResult(
                    ok=False,
                    run_dir=run_dir,
                    note=f"metrics status not ok for task={task_id} model={model_id}",
                )

            run_plan = metrics.get("run_plan", {}) or {}
            if str(run_plan.get("init_id", "")) != "pretrained":
                return KFBaselineSmokeResult(
                    ok=False,
                    run_dir=run_dir,
                    note=f"init_id mismatch for task={task_id} model={model_id}: {run_plan.get('init_id')}",
                )
            if str(run_plan.get("track_id", "")) != "frozen":
                return KFBaselineSmokeResult(
                    ok=False,
                    run_dir=run_dir,
                    note=f"track_id mismatch for task={task_id} model={model_id}: {run_plan.get('track_id')}",
                )

            ledger = metrics.get("budgets", {}) or {}
            if int(ledger.get("train_updates_used", -1)) != 0:
                return KFBaselineSmokeResult(
                    ok=False,
                    run_dir=run_dir,
                    note=f"train_updates_used must be 0 for task={task_id} model={model_id}",
                )
            if int(ledger.get("adapt_updates_used", -1)) != 0:
                return KFBaselineSmokeResult(
                    ok=False,
                    run_dir=run_dir,
                    note=f"adapt_updates_used must be 0 for task={task_id} model={model_id}",
                )

            adapter_meta = metrics.get("adapter_meta", {}) or {}
            if str(adapter_meta.get("baseline", "")) != "mb_kf":
                return KFBaselineSmokeResult(
                    ok=False,
                    run_dir=run_dir,
                    note=f"adapter_meta.baseline mismatch for task={task_id} model={model_id}",
                )
            if str(adapter_meta.get("mode", "")) != str(expected_mode[model_id]):
                return KFBaselineSmokeResult(
                    ok=False,
                    run_dir=run_dir,
                    note=f"adapter_meta.mode mismatch for task={task_id} model={model_id}",
                )

            preds = np.load(run_dir / "artifacts" / "preds_test.npz")
            if "x_hat" not in preds.files:
                return KFBaselineSmokeResult(
                    ok=False,
                    run_dir=run_dir,
                    note=f"preds_test.npz missing x_hat for task={task_id} model={model_id}",
                )
            x_hat = preds["x_hat"]
            if x_hat.ndim != 3:
                return KFBaselineSmokeResult(
                    ok=False,
                    run_dir=run_dir,
                    note=f"x_hat rank mismatch for task={task_id} model={model_id}: shape={x_hat.shape}",
                )
            dims = metrics.get("dims", {}) or {}
            x_dim = int(dims.get("x_dim", -1))
            T = int(dims.get("T", -1))
            if x_hat.shape[1] != T or x_hat.shape[2] != x_dim:
                return KFBaselineSmokeResult(
                    ok=False,
                    run_dir=run_dir,
                    note=(
                        f"x_hat shape mismatch for task={task_id} model={model_id}: "
                        f"x_hat={x_hat.shape} expected [N,{T},{x_dim}]"
                    ),
                )

    return KFBaselineSmokeResult(
        ok=True,
        run_dir=last_run_dir,
        note="KF baselines pretrained,frozen smoke passed (artifacts + ledger + shape checks)",
    )

