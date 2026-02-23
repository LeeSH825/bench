from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class GpuAllModelsSmokeResult:
    ok: bool
    skipped: bool
    run_dir: Path
    note: str


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    cp = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    out = (cp.stdout or "") + (cp.stderr or "")
    return cp.returncode, out


def _gpu_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _set_determinism(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[call-arg]
    except Exception:
        pass


def _find_run_dir(
    bench_root: Path,
    *,
    suite_name: str,
    task_id: str,
    model_id: str,
    track: str,
    init_id: str,
) -> Path:
    root = bench_root / "runs" / suite_name / task_id / model_id / track / "seed_0"
    scen_dirs = sorted(root.glob("scenario_*"))
    for scen in scen_dirs:
        cand = scen / f"init_{init_id}"
        if cand.exists():
            return cand
        if (scen / "metrics.json").exists():
            return scen
    return Path("")


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def run_gpu_all_models_device_smoke(suite_yaml: Path) -> GpuAllModelsSmokeResult:
    if not _gpu_available():
        return GpuAllModelsSmokeResult(
            ok=True,
            skipped=True,
            run_dir=Path(""),
            note="GPU not available; skipped all-model GPU smoke",
        )

    _set_determinism(seed=0)

    bench_root = _bench_root()
    suite_yaml = suite_yaml.expanduser().resolve()
    suite_name = "gpu_models_smoke"
    task_id = "A_linear_gpu_models_smoke_v0"

    shutil.rmtree(bench_root / "runs" / suite_name, ignore_errors=True)
    summary_csv = bench_root / "reports" / "summary_gpu_models_smoke.csv"
    if summary_csv.exists():
        summary_csv.unlink()

    env = os.environ.copy()
    env["BENCH_DATA_CACHE"] = str((bench_root / "bench_data_cache").resolve())
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
        return GpuAllModelsSmokeResult(
            ok=False,
            skipped=False,
            run_dir=Path(""),
            note=f"smoke_data failed:\n{out_data}",
        )

    cases: List[Dict[str, object]] = [
        {"model_id": "kalmannet_tsp", "plan": "trained:frozen", "expect_train": True},
        {"model_id": "adaptive_knet", "plan": "trained:frozen", "expect_train": True},
        {"model_id": "split_knet", "plan": "trained:frozen", "expect_train": True},
        {"model_id": "maml_knet", "plan": "trained:frozen", "expect_train": True},
    ]

    last_run_dir = Path("")
    for case in cases:
        model_id = str(case["model_id"])
        plan = str(case["plan"])
        expect_train = bool(case["expect_train"])
        init_id, track = plan.split(":", 1)

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
            plan,
            "--device",
            "cuda",
        ]
        rc_run, out_run = _run(cmd_run, bench_root, env)
        if rc_run != 0:
            return GpuAllModelsSmokeResult(
                ok=False,
                skipped=False,
                run_dir=last_run_dir,
                note=f"{model_id} ({plan}) run_suite failed:\n{out_run}",
            )

        run_dir = _find_run_dir(
            bench_root=bench_root,
            suite_name=suite_name,
            task_id=task_id,
            model_id=model_id,
            track=track,
            init_id=init_id,
        )
        if not run_dir.exists():
            return GpuAllModelsSmokeResult(
                ok=False,
                skipped=False,
                run_dir=last_run_dir,
                note=f"{model_id} ({plan}) run_dir not found",
            )
        last_run_dir = run_dir

        required = [
            "run_plan.json",
            "budget_ledger.json",
            "metrics.json",
            "metrics_step.csv",
            "timing.csv",
        ]
        if expect_train:
            required.extend(["checkpoints/model.pt", "checkpoints/train_state.json"])
        missing = [p for p in required if not (run_dir / p).exists()]
        if missing:
            return GpuAllModelsSmokeResult(
                ok=False,
                skipped=False,
                run_dir=run_dir,
                note=f"{model_id} ({plan}) missing artifacts: {missing}",
            )

        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if str(metrics.get("status", "")).lower() != "ok":
            return GpuAllModelsSmokeResult(
                ok=False,
                skipped=False,
                run_dir=run_dir,
                note=f"{model_id} ({plan}) metrics status is not ok: {metrics.get('status')}",
            )

        run_plan = metrics.get("run_plan", {}) or {}
        dev_resolved = str(run_plan.get("device_resolved", "")).lower()
        if "cuda" not in dev_resolved:
            return GpuAllModelsSmokeResult(
                ok=False,
                skipped=False,
                run_dir=run_dir,
                note=f"{model_id} ({plan}) expected run_plan.device_resolved contains cuda, got '{dev_resolved}'",
            )

        adapter_meta = metrics.get("adapter_meta", {}) or {}
        runtime_device = str(adapter_meta.get("runtime_device", "")).lower()
        if "cuda" not in runtime_device:
            return GpuAllModelsSmokeResult(
                ok=False,
                skipped=False,
                run_dir=run_dir,
                note=(
                    f"{model_id} ({plan}) expected adapter_meta.runtime_device contains cuda, "
                    f"got '{runtime_device}'"
                ),
            )

        budgets = metrics.get("budgets", {}) or {}
        adapt_used = _to_int(budgets.get("adapt_updates_used"), default=-1)
        if adapt_used != 0:
            return GpuAllModelsSmokeResult(
                ok=False,
                skipped=False,
                run_dir=run_dir,
                note=f"{model_id} ({plan}) frozen track violation: adapt_updates_used={adapt_used}",
            )

        train_used = _to_int(budgets.get("train_updates_used"), default=0)
        train_max = _to_int((run_plan.get("budgets", {}) or {}).get("train_max_updates"), default=0)
        if expect_train:
            if train_used <= 0:
                return GpuAllModelsSmokeResult(
                    ok=False,
                    skipped=False,
                    run_dir=run_dir,
                    note=f"{model_id} ({plan}) expected train_updates_used>0, got {train_used}",
                )
            if train_max > 0 and train_used > train_max:
                return GpuAllModelsSmokeResult(
                    ok=False,
                    skipped=False,
                    run_dir=run_dir,
                    note=f"{model_id} ({plan}) train_updates_used exceeded budget ({train_used}>{train_max})",
                )
        else:
            if train_used != 0:
                return GpuAllModelsSmokeResult(
                    ok=False,
                    skipped=False,
                    run_dir=run_dir,
                    note=f"{model_id} ({plan}) expected train_updates_used==0 for untrained plan, got {train_used}",
                )

    return GpuAllModelsSmokeResult(
        ok=True,
        skipped=False,
        run_dir=last_run_dir,
        note="all models passed GPU smoke (cuda device + artifacts + frozen fairness)",
    )


def test_gpu_all_models_device_smoke_pytest() -> None:
    try:
        import pytest  # type: ignore
    except Exception:  # pragma: no cover
        return

    if not _gpu_available():
        pytest.skip("GPU not available")

    suite_yaml = _bench_root() / "bench" / "configs" / "suite_gpu_models_smoke.yaml"
    res = run_gpu_all_models_device_smoke(suite_yaml=suite_yaml)
    assert res.ok and (not res.skipped), res.note
