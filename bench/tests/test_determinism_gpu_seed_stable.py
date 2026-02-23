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
except Exception:  # pragma: no cover - best effort in environments without torch
    torch = None  # type: ignore


GPU_MSE_DB_ABS_TOL = 1e-5
GPU_MSE_DB_REL_TOL = 1e-5
GPU_RECOVERY_K_ABS_TOL = 1.0


@dataclass
class DeterminismGpuResult:
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


def _find_run_dir(bench_root: Path, track: str, init_id: str) -> Path:
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
        # single-plan runs may not have init_* subfolder
        cand = scen / f"init_{init_id}"
        if cand.exists():
            return cand
        if (scen / "metrics.json").exists():
            return scen
    return Path("")


def _read_stability_metrics(run_dir: Path) -> Tuple[Dict[str, float], Dict[str, object]]:
    obj = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if str(obj.get("status", "")).lower() != "ok":
        raise RuntimeError(f"metrics status not ok at {run_dir}")
    acc = obj.get("accuracy", {}) or {}
    rec = obj.get("shift_recovery", {}) or {}
    out = {
        "mse_db": float(acc.get("mse_db", 0.0)),
    }
    rk = rec.get("recovery_k")
    if rk is not None:
        out["recovery_k"] = float(rk)
    return out, obj


def _mse_db_within_tol(a: float, b: float) -> Tuple[bool, float, float]:
    abs_delta = abs(float(a) - float(b))
    denom = max(abs(float(a)), abs(float(b)), 1e-12)
    rel_delta = abs_delta / denom
    ok = (abs_delta <= GPU_MSE_DB_ABS_TOL) or (rel_delta <= GPU_MSE_DB_REL_TOL)
    return ok, abs_delta, rel_delta


def run_determinism_gpu_seed_stable(suite_yaml: Path) -> DeterminismGpuResult:
    if not _gpu_available():
        return DeterminismGpuResult(
            ok=True,
            skipped=True,
            run_dir=Path(""),
            note="GPU not available; skipped GPU determinism smoke",
        )

    _set_determinism(seed=0)

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
        return DeterminismGpuResult(ok=False, skipped=False, run_dir=Path(""), note=f"smoke_data failed:\n{out_data}")

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
        "--device",
        "cuda",
    ]

    rc1, out1 = _run(cmd_run, bench_root, env)
    if rc1 != 0:
        return DeterminismGpuResult(ok=False, skipped=False, run_dir=Path(""), note=f"run_suite #1 failed:\n{out1}")

    run_dir = _find_run_dir(bench_root, track="frozen", init_id="trained")
    if not run_dir.exists():
        return DeterminismGpuResult(ok=False, skipped=False, run_dir=Path(""), note="GPU run_dir not found")

    required = [
        "run_plan.json",
        "budget_ledger.json",
        "checkpoints/model.pt",
        "metrics.json",
        "metrics_step.csv",
        "timing.csv",
    ]
    missing = [p for p in required if not (run_dir / p).exists()]
    if missing:
        return DeterminismGpuResult(ok=False, skipped=False, run_dir=run_dir, note=f"run #1 missing artifacts: {missing}")

    m1, obj1 = _read_stability_metrics(run_dir)
    dev1 = str((((obj1.get("run_plan", {}) or {}).get("device_resolved")) or "")).lower()
    if "cuda" not in dev1:
        return DeterminismGpuResult(
            ok=False,
            skipped=False,
            run_dir=run_dir,
            note=f"expected device_resolved contains cuda, got '{dev1}'",
        )

    rc2, out2 = _run(cmd_run, bench_root, env)
    if rc2 != 0:
        return DeterminismGpuResult(ok=False, skipped=False, run_dir=run_dir, note=f"run_suite #2 failed:\n{out2}")

    m2, _obj2 = _read_stability_metrics(run_dir)

    mse_ok, abs_delta, rel_delta = _mse_db_within_tol(m1["mse_db"], m2["mse_db"])
    if not mse_ok:
        return DeterminismGpuResult(
            ok=False,
            skipped=False,
            run_dir=run_dir,
            note=(
                "GPU mse_db drift exceeded D19 tolerance: "
                f"run1={m1['mse_db']} run2={m2['mse_db']} "
                f"abs_delta={abs_delta} rel_delta={rel_delta} "
                f"(abs_tol={GPU_MSE_DB_ABS_TOL}, rel_tol={GPU_MSE_DB_REL_TOL})"
            ),
        )

    rk1: Optional[float] = m1.get("recovery_k")
    rk2: Optional[float] = m2.get("recovery_k")
    if rk1 is not None and rk2 is not None:
        rk_abs = abs(float(rk1) - float(rk2))
        if rk_abs > GPU_RECOVERY_K_ABS_TOL:
            return DeterminismGpuResult(
                ok=False,
                skipped=False,
                run_dir=run_dir,
                note=(
                    "GPU recovery_k drift exceeded D19 tolerance: "
                    f"run1={rk1} run2={rk2} abs_delta={rk_abs} "
                    f"(abs_tol={GPU_RECOVERY_K_ABS_TOL})"
                ),
            )

    return DeterminismGpuResult(
        ok=True,
        skipped=False,
        run_dir=run_dir,
        note=(
            "GPU determinism stable for trained,frozen "
            f"(mse_db abs_delta={abs_delta:.3e}, rel_delta={rel_delta:.3e})"
        ),
    )


def test_determinism_gpu_seed_stable_pytest() -> None:
    """
    Optional pytest entrypoint:
    - no GPU: pytest.skip
    - GPU: enforce D19 tolerances
    """
    try:
        import pytest  # type: ignore
    except Exception:  # pragma: no cover
        return

    if not _gpu_available():
        pytest.skip("GPU not available")

    suite_yaml = _bench_root() / "bench" / "configs" / "suite_plan_matrix_smoke.yaml"
    res = run_determinism_gpu_seed_stable(suite_yaml=suite_yaml)
    assert res.ok and (not res.skipped), res.note
