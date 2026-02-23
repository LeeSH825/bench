from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class CacheHitResult:
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
    root = bench_root / "runs" / "cache_smoke" / "A_linear_cache_smoke_v0" / "kalmannet_tsp" / "frozen" / "seed_0"
    cands = sorted(root.glob("scenario_*"))
    if not cands:
        return Path("")
    return cands[0]


def run_cache_hit_skips_train(suite_yaml: Path) -> CacheHitResult:
    bench_root = _bench_root()
    suite_yaml = suite_yaml.expanduser().resolve()
    task_id = "A_linear_cache_smoke_v0"

    # Ensure deterministic miss->hit sequence.
    shutil.rmtree(bench_root / "runs" / "cache_smoke", ignore_errors=True)
    shutil.rmtree(bench_root / "runs" / "_model_cache" / "cache_smoke", ignore_errors=True)
    summary_csv = bench_root / "reports" / "summary_cache_smoke.csv"
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
        return CacheHitResult(ok=False, run_dir=Path(""), note=f"smoke_data failed:\n{out_data}")

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
        return CacheHitResult(ok=False, run_dir=Path(""), note=f"run_suite #1 failed:\n{out1}")

    run_dir = _find_run_dir(bench_root)
    if not run_dir.exists():
        return CacheHitResult(ok=False, run_dir=Path(""), note="run_dir not found after run_suite #1")

    required = [
        "run_plan.json",
        "budget_ledger.json",
        "checkpoints/model.pt",
        "checkpoints/train_state.json",
        "metrics.json",
    ]
    missing = [p for p in required if not (run_dir / p).exists()]
    if missing:
        return CacheHitResult(ok=False, run_dir=run_dir, note=f"first run missing artifacts: {missing}")

    ledger1 = json.loads((run_dir / "budget_ledger.json").read_text(encoding="utf-8"))
    if bool(ledger1.get("train_skipped", False)):
        return CacheHitResult(ok=False, run_dir=run_dir, note="first run must be cache miss (train_skipped=false)")
    if bool(ledger1.get("cache_hit", False)):
        return CacheHitResult(ok=False, run_dir=run_dir, note="first run unexpectedly reported cache_hit=true")
    if int(ledger1.get("train_updates_used", 0)) <= 0:
        return CacheHitResult(ok=False, run_dir=run_dir, note="first run expected train_updates_used > 0")

    rc2, out2 = _run(cmd_run, bench_root, env)
    if rc2 != 0:
        return CacheHitResult(ok=False, run_dir=run_dir, note=f"run_suite #2 failed:\n{out2}")

    if not (run_dir / "checkpoints" / "model.pt").exists():
        return CacheHitResult(ok=False, run_dir=run_dir, note="second run missing checkpoints/model.pt")

    ledger2 = json.loads((run_dir / "budget_ledger.json").read_text(encoding="utf-8"))
    if not bool(ledger2.get("train_skipped", False)):
        return CacheHitResult(ok=False, run_dir=run_dir, note="second run expected train_skipped=true")
    if not bool(ledger2.get("cache_hit", False)):
        return CacheHitResult(ok=False, run_dir=run_dir, note="second run expected cache_hit=true")
    if int(ledger2.get("train_updates_used", -1)) != 0:
        return CacheHitResult(
            ok=False,
            run_dir=run_dir,
            note=f"second run expected train_updates_used=0, got {ledger2.get('train_updates_used')}",
        )

    return CacheHitResult(
        ok=True,
        run_dir=run_dir,
        note="cache miss->hit verified; second run skipped training with auditable ledger flags",
    )
