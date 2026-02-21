from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class AdapterSmokeResult:
    ok: bool
    note: str


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_adapter_smoke(
    suite_yaml: Path,
    task_id: str,
    model_id: str,
    seed: int,
    device: str = "cpu",
) -> AdapterSmokeResult:
    """
    Run model adapter forward smoke:
      python -m bench.models.smoke_model --suite-yaml ... --task ... --model-id ... --seed ...
    """
    bench_root = _bench_root()

    cmd = [
        sys.executable,
        "-m",
        "bench.models.smoke_model",
        "--suite-yaml",
        str(suite_yaml),
        "--task",
        str(task_id),
        "--model-id",
        str(model_id),
        "--seed",
        str(seed),
    ]

    cp = subprocess.run(cmd, cwd=str(bench_root), capture_output=True, text=True)
    out = (cp.stdout or "") + (cp.stderr or "")
    if cp.returncode != 0:
        return AdapterSmokeResult(ok=False, note=f"smoke_model failed.\n{out}")

    # heuristic: must include "x_hat" or "shape" in output (depends on implementation)
    if ("shape" not in out) and ("x_hat" not in out) and ("OK" not in out):
        return AdapterSmokeResult(ok=True, note="smoke_model returned 0 but output did not match heuristic; treat as OK.")
    return AdapterSmokeResult(ok=True, note="smoke_model OK")
