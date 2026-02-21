from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class RunnerSmokeResult:
    ok: bool
    run_dir: Path
    note: str


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_run_dir(output: str) -> Optional[Path]:
    # smoke_run prints:
    # [smoke_run] result:
    #   run_dir: /path/...
    for line in output.splitlines():
        if "run_dir:" in line:
            p = line.split("run_dir:", 1)[-1].strip()
            if p:
                return Path(p).expanduser().resolve()
    return None


def run_runner_smoke(
    suite_yaml: Path,
    task_id: str,
    model_id: str,
    seed: int,
    device: str = "cpu",
    track: str = "frozen",
) -> RunnerSmokeResult:
    bench_root = _bench_root()

    cmd = [
        sys.executable,
        "-m",
        "bench.runners.smoke_run",
        "--suite-yaml",
        str(suite_yaml),
        "--task-id",
        str(task_id),
        "--model-id",
        str(model_id),
        "--seed",
        str(seed),
        "--track",
        str(track),
        "--device",
        str(device),
    ]

    cp = subprocess.run(cmd, cwd=str(bench_root), capture_output=True, text=True)
    out = (cp.stdout or "") + (cp.stderr or "")
    run_dir = _parse_run_dir(out) or Path("")

    if cp.returncode != 0:
        return RunnerSmokeResult(ok=False, run_dir=run_dir, note=f"smoke_run failed.\n{out}")

    # require artifacts
    required = ["config_snapshot.yaml", "metrics.json", "metrics_step.csv", "timing.csv"]
    missing = []
    for f in required:
        if not (run_dir / f).exists():
            missing.append(f)

    if missing:
        return RunnerSmokeResult(
            ok=False,
            run_dir=run_dir,
            note=f"smoke_run returned 0 but missing artifacts: {missing}\nOutput:\n{out}",
        )

    # require metrics status ok
    try:
        import json

        m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if str(m.get("status", "")).lower() != "ok":
            return RunnerSmokeResult(ok=False, run_dir=run_dir, note=f"metrics.json status is not ok: {m.get('status')}")
    except Exception as e:
        return RunnerSmokeResult(ok=False, run_dir=run_dir, note=f"failed to read/parse metrics.json: {type(e).__name__}: {e}")

    return RunnerSmokeResult(ok=True, run_dir=run_dir, note="runner smoke OK (run_dir + required artifacts present)")
