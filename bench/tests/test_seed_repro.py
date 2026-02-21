from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class SeedReproResult:
    ok: bool
    cache_root: Path
    npz_path: Path
    mtime1: float
    mtime2: float
    note: str


def _run(cmd: list[str], cwd: Path) -> Tuple[int, str]:
    cp = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    out = (cp.stdout or "") + (cp.stderr or "")
    return cp.returncode, out


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_seed_repro(
    suite_yaml: Path,
    task_id: str,
    seed: int,
    device: str = "cpu",
    cache_root: Optional[Path] = None,
) -> SeedReproResult:
    """
    Run smoke_data twice with same seed and confirm cache hit:
      - test.npz exists after first run
      - second run does not modify mtime (or changes negligibly)
    """
    bench_root = _bench_root()
    env = os.environ.copy()
    if cache_root is not None:
        env["BENCH_DATA_CACHE"] = str(cache_root)

    # We rely on the project-provided CLI:
    # python -m bench.tasks.smoke_data --suite-yaml ... --task ... --seed ...
    cmd = [
        sys.executable,
        "-m",
        "bench.tasks.smoke_data",
        "--suite-yaml",
        str(suite_yaml),
        "--task",
        str(task_id),
        "--seed",
        str(seed),
    ]

    # Run #1
    cp1 = subprocess.run(cmd, cwd=str(bench_root), env=env, capture_output=True, text=True)
    if cp1.returncode != 0:
        return SeedReproResult(
            ok=False,
            cache_root=Path(env.get("BENCH_DATA_CACHE", str(bench_root / "bench_data_cache"))),
            npz_path=Path(""),
            mtime1=0.0,
            mtime2=0.0,
            note=f"smoke_data run#1 failed.\n{cp1.stdout}\n{cp1.stderr}",
        )

    # Locate cache root from stdout if possible; else fallback to env/default
    # Expected line contains: cache_root=/path
    cache_root_used = None
    for line in (cp1.stdout or "").splitlines():
        if "cache_root=" in line:
            cache_root_used = line.split("cache_root=", 1)[-1].strip()
            break
    if cache_root_used is None:
        cache_root_used = env.get("BENCH_DATA_CACHE") or str(bench_root / "bench_data_cache")
    cache_root_used_p = Path(cache_root_used).expanduser().resolve()

    # Find one scenario directory and its test.npz
    task_root = cache_root_used_p / "shift" / task_id
    # fallback if suite name differs, try scan
    if not task_root.exists():
        # best effort: find <cache_root>/*/<task_id>
        found = list(cache_root_used_p.glob(f"*/{task_id}"))
        if found:
            task_root = found[0]

    scen_dirs = sorted(task_root.glob("scenario_*"))
    if not scen_dirs:
        return SeedReproResult(
            ok=False,
            cache_root=cache_root_used_p,
            npz_path=Path(""),
            mtime1=0.0,
            mtime2=0.0,
            note=f"no scenario_* dirs found under {task_root}",
        )

    # pick first scenario
    scen = scen_dirs[0]
    npz_path = scen / f"seed_{seed}" / "test.npz"
    if not npz_path.exists():
        return SeedReproResult(
            ok=False,
            cache_root=cache_root_used_p,
            npz_path=npz_path,
            mtime1=0.0,
            mtime2=0.0,
            note=f"expected test.npz missing: {npz_path}",
        )

    mtime1 = npz_path.stat().st_mtime

    # tiny delay to make mtime differences visible if any
    time.sleep(1.2)

    # Run #2
    cp2 = subprocess.run(cmd, cwd=str(bench_root), env=env, capture_output=True, text=True)
    if cp2.returncode != 0:
        return SeedReproResult(
            ok=False,
            cache_root=cache_root_used_p,
            npz_path=npz_path,
            mtime1=mtime1,
            mtime2=0.0,
            note=f"smoke_data run#2 failed.\n{cp2.stdout}\n{cp2.stderr}",
        )

    mtime2 = npz_path.stat().st_mtime

    # Rule: cache hit => file should not be rewritten
    # Allow tiny differences on filesystems with coarse timestamp resolution.
    ok = abs(mtime2 - mtime1) < 0.5

    note = "cache hit (mtime unchanged)" if ok else f"cache possibly rewritten (mtime1={mtime1}, mtime2={mtime2})"
    return SeedReproResult(ok=ok, cache_root=cache_root_used_p, npz_path=npz_path, mtime1=mtime1, mtime2=mtime2, note=note)
