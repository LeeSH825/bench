from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: str | None = None) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def get_pip_freeze_text() -> str:
    """
    Return `pip freeze` output (best effort).
    Uses current interpreter: `python -m pip freeze`.
    """
    return _run([sys.executable, "-m", "pip", "freeze"])


def collect_env_info(
    bench_root: str | None = None,
    third_party_dir: str | None = None,
) -> dict[str, Any]:
    """
    Collect environment metadata for reproducibility.

    Intended to be written per-run into run_dir as:
    - env.json
    - pip_freeze.txt

    Includes (best effort):
    - python version, platform
    - torch version, cuda availability
    - bench git commit + dirty status (if git repo)
    - third_party git submodule status (if present)
    """
    info: dict[str, Any] = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    # torch optional
    try:
        import torch  # type: ignore

        info.update(
            {
                "torch_version": getattr(torch, "__version__", ""),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": getattr(torch.version, "cuda", ""),
            }
        )
    except Exception:
        info.update({"torch_version": "", "cuda_available": False, "cuda_version": ""})

    # git info (bench)
    if bench_root is None:
        # heuristic: repo root is two levels above this file: bench/bench/utils/env_detect.py
        bench_root = str(Path(__file__).resolve().parents[2])

    info["bench_root"] = bench_root
    info["git"] = {
        "bench_commit": _run(["git", "rev-parse", "HEAD"], cwd=bench_root),
        "bench_status": _run(["git", "status", "--porcelain"], cwd=bench_root),
        "bench_remote": _run(["git", "remote", "-v"], cwd=bench_root),
    }

    # third_party info
    if third_party_dir is None:
        # default convention: /mnt/data/third_party or /mnt/data/bench/third_party may be used
        # We just record whatever exists.
        candidates = [
            str(Path(bench_root).parent / "third_party"),
            str(Path(bench_root) / "third_party"),
        ]
        third_party_dir = next((p for p in candidates if Path(p).exists()), candidates[0])

    info["third_party_dir"] = third_party_dir

    # best-effort: git submodule status from bench root
    sub_status = _run(["git", "submodule", "status", "--recursive"], cwd=bench_root)
    info["git"]["submodules_status"] = sub_status

    # best-effort: per-repo HEAD if each is a git repo
    tp = Path(third_party_dir)
    repos: dict[str, Any] = {}
    if tp.exists():
        for child in tp.iterdir():
            if not child.is_dir():
                continue
            head = _run(["git", "rev-parse", "HEAD"], cwd=str(child))
            dirty = _run(["git", "status", "--porcelain"], cwd=str(child))
            if head:
                repos[child.name] = {"commit": head, "status": dirty}
    info["third_party_git"] = repos

    return info


def to_pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

