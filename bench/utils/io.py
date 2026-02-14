from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """
    Always accept str/Path and return Path.
    This avoids: "'str' object has no attribute 'mkdir'".
    """
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_text_best_effort(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def _write_text(p: Path, s: str) -> None:
    ensure_dir(p.parent)
    p.write_text(s, encoding="utf-8")


def _write_json(p: Path, obj: Any) -> None:
    _write_text(p, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def get_pip_freeze_text() -> str:
    """
    Best-effort pip freeze.
    Never raises; returns a short diagnostic on failure.
    """
    try:
        cp = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        out = (cp.stdout or "").strip()
        err = (cp.stderr or "").strip()
        if out:
            return out + "\n"
        if err:
            return f"# pip freeze produced no stdout\n# stderr:\n{err}\n"
        return "# pip freeze produced no output\n"
    except Exception as e:
        return f"# pip freeze failed: {type(e).__name__}: {e}\n"


def _git(cmd: list[str], cwd: Path) -> Optional[str]:
    try:
        cp = subprocess.run(cmd, cwd=str(cwd), check=False, capture_output=True, text=True)
        out = (cp.stdout or "").strip()
        if cp.returncode == 0 and out:
            return out
        return None
    except Exception:
        return None


def write_env_snapshot(run_dir: PathLike) -> None:
    """
    Writes:
      - env.json
      - pip_freeze.txt
      - (if exists) requirements.lock copy
    Accepts PathLike safely.
    """
    rd = ensure_dir(run_dir)

    env: Dict[str, Any] = {
        "python": sys.version.replace(os.linesep, " "),
        "executable": sys.executable,
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "BENCH_DATA_CACHE": os.environ.get("BENCH_DATA_CACHE"),
        },
    }

    # torch info (best-effort)
    try:
        import torch  # type: ignore

        env["torch"] = {
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": getattr(torch.backends.cudnn, "version", lambda: None)(),
        }
    except Exception as e:
        env["torch"] = {"error": f"{type(e).__name__}: {e}"}

    _write_json(rd / "env.json", env)
    _write_text(rd / "pip_freeze.txt", get_pip_freeze_text())

    # copy requirements.lock if exists (bench root 기준)
    try:
        bench_root = Path(__file__).resolve().parents[2]  # .../bench/bench/utils/io.py -> .../bench
        lock = bench_root / "requirements.lock"
        if lock.exists():
            shutil.copyfile(str(lock), str(rd / "requirements.lock"))
    except Exception:
        pass


def write_git_snapshot(run_dir: PathLike) -> None:
    """
    Writes git_versions.txt with:
      - bench repo commit
      - submodule status (if any)
    Accepts PathLike safely.
    """
    rd = ensure_dir(run_dir)
    bench_root = Path(__file__).resolve().parents[2]

    lines: list[str] = []
    head = _git(["git", "rev-parse", "HEAD"], bench_root)
    if head:
        lines.append(f"bench: {head}")
    else:
        lines.append("bench: (no git or not a repo)")

    sub = _git(["git", "submodule", "status", "--recursive"], bench_root)
    if sub:
        lines.append("\n[submodules]\n" + sub)

    _write_text(rd / "git_versions.txt", "\n".join(lines) + "\n")

