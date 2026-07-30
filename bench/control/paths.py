"""Control-plane root resolution and path safety.

All new control-plane state is written under a single root that is *separate*
from the existing `runs/` tree, so this tranche cannot touch, overwrite, or
race any historical artifact:

    <control_root>/
      registry.sqlite3          run registry (WAL)
      runs/<experiment_id>/<run_id>/     immutable run directories

The default root is ``<repo>/control``. It is overridable via the
``BENCH_CONTROL_ROOT`` environment variable, which is what tests use to get a
fully isolated ``tmp_path`` root.

:func:`safe_relative_path` implements the path allowlist from design doc 03 §17:
absolute paths, ``..`` traversal, and symlink escapes are rejected rather than
normalized, because silently normalizing an escape attempt hides the bug.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

#: Environment variable that overrides the control root.
CONTROL_ROOT_ENV = "BENCH_CONTROL_ROOT"

#: Registry filename inside the control root.
REGISTRY_FILENAME = "registry.sqlite3"


class UnsafePathError(ValueError):
    """Raised when a path would escape its allowed root."""


def repo_root() -> Path:
    """Absolute path of the repository root (the parent of the `bench` package)."""
    return Path(__file__).resolve().parents[2]


def control_root(explicit: Optional[str | os.PathLike[str]] = None) -> Path:
    """Resolve the control-plane root directory.

    Precedence: explicit argument, then ``$BENCH_CONTROL_ROOT``, then
    ``<repo>/control``. The directory is *not* created here — creation is the
    caller's decision so that read-only consumers (the API, the dashboard) do
    not materialize state as a side effect of pointing at a missing root.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(CONTROL_ROOT_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "control"


def registry_path(root: Optional[str | os.PathLike[str]] = None) -> Path:
    """Path of the SQLite registry inside *root*."""
    return control_root(root) / REGISTRY_FILENAME


def runs_root(root: Optional[str | os.PathLike[str]] = None) -> Path:
    """Root of the immutable control-plane run tree."""
    return control_root(root) / "runs"


def legacy_runs_root() -> Path:
    """Root of the pre-existing deterministic run tree.

    Read-only for the control plane: the legacy importer reads from here and
    never writes.
    """
    return repo_root() / "runs"


def safe_relative_path(base: Path, candidate: str | os.PathLike[str]) -> Path:
    """Resolve *candidate* underneath *base*, rejecting escapes.

    Rejects absolute paths and any result that does not stay inside *base*
    after full symlink resolution.
    """
    text = str(candidate)
    if not text or text in (".", "/"):
        raise UnsafePathError(f"empty or root-only relative path: {candidate!r}")
    raw = Path(text)
    if raw.is_absolute():
        raise UnsafePathError(f"absolute path not allowed: {candidate!r}")
    if any(part == ".." for part in raw.parts):
        raise UnsafePathError(f"parent traversal not allowed: {candidate!r}")

    base_resolved = base.resolve()
    target = (base_resolved / raw).resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise UnsafePathError(
            f"path {candidate!r} escapes its allowed root {base_resolved}"
        )
    return target


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if absent and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
