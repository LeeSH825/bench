"""Immutable run directory allocation.

Every run gets a brand-new directory:

    <control_root>/runs/<experiment_id>/<run_id>/

Because ``run_id`` is a freshly allocated UUIDv7, launching the *same*
configuration twice necessarily produces two directories. This is the direct
countermeasure to risk R-01 / DND-004: the existing runner derives its path
deterministically from the config (`run_suite.py` `output_dir_template`) and
therefore overwrites a previous run of the same config. Nothing here can do
that.

Allocation is atomic: the leaf directory is created with ``mkdir(exist_ok=False)``,
so if the same id were ever proposed twice, the second attempt raises instead of
quietly joining an existing run's directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config.schema import ResolvedRunSpec
from .identity import ExperimentId, RunId
from .paths import ensure_dir, runs_root

#: Subdirectories created inside every run directory (design doc 03 §15).
RUN_SUBDIRECTORIES = ("checkpoints", "artifacts", "provenance", "tmp")

#: Canonical filenames inside a run directory.
RESOLVED_SPEC_FILENAME = "resolved_run_spec.json"
ORIGINAL_CONFIG_FILENAME = "original_config.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"
EVENTS_FILENAME = "events.jsonl"
STDOUT_FILENAME = "stdout.log"
STDERR_FILENAME = "stderr.log"
FAILURE_FILENAME = "failure.json"


class AllocationError(RuntimeError):
    """Raised when a run directory cannot be allocated."""


@dataclass(frozen=True)
class RunLocation:
    """Filesystem layout of one allocated run."""

    run_id: RunId
    experiment_id: ExperimentId
    root: Path

    @property
    def resolved_spec_path(self) -> Path:
        return self.root / RESOLVED_SPEC_FILENAME

    @property
    def original_config_path(self) -> Path:
        return self.root / ORIGINAL_CONFIG_FILENAME

    @property
    def manifest_path(self) -> Path:
        return self.root / RUN_MANIFEST_FILENAME

    @property
    def events_path(self) -> Path:
        return self.root / EVENTS_FILENAME

    @property
    def stdout_path(self) -> Path:
        return self.root / STDOUT_FILENAME

    @property
    def stderr_path(self) -> Path:
        return self.root / STDERR_FILENAME

    @property
    def failure_path(self) -> Path:
        return self.root / FAILURE_FILENAME

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def provenance_dir(self) -> Path:
        return self.root / "provenance"

    @property
    def tmp_dir(self) -> Path:
        """Scratch space. Partial artifacts must never appear outside here."""
        return self.root / "tmp"

    def relative_to_runs_root(self, control_root: Optional[str | os.PathLike[str]] = None) -> str:
        return str(self.root.relative_to(runs_root(control_root)))


def allocate_run_directory(
    *,
    run_id: RunId,
    experiment_id: ExperimentId,
    control_root: Optional[str | os.PathLike[str]] = None,
) -> RunLocation:
    """Create and return a fresh run directory.

    Raises :class:`AllocationError` if the directory already exists — that would
    mean a run id collision, which must be loud, never merged into.
    """
    base = runs_root(control_root) / experiment_id.value
    ensure_dir(base)
    leaf = base / run_id.value
    try:
        leaf.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise AllocationError(
            f"run directory {leaf} already exists; refusing to reuse an allocated run id"
        ) from exc
    for name in RUN_SUBDIRECTORIES:
        (leaf / name).mkdir(exist_ok=True)
    return RunLocation(run_id=run_id, experiment_id=experiment_id, root=leaf)


def write_run_spec(location: RunLocation, spec: ResolvedRunSpec) -> Path:
    """Persist the resolved spec into the run directory.

    Written through ``tmp/`` and atomically renamed, so a reader can never
    observe a half-written spec.
    """
    return atomic_write_text(location.resolved_spec_path, spec.to_json(), tmp_dir=location.tmp_dir)


def atomic_write_text(path: Path, text: str, *, tmp_dir: Optional[Path] = None) -> Path:
    """Write *text* to *path* atomically (temp file → fsync → rename).

    The rename is atomic on POSIX when source and destination share a
    filesystem, which is why the temp file defaults to the destination's own
    directory rather than the system temp dir.
    """
    directory = tmp_dir or path.parent
    ensure_dir(directory)
    ensure_dir(path.parent)
    temp_path = directory / f".{path.name}.{os.getpid()}.tmp"
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)
    # fsync the containing directory so the rename itself is durable.
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path
