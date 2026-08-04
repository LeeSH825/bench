"""Atomic checkpoint publication.

The ordering here is the whole point (ADR-CSR-008). A checkpoint becomes
visible to the catalog only after its bytes are durable, so a crash can leave
"files with no row" (recoverable by the reconciler) but never "row with no
bytes", which would be a checkpoint that lies about being resumable
(DND-CSR-010).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Optional

#: Temp suffix. Files carrying it are never listed as checkpoints.
TEMP_SUFFIX = ".tmp-write"

_CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Digest a file without reading it entirely into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_path(path: Path) -> None:
    handle = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def fsync_dir(path: Path) -> None:
    """fsync a directory so a rename is durable.

    Not available on every platform; a failure here must not fail the write,
    because the rename itself has already happened.
    """
    try:
        _fsync_path(path)
    except (OSError, AttributeError):  # pragma: no cover - platform dependent
        pass


def write_bytes_durably(path: Path, data: bytes) -> None:
    """Write to a temp file in the same directory, fsync, then atomically move."""
    temp = path.with_name(path.name + TEMP_SUFFIX)
    with open(temp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    fsync_dir(path.parent)


def write_via_callback_durably(
    path: Path,
    writer: Callable[[Path], None],
    *,
    fault_hook: Optional[Callable[[str, Path], None]] = None,
) -> tuple[Path, str, int]:
    """Materialise a payload through ``writer`` and publish it atomically.

    ``writer`` receives the temporary path; this is what lets ``torch.save``
    own the serialisation while this module owns durability.

    Returns ``(final_path, sha256, size_bytes)``. The digest is computed on the
    temporary file *before* the rename, so what is catalogued is exactly what
    was made durable.

    ``fault_hook`` exists so fault-injection tests can crash the process at a
    named point without this module knowing anything about testing.
    """
    temp = path.with_name(path.name + TEMP_SUFFIX)
    if temp.exists():
        temp.unlink()

    writer(temp)

    # The callback may have used buffered IO of its own; force it down.
    with open(temp, "rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())

    _hook(fault_hook, "after_payload_write", temp)

    digest = sha256_file(temp)
    size = temp.stat().st_size

    _hook(fault_hook, "before_payload_rename", temp)
    os.replace(temp, path)
    fsync_dir(path.parent)
    _hook(fault_hook, "after_payload_rename", path)

    return path, digest, size


def _hook(hook: Optional[Callable[[str, Path], None]], name: str, path: Path) -> None:
    if hook is not None:
        hook(name, path)


def cleanup_temp_files(directory: Path, *, older_than_seconds: float = 0.0) -> list[Path]:
    """Report leftover temp files. Deletion is deliberately conservative.

    A temp file may belong to a *live* writer in another process, so this
    returns candidates rather than removing anything by default; only files
    older than ``older_than_seconds`` are removed.
    """
    import time

    removed: list[Path] = []
    if not directory.exists():
        return removed
    now = time.time()
    for candidate in directory.rglob(f"*{TEMP_SUFFIX}"):
        if not candidate.is_file():
            continue
        if older_than_seconds > 0 and (now - candidate.stat().st_mtime) < older_than_seconds:
            continue
        if older_than_seconds <= 0:
            continue
        candidate.unlink()
        removed.append(candidate)
    return removed


def list_temp_files(directory: Path) -> list[Path]:
    """Leftover temp files, for reporting. Never treated as checkpoints."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob(f"*{TEMP_SUFFIX}") if p.is_file())
