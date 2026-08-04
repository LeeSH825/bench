"""Repository and environment provenance capture.

Every run records the exact code state it ran against, including whether the
working tree was dirty and what revision each submodule was pinned at. The audit
listed "third-party dirty revision unrecorded" as an active risk (R-18); this
closes it for every control-plane run.

All git calls use an argv list with a timeout and never a shell string, and every
one of them degrades to ``None`` rather than raising — provenance capture must
not be able to prevent a run from starting.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from .canonical import content_hash
from .config.schema import ProvenanceSection
from .paths import repo_root


def _git(args: list[str], *, cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(  # noqa: S603 - argv list, no shell
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_commit(root: Optional[Path] = None) -> Optional[str]:
    return _git(["rev-parse", "HEAD"], cwd=root or repo_root())


def git_dirty(root: Optional[Path] = None) -> Optional[bool]:
    status = _git(["status", "--porcelain"], cwd=root or repo_root())
    if status is None:
        return None
    return bool(status.strip())


def submodule_revisions(root: Optional[Path] = None) -> dict[str, Any]:
    """Map each submodule path to its revision and dirty flag.

    ``git submodule status`` prefixes a revision with ``+`` when the checked-out
    commit differs from the index, and ``-`` when it is uninitialized. Both are
    recorded rather than normalized away.
    """
    output = _git(["submodule", "status"], cwd=root or repo_root())
    if output is None:
        return {}
    revisions: dict[str, Any] = {}
    for line in output.splitlines():
        text = line.strip()
        if not text:
            continue
        marker = text[0] if text[0] in "+-U" else " "
        body = text[1:] if marker != " " else text
        parts = body.split()
        if len(parts) < 2:
            continue
        revisions[parts[1]] = {
            "revision": parts[0],
            "modified_content": marker == "+",
            "uninitialized": marker == "-",
            "merge_conflict": marker == "U",
        }
    return revisions


def environment_document() -> dict[str, Any]:
    """Interpreter/platform/library facts that affect numerics."""
    document: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            document["torch_version"] = str(torch.__version__)
            document["torch_cuda_version"] = getattr(torch.version, "cuda", None)
            document["cuda_available"] = bool(torch.cuda.is_available())
            if document["cuda_available"]:
                document["cuda_devices"] = [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ]
        except Exception:
            pass
    return document


def environment_fingerprint() -> str:
    """Content hash of :func:`environment_document`."""
    return content_hash(environment_document())


def repository_provenance(root: Optional[Path] = None) -> ProvenanceSection:
    """Build the provenance section attached to a run spec."""
    base = root or repo_root()
    return ProvenanceSection(
        git_commit=git_commit(base),
        git_dirty=git_dirty(base),
        submodule_revisions=submodule_revisions(base),
        environment_fingerprint=environment_fingerprint(),
    )
