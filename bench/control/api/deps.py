"""Shared service objects for the API layer.

The registry connection and worker manager are process-wide singletons created
lazily on first use. FastAPI's dependency system injects them, which is what
keeps the routers free of construction logic and makes them trivially testable
with an overridden control root.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..paths import control_root, registry_path
from ..process.manager import WorkerManager
from ..registry.sqlite import SqliteRegistry

#: Reentrant on purpose: `get_manager()` needs a registry, so it calls
#: `get_registry()` while already holding this lock. With a plain Lock that is a
#: hard deadlock — and it only triggers when *both* singletons are still unset,
#: i.e. on the very first request after start-up if that request happens to be
#: one that depends on the manager before the registry.
_LOCK = threading.RLock()
_REGISTRY: Optional[SqliteRegistry] = None
_MANAGER: Optional[WorkerManager] = None
_ROOT: Optional[Path] = None


def configure(root: Optional[str | os.PathLike[str]] = None) -> None:
    """Point the service at a control root, discarding any cached state.

    Tests call this with a ``tmp_path`` so nothing touches the real registry.
    """
    global _REGISTRY, _MANAGER, _ROOT
    with _LOCK:
        if _REGISTRY is not None:
            try:
                _REGISTRY.close()
            except Exception:
                pass
        _REGISTRY = None
        _MANAGER = None
        _ROOT = control_root(root)


def active_root() -> Path:
    return _ROOT if _ROOT is not None else control_root()


def get_registry() -> SqliteRegistry:
    """Return the process-wide registry, creating it on first use."""
    global _REGISTRY
    if _REGISTRY is None:
        with _LOCK:
            if _REGISTRY is None:
                _REGISTRY = SqliteRegistry(registry_path(active_root()))
    return _REGISTRY


def get_manager() -> WorkerManager:
    """Return the process-wide worker manager.

    The read-only API uses it only for liveness/orphan *reporting*. It exposes
    no endpoint that launches, stops, or kills anything in this tranche.
    """
    global _MANAGER
    if _MANAGER is None:
        with _LOCK:
            if _MANAGER is None:
                _MANAGER = WorkerManager(get_registry(), control_root_path=active_root())
    return _MANAGER


@lru_cache(maxsize=1)
def service_start_time() -> float:
    import time

    return time.time()
