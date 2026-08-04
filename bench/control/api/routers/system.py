"""System health, GPU inventory, and capability endpoints."""

from __future__ import annotations

import os
import platform
import socket
import time
from typing import Any

from fastapi import APIRouter, Depends

from ... import CONTROL_PLANE_VERSION
from ...capabilities import all_capabilities
from ...config.schema import CONFIG_SCHEMA_VERSION
from ...events.schema import EVENT_SCHEMA_VERSION
from ...process.manager import WorkerManager
from ...registry.schema import (
    ACTIVE_STATES_THIS_TRANCHE,
    ALLOWED_TRANSITIONS,
    REGISTRY_SCHEMA_VERSION,
    RunState,
)
from ...registry.sqlite import SqliteRegistry
from ...telemetry import gpu_inventory, psutil_available
from ..deps import active_root, get_manager, get_registry, service_start_time

router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("/health")
def health(
    registry: SqliteRegistry = Depends(get_registry),
    manager: WorkerManager = Depends(get_manager),
) -> dict[str, Any]:
    """Per-subsystem health.

    Each component reports its own status so a degraded telemetry collector is
    distinguishable from an unreachable registry (acceptance A-06). The overall
    status is the worst component status, never an optimistic average.
    """
    components: dict[str, dict[str, Any]] = {}

    try:
        version = registry.schema_version
        total = registry.count_runs()
        components["registry"] = {
            "status": "ok" if version == REGISTRY_SCHEMA_VERSION else "degraded",
            "schema_version": version,
            "expected_schema_version": REGISTRY_SCHEMA_VERSION,
            "path": str(registry.path),
            "runs": total,
        }
    except Exception as exc:
        components["registry"] = {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}

    try:
        candidates = manager.find_orphan_candidates()
        components["worker_manager"] = {
            "status": "ok" if not candidates else "degraded",
            "orphan_candidates": len(candidates),
            "detail": (
                None
                if not candidates
                else f"{len(candidates)} run(s) have a missing or stale worker"
            ),
        }
    except Exception as exc:
        components["worker_manager"] = {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}

    gpus = []
    try:
        gpus = gpu_inventory()
        components["telemetry"] = {
            "status": "ok" if psutil_available() else "degraded",
            "psutil_available": psutil_available(),
            "gpu_count": len(gpus),
            "detail": (
                None
                if psutil_available()
                else "psutil is not installed; CPU/RAM telemetry is unavailable"
            ),
        }
    except Exception as exc:
        components["telemetry"] = {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}

    ranking = {"ok": 0, "degraded": 1, "error": 2}
    overall = max(
        (component.get("status", "error") for component in components.values()),
        key=lambda status: ranking.get(status, 2),
        default="ok",
    )
    return {
        "status": overall,
        "components": components,
        "control_plane_version": CONTROL_PLANE_VERSION,
        "control_root": str(active_root()),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "uptime_seconds": time.time() - service_start_time(),
        "python": platform.python_version(),
        "read_only": True,
    }


@router.get("/gpus")
def gpus(registry: SqliteRegistry = Depends(get_registry)) -> dict[str, Any]:
    """Visible GPU devices and any leases held against them.

    An empty ``devices`` list means "no NVIDIA GPU visible", not "0 % busy".
    """
    devices = gpu_inventory()
    return {
        "available": bool(devices),
        "devices": devices,
        "leases": registry.active_gpu_leases(),
        "note": (
            None
            if devices
            else "No NVIDIA device is visible to this process (no NVML bindings and no nvidia-smi)."
        ),
    }


@router.get("/workers")
def workers(registry: SqliteRegistry = Depends(get_registry)) -> dict[str, Any]:
    return {"workers": [worker.as_dict() for worker in registry.list_workers()]}


@router.get("/state-machine")
def state_machine() -> dict[str, Any]:
    """The run state machine, including which states this build can produce.

    Exposed so the UI can render state semantics without hard-coding them, and
    so it is unambiguous that `STOP_REQUESTED`/`CHECKPOINTING`/`INTERRUPTED`/
    `RESUMING` exist in the schema but are not reachable in this build.
    """
    return {
        "states": [state.value for state in RunState],
        "active_states_this_build": sorted(state.value for state in ACTIVE_STATES_THIS_TRANCHE),
        "schema_only_states": sorted(
            state.value for state in RunState if state not in ACTIVE_STATES_THIS_TRANCHE
        ),
        "transitions": {
            source.value: sorted(target.value for target in targets)
            for source, targets in ALLOWED_TRANSITIONS.items()
        },
    }


def _writes_on() -> bool:
    from ..write_mode import writes_enabled

    return writes_enabled()


capabilities_router = APIRouter(prefix="/api/v1", tags=["capabilities"])


@capabilities_router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    """Declared model capabilities and this build's own feature set.

    ``control_plane`` states plainly which control features are absent. The UI
    uses it to decide what *not* to render — a button whose backing feature does
    not exist must not appear at all.
    """
    return {
        "schema_versions": {
            "control_plane": CONTROL_PLANE_VERSION,
            "config": CONFIG_SCHEMA_VERSION,
            "registry": REGISTRY_SCHEMA_VERSION,
            "event": EVENT_SCHEMA_VERSION,
            "api": "v1",
        },
        "control_plane": {
            "read_only_dashboard": True,
            "launch_from_ui": False,
            "graceful_stop": False,
            "force_terminate": False,
            # The stop and resume *backends* exist and are certified; what does
            # not exist is a write surface here. These flags describe this API,
            # so they stay false until a write-control tranche exposes them
            # (ADR-CSR-014).
            # These describe *this process*: they flip only when write mode is
            # explicitly enabled, which also requires a loopback bind.
            "write_control_available": True,
            "write_control_enabled": _writes_on(),
            "write_mode_loopback_only": True,
            "graceful_stop_api": _writes_on(),
            "resume_api": _writes_on(),
            "dash_stop_control": _writes_on(),
            "dash_resume_control": _writes_on(),
            "exact_resume": False,
            "warm_start_api": False,
            "checkpoint_catalog_write": False,
            "multi_gpu_queue": False,
            "shared_gpu_execution": False,
            "authentication": False,
            "websocket_stream": False,
            "graceful_stop_backend": True,
            "exact_resume_backend": True,
            "checkpoint_catalog_read": True,
            "notes": (
                "This build observes runs; it does not control them. Launching, stopping, "
                "and resuming are CLI operations (bench.control.cli). The stop and "
                "exact-resume backends are implemented and certified for a specific "
                "envelope — see /api/v1/capabilities/exact-resume — but no write endpoint "
                "and no dashboard action control is exposed."
            ),
        },
        "models": all_capabilities(),
    }


@capabilities_router.get("/capabilities/exact-resume")
def exact_resume_capabilities() -> dict[str, Any]:
    """The exact-resume certification matrix.

    Certification is keyed on the full execution envelope, not on a model name:
    a CPU/fp32/single-worker result says nothing about GPU, AMP, or a
    multi-worker loader, and is not generalised to them (ADR-CSR-013,
    DND-CSR-008). Anything absent from this matrix is uncertified.
    """
    from ...checkpoints.certification import (
        UNCERTIFIED_ENVELOPES,
        certification_matrix,
    )
    from ...checkpoints.schema import (
        CHECKPOINT_SCHEMA_VERSION,
        RESUME_BOUNDARY_OPTIMIZER_UPDATE,
    )

    matrix = certification_matrix()
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "resume_boundary": RESUME_BOUNDARY_OPTIMIZER_UPDATE,
        "certified": [row for row in matrix if row["certified"]],
        "not_certified": [row for row in matrix if not row["certified"]],
        "uncertified_envelopes": UNCERTIFIED_ENVELOPES,
        "resume_is_read_only_here": True,
    }
