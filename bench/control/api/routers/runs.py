"""Run listing, detail, events, and artifacts — all read-only.

No endpoint here mutates run state. The UI observes; the worker and the manager
own transitions (design doc 05, DND-010). Control endpoints (`stop`, `resume`,
`terminate`) are absent rather than stubbed — an endpoint that exists but
refuses is still a promise, and this build makes no such promise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ...capabilities import capabilities_for
from ...events.reader import EventReader
from ...events.schema import EventType
from ...process.manager import DEFAULT_HEARTBEAT_TIMEOUT_SECONDS, WorkerManager
from ...process.signals import describe_exit_code
from ...registry.schema import RunState
from ...registry.sqlite import SqliteRegistry
from ..deps import get_manager, get_registry

router = APIRouter(prefix="/api/v1", tags=["runs"])


def _run_dir(record: Any) -> Optional[Path]:
    return Path(record.run_dir) if record.run_dir else None


def _identity_block(record: Any) -> dict[str, Any]:
    """Identity plus declared capability, which the UI shows as badges.

    Every field the UI needs to display a model honestly is assembled here so
    that no page has to guess: paper fidelity and exact-resume certification
    come from the capability declaration, never from "it ran, so it works".
    """
    capability = capabilities_for(record.model_id)
    variant = record.variant_id or ""
    return {
        "model_id": record.model_id,
        "implementation_id": record.implementation_id,
        "init_id": record.init_id,
        "variant_id": variant,
        "variant_id_short": variant.replace("sha256:", "")[:12],
        "display_name": capability.display_name,
        "trainable": capability.trainable,
        "paper_fidelity_status": capability.paper_fidelity_status,
        "paper_fidelity_note": capability.paper_fidelity_note,
        "supports_exact_resume": capability.supports_exact_resume,
        "supports_warm_start": capability.supports_warm_start,
        "supports_graceful_stop": capability.supports_graceful_stop,
        "event_instrumentation": capability.event_instrumentation,
        "instrumentation_note": capability.instrumentation_note,
    }


@router.get("/runs")
def list_runs(
    state: Optional[str] = Query(None, description="Filter by run state"),
    experiment_id: Optional[str] = Query(None),
    model_id: Optional[str] = Query(None),
    variant_id: Optional[str] = Query(None),
    include_legacy: bool = Query(True),
    active_only: bool = Query(False),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    registry: SqliteRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """Bounded run listing. ``limit`` is capped server-side at 1000."""
    if state is not None:
        try:
            RunState(state)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown state {state!r}") from exc
    records = registry.list_runs(
        state=state,
        experiment_id=experiment_id,
        model_id=model_id,
        variant_id=variant_id,
        include_legacy=include_legacy,
        active_only=active_only,
        limit=limit,
        offset=offset,
    )
    return {
        "runs": [{**record.as_dict(), "identity": _identity_block(record)} for record in records],
        "count": len(records),
        "total": registry.count_runs(include_legacy=include_legacy),
        "limit": limit,
        "offset": offset,
    }


@router.get("/runs/{run_id}")
def get_run(
    run_id: str,
    registry: SqliteRegistry = Depends(get_registry),
    manager: WorkerManager = Depends(get_manager),
) -> dict[str, Any]:
    """Full run detail, including worker liveness and legacy provenance."""
    record = registry.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")

    directory = _run_dir(record)
    reader = EventReader(directory / "events.jsonl") if directory else None
    legacy = registry.legacy_mapping(run_id)

    return {
        **record.as_dict(),
        "identity": _identity_block(record),
        "exit_code_description": describe_exit_code(record.exit_code),
        "worker": manager.describe_worker(run_id),
        "transitions": registry.list_transitions(run_id),
        "checkpoints": registry.list_checkpoints(run_id),
        "has_event_journal": bool(reader is not None and reader.exists),
        "last_journal_event_id": (reader.last_event_id() if reader and reader.exists else 0),
        "legacy_mapping": legacy,
        "inspector_deep_link": _inspector_link(record),
        # Single source of truth for the UI: the dashboard renders these and
        # never recomputes the conditions itself (a second implementation is a
        # second set of bugs). The backend re-checks on the actual request.
        "action_eligibility": _eligibility_block(record, registry, manager),
    }


def _eligibility_block(record: Any, registry: SqliteRegistry, manager: Any) -> dict[str, Any]:
    from ...action_eligibility import (
        best_resume_checkpoint,
        resume_eligibility,
        stop_eligibility,
    )
    from ...checkpoints.resume_coordinator import ACTION_RESUME_EXACT
    from ..write_mode import writes_enabled

    run_id = record.run_id
    stop = stop_eligibility(
        run=record,
        worker=manager.describe_worker(run_id),
        open_action=registry.open_action(run_id, action="stop"),
    )
    checkpoint = best_resume_checkpoint(registry=registry, run_id=run_id)
    resume = resume_eligibility(
        checkpoint_row=checkpoint,
        parent_run=record,
        registry=registry,
        open_action=registry.open_action(run_id, action=ACTION_RESUME_EXACT),
    )
    return {
        "write_control_enabled": writes_enabled(),
        "stop_action": stop.as_dict(),
        "resume_action": {
            **resume.as_dict(),
            "checkpoint_id": (checkpoint or {}).get("checkpoint_id"),
        },
    }


def _inspector_link(record: Any) -> Optional[dict[str, str]]:
    """Deep link into the existing Streamlit Run Inspector.

    The Inspector already accepts a ``?run=`` query parameter and matches it
    against either an absolute run directory or one relative to its runs root
    (``viz/app/components/overlay_picker.py``). No change to the Streamlit app
    was needed — the control plane just has to know where the visualization
    artifacts are.

    Returns ``None`` when the run has no ``meta.json``, because the Inspector
    only indexes directories that have one; offering a link that leads to "no
    valid runs found" would be worse than offering none.
    """
    if not record.run_dir:
        return None
    directory = Path(record.run_dir)
    candidates = [directory, directory / "legacy"]
    for candidate in candidates:
        try:
            if (candidate / "meta.json").exists():
                return {"run_path": str(candidate), "query_param": "run"}
            matches = sorted(candidate.rglob("meta.json"))
            if matches:
                return {"run_path": str(matches[0].parent), "query_param": "run"}
        except Exception:
            continue
    return None


@router.get("/runs/{run_id}/events")
def get_events(
    run_id: str,
    after_event_id: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
    event_type: Optional[str] = Query(None),
    registry: SqliteRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """Cursor-paginated event journal read.

    Poll with ``after_event_id = next_cursor`` to stream. The same cursor
    semantics a WebSocket would need are provided here, which is why bounded
    polling is a complete MVP rather than a placeholder (design doc 05 §6).
    """
    record = registry.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    directory = _run_dir(record)
    if directory is None:
        return {"events": [], "next_cursor": after_event_id, "has_more": False, "warnings": []}

    types: Optional[list[EventType]] = None
    if event_type:
        try:
            types = [EventType(item) for item in event_type.split(",") if item]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    reader = EventReader(directory / "events.jsonl")
    if not reader.exists:
        return {
            "events": [],
            "next_cursor": after_event_id,
            "has_more": False,
            "warnings": [],
            "journal_present": False,
        }
    page = reader.scan(after_event_id=after_event_id, limit=limit, event_types=types)
    return {**page.as_dict(), "journal_present": True}


@router.get("/runs/{run_id}/metrics")
def get_metrics(
    run_id: str,
    names: Optional[str] = Query(None, description="Comma-separated metric names"),
    limit_per_series: int = Query(2000, ge=10, le=20000),
    registry: SqliteRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """Metric series for charting, bounded per series."""
    record = registry.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    directory = _run_dir(record)
    if directory is None:
        return {"series": {}}
    reader = EventReader(directory / "events.jsonl")
    if not reader.exists:
        return {"series": {}, "journal_present": False}
    wanted = [item for item in (names or "").split(",") if item] or None
    return {
        "series": reader.metric_series(names=wanted, limit_per_name=limit_per_series),
        "journal_present": True,
    }


@router.get("/runs/{run_id}/resources")
def get_resources(
    run_id: str,
    limit: int = Query(2000, ge=10, le=20000),
    registry: SqliteRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """Resource telemetry samples for this run."""
    record = registry.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    directory = _run_dir(record)
    if directory is None:
        return {"samples": []}
    reader = EventReader(directory / "events.jsonl")
    if not reader.exists:
        return {"samples": [], "journal_present": False}
    return {"samples": reader.resource_samples(limit=limit), "journal_present": True}


@router.get("/runs/{run_id}/artifacts")
def get_artifacts(
    run_id: str,
    registry: SqliteRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """Registered artifacts plus what is actually on disk.

    Both are reported because they can disagree: a run killed mid-write may have
    files with no registry row. Showing only the registry would hide them.
    """
    record = registry.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    registered = [artifact.as_dict() for artifact in registry.list_artifacts(run_id)]

    on_disk: list[dict[str, Any]] = []
    directory = _run_dir(record)
    if directory is not None and directory.exists():
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            # `tmp/` holds partial writes by construction — never advertise it.
            if "tmp" in path.relative_to(directory).parts:
                continue
            try:
                on_disk.append(
                    {
                        "path": str(path.relative_to(directory)),
                        "bytes": path.stat().st_size,
                    }
                )
            except Exception:
                continue
    return {
        "run_id": run_id,
        "registered": registered,
        "on_disk": on_disk[:500],
        "on_disk_truncated": len(on_disk) > 500,
        "failure_present": bool(directory and (directory / "failure.json").exists()),
    }


@router.get("/runs/{run_id}/checkpoints")
def get_checkpoints(
    run_id: str,
    registry: SqliteRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """Catalogued checkpoints for a run.

    Read-only, like everything else on this surface: creating, validating, or
    resuming a checkpoint is a CLI operation in this tranche (ADR-CSR-014).
    ``validation_status`` is reported as stored — this endpoint does not
    re-hash payloads, because a GET must not do unbounded disk work.
    """
    record = registry.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")

    rows = registry.list_checkpoints(run_id)
    checkpoints: list[dict[str, Any]] = []
    for row in rows:
        entry = {k: v for k, v in row.items() if k != "manifest_json"}
        entry["complete"] = bool(row.get("complete"))
        checkpoints.append(entry)

    return {
        "run_id": run_id,
        "count": len(checkpoints),
        "checkpoints": checkpoints,
        "latest_checkpoint_id": record.latest_checkpoint_id,
        "best_checkpoint_id": record.best_checkpoint_id,
        "resume_is_read_only_here": True,
    }


@router.get("/checkpoints/{checkpoint_id}")
def get_checkpoint(
    checkpoint_id: str,
    registry: SqliteRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """One checkpoint, including its stored manifest."""
    row = registry.get_checkpoint(checkpoint_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown checkpoint {checkpoint_id}")

    document = {k: v for k, v in row.items() if k != "manifest_json"}
    document["complete"] = bool(row.get("complete"))
    manifest_json = row.get("manifest_json") or ""
    if manifest_json:
        try:
            document["manifest"] = json.loads(manifest_json)
        except json.JSONDecodeError:
            document["manifest"] = None
    return document


@router.get("/runs/{run_id}/lineage")
def get_lineage(
    run_id: str,
    registry: SqliteRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """Resume lineage for a run.

    A resumed run is a *child*: it has its own run_id and directory, and points
    back at the parent and the checkpoint it continued from (ADR-CSR-003). This
    reports both directions so a researcher can tell a continuous run from a
    resumed one before comparing their numbers.
    """
    record = registry.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")

    children = [
        {
            "run_id": child.run_id,
            "state": str(child.state),
            "resumed_from_checkpoint_id": child.resumed_from_checkpoint_id,
            "created_at": child.created_at,
        }
        for child in registry.list_runs(limit=1000)
        if child.resumed_from_run_id == run_id or child.parent_run_id == run_id
    ]

    return {
        "run_id": run_id,
        "parent_run_id": record.parent_run_id,
        "resumed_from_run_id": record.resumed_from_run_id,
        "resumed_from_checkpoint_id": record.resumed_from_checkpoint_id,
        "is_resumed_child": bool(record.resumed_from_run_id),
        "children": children,
        "variant_id": record.variant_id,
        "variant_note": (
            "An exact-resume child inherits the parent's variant_id: a resume is "
            "execution lineage, not a new model variant."
        ),
    }


@router.get("/runs/{run_id}/actions")
def get_actions(
    run_id: str,
    registry: SqliteRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """Stop-request history for a run (read-only).

    Requesting a stop is a CLI operation; this endpoint only reports what was
    requested and how it settled.
    """
    record = registry.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    actions = registry.list_actions(run_id)
    return {
        "run_id": run_id,
        "count": len(actions),
        "actions": actions,
        "open_action": registry.open_action(run_id),
        "write_actions_exposed": False,
    }


@router.get("/runs/{run_id}/logs")
def get_logs(
    run_id: str,
    stream: str = Query("stdout", pattern="^(stdout|stderr)$"),
    max_bytes: int = Query(64_000, ge=1_000, le=1_000_000),
    registry: SqliteRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """Bounded tail of a captured log stream.

    Reads only the last ``max_bytes`` by seeking, so tailing a multi-gigabyte
    log costs the same as tailing a small one (risk R-07).
    """
    record = registry.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    directory = _run_dir(record)
    if directory is None:
        return {"stream": stream, "text": "", "truncated": False, "size_bytes": 0}
    path = directory / ("stdout.log" if stream == "stdout" else "stderr.log")
    if not path.exists():
        return {"stream": stream, "text": "", "truncated": False, "size_bytes": 0, "present": False}
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        data = handle.read()
    return {
        "stream": stream,
        "text": data.decode("utf-8", errors="replace"),
        "truncated": size > max_bytes,
        "size_bytes": size,
        "present": True,
    }


@router.get("/orphan-candidates")
def orphan_candidates(
    heartbeat_timeout_seconds: float = Query(DEFAULT_HEARTBEAT_TIMEOUT_SECONDS, gt=0),
    manager: WorkerManager = Depends(get_manager),
) -> dict[str, Any]:
    """Runs whose worker appears to have vanished.

    Reporting only. Adjudicating an orphan is a researcher decision and is not
    exposed as an API action in this tranche.
    """
    candidates = manager.find_orphan_candidates(
        heartbeat_timeout_seconds=heartbeat_timeout_seconds
    )
    return {"candidates": [candidate.as_dict() for candidate in candidates]}
