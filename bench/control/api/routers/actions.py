"""Write-control action routes.

Registered **only** when write mode is explicitly enabled, so in the default
build these paths do not exist at all.

These handlers are deliberately thin. They validate the request, delegate to
the already-certified durable action services, and return the action resource.
No handler starts training, restores a checkpoint, spawns a process, or mutates
SQLite directly — a request must never be able to block on a worker.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Response
from pydantic import BaseModel, Field

from ...action_eligibility import best_resume_checkpoint
from ...checkpoints.resume_coordinator import (
    ACTION_RESUME_EXACT,
    ResumeConflict,
    ResumeCoordinator,
    ResumeRejected,
)
from ...process.manager import WorkerManager
from ...registry.schema import RunState
from ...registry.sqlite import SqliteRegistry
from ...training_path import TrainingPathId
from ..deps import get_manager, get_registry
from ..write_mode import REQUEST_HEADER, REQUEST_HEADER_VALUE

router = APIRouter(prefix="/api/v1", tags=["actions"])

ACTION_STOP = "stop"
ACTION_TYPE_STOP = "STOP_GRACEFUL"
#: Duplicated from launch_coordinator rather than imported: this module is
#: loaded while the app is being built, and the coordinator pulls in the config
#: and process layers.
ACTION_TYPE_LAUNCH = "LAUNCH_RUN"
RESPONSE_SCHEMA_VERSION = 1

#: Action states that are finished. A finished action is returned as 200, a
#: live one as 202, so a client can tell "already done" from "in progress".
_TERMINAL_ACTION_STATES = frozenset({"COMPLETED", "FAILED", "REJECTED"})


class StopRequest(BaseModel):
    expected_state_version: Optional[int] = Field(
        None, description="Optimistic concurrency guard against the run's state_version"
    )
    reason: Optional[str] = Field(None, max_length=500)


class ResumeRequest(BaseModel):
    expected_parent_state_version: Optional[int] = Field(None)


def _require_write_header(value: Optional[str]) -> None:
    if value != REQUEST_HEADER_VALUE:
        raise HTTPException(
            status_code=400,
            detail={
                "reason_code": "MISSING_CONTROL_HEADER",
                "message": (
                    f"write requests must carry {REQUEST_HEADER}: {REQUEST_HEADER_VALUE}. "
                    "This blocks plain HTML form posts and naive cross-site fetches, "
                    "which cannot set custom headers without a CORS preflight this "
                    "service never grants."
                ),
            },
        )


def _require_idempotency_key(value: Optional[str]) -> str:
    key = (value or "").strip()
    if not key:
        raise HTTPException(
            status_code=400,
            detail={
                "reason_code": "MISSING_IDEMPOTENCY_KEY",
                "message": (
                    "Idempotency-Key is required so a retried request cannot create a "
                    "second action, child run, or worker."
                ),
            },
        )
    if len(key) > 200:
        raise HTTPException(
            status_code=400,
            detail={"reason_code": "IDEMPOTENCY_KEY_TOO_LONG",
                    "message": "Idempotency-Key must be at most 200 characters."},
        )
    return key


def _action_resource(
    action: dict[str, Any], *, reused: bool = False, checkpoint_id: Optional[str] = None
) -> dict[str, Any]:
    """Public shape of an action. The idempotency key is never echoed back."""
    state = str(action.get("status") or "")
    run_id = action.get("run_id")
    if run_id is None and str(action.get("action")) == ACTION_TYPE_LAUNCH:
        # A launch action has no run when it is requested — it *creates* one.
        # Without this, polling the action loses the link to the run that was
        # just started, which is the only handle the operator has on it.
        run_id = action.get("result_child_run_id")
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "action_id": action.get("action_id"),
        "action_type": (
            ACTION_TYPE_STOP if str(action.get("action")) == ACTION_STOP
            else str(action.get("action"))
        ),
        "state": state,
        "terminal": state in _TERMINAL_ACTION_STATES,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "child_run_id": action.get("result_child_run_id"),
        "result_checkpoint_id": action.get("result_checkpoint_id"),
        "requested_at": action.get("requested_at"),
        "acknowledged_at": action.get("acknowledged_at"),
        "completed_at": action.get("completed_at"),
        "error": action.get("failure_reason"),
        "idempotency_reused": bool(reused),
        "status_url": f"/api/v1/actions/{action.get('action_id')}",
    }


@router.post("/runs/{run_id}/actions/stop", status_code=202)
def request_stop(
    run_id: str,
    response: Response,
    payload: StopRequest = Body(default_factory=StopRequest),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    control_request: Optional[str] = Header(None, alias=REQUEST_HEADER),
    registry: SqliteRegistry = Depends(get_registry),
    manager: WorkerManager = Depends(get_manager),
) -> dict[str, Any]:
    """Record a durable graceful-stop request.

    Returns as soon as the row is durable. The worker honours it at its next
    checkpoint-safe boundary; this handler never waits for that.
    """
    _require_write_header(control_request)
    key = _require_idempotency_key(idempotency_key)

    record = registry.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail={
            "reason_code": "UNKNOWN_RUN", "message": f"unknown run {run_id}"})

    existing = registry.get_action_by_key(key)
    if existing is not None:
        if str(existing.get("run_id")) != run_id or str(existing.get("action")) != ACTION_STOP:
            raise HTTPException(status_code=409, detail={
                "reason_code": "IDEMPOTENCY_KEY_REUSED",
                "message": "This Idempotency-Key was already used for a different request."})
        if str(existing.get("status")) in _TERMINAL_ACTION_STATES:
            response.status_code = 200
        return _action_resource(existing, reused=True)

    if str(record.training_path_id) != str(TrainingPathId.CONTROL_RESUMABLE_V1):
        raise HTTPException(status_code=422, detail={
            "reason_code": "TRAINING_PATH_NOT_RESUMABLE",
            "message": (
                f"Safe stop unavailable: this run used {record.training_path_id}. "
                "Graceful stop is certified only for control_resumable_v1."),
            "training_path_id": record.training_path_id})

    if record.state is not RunState.RUNNING:
        raise HTTPException(status_code=409, detail={
            "reason_code": "RUN_NOT_RUNNING",
            "message": f"the run is {record.state.value}, not RUNNING",
            "run_state": record.state.value})

    # The eligibility read model and the enforcement path must agree
    # (ADR-WC-020): a RUNNING run whose worker is gone cannot honour a stop,
    # so refuse rather than record an action nothing will ever act on.
    worker = manager.describe_worker(run_id)
    if worker and worker.get("known") and not worker.get("pid_alive", True):
        raise HTTPException(status_code=409, detail={
            "reason_code": "NO_LIVE_WORKER",
            "message": (
                "the worker process for this run is no longer alive; reconcile the "
                "run instead of requesting a graceful stop")})

    if (payload.expected_state_version is not None
            and int(payload.expected_state_version) != int(record.state_version)):
        raise HTTPException(status_code=409, detail={
            "reason_code": "STALE_STATE_VERSION",
            "message": (
                f"run is at state_version {record.state_version}, request expected "
                f"{payload.expected_state_version}"),
            "current_state_version": record.state_version})

    try:
        action = registry.request_action(
            run_id=run_id, action=ACTION_STOP, idempotency_key=key,
            requested_by="api",
            parameters={"reason": payload.reason or "operator_requested"},
            expected_state_version=payload.expected_state_version,
        )
    except Exception as exc:  # pragma: no cover - registry failure
        raise HTTPException(status_code=503, detail={
            "reason_code": "REGISTRY_UNAVAILABLE", "message": str(exc)}) from exc

    return _action_resource(action)


@router.post("/checkpoints/{checkpoint_id}/actions/resume", status_code=202)
def request_resume(
    checkpoint_id: str,
    response: Response,
    payload: ResumeRequest = Body(default_factory=ResumeRequest),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    control_request: Optional[str] = Header(None, alias=REQUEST_HEADER),
    registry: SqliteRegistry = Depends(get_registry),
    manager: WorkerManager = Depends(get_manager),
) -> dict[str, Any]:
    """Launch an immutable resumed child from a validated checkpoint.

    The coordinator validates eligibility, allocates the child, and launches
    one worker. `COMPLETED` here means the child *launched*, never that its
    training finished.
    """
    _require_write_header(control_request)
    key = _require_idempotency_key(idempotency_key)

    row = registry.get_checkpoint(checkpoint_id)
    if row is None:
        raise HTTPException(status_code=404, detail={
            "reason_code": "UNKNOWN_CHECKPOINT",
            "message": f"unknown checkpoint {checkpoint_id}"})

    existing = registry.get_action_by_key(key)
    if existing is not None:
        import json as _json
        stored = {}
        try:
            stored = _json.loads(existing.get("parameters_json") or "{}")
        except Exception:
            stored = {}
        if (str(existing.get("action")) != ACTION_RESUME_EXACT
                or str(stored.get("checkpoint_id")) != checkpoint_id):
            raise HTTPException(status_code=409, detail={
                "reason_code": "IDEMPOTENCY_KEY_REUSED",
                "message": "This Idempotency-Key was already used for a different request."})
        if str(existing.get("status")) in _TERMINAL_ACTION_STATES:
            response.status_code = 200
        return _action_resource(existing, reused=True, checkpoint_id=checkpoint_id)

    coordinator = ResumeCoordinator(
        registry=registry, manager=manager, control_root=manager.control_root
    )
    try:
        outcome = coordinator.request_resume(
            checkpoint_id=checkpoint_id,
            idempotency_key=key,
            expected_parent_state_version=payload.expected_parent_state_version,
            requested_by="api",
        )
    except ResumeConflict as exc:
        raise HTTPException(status_code=409, detail={
            "reason_code": "STALE_OR_CONFLICTING_REQUEST", "message": str(exc)}) from exc
    except ResumeRejected as exc:
        codes = list(exc.reason_codes)
        # A checkpoint that exists but cannot be trusted is a conflict; an
        # envelope this build never certified is unprocessable.
        conflictish = {"CHECKPOINT_NOT_VALID", "PARENT_NOT_TERMINAL",
                       "UNKNOWN_CHECKPOINT", "UNKNOWN_PARENT"}
        status = 409 if any(c in conflictish for c in codes) else 422
        raise HTTPException(status_code=status, detail={
            "reason_code": codes[0] if codes else "NOT_ELIGIBLE",
            "reason_codes": codes, "message": str(exc)}) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail={
            "reason_code": "MANAGER_UNAVAILABLE", "message": str(exc)}) from exc

    action = registry.get_action(outcome.action_id) or {}
    return _action_resource(action, reused=outcome.reused_existing,
                            checkpoint_id=checkpoint_id)


class LaunchRequest(BaseModel):
    preset_id: str = Field(..., max_length=200)
    preset_digest: str = Field(..., max_length=200)
    task_id: Optional[str] = Field(None, max_length=200)
    model_id: Optional[str] = Field(None, max_length=200)
    init_id: Optional[str] = Field(None, max_length=64)
    overrides: dict[str, Any] = Field(default_factory=dict)
    expected_structural_config_hash: Optional[str] = Field(None, max_length=200)
    expected_operational_config_hash: Optional[str] = Field(None, max_length=200)


@router.post("/runs/launch", status_code=202)
def launch_run(
    response: Response,
    payload: LaunchRequest = Body(...),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    control_request: Optional[str] = Header(None, alias=REQUEST_HEADER),
    registry: SqliteRegistry = Depends(get_registry),
    manager: WorkerManager = Depends(get_manager),
) -> dict[str, Any]:
    """Allocate an immutable run from a tracked preset and launch one worker.

    The handler validates, records a durable action, and delegates. It never
    spawns a process itself and never waits for the workload.
    """
    from ...launch_coordinator import (
        ACTION_LAUNCH_RUN,
        LaunchConflict,
        LaunchCoordinator,
        LaunchRejected,
    )

    _require_write_header(control_request)
    key = _require_idempotency_key(idempotency_key)

    coordinator = LaunchCoordinator(
        registry=registry, manager=manager, control_root=manager.control_root)
    try:
        outcome = coordinator.request_launch(
            preset_id=payload.preset_id, preset_digest=payload.preset_digest,
            idempotency_key=key, task_id=payload.task_id, model_id=payload.model_id,
            init_id=payload.init_id, overrides=payload.overrides,
            expected_structural_config_hash=payload.expected_structural_config_hash,
            expected_operational_config_hash=payload.expected_operational_config_hash,
        )
    except LaunchConflict as exc:
        raise HTTPException(status_code=409, detail={
            "reason_code": "STALE_OR_CONFLICTING_REQUEST", "message": str(exc)}) from exc
    except LaunchRejected as exc:
        codes = list(exc.reason_codes)
        status = 404 if "UNKNOWN_PRESET" in codes else 422
        raise HTTPException(status_code=status, detail={
            "reason_code": codes[0] if codes else "NOT_LAUNCHABLE",
            "reason_codes": codes, "message": str(exc)}) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail={
            "reason_code": "MANAGER_UNAVAILABLE", "message": str(exc)}) from exc

    action = registry.get_action(outcome.action_id) or {}
    body = _action_resource(action, reused=outcome.reused_existing)
    body["action_type"] = ACTION_LAUNCH_RUN
    body["run_id"] = outcome.run_id
    body["run_url"] = f"/api/v1/runs/{outcome.run_id}" if outcome.run_id else None
    body["error"] = body.get("error") or outcome.reason
    if outcome.reused_existing and str(action.get("status")) in _TERMINAL_ACTION_STATES:
        response.status_code = 200
    return body


@router.get("/actions/{action_id}")
def get_action(
    action_id: str,
    registry: SqliteRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """Poll an action. Durable state is the authority, not client memory."""
    action = registry.get_action(action_id)
    if action is None:
        raise HTTPException(status_code=404, detail={
            "reason_code": "UNKNOWN_ACTION", "message": f"unknown action {action_id}"})
    import json as _json
    checkpoint_id = None
    try:
        checkpoint_id = _json.loads(action.get("parameters_json") or "{}").get("checkpoint_id")
    except Exception:
        checkpoint_id = None
    return _action_resource(action, checkpoint_id=checkpoint_id)
