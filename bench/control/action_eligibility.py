"""Single source of truth for "may this action be requested?".

The API returns this; the UI renders it. The UI must never recompute the
conditions, because a second implementation is a second set of bugs and the
two will drift. The backend re-checks everything anyway when the action is
actually requested — the read model is for explanation, not enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .registry.schema import RunState
from .training_path import TrainingPathId

#: Machine-readable codes. The UI keys off these, not off message text.
STOP_OK = None
STOP_NOT_RUNNING = "RUN_NOT_RUNNING"
STOP_NO_WORKER = "NO_LIVE_WORKER"
STOP_PATH_NOT_RESUMABLE = "TRAINING_PATH_NOT_RESUMABLE"
STOP_ALREADY_REQUESTED = "STOP_ALREADY_REQUESTED"

RESUME_NO_CHECKPOINT = "NO_ELIGIBLE_CHECKPOINT"
RESUME_ALREADY_REQUESTED = "RESUME_ALREADY_REQUESTED"


@dataclass
class ActionEligibility:
    eligible: bool
    reason_code: Optional[str] = None
    reason: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "eligible": bool(self.eligible),
            "reason_code": self.reason_code,
            "reason": self.reason,
            **self.extra,
        }


def stop_eligibility(*, run: Any, worker: Optional[dict[str, Any]] = None,
                     open_action: Optional[dict[str, Any]] = None) -> ActionEligibility:
    """Whether a graceful stop can be requested for this run right now."""
    if run is None:
        return ActionEligibility(False, "UNKNOWN_RUN", "This run does not exist.")

    path = str(getattr(run, "training_path_id", "") or "")
    if path != str(TrainingPathId.CONTROL_RESUMABLE_V1):
        return ActionEligibility(
            False, STOP_PATH_NOT_RESUMABLE,
            f"Safe stop unavailable: this run used {path or 'an unknown path'}. "
            "Graceful stop is only available for runs on the certified resumable "
            "path, because only that loop has a checkpoint-safe boundary.",
            {"training_path_id": path},
        )

    if run.state is not RunState.RUNNING:
        return ActionEligibility(
            False, STOP_NOT_RUNNING,
            f"Safe stop unavailable: the run is {run.state.value}, not RUNNING.",
            {"run_state": run.state.value},
        )

    if worker is not None and worker.get("known") and not worker.get("pid_alive", True):
        return ActionEligibility(
            False, STOP_NO_WORKER,
            "Safe stop unavailable: the worker process is no longer alive. "
            "Reconcile the run instead.",
        )

    if open_action is not None:
        return ActionEligibility(
            False, STOP_ALREADY_REQUESTED,
            "A stop has already been requested for this run and is still being "
            "handled at the next safe boundary.",
            {"action_id": open_action.get("action_id"),
             "action_state": open_action.get("status")},
        )

    return ActionEligibility(True)


def resume_eligibility(*, checkpoint_row: Optional[dict[str, Any]], parent_run: Any,
                       registry: Any = None,
                       open_action: Optional[dict[str, Any]] = None,
                       manifest: Any = None) -> ActionEligibility:
    """Whether an exact resume can be launched from this checkpoint.

    Delegates the substantive decision to the checkpoint eligibility evaluator
    so API, UI and the coordinator all agree.
    """
    from .checkpoints.eligibility import evaluate_resume_eligibility

    if checkpoint_row is None and manifest is None:
        return ActionEligibility(
            False, RESUME_NO_CHECKPOINT,
            "Exact resume unavailable: this run has no launch-eligible interrupt "
            "checkpoint.",
        )

    if open_action is not None:
        return ActionEligibility(
            False, RESUME_ALREADY_REQUESTED,
            "A resume has already been requested for this checkpoint.",
            {"action_id": open_action.get("action_id"),
             "action_state": open_action.get("status")},
        )

    report = evaluate_resume_eligibility(
        manifest=manifest, checkpoint_row=checkpoint_row,
        parent_run=parent_run, registry=registry,
    )
    if report.eligible:
        return ActionEligibility(
            True, extra={"certification_id": report.certification_id,
                         "training_path_id": report.training_path_id,
                         "checkpoint_schema_version": report.checkpoint_schema_version}
        )
    return ActionEligibility(
        False,
        report.reason_codes[0] if report.reason_codes else "NOT_ELIGIBLE",
        "Exact resume unavailable: " + " ".join(report.messages),
        {"reason_codes": report.reason_codes,
         "training_path_id": report.training_path_id,
         "checkpoint_schema_version": report.checkpoint_schema_version},
    )


def best_resume_checkpoint(*, registry: Any, run_id: str) -> Optional[dict[str, Any]]:
    """The interrupt checkpoint a resume would target, if any."""
    rows = [
        row for row in registry.list_checkpoints(run_id)
        if str(row.get("kind")) == "interrupt"
        and str(row.get("validation_status")) == "VALID"
    ]
    if not rows:
        return None
    return sorted(rows, key=lambda r: int(r.get("global_step") or 0))[-1]
