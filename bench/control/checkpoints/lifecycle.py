"""Graceful-stop and resume lifecycle orchestration.

The ordering here is the safety property (ADR-CSR-012, DND-CSR-004):

    RUNNING → STOP_REQUESTED → CHECKPOINTING → [checkpoint durable + validated]
    → INTERRUPTED

``INTERRUPTED`` is recorded only after the interrupt checkpoint exists *and*
validates. If the checkpoint cannot be written, the run is ``FAILED`` with exit
code 50 rather than a terminal state that implies resumable state exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..registry.schema import RunState
from .schema import CheckpointKind, TrainingCursor

#: Exit codes for the stop paths. 10 means "stopped cleanly and is resumable".
EXIT_INTERRUPTED = 10
#: 50 means "asked to stop, could not persist state". Deliberately distinct
#: from the ordinary failure code so an operator can tell the two apart.
EXIT_CHECKPOINT_FAILED = 50


@dataclass
class StopOutcome:
    state: RunState
    exit_code: int
    checkpoint_id: Optional[str] = None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": str(self.state),
            "exit_code": int(self.exit_code),
            "checkpoint_id": self.checkpoint_id,
            "reason": self.reason,
        }


def settle_graceful_stop(
    *,
    run_id: str,
    registry: Any,
    service: Any,
    cursor: TrainingCursor,
    adapter: Any,
    rng: Any,
    identity: dict[str, Any],
    action_id: Optional[str] = None,
    structural_config_hash: str = "",
    dataset_fingerprint: str = "",
    batch_plan: Optional[Any] = None,
    resolved_run_spec: Optional[dict[str, Any]] = None,
    progress: Optional[dict[str, Any]] = None,
    event_writer: Optional[Any] = None,
    training_path_id: Optional[str] = None,
    training_path_contract_version: Optional[int] = None,
) -> StopOutcome:
    """Write the interrupt checkpoint, then record the terminal state.

    Called from ordinary control flow at a safe boundary — never from a signal
    handler (ADR-CSR-010).
    """
    current = registry.get_run(run_id)
    if current is None:
        raise ValueError(f"unknown run_id {run_id!r}")

    # RUNNING → STOP_REQUESTED (skipped if a requester already moved it).
    if current.state is RunState.RUNNING:
        registry.transition(
            run_id,
            to_state=RunState.STOP_REQUESTED,
            actor="worker",
            reason="stop request acknowledged at safe boundary",
        )

    registry.transition(
        run_id,
        to_state=RunState.CHECKPOINTING,
        actor="worker",
        reason="writing interrupt checkpoint",
    )
    if event_writer is not None:
        event_writer.status(
            "CHECKPOINTING", phase="train", message="writing interrupt checkpoint"
        )

    try:
        state = adapter.capture_training_state(cursor)
        extra = dict(state.extra_state or {})
        extra["progress"] = dict(progress or {})
        state.extra_state = extra

        saved = service.save(
            run_id=run_id,
            kind=CheckpointKind.INTERRUPT,
            cursor=cursor,
            adapter_state=state,
            rng=rng,
            identity=identity,
            structural_config_hash=structural_config_hash,
            dataset_fingerprint=dataset_fingerprint,
            batch_plan=batch_plan,
            resolved_run_spec=resolved_run_spec,
            capabilities=adapter.checkpoint_capabilities(),
            # Supplying the path makes this a Checkpoint v2 package, which is
            # what a resumed child launch requires. Without it the package is
            # a valid v1 artifact that is deliberately not launch-eligible.
            training_path_id=training_path_id,
            training_path_contract_version=training_path_contract_version,
        )
        # Validate *before* claiming the run is resumable.
        report = service.validate(saved.checkpoint_id)
        report.raise_if_invalid()
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if event_writer is not None:
            event_writer.failure(f"interrupt checkpoint failed: {reason}", phase="train")
        registry.transition(
            run_id,
            to_state=RunState.FAILED,
            actor="worker",
            reason=f"interrupt checkpoint failed: {reason}",
            fields={"exit_code": EXIT_CHECKPOINT_FAILED, "terminal_reason": "checkpoint_failed"},
        )
        if action_id:
            registry.fail_action(action_id, reason=reason)
        return StopOutcome(
            state=RunState.FAILED, exit_code=EXIT_CHECKPOINT_FAILED, reason=reason
        )

    registry.transition(
        run_id,
        to_state=RunState.INTERRUPTED,
        actor="worker",
        reason="interrupt checkpoint written and validated",
        fields={"exit_code": EXIT_INTERRUPTED, "terminal_reason": "interrupted"},
    )
    if action_id:
        registry.complete_action(action_id, result_checkpoint_id=saved.checkpoint_id)
    if event_writer is not None:
        event_writer.status(
            "INTERRUPTED", phase="train", message=f"interrupted at update {cursor.global_update}"
        )

    return StopOutcome(
        state=RunState.INTERRUPTED,
        exit_code=EXIT_INTERRUPTED,
        checkpoint_id=saved.checkpoint_id,
        reason="graceful stop",
    )


@dataclass
class ResumePlan:
    """What a resume would do, resolved and validated before anything is created."""

    parent_run_id: str
    checkpoint_id: str
    cursor: TrainingCursor
    manifest: Any
    identity: dict[str, Any]

    def lineage(self) -> dict[str, Any]:
        return {
            "parent_run_id": self.parent_run_id,
            "resumed_from_run_id": self.parent_run_id,
            "resumed_from_checkpoint_id": self.checkpoint_id,
        }


def plan_resume(
    *,
    checkpoint_id: str,
    registry: Any,
    service: Any,
    expected: Optional[dict[str, Any]] = None,
) -> ResumePlan:
    """Resolve and validate a resume without mutating anything.

    A resume produces a **child run**; the parent's directory, events, and
    checkpoints are never touched (ADR-CSR-003, DND-CSR-003).
    """
    row = registry.get_checkpoint(checkpoint_id)
    if row is None:
        raise ValueError(f"unknown checkpoint {checkpoint_id!r}")

    parent_run_id = str(row["run_id"])
    parent = registry.get_run(parent_run_id)
    if parent is None:
        raise ValueError(f"checkpoint {checkpoint_id!r} references unknown run {parent_run_id!r}")

    if parent.state not in (
        RunState.INTERRUPTED,
        RunState.COMPLETED,
        RunState.FAILED,
        RunState.ORPHANED,
        RunState.CANCELLED,
    ):
        raise ValueError(
            f"refusing to resume from run {parent_run_id} in state {parent.state}: "
            "resume requires a run that is no longer executing"
        )

    manifest, cursor, _state, _rng, _payload = service.load(checkpoint_id, expected=expected)
    identity = {
        "model_id": manifest.model_id,
        "implementation_id": manifest.implementation_id,
        # Exact resume inherits the parent's variant: a resume is execution
        # lineage, not a new algorithm variant (ADR-CSR-004).
        "variant_id": manifest.variant_id,
    }
    return ResumePlan(
        parent_run_id=parent_run_id,
        checkpoint_id=checkpoint_id,
        cursor=cursor,
        manifest=manifest,
        identity=identity,
    )
