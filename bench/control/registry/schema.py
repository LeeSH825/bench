"""Run state machine and registry record types.

The state vocabulary is the full one from design doc 03 §6, including the four
states this tranche does **not** drive (`STOP_REQUESTED`, `CHECKPOINTING`,
`INTERRUPTED`, `RESUMING`). They are present so Phase 2/5 can enable graceful
stop and resume without a schema migration — but no code path in this tranche
transitions into them, and the API exposes no action that would.

That distinction matters: a state existing in an enum is not a feature. See
:data:`ACTIVE_STATES_THIS_TRANCHE`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

#: Version of the registry SQL schema. Must equal the highest applied migration.
REGISTRY_SCHEMA_VERSION = 1


class RunState(str, enum.Enum):
    """Lifecycle state of a run."""

    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOP_REQUESTED = "STOP_REQUESTED"
    CHECKPOINTING = "CHECKPOINTING"
    INTERRUPTED = "INTERRUPTED"
    RESUMING = "RESUMING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    ORPHANED = "ORPHANED"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Terminal states. Once reached, a run never returns to a non-terminal state
#: (design doc 03 §5.1).
#:
#: ``ORPHANED`` is deliberately **not** terminal: it means "the worker vanished
#: and we do not know what happened". A researcher must inspect PID identity,
#: heartbeat, and checkpoint integrity and then decide. Auto-converting it to
#: FAILED would destroy exactly the information that makes it useful
#: (design doc 03 §6).
TERMINAL_STATES = frozenset({RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED})

#: States this tranche's code can actually produce. Anything else is schema-only.
ACTIVE_STATES_THIS_TRANCHE = frozenset(
    {
        RunState.CREATED,
        RunState.VALIDATING,
        RunState.QUEUED,
        RunState.STARTING,
        RunState.RUNNING,
        RunState.COMPLETED,
        RunState.FAILED,
        RunState.CANCELLED,
        RunState.ORPHANED,
    }
)

#: Allowed state transitions (design doc 03 §6).
ALLOWED_TRANSITIONS: Mapping[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.VALIDATING, RunState.CANCELLED}),
    RunState.VALIDATING: frozenset({RunState.QUEUED, RunState.FAILED, RunState.CANCELLED}),
    RunState.QUEUED: frozenset({RunState.STARTING, RunState.CANCELLED}),
    RunState.STARTING: frozenset({RunState.RUNNING, RunState.FAILED, RunState.ORPHANED}),
    RunState.RUNNING: frozenset(
        {
            RunState.STOP_REQUESTED,
            RunState.CHECKPOINTING,
            RunState.COMPLETED,
            RunState.FAILED,
            RunState.ORPHANED,
        }
    ),
    RunState.STOP_REQUESTED: frozenset(
        {RunState.CHECKPOINTING, RunState.CANCELLED, RunState.FAILED}
    ),
    RunState.CHECKPOINTING: frozenset(
        {RunState.RUNNING, RunState.INTERRUPTED, RunState.COMPLETED, RunState.FAILED}
    ),
    RunState.INTERRUPTED: frozenset({RunState.RESUMING}),
    RunState.RESUMING: frozenset({RunState.RUNNING, RunState.FAILED}),
    # An orphaned run is resolved by researcher action, not by the state machine.
    # The only permitted moves are explicit adjudications.
    RunState.ORPHANED: frozenset({RunState.FAILED, RunState.CANCELLED}),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
}


class InvalidTransitionError(ValueError):
    """Raised when a state transition is not permitted."""

    def __init__(self, from_state: RunState, to_state: RunState):
        self.from_state = from_state
        self.to_state = to_state
        allowed = sorted(state.value for state in ALLOWED_TRANSITIONS.get(from_state, frozenset()))
        super().__init__(
            f"transition {from_state.value} → {to_state.value} is not allowed; "
            f"allowed from {from_state.value}: {allowed or '(terminal)'}"
        )


def is_terminal(state: RunState) -> bool:
    return state in TERMINAL_STATES


def validate_transition(from_state: RunState, to_state: RunState) -> None:
    """Raise :class:`InvalidTransitionError` if the transition is not allowed."""
    if to_state not in ALLOWED_TRANSITIONS.get(from_state, frozenset()):
        raise InvalidTransitionError(from_state, to_state)


# --------------------------------------------------------------------------- #
# Record types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    name: str
    description: str = ""
    tags: tuple[str, ...] = ()
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class RunRecord:
    """Current registry state for one run (design doc 03 §5)."""

    run_id: str
    experiment_id: str
    state: RunState
    state_version: int
    created_at: str
    updated_at: str

    # identity
    model_id: str = ""
    implementation_id: str = ""
    init_id: str = ""
    variant_id: str = ""
    task_id: str = ""
    scenario_id: str = ""
    seed: int = 0

    # execution
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    host: Optional[str] = None
    pid: Optional[int] = None
    process_group_id: Optional[int] = None
    heartbeat_at: Optional[str] = None
    worker_instance_id: Optional[str] = None
    gpu_lease_id: Optional[str] = None
    device: Optional[str] = None

    # progress
    phase: Optional[str] = None
    subphase: Optional[str] = None
    global_step: int = 0
    epoch: int = 0
    batch_cursor: int = 0
    last_event_id: int = 0

    # lineage / outcome
    latest_checkpoint_id: Optional[str] = None
    best_checkpoint_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    resumed_from_run_id: Optional[str] = None
    resumed_from_checkpoint_id: Optional[str] = None
    exit_code: Optional[int] = None
    terminal_reason: Optional[str] = None
    error_summary: Optional[str] = None

    # location / config identity
    run_dir: str = ""
    structural_config_hash: str = ""
    operational_config_hash: str = ""
    resolved_spec_hash: str = ""

    # legacy import
    legacy: bool = False
    status_confidence: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "state": self.state.value,
            "state_version": self.state_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "host": self.host,
            "pid": self.pid,
            "process_group_id": self.process_group_id,
            "heartbeat_at": self.heartbeat_at,
            "worker_instance_id": self.worker_instance_id,
            "gpu_lease_id": self.gpu_lease_id,
            "device": self.device,
            "phase": self.phase,
            "subphase": self.subphase,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "batch_cursor": self.batch_cursor,
            "last_event_id": self.last_event_id,
            "latest_checkpoint_id": self.latest_checkpoint_id,
            "best_checkpoint_id": self.best_checkpoint_id,
            "parent_run_id": self.parent_run_id,
            "resumed_from_run_id": self.resumed_from_run_id,
            "resumed_from_checkpoint_id": self.resumed_from_checkpoint_id,
            "exit_code": self.exit_code,
            "terminal_reason": self.terminal_reason,
            "error_summary": self.error_summary,
            "model_id": self.model_id,
            "implementation_id": self.implementation_id,
            "init_id": self.init_id,
            "variant_id": self.variant_id,
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "run_dir": self.run_dir,
            "structural_config_hash": self.structural_config_hash,
            "operational_config_hash": self.operational_config_hash,
            "resolved_spec_hash": self.resolved_spec_hash,
            "legacy": self.legacy,
            "status_confidence": self.status_confidence,
            "is_terminal": is_terminal(self.state),
        }


@dataclass(frozen=True)
class WorkerRecord:
    """One worker process instance.

    ``process_start_time`` and ``worker_token`` together defend against PID
    reuse (acceptance P-06): a recycled PID belonging to an unrelated process
    will not match the recorded start time, so it is never mistaken for a live
    worker and never killed.
    """

    worker_instance_id: str
    run_id: str
    host: str
    pid: int
    process_group_id: int
    process_start_time: float
    worker_token: str
    started_at: str
    last_heartbeat_at: Optional[str] = None
    state: str = "STARTING"
    exit_code: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "worker_instance_id": self.worker_instance_id,
            "run_id": self.run_id,
            "host": self.host,
            "pid": self.pid,
            "process_group_id": self.process_group_id,
            "process_start_time": self.process_start_time,
            "worker_token": self.worker_token,
            "started_at": self.started_at,
            "last_heartbeat_at": self.last_heartbeat_at,
            "state": self.state,
            "exit_code": self.exit_code,
        }


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    run_id: str
    kind: str
    uri: str
    sha256: Optional[str] = None
    bytes_: int = 0
    media_type: str = "application/octet-stream"
    created_at: str = ""
    complete: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "kind": self.kind,
            "uri": self.uri,
            "sha256": self.sha256,
            "bytes": self.bytes_,
            "media_type": self.media_type,
            "created_at": self.created_at,
            "complete": self.complete,
            "metadata": dict(self.metadata),
        }
