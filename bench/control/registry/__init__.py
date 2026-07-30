"""SQLite-backed run registry: the authoritative state of record.

Division of responsibility with the event journal (design doc 05, DND-015):

* **registry (here)** — current state, state transitions, optimistic
  concurrency, indexed queries, worker/lease bookkeeping. Small, queryable,
  transactional.
* **event journal** (`bench.control.events`) — the append-only, high-rate,
  portable audit trail. Never queried transactionally.

Neither is sufficient alone: a DB-only design loses portable recovery, a
JSONL-only design loses concurrent state transitions.
"""

from __future__ import annotations

from .schema import (  # noqa: F401
    ALLOWED_TRANSITIONS,
    REGISTRY_SCHEMA_VERSION,
    TERMINAL_STATES,
    ExperimentRecord,
    InvalidTransitionError,
    RunRecord,
    RunState,
    WorkerRecord,
    is_terminal,
    validate_transition,
)
from .sqlite import ConcurrencyError, RegistryError, SqliteRegistry, open_registry  # noqa: F401

__all__ = [
    "ALLOWED_TRANSITIONS",
    "REGISTRY_SCHEMA_VERSION",
    "TERMINAL_STATES",
    "ConcurrencyError",
    "ExperimentRecord",
    "InvalidTransitionError",
    "RegistryError",
    "RunRecord",
    "RunState",
    "SqliteRegistry",
    "WorkerRecord",
    "is_terminal",
    "open_registry",
    "validate_transition",
]
