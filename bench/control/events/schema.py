"""Event journal schema.

One event is one JSONL line. The journal is append-only and is the portable
audit trail; the SQLite registry holds current state. Neither replaces the
other (DND-015).

Metric naming follows design doc 03 §7.2. The namespace matters more than it
looks: `metrics_step.csv` in the legacy runner indexes by *sequence time*, while
the control plane's `step` is an *optimizer global step*. Mixing them produces
charts that are silently wrong, so the two never share a field —
:data:`STEP_TYPES` makes the axis explicit on every metric event.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

#: Version of the event document. Readers accept this version and one below.
EVENT_SCHEMA_VERSION = 1

#: Minimum event schema version this build can read.
MIN_READABLE_EVENT_SCHEMA_VERSION = 1

#: Payloads larger than this are rejected: a large tensor belongs in an artifact
#: with only its URI and hash in the event (design doc 03 §7.3).
MAX_PAYLOAD_BYTES = 16 * 1024

#: Log/message text is truncated at this length before writing.
MAX_MESSAGE_CHARS = 8 * 1024


class EventType(str, enum.Enum):
    STATUS = "status"
    METRIC = "metric"
    LOG = "log"
    RESOURCE = "resource"
    CHECKPOINT = "checkpoint"
    ARTIFACT = "artifact"
    CONTROL = "control"
    WARNING = "warning"
    FAILURE = "failure"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Which axis a metric's ``step`` is measured on.
STEP_TYPES = ("global_step", "epoch", "validation_index", "sequence_index", "wall_clock")

#: Event types that must be durable immediately (fsync) because losing them
#: would misrepresent the run's outcome.
DURABLE_EVENT_TYPES = frozenset(
    {EventType.STATUS, EventType.FAILURE, EventType.CHECKPOINT}
)

# -- canonical metric names (design doc 03 §7.2) ----------------------------- #

METRIC_TRAIN_LOSS = "loss/train_total"
METRIC_VALIDATION_LOSS = "loss/validation_total"
METRIC_TEST_MSE = "metric/test_mse"
METRIC_TEST_MSE_DB = "metric/test_mse_db"
METRIC_GLOBAL_STEP = "progress/global_step"
METRIC_EPOCH = "progress/epoch"
METRIC_THROUGHPUT = "throughput/sequences_per_sec"
METRIC_UPDATE_LATENCY_MS = "latency/update_ms"


def utc_now() -> str:
    """RFC 3339 UTC timestamp with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class EventValidationError(ValueError):
    """Raised when an event cannot be written as specified."""


@dataclass(frozen=True)
class Event:
    """One journal entry.

    ``event_id`` is monotonic and gap-free per run, assigned by the writer under
    its own lock. It is the cursor used by the polling API — which is why it must
    never go backwards or repeat (acceptance R-05).
    """

    event_id: int
    run_id: str
    event_type: EventType
    timestamp: str = field(default_factory=utc_now)
    phase: Optional[str] = None
    subphase: Optional[str] = None
    step_type: Optional[str] = None
    step: Optional[int] = None
    name: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    level: Optional[str] = None
    message: Optional[str] = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = EVENT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "phase": self.phase,
            "subphase": self.subphase,
            "step_type": self.step_type,
            "step": self.step,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "level": self.level,
            "message": self.message,
            "payload": dict(self.payload),
        }

    @staticmethod
    def from_dict(document: Mapping[str, Any]) -> "Event":
        version = int(document.get("schema_version", EVENT_SCHEMA_VERSION))
        if version > EVENT_SCHEMA_VERSION or version < MIN_READABLE_EVENT_SCHEMA_VERSION:
            raise EventValidationError(
                f"event schema_version={version} outside readable range "
                f"[{MIN_READABLE_EVENT_SCHEMA_VERSION}, {EVENT_SCHEMA_VERSION}]"
            )
        return Event(
            event_id=int(document["event_id"]),
            run_id=str(document["run_id"]),
            event_type=EventType(str(document["event_type"])),
            timestamp=str(document.get("timestamp") or ""),
            phase=document.get("phase"),
            subphase=document.get("subphase"),
            step_type=document.get("step_type"),
            step=document.get("step"),
            name=document.get("name"),
            value=document.get("value"),
            unit=document.get("unit"),
            level=document.get("level"),
            message=document.get("message"),
            payload=dict(document.get("payload") or {}),
            schema_version=version,
        )
