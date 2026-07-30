"""Append-only event journal and the adapter-facing observer contract."""

from __future__ import annotations

from .observer import (  # noqa: F401
    JournalObserver,
    NullObserver,
    RunObserver,
    active_observer,
    set_active_observer,
)
from .reader import EventPage, EventReader, RecoveryWarning  # noqa: F401
from .schema import (  # noqa: F401
    EVENT_SCHEMA_VERSION,
    METRIC_EPOCH,
    METRIC_GLOBAL_STEP,
    METRIC_TEST_MSE,
    METRIC_TEST_MSE_DB,
    METRIC_TRAIN_LOSS,
    METRIC_VALIDATION_LOSS,
    Event,
    EventType,
    EventValidationError,
)
from .writer import EventWriter  # noqa: F401

__all__ = [
    "EVENT_SCHEMA_VERSION",
    "Event",
    "EventPage",
    "EventReader",
    "EventType",
    "EventValidationError",
    "EventWriter",
    "JournalObserver",
    "METRIC_EPOCH",
    "METRIC_GLOBAL_STEP",
    "METRIC_TEST_MSE",
    "METRIC_TEST_MSE_DB",
    "METRIC_TRAIN_LOSS",
    "METRIC_VALIDATION_LOSS",
    "NullObserver",
    "RecoveryWarning",
    "RunObserver",
    "active_observer",
    "set_active_observer",
]
