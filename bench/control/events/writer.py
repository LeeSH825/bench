"""Append-only JSONL event writer.

Durability policy
-----------------

Flushing every event would serialize training on the disk; flushing none would
lose the run's outcome on a crash. The split used here:

* every event is written and ``flush()``-ed to the OS immediately, so another
  process reading the file sees it right away (this is what makes live tailing
  work at all);
* events in :data:`~bench.control.events.schema.DURABLE_EVENT_TYPES` (status,
  failure, checkpoint) additionally ``fsync()``, so an abrupt power loss cannot
  erase the record of how a run ended.

High-rate metric events therefore cost one ``write`` syscall and no ``fsync``.

Ordering
--------

``event_id`` is assigned under a lock held for the whole append, so ids are
monotonic and gap-free even with several threads (the telemetry sampler runs on
its own thread). Multi-*process* writers to one journal are not supported and
not needed: exactly one worker owns a run.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

from ..paths import ensure_dir
from .schema import (
    DURABLE_EVENT_TYPES,
    EVENT_SCHEMA_VERSION,
    MAX_MESSAGE_CHARS,
    MAX_PAYLOAD_BYTES,
    Event,
    EventType,
    EventValidationError,
    utc_now,
)


def _truncate(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    if len(text) <= MAX_MESSAGE_CHARS:
        return text
    return text[:MAX_MESSAGE_CHARS] + f"… [truncated, {len(text)} chars total]"


class EventWriter:
    """Appends events to one run's ``events.jsonl``.

    Use as a context manager, or call :meth:`close` explicitly.
    """

    def __init__(self, path: str | os.PathLike[str], run_id: str, *, start_event_id: Optional[int] = None):
        self.path = Path(path)
        self.run_id = str(run_id)
        ensure_dir(self.path.parent)
        self._lock = threading.Lock()
        # Resuming an existing journal (e.g. the worker reopening after the
        # manager wrote a startup event) must not restart the id sequence.
        self._next_id = int(start_event_id) if start_event_id is not None else self._scan_last_id() + 1
        self._handle = self.path.open("a", encoding="utf-8")
        self._closed = False

    def _scan_last_id(self) -> int:
        """Highest event_id already in the file, or 0 if none/unreadable."""
        if not self.path.exists():
            return 0
        last = 0
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = max(last, int(json.loads(line)["event_id"]))
                except Exception:
                    # A partial or corrupt tail line does not stop us from
                    # continuing the sequence; the reader reports it separately.
                    continue
        return last

    @property
    def last_event_id(self) -> int:
        return self._next_id - 1

    # -- core append ---------------------------------------------------------

    def append(
        self,
        event_type: EventType,
        *,
        phase: Optional[str] = None,
        subphase: Optional[str] = None,
        step_type: Optional[str] = None,
        step: Optional[int] = None,
        name: Optional[str] = None,
        value: Optional[float] = None,
        unit: Optional[str] = None,
        level: Optional[str] = None,
        message: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Event:
        """Append one event and return it (with its assigned id)."""
        if self._closed:
            raise EventValidationError("event writer is closed")

        document_payload = dict(payload or {})
        encoded_payload = json.dumps(document_payload, ensure_ascii=False, default=str)
        if len(encoded_payload.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise EventValidationError(
                f"event payload is {len(encoded_payload)} bytes, over the "
                f"{MAX_PAYLOAD_BYTES} byte limit. Store the data as an artifact and "
                "reference its URI and hash from the event instead."
            )

        with self._lock:
            event = Event(
                event_id=self._next_id,
                run_id=self.run_id,
                event_type=event_type,
                timestamp=utc_now(),
                phase=phase,
                subphase=subphase,
                step_type=step_type,
                step=step,
                name=name,
                value=(float(value) if value is not None else None),
                unit=unit,
                level=level,
                message=_truncate(message),
                payload=document_payload,
                schema_version=EVENT_SCHEMA_VERSION,
            )
            self._next_id += 1
            line = json.dumps(event.as_dict(), ensure_ascii=False, default=str)
            self._handle.write(line + "\n")
            self._handle.flush()
            if event_type in DURABLE_EVENT_TYPES:
                os.fsync(self._handle.fileno())
        return event

    # -- typed conveniences --------------------------------------------------

    def status(self, state: str, *, phase: Optional[str] = None, message: Optional[str] = None, **payload: Any) -> Event:
        return self.append(
            EventType.STATUS,
            phase=phase,
            name=state,
            message=message,
            payload=payload,
        )

    def metric(
        self,
        name: str,
        value: float,
        *,
        step: Optional[int] = None,
        step_type: str = "global_step",
        phase: Optional[str] = None,
        unit: Optional[str] = None,
        **payload: Any,
    ) -> Event:
        return self.append(
            EventType.METRIC,
            name=name,
            value=value,
            step=step,
            step_type=step_type,
            phase=phase,
            unit=unit,
            payload=payload,
        )

    def log(self, message: str, *, level: str = "INFO", phase: Optional[str] = None, **payload: Any) -> Event:
        return self.append(EventType.LOG, level=level, message=message, phase=phase, payload=payload)

    def warning(self, message: str, *, phase: Optional[str] = None, **payload: Any) -> Event:
        return self.append(EventType.WARNING, level="WARNING", message=message, phase=phase, payload=payload)

    def failure(self, message: str, *, phase: Optional[str] = None, **payload: Any) -> Event:
        return self.append(EventType.FAILURE, level="ERROR", message=message, phase=phase, payload=payload)

    def resource(self, sample: Mapping[str, Any]) -> Event:
        return self.append(EventType.RESOURCE, name="resource_sample", payload=dict(sample))

    def artifact(self, *, kind: str, uri: str, sha256: Optional[str] = None, bytes_: int = 0, **payload: Any) -> Event:
        return self.append(
            EventType.ARTIFACT,
            name=kind,
            payload={"uri": uri, "sha256": sha256, "bytes": bytes_, **payload},
        )

    def checkpoint(self, *, checkpoint_id: str, uri: str, **payload: Any) -> Event:
        return self.append(
            EventType.CHECKPOINT,
            name="checkpoint",
            payload={"checkpoint_id": checkpoint_id, "uri": uri, **payload},
        )

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            finally:
                self._handle.close()
                self._closed = True

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
