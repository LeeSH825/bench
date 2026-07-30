"""Bounded, crash-tolerant JSONL event reader.

Two properties matter more than speed here.

**Partial-tail recovery.** A worker killed mid-``write`` leaves a truncated last
line. The reader returns every valid event before it and reports the damage as a
:class:`RecoveryWarning` rather than raising — losing the whole journal because
its last 30 bytes are incomplete would be absurd (acceptance R-04). A malformed
line in the *middle* of the file is different: that indicates real corruption,
not a crash tail, and is reported separately.

**Bounded reads.** Every query takes a cursor and a limit. The dashboard polls
``after_event_id`` and gets at most ``limit`` events, so a 500 MB journal never
becomes a 500 MB HTTP response (risk R-07).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from .schema import Event, EventType


@dataclass(frozen=True)
class RecoveryWarning:
    """A line the reader could not parse."""

    line_number: int
    byte_offset: int
    reason: str
    is_tail: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "line_number": self.line_number,
            "byte_offset": self.byte_offset,
            "reason": self.reason,
            "is_tail": self.is_tail,
        }


@dataclass(frozen=True)
class EventPage:
    """A bounded slice of a run's journal."""

    events: tuple[Event, ...]
    next_cursor: int
    has_more: bool
    warnings: tuple[RecoveryWarning, ...] = field(default_factory=tuple)
    total_scanned: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "events": [event.as_dict() for event in self.events],
            "next_cursor": self.next_cursor,
            "has_more": self.has_more,
            "warnings": [warning.as_dict() for warning in self.warnings],
            "count": len(self.events),
        }


class EventReader:
    """Reads one run's ``events.jsonl``."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def _iter_lines(self) -> Iterator[tuple[int, int, str]]:
        """Yield ``(line_number, byte_offset, text)`` for each line."""
        if not self.path.exists():
            return
        with self.path.open("rb") as handle:
            offset = 0
            for line_number, raw in enumerate(handle, start=1):
                yield line_number, offset, raw.decode("utf-8", errors="replace")
                offset += len(raw)

    def scan(
        self,
        *,
        after_event_id: int = 0,
        limit: int = 1000,
        event_types: Optional[Sequence[EventType | str]] = None,
    ) -> EventPage:
        """Return up to *limit* events with ``event_id > after_event_id``.

        A line is treated as a crash tail only if it is the final line **and**
        does not end with a newline. Anything else that fails to parse is mid-file
        corruption and is reported with ``is_tail=False``.
        """
        wanted: Optional[set[str]] = None
        if event_types:
            wanted = {
                item.value if isinstance(item, EventType) else str(item) for item in event_types
            }

        # Determine whether the file's last byte is a newline; without that, the
        # final line is by definition incomplete.
        ends_with_newline = True
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                ends_with_newline = handle.read(1) == b"\n"

        collected: list[Event] = []
        warnings: list[RecoveryWarning] = []
        scanned = 0
        has_more = False
        max_seen = int(after_event_id)

        lines = list(self._iter_lines())
        last_line_number = lines[-1][0] if lines else 0

        for line_number, byte_offset, text in lines:
            scanned += 1
            stripped = text.strip()
            if not stripped:
                continue
            is_final_line = line_number == last_line_number
            try:
                document = json.loads(stripped)
                event = Event.from_dict(document)
            except Exception as exc:
                warnings.append(
                    RecoveryWarning(
                        line_number=line_number,
                        byte_offset=byte_offset,
                        reason=f"{type(exc).__name__}: {exc}",
                        is_tail=bool(is_final_line and not ends_with_newline),
                    )
                )
                continue
            max_seen = max(max_seen, event.event_id)
            if event.event_id <= after_event_id:
                continue
            if wanted is not None and event.event_type.value not in wanted:
                continue
            if len(collected) >= limit:
                has_more = True
                break
            collected.append(event)

        next_cursor = collected[-1].event_id if collected else int(after_event_id)
        return EventPage(
            events=tuple(collected),
            next_cursor=next_cursor,
            has_more=has_more,
            warnings=tuple(warnings),
            total_scanned=scanned,
        )

    def tail(self, *, limit: int = 200, event_types: Optional[Sequence[EventType | str]] = None) -> EventPage:
        """Return the *last* ``limit`` matching events.

        Used for the bounded log tail in the dashboard: a researcher opening a
        run wants its most recent lines, not its first 200.
        """
        page = self.scan(after_event_id=0, limit=10**9, event_types=event_types)
        events = page.events[-limit:] if limit > 0 else ()
        return EventPage(
            events=events,
            next_cursor=events[-1].event_id if events else 0,
            has_more=len(page.events) > len(events),
            warnings=page.warnings,
            total_scanned=page.total_scanned,
        )

    def last_event_id(self) -> int:
        """Highest valid ``event_id`` in the journal (0 if empty)."""
        highest = 0
        for _, _, text in self._iter_lines():
            stripped = text.strip()
            if not stripped:
                continue
            try:
                highest = max(highest, int(json.loads(stripped)["event_id"]))
            except Exception:
                continue
        return highest

    def metric_series(
        self, *, names: Optional[Sequence[str]] = None, limit_per_name: int = 2000
    ) -> dict[str, list[dict[str, Any]]]:
        """Collect metric events grouped by metric name, bounded per series.

        Down-samples by keeping the most recent ``limit_per_name`` points of each
        series, so a very long run still renders. The dashboard states when a
        series has been trimmed rather than silently showing a partial curve.
        """
        wanted = set(names) if names else None
        series: dict[str, list[dict[str, Any]]] = {}
        page = self.scan(after_event_id=0, limit=10**9, event_types=[EventType.METRIC])
        for event in page.events:
            if event.name is None:
                continue
            if wanted is not None and event.name not in wanted:
                continue
            series.setdefault(event.name, []).append(
                {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "step": event.step,
                    "step_type": event.step_type,
                    "value": event.value,
                    "phase": event.phase,
                    "unit": event.unit,
                }
            )
        return {name: points[-limit_per_name:] for name, points in series.items()}

    def resource_samples(self, *, limit: int = 2000) -> list[dict[str, Any]]:
        """Most recent resource samples, newest last."""
        page = self.scan(after_event_id=0, limit=10**9, event_types=[EventType.RESOURCE])
        samples = [
            {"event_id": event.event_id, "timestamp": event.timestamp, **dict(event.payload)}
            for event in page.events
        ]
        return samples[-limit:]
