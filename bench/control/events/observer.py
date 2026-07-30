"""The observer contract shared by the runner and model adapters.

This is the *only* sanctioned way for training code to report progress. It is
deliberately tiny — four methods — because every adapter has to implement
against it, and a wide interface guarantees drift (risk R-14).

Two implementations ship here:

* :class:`NullObserver` — does nothing. This is the default, so an adapter can
  call observer methods unconditionally and remain perfectly usable under the
  existing CLI, where no control plane is present. Instrumenting an adapter must
  never make it depend on the control plane.
* :class:`JournalObserver` — writes to a run's event journal and mirrors
  progress into the registry.

What this contract is *not*: it is not a metrics store, and stdout is never a
substitute for it. If an adapter has no observer wiring, its coverage is
reported as such in `bench.control.capabilities` — parsing its stdout to invent
metrics is forbidden (DND-006).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from ..registry.sqlite import SqliteRegistry
from .schema import EventType
from .writer import EventWriter


@runtime_checkable
class RunObserver(Protocol):
    """Minimal reporting surface implemented by the control plane.

    Adapters receive an object satisfying this protocol and call it during
    training. All methods must be safe to call at high frequency and must never
    raise into the training loop.
    """

    def status(self, state: str, *, phase: Optional[str] = None, message: Optional[str] = None, **payload: Any) -> None:
        """Report a lifecycle/phase transition."""

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
    ) -> None:
        """Report one scalar measurement on an explicit step axis."""

    def log(self, message: str, *, level: str = "INFO", phase: Optional[str] = None, **payload: Any) -> None:
        """Report a human-readable message."""

    def artifact(self, *, kind: str, uri: str, sha256: Optional[str] = None, bytes_: int = 0, **payload: Any) -> None:
        """Report that an output file was produced."""


class NullObserver:
    """No-op observer. The default everywhere.

    Its existence is what lets adapter instrumentation be additive: an adapter
    calls ``self._observer.metric(...)`` and, under the legacy CLI, nothing
    happens and nothing is imported.
    """

    def status(self, state: str, *, phase: Optional[str] = None, message: Optional[str] = None, **payload: Any) -> None:
        return None

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
    ) -> None:
        return None

    def log(self, message: str, *, level: str = "INFO", phase: Optional[str] = None, **payload: Any) -> None:
        return None

    def artifact(self, *, kind: str, uri: str, sha256: Optional[str] = None, bytes_: int = 0, **payload: Any) -> None:
        return None


class JournalObserver:
    """Writes observations to a run's event journal.

    Failure isolation: every method swallows its own exceptions. A full disk or a
    transient SQLite lock must degrade observability, never kill a training run
    that is otherwise healthy. Swallowed failures are counted in
    :attr:`dropped_events` and surfaced once as a warning event, so the loss is
    visible rather than silent.

    Registry mirroring is *throttled*: the journal takes every metric, but the
    registry's ``global_step`` column is only refreshed every
    ``registry_update_interval`` metric events. Writing a SQLite row per
    optimizer step would make the registry the training bottleneck.
    """

    def __init__(
        self,
        writer: EventWriter,
        *,
        registry: Optional[SqliteRegistry] = None,
        run_id: Optional[str] = None,
        registry_update_interval: int = 25,
    ):
        self._writer = writer
        self._registry = registry
        self._run_id = run_id or writer.run_id
        self._registry_update_interval = max(1, int(registry_update_interval))
        self._metric_count = 0
        self._latest_step: Optional[int] = None
        self._current_phase: Optional[str] = None
        self.dropped_events = 0
        self._reported_drop = False

    # -- internals -----------------------------------------------------------

    def _on_drop(self, exc: BaseException) -> None:
        self.dropped_events += 1
        if self._reported_drop:
            return
        self._reported_drop = True
        try:
            self._writer.warning(
                "observer dropped at least one event; observability is degraded but the "
                f"run continues. First failure: {type(exc).__name__}: {exc}"
            )
        except Exception:
            # If even the warning cannot be written, there is nothing further to
            # do without endangering the run.
            pass

    def _mirror_progress(self, *, phase: Optional[str], step: Optional[int], force: bool) -> None:
        if self._registry is None:
            return
        if not force and (self._metric_count % self._registry_update_interval) != 0:
            return
        try:
            self._registry.update_progress(
                self._run_id,
                phase=phase or self._current_phase,
                global_step=step if step is not None else self._latest_step,
                last_event_id=self._writer.last_event_id,
            )
        except Exception as exc:
            self._on_drop(exc)

    # -- RunObserver ---------------------------------------------------------

    def status(self, state: str, *, phase: Optional[str] = None, message: Optional[str] = None, **payload: Any) -> None:
        if phase:
            self._current_phase = phase
        try:
            self._writer.status(state, phase=phase, message=message, **payload)
        except Exception as exc:
            self._on_drop(exc)
            return
        # Status changes are rare and meaningful — always mirror them.
        self._mirror_progress(phase=phase, step=None, force=True)

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
    ) -> None:
        if step is not None and step_type == "global_step":
            self._latest_step = int(step)
        try:
            self._writer.metric(
                name,
                value,
                step=step,
                step_type=step_type,
                phase=phase or self._current_phase,
                unit=unit,
                **payload,
            )
        except Exception as exc:
            self._on_drop(exc)
            return
        self._metric_count += 1
        self._mirror_progress(phase=phase, step=step, force=False)

    def log(self, message: str, *, level: str = "INFO", phase: Optional[str] = None, **payload: Any) -> None:
        try:
            self._writer.log(message, level=level, phase=phase or self._current_phase, **payload)
        except Exception as exc:
            self._on_drop(exc)

    def artifact(self, *, kind: str, uri: str, sha256: Optional[str] = None, bytes_: int = 0, **payload: Any) -> None:
        try:
            self._writer.artifact(kind=kind, uri=uri, sha256=sha256, bytes_=bytes_, **payload)
        except Exception as exc:
            self._on_drop(exc)

    # -- extras used by the worker (not part of the adapter-facing protocol) --

    def resource(self, sample: Mapping[str, Any]) -> None:
        try:
            self._writer.resource(sample)
        except Exception as exc:
            self._on_drop(exc)

    def warning(self, message: str, *, phase: Optional[str] = None, **payload: Any) -> None:
        try:
            self._writer.warning(message, phase=phase or self._current_phase, **payload)
        except Exception as exc:
            self._on_drop(exc)

    def failure(self, message: str, *, phase: Optional[str] = None, **payload: Any) -> None:
        try:
            self._writer.failure(message, phase=phase or self._current_phase, **payload)
        except Exception as exc:
            self._on_drop(exc)

    def flush_progress(self) -> None:
        """Force a registry progress mirror (called at phase boundaries)."""
        self._mirror_progress(phase=self._current_phase, step=self._latest_step, force=True)


#: Process-global observer, set by the worker so that adapter code deep in the
#: call stack can report without every intermediate function growing a parameter.
#:
#: This is a pragmatic concession: threading an observer through
#: `run_suite.run_one` → adapter → inner loop would be a large, risky refactor of
#: a 2974-line module. The global defaults to NullObserver, so the legacy CLI is
#: entirely unaffected.
_ACTIVE_OBSERVER: RunObserver = NullObserver()


def set_active_observer(observer: Optional[RunObserver]) -> None:
    """Install the process-wide observer (or reset it with ``None``)."""
    global _ACTIVE_OBSERVER
    _ACTIVE_OBSERVER = observer if observer is not None else NullObserver()


def active_observer() -> RunObserver:
    """Return the process-wide observer; never ``None``."""
    return _ACTIVE_OBSERVER
