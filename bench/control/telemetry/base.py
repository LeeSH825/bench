"""Resource telemetry contracts and the sampling loop.

Design rules that the collectors must honour:

* **A collector failure is never a run failure.** Every collector call is
  wrapped; the error is recorded in :attr:`ResourceSample.collector_errors` and
  sampling continues. A missing NVML library must not kill a training job.
* **Availability is explicit.** On a CPU-only host the GPU section is ``None``,
  not zeros. Zeros would be plotted as a flat line at 0 % utilization, which
  reads as "idle GPU" rather than "no GPU".
* **Whole-device and per-process metrics are separate fields**, with an
  ``attribution_quality`` flag, because NVML can usually report a process's GPU
  *memory* but not its share of GPU *utilization* (design doc 03 §8).
* **Gaps are gaps.** The sampler records its own timestamps; the UI must not
  interpolate across a collection outage (acceptance T-05).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class GpuSample:
    """One GPU device reading.

    ``device_*`` fields describe the **whole device** (shared with every other
    process on it). ``process_memory_used_bytes`` is this run's own allocation,
    when NVML can attribute it.

    ``torch_allocator_*`` are PyTorch's *caching allocator* numbers. They are
    deliberately labelled separately and are always smaller than
    ``process_memory_used_bytes``: the allocator's reserved pool, CUDA context,
    and cuDNN workspaces are not the same thing, and conflating them makes
    "why does my 4 GB model use 6 GB" impossible to answer.
    """

    backend: str
    device_index: int
    device_uuid: Optional[str] = None
    device_name: Optional[str] = None
    device_utilization_percent: Optional[float] = None
    device_memory_used_bytes: Optional[int] = None
    device_memory_total_bytes: Optional[int] = None
    process_memory_used_bytes: Optional[int] = None
    temperature_c: Optional[float] = None
    power_w: Optional[float] = None
    torch_allocator_allocated_bytes: Optional[int] = None
    torch_allocator_reserved_bytes: Optional[int] = None
    attribution_quality: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device_index": self.device_index,
            "device_uuid": self.device_uuid,
            "device_name": self.device_name,
            "device_utilization_percent": self.device_utilization_percent,
            "device_memory_used_bytes": self.device_memory_used_bytes,
            "device_memory_total_bytes": self.device_memory_total_bytes,
            "process_memory_used_bytes": self.process_memory_used_bytes,
            "temperature_c": self.temperature_c,
            "power_w": self.power_w,
            "torch_allocator_allocated_bytes": self.torch_allocator_allocated_bytes,
            "torch_allocator_reserved_bytes": self.torch_allocator_reserved_bytes,
            "attribution_quality": self.attribution_quality,
        }


@dataclass(frozen=True)
class ResourceSample:
    """One telemetry tick (design doc 03 §8)."""

    timestamp: str
    run_id: str
    pid: Optional[int] = None
    process_tree_cpu_percent: Optional[float] = None
    process_tree_rss_bytes: Optional[int] = None
    process_count: Optional[int] = None
    system_cpu_percent: Optional[float] = None
    system_ram_used_bytes: Optional[int] = None
    system_ram_total_bytes: Optional[int] = None
    gpu: Optional[GpuSample] = None
    run_dir_bytes: Optional[int] = None
    disk_free_bytes: Optional[int] = None
    collector_errors: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "pid": self.pid,
            "process_tree_cpu_percent": self.process_tree_cpu_percent,
            "process_tree_rss_bytes": self.process_tree_rss_bytes,
            "process_count": self.process_count,
            "system_cpu_percent": self.system_cpu_percent,
            "system_ram_used_bytes": self.system_ram_used_bytes,
            "system_ram_total_bytes": self.system_ram_total_bytes,
            "gpu": self.gpu.as_dict() if self.gpu is not None else None,
            "disk": {
                "run_dir_bytes": self.run_dir_bytes,
                "free_bytes": self.disk_free_bytes,
            },
            "collector_errors": list(self.collector_errors),
        }


class Collector(Protocol):
    """A source of resource readings."""

    name: str

    def available(self) -> bool:
        """Whether this collector can produce readings on this host."""

    def collect(self) -> dict[str, Any]:
        """Return a partial :class:`ResourceSample` field mapping."""


class TelemetrySampler:
    """Background thread that samples collectors at a fixed interval.

    Runs as a **daemon** thread: if the training loop exits, telemetry must not
    keep the process alive. The sink is called with a completed
    :class:`ResourceSample`; exceptions from the sink are swallowed for the same
    reason collector exceptions are.
    """

    def __init__(
        self,
        *,
        run_id: str,
        collectors: Sequence[Collector],
        sink: Callable[[ResourceSample], None],
        interval_seconds: float = 2.0,
        pid: Optional[int] = None,
    ):
        self.run_id = run_id
        self.collectors = list(collectors)
        self.sink = sink
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.pid = pid
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def sample_once(self) -> ResourceSample:
        """Collect one sample from every available collector."""
        fields: dict[str, Any] = {}
        errors: list[str] = []
        for collector in self.collectors:
            try:
                if not collector.available():
                    continue
                fields.update(collector.collect())
            except Exception as exc:
                errors.append(f"{collector.name}: {type(exc).__name__}: {exc}")
        return ResourceSample(
            timestamp=utc_now(),
            run_id=self.run_id,
            pid=self.pid,
            collector_errors=tuple(errors),
            **{key: value for key, value in fields.items() if key in _SAMPLE_FIELDS},
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sink(self.sample_once())
            except Exception:
                # Telemetry must never take down the run it is observing.
                pass
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name=f"telemetry-{self.run_id[:8]}", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            self._thread = None

    def __enter__(self) -> "TelemetrySampler":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


#: Field names a collector may contribute to a :class:`ResourceSample`.
_SAMPLE_FIELDS = frozenset(
    {
        "process_tree_cpu_percent",
        "process_tree_rss_bytes",
        "process_count",
        "system_cpu_percent",
        "system_ram_used_bytes",
        "system_ram_total_bytes",
        "gpu",
        "run_dir_bytes",
        "disk_free_bytes",
    }
)
