"""Resource telemetry: CPU/RAM/disk and NVIDIA GPU sampling."""

from __future__ import annotations

from typing import Optional, Sequence

from .base import Collector, GpuSample, ResourceSample, TelemetrySampler  # noqa: F401
from .cpu import CpuCollector, process_alive, process_start_time, psutil_available  # noqa: F401
from .nvidia import NvidiaCollector, gpu_inventory  # noqa: F401


def default_collectors(
    *,
    pid: Optional[int] = None,
    run_dir: Optional[object] = None,
    device: Optional[str] = None,
) -> list[Collector]:
    """Build the collector set appropriate for *device*.

    The GPU collector is only added when the run actually targets CUDA. Sampling
    a GPU that the run does not use would attribute unrelated load to it.
    """
    collectors: list[Collector] = [CpuCollector(pid=pid, run_dir=run_dir)]  # type: ignore[arg-type]
    device_text = str(device or "").strip().lower()
    if device_text.startswith("cuda"):
        index = 0
        if ":" in device_text:
            try:
                index = int(device_text.split(":", 1)[1])
            except ValueError:
                index = 0
        gpu_collector = NvidiaCollector(device_index=index, pid=pid)
        if gpu_collector.available():
            collectors.append(gpu_collector)
    return collectors


__all__ = [
    "Collector",
    "CpuCollector",
    "GpuSample",
    "NvidiaCollector",
    "ResourceSample",
    "TelemetrySampler",
    "default_collectors",
    "gpu_inventory",
    "process_alive",
    "process_start_time",
    "psutil_available",
]
