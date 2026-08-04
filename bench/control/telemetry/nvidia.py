"""NVIDIA GPU telemetry via NVML, falling back to ``nvidia-smi``.

Backend selection, in order:

1. **NVML** (``pynvml`` / ``nvidia-ml-py``) — preferred: gives per-process
   memory attribution, temperature, and power in one cheap in-process call.
2. **``nvidia-smi``** — fallback when the NVML bindings are absent. Parsed from
   ``--format=csv,noheader,nounits`` output, invoked as an argv list (never a
   shell string, per design doc 03 §17).
3. **Unavailable** — CPU-only host. ``available()`` is ``False`` and no GPU
   section appears in the sample.

Attribution honesty (design doc 03 §8): NVML reports *whole-device*
utilization; it cannot split utilization per process. It *can* report per-process
memory. The sample therefore carries both, plus ``attribution_quality``:

* ``memory_only``   — device utilization is shared, memory is attributed to this PID
* ``device_only``   — nothing could be attributed to this PID (nvidia-smi fallback,
                      or a container that hides process lists)
* ``unavailable``   — no GPU reading at all
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Optional

from .base import GpuSample

try:  # pragma: no cover - import guard
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore


def _torch_allocator_bytes(device_index: int) -> tuple[Optional[int], Optional[int]]:
    """PyTorch caching-allocator numbers, if torch is loaded with CUDA.

    Imported lazily and defensively: the API/dashboard processes have no reason
    to import torch, and doing so would cost seconds of startup.
    """
    torch = __import__("sys").modules.get("torch")
    if torch is None:
        return None, None
    try:
        if not torch.cuda.is_available():  # type: ignore[union-attr]
            return None, None
        allocated = int(torch.cuda.memory_allocated(device_index))  # type: ignore[union-attr]
        reserved = int(torch.cuda.memory_reserved(device_index))  # type: ignore[union-attr]
        return allocated, reserved
    except Exception:
        return None, None


class NvidiaCollector:
    """GPU readings for one device index."""

    name = "nvidia"

    def __init__(self, *, device_index: int = 0, pid: Optional[int] = None):
        self.device_index = int(device_index)
        self.pid = int(pid or os.getpid())
        self._backend: Optional[str] = None
        self._handle: Any = None
        self._init_backend()

    # -- backend setup -------------------------------------------------------

    def _init_backend(self) -> None:
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                self._backend = "nvml"
                return
            except Exception:
                self._handle = None
        if _nvidia_smi_path() is not None:
            self._backend = "nvidia_smi"
            return
        self._backend = None

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def available(self) -> bool:
        return self._backend is not None

    # -- collection ----------------------------------------------------------

    def collect(self) -> dict[str, Any]:
        if self._backend == "nvml":
            return {"gpu": self._collect_nvml()}
        if self._backend == "nvidia_smi":
            return {"gpu": self._collect_smi()}
        return {}

    def _collect_nvml(self) -> GpuSample:
        assert pynvml is not None
        handle = self._handle

        def _try(function: Any, *args: Any) -> Any:
            try:
                return function(*args)
            except Exception:
                return None

        memory = _try(pynvml.nvmlDeviceGetMemoryInfo, handle)
        utilization = _try(pynvml.nvmlDeviceGetUtilizationRates, handle)
        temperature = _try(pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU)
        power_mw = _try(pynvml.nvmlDeviceGetPowerUsage, handle)
        uuid_value = _try(pynvml.nvmlDeviceGetUUID, handle)
        name_value = _try(pynvml.nvmlDeviceGetName, handle)

        process_memory: Optional[int] = None
        attribution = "device_only"
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for entry in processes:
                if int(getattr(entry, "pid", -1)) == self.pid:
                    used = getattr(entry, "usedGpuMemory", None)
                    if used is not None:
                        process_memory = int(used)
                        attribution = "memory_only"
                    break
        except Exception:
            attribution = "device_only"

        allocated, reserved = _torch_allocator_bytes(self.device_index)
        return GpuSample(
            backend="nvml",
            device_index=self.device_index,
            device_uuid=_as_text(uuid_value),
            device_name=_as_text(name_value),
            device_utilization_percent=(float(utilization.gpu) if utilization is not None else None),
            device_memory_used_bytes=(int(memory.used) if memory is not None else None),
            device_memory_total_bytes=(int(memory.total) if memory is not None else None),
            process_memory_used_bytes=process_memory,
            temperature_c=(float(temperature) if temperature is not None else None),
            power_w=(float(power_mw) / 1000.0 if power_mw is not None else None),
            torch_allocator_allocated_bytes=allocated,
            torch_allocator_reserved_bytes=reserved,
            attribution_quality=attribution,
        )

    def _collect_smi(self) -> Optional[GpuSample]:
        binary = _nvidia_smi_path()
        if binary is None:
            return None
        query = (
            "index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        )
        result = subprocess.run(
            [
                binary,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
                f"--id={self.device_index}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        parts = [item.strip() for item in result.stdout.strip().splitlines()[0].split(",")]
        if len(parts) < 8:
            return None
        allocated, reserved = _torch_allocator_bytes(self.device_index)
        return GpuSample(
            backend="nvidia_smi",
            device_index=_as_int(parts[0]) or self.device_index,
            device_uuid=parts[1] or None,
            device_name=parts[2] or None,
            device_utilization_percent=_as_float(parts[3]),
            device_memory_used_bytes=_mib_to_bytes(_as_float(parts[4])),
            device_memory_total_bytes=_mib_to_bytes(_as_float(parts[5])),
            process_memory_used_bytes=None,
            temperature_c=_as_float(parts[6]),
            power_w=_as_float(parts[7]),
            torch_allocator_allocated_bytes=allocated,
            torch_allocator_reserved_bytes=reserved,
            # nvidia-smi's per-GPU query gives no per-process attribution.
            attribution_quality="device_only",
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _nvidia_smi_path() -> Optional[str]:
    from shutil import which

    return which("nvidia-smi")


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _as_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mib_to_bytes(value: Optional[float]) -> Optional[int]:
    return int(value * 1024 * 1024) if value is not None else None


def gpu_inventory() -> list[dict[str, Any]]:
    """Enumerate visible GPUs for the ``/system/gpus`` endpoint.

    Returns an empty list on a CPU-only host — the API reports availability
    explicitly rather than pretending a device exists.
    """
    devices: list[dict[str, Any]] = []
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            count = int(pynvml.nvmlDeviceGetCount())
            for index in range(count):
                collector = NvidiaCollector(device_index=index)
                sample = collector._collect_nvml() if collector.backend == "nvml" else None
                if sample is not None:
                    devices.append(sample.as_dict())
            if devices:
                return devices
        except Exception:
            devices = []
    binary = _nvidia_smi_path()
    if binary is None:
        return []
    try:
        result = subprocess.run(
            [binary, "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        indices = [
            int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()
        ]
    except Exception:
        return []
    for index in indices:
        sample = NvidiaCollector(device_index=index)._collect_smi()
        if sample is not None:
            devices.append(sample.as_dict())
    return devices
