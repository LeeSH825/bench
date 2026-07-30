"""CPU, memory, and disk telemetry.

``psutil`` is an **optional** dependency. Without it this collector reports
``available() == False`` and the sample simply has no CPU fields — the run,
the API, and the dashboard all keep working. That is deliberate: adding a hard
dependency to the training environment for a progress bar is not a good trade
(design doc: dependency policy).

Process-tree aggregation (acceptance T-03): a training run's real CPU and RSS
usage includes DataLoader workers, so the collector walks the worker's children
rather than reporting the parent alone.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Optional

try:  # pragma: no cover - import guard
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil is optional
    psutil = None  # type: ignore


class CpuCollector:
    """Process-tree and system CPU/RAM/disk readings."""

    name = "cpu"

    def __init__(self, *, pid: Optional[int] = None, run_dir: Optional[Path] = None):
        self.pid = int(pid or os.getpid())
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self._process: Any = None
        if psutil is not None:
            try:
                self._process = psutil.Process(self.pid)
                # psutil's cpu_percent is measured between successive calls; the
                # first call always returns 0.0. Prime it here so the first real
                # sample carries a meaningful number.
                self._process.cpu_percent(interval=None)
            except Exception:
                self._process = None

    def available(self) -> bool:
        return psutil is not None

    def _process_tree(self) -> tuple[Optional[float], Optional[int], Optional[int]]:
        if self._process is None:
            return None, None, None
        try:
            members = [self._process, *self._process.children(recursive=True)]
        except Exception:
            return None, None, None
        cpu_total = 0.0
        rss_total = 0
        counted = 0
        for member in members:
            try:
                cpu_total += float(member.cpu_percent(interval=None))
                rss_total += int(member.memory_info().rss)
                counted += 1
            except Exception:
                # A child that exited between listing and reading is normal.
                continue
        if counted == 0:
            return None, None, None
        return cpu_total, rss_total, counted

    def _disk(self) -> tuple[Optional[int], Optional[int]]:
        run_dir_bytes: Optional[int] = None
        free_bytes: Optional[int] = None
        if self.run_dir is not None:
            try:
                free_bytes = int(shutil.disk_usage(self.run_dir).free)
            except Exception:
                free_bytes = None
            try:
                run_dir_bytes = sum(
                    path.stat().st_size for path in self.run_dir.rglob("*") if path.is_file()
                )
            except Exception:
                run_dir_bytes = None
        return run_dir_bytes, free_bytes

    def collect(self) -> dict[str, Any]:
        assert psutil is not None  # guarded by available()
        cpu_percent, rss_bytes, process_count = self._process_tree()
        run_dir_bytes, free_bytes = self._disk()
        try:
            virtual_memory = psutil.virtual_memory()
            system_ram_used = int(virtual_memory.total - virtual_memory.available)
            system_ram_total = int(virtual_memory.total)
        except Exception:
            system_ram_used = None
            system_ram_total = None
        try:
            system_cpu = float(psutil.cpu_percent(interval=None))
        except Exception:
            system_cpu = None
        return {
            "process_tree_cpu_percent": cpu_percent,
            "process_tree_rss_bytes": rss_bytes,
            "process_count": process_count,
            "system_cpu_percent": system_cpu,
            "system_ram_used_bytes": system_ram_used,
            "system_ram_total_bytes": system_ram_total,
            "run_dir_bytes": run_dir_bytes,
            "disk_free_bytes": free_bytes,
        }


def psutil_available() -> bool:
    return psutil is not None


def process_start_time(pid: int) -> Optional[float]:
    """Creation time of *pid* as a Unix timestamp, or ``None``.

    Together with the PID this forms the identity used to defend against PID
    reuse (acceptance P-06). Falls back to ``/proc`` so the defence still works
    on Linux without psutil.
    """
    if psutil is not None:
        try:
            return float(psutil.Process(pid).create_time())
        except Exception:
            return None
    # /proc/<pid>/stat field 22 is starttime in clock ticks since boot.
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
            content = handle.read()
        # The comm field may contain spaces inside parentheses; split after it.
        tail = content[content.rindex(")") + 2:].split()
        ticks = float(tail[19])
        hertz = os.sysconf("SC_CLK_TCK")
        with open("/proc/stat", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("btime"):
                    boot_time = float(line.split()[1])
                    return boot_time + ticks / hertz
    except Exception:
        return None
    return None


def process_alive(pid: int) -> bool:
    """Whether *pid* is a live process (says nothing about its identity).

    A **zombie** counts as dead. This is not pedantry: when the manager launches
    a worker and the worker is SIGKILLed, the worker becomes a zombie child of
    the manager until it is reaped. ``psutil.pid_exists`` and ``kill(pid, 0)``
    both report a zombie as existing, so without this check a killed run would
    look alive forever and would never be classified as ORPHANED.
    """
    if pid <= 0:
        return False
    if psutil is not None:
        try:
            process = psutil.Process(pid)
            return process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but is owned by another user — not ours to inspect further.
        return True
    except Exception:
        return False
    # /proc/<pid>/stat field 3 is the process state; 'Z' is a zombie.
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
            content = handle.read()
        state = content[content.rindex(")") + 2:].split()[0]
        return state != "Z"
    except Exception:
        return True
