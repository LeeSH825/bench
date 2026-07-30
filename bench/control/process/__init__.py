"""Worker process supervision: launch, signals, exit codes, orphan detection."""

from __future__ import annotations

from .executors import ExecutionError, Executor, SuiteExecutor, SyntheticExecutor, build_executor  # noqa: F401
from .manager import (  # noqa: F401
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    LaunchResult,
    OrphanCandidate,
    WorkerManager,
)
from .signals import ExitCode, SignalHandler, describe_exit_code, signal_process_group  # noqa: F401

__all__ = [
    "DEFAULT_HEARTBEAT_INTERVAL_SECONDS",
    "DEFAULT_HEARTBEAT_TIMEOUT_SECONDS",
    "ExecutionError",
    "ExitCode",
    "Executor",
    "LaunchResult",
    "OrphanCandidate",
    "SignalHandler",
    "SuiteExecutor",
    "SyntheticExecutor",
    "WorkerManager",
    "build_executor",
    "describe_exit_code",
    "signal_process_group",
]
