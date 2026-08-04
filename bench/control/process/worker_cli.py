"""Worker entry point.

    python -m bench.control.process.worker_cli \
        --run-id <run_id> \
        --registry <sqlite_path> \
        --run-spec <resolved_run_spec.json>

Responsibilities, in order:

1. load and verify the resolved run spec,
2. take the run from STARTING to RUNNING in the registry,
3. start the heartbeat thread and the telemetry sampler,
4. install the process-wide observer so instrumented adapters can report,
5. execute the workload,
6. record exactly one terminal transition, then exit with the matching code.

Failure taxonomy, which is the point of this module:

* **ordinary failure** — the workload raised. The worker catches it, writes
  ``failure.json`` plus a traceback artifact, emits a ``failure`` event,
  transitions to FAILED, and exits 40. The run's outcome is fully recorded.
* **abrupt death** — SIGKILL, OOM-killer, power loss. The worker records
  *nothing*, because it is gone. The run stays RUNNING with a frozen heartbeat
  until the manager's orphan detector verifies the PID is absent and marks it
  ORPHANED. It is never mistaken for COMPLETED, which is the invariant that
  matters (design doc 06, P-05).

This module deliberately does **not** implement graceful stop or checkpointing.
On SIGINT/SIGTERM it records CANCELLED and exits 70; it does not pretend to have
saved resumable state.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Optional

from ..allocation import RunLocation, atomic_write_text
from ..config.resolver import resolved_from_dict
from ..config.schema import ResolvedRunSpec
from ..events.observer import JournalObserver, set_active_observer
from ..events.writer import EventWriter
from ..identity import ExperimentId, RunId
from ..registry.schema import RunState
from ..registry.sqlite import SqliteRegistry
from ..telemetry import TelemetrySampler, default_collectors
from .executors import ExecutionError, build_executor
from ..checkpoints import CheckpointService
from ..checkpoints.lifecycle import settle_graceful_stop
from ..checkpoints.stop import StopCoordinator
from ..training_path import TRAINING_PATH_CONTRACT_VERSION

#: States the runner may already have reached on its own. The worker must not
#: overwrite these with COMPLETED.
_RUNNER_SETTLED_STATES = frozenset({RunState.INTERRUPTED, RunState.FAILED})


def _build_interrupt_settlement(
    *, spec, location, registry, writer, control_root, stop_coordinator
):
    """Return the callback the runner invokes at an interrupted safe boundary.

    Closes over the registry, event writer and checkpoint service so the runner
    itself stays free of control-plane wiring. The live adapter is supplied by
    the caller, because only it has one in scope.
    """

    def _settle(*, adapter, cursor, progress, batch_plan):
        from ..checkpoints import capture_rng

        service = CheckpointService(
            location.root, registry=registry, control_root=control_root
        )
        outcome = settle_graceful_stop(
            run_id=spec.run_id.value,
            registry=registry,
            service=service,
            cursor=cursor,
            adapter=adapter,
            rng=capture_rng(),
            identity={
                "model_id": spec.model_id.value,
                "implementation_id": spec.implementation_id.value,
                "variant_id": spec.variant_id.value,
            },
            action_id=stop_coordinator.decision.action_id,
            structural_config_hash=spec.structural_config_hash,
            dataset_fingerprint=str(spec.draft.dataset.fingerprint or ""),
            batch_plan=batch_plan,
            resolved_run_spec=spec.as_dict(),
            progress=progress.as_dict(),
            event_writer=writer,
            training_path_id=str(spec.draft.execution.training_path_id),
            training_path_contract_version=TRAINING_PATH_CONTRACT_VERSION,
        )
        return outcome.as_dict()

    return _settle
from .signals import ExitCode, SignalHandler


class HeartbeatThread(threading.Thread):
    """Refreshes the run's heartbeat until stopped.

    Daemon thread: if the main thread dies unexpectedly, the heartbeat must stop
    with it. A heartbeat that outlives the work it claims to represent is worse
    than no heartbeat, because it would keep a dead run looking alive.
    """

    def __init__(
        self,
        registry: SqliteRegistry,
        run_id: str,
        *,
        worker_instance_id: Optional[str],
        interval_seconds: float,
    ):
        super().__init__(name=f"heartbeat-{run_id[:8]}", daemon=True)
        self.registry = registry
        self.run_id = run_id
        self.worker_instance_id = worker_instance_id
        self.interval_seconds = max(1.0, float(interval_seconds))
        # NOT `self._stop`: threading.Thread has a private `_stop()` method that
        # `join()` calls internally, and shadowing it with an Event makes every
        # join raise "'Event' object is not callable".
        self._stop_event = threading.Event()
        self.beats = 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.registry.record_heartbeat(
                    self.run_id, worker_instance_id=self.worker_instance_id
                )
                self.beats += 1
            except Exception:
                # A transient SQLite lock must not kill the run; the next beat
                # will catch up, and prolonged failure surfaces as staleness.
                pass
            self._stop_event.wait(self.interval_seconds)

    def stop(self, *, timeout: float = 3.0) -> None:
        self._stop_event.set()
        self.join(timeout=timeout)


def _optional_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _load_spec(path: Path) -> ResolvedRunSpec:
    document = json.loads(path.read_text(encoding="utf-8"))
    return resolved_from_dict(document)


def _write_failure(
    location: RunLocation,
    *,
    run_id: str,
    phase: Optional[str],
    exc: BaseException,
) -> tuple[Path, Path]:
    """Persist a failure record and the full traceback."""
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    traceback_path = location.artifacts_dir / "traceback.txt"
    atomic_write_text(traceback_path, text, tmp_dir=location.tmp_dir)
    failure = {
        "run_id": run_id,
        "phase": phase,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback_artifact": str(traceback_path.relative_to(location.root)),
    }
    failure_path = location.failure_path
    atomic_write_text(
        failure_path, json.dumps(failure, indent=2, sort_keys=True), tmp_dir=location.tmp_dir
    )
    return failure_path, traceback_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.control.process.worker_cli",
        description="Execute one benchmark run as a supervised, detached worker process.",
    )
    parser.add_argument("--run-id", required=True, help="Run id assigned by the control plane")
    parser.add_argument("--registry", required=True, help="Path to the SQLite registry")
    parser.add_argument("--run-spec", required=True, help="Path to resolved_run_spec.json")
    parser.add_argument(
        "--executor",
        default=None,
        help="Executor name (default: from the spec's bench_context, else 'suite')",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=float(os.environ.get("BENCH_CONTROL_HEARTBEAT_INTERVAL", "10")),
        help="Seconds between heartbeats",
    )
    parser.add_argument(
        "--fail-at-step",
        type=int,
        # Also readable from the environment so the manager (whose argv is
        # fixed by the worker CLI contract) can inject it for failure tests.
        default=_optional_int(os.environ.get("BENCH_CONTROL_FAIL_AT_STEP")),
        help="Synthetic executor only: raise an ordinary failure at this optimizer step",
    )
    parser.add_argument(
        "--step-sleep",
        type=float,
        default=float(os.environ.get("BENCH_CONTROL_STEP_SLEEP", "0") or 0),
        help=(
            "Synthetic executor only: seconds to pause per optimizer step. Used to make a "
            "demo run last long enough to observe live, and by the live-update tests."
        ),
    )
    args = parser.parse_args(argv)

    run_id = str(args.run_id)
    spec_path = Path(args.run_spec).expanduser().resolve()

    # ---- worker-protocol phase: nothing below this is recoverable ---------- #
    try:
        registry = SqliteRegistry(args.registry, migrate=False)
        spec = _load_spec(spec_path)
        if spec.run_id.value != run_id:
            raise ExecutionError(
                f"--run-id {run_id} does not match the spec's run_id {spec.run_id.value}"
            )
        record = registry.get_run(run_id)
        if record is None:
            raise ExecutionError(f"run {run_id} is not present in the registry")
        location = RunLocation(
            run_id=RunId(run_id),
            experiment_id=ExperimentId(spec.draft.experiment.experiment_id),
            root=Path(record.run_dir),
        )
    except Exception as exc:  # noqa: BLE001 - last-resort protocol failure
        print(f"[worker] protocol failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return int(ExitCode.WORKER_PROTOCOL_FAILURE)

    worker_instance_id = os.environ.get("BENCH_CONTROL_WORKER_INSTANCE_ID")
    writer = EventWriter(location.events_path, run_id)
    observer = JournalObserver(writer, registry=registry, run_id=run_id)
    set_active_observer(observer)

    heartbeat = HeartbeatThread(
        registry,
        run_id,
        worker_instance_id=worker_instance_id,
        interval_seconds=args.heartbeat_interval,
    )
    sampler: Optional[TelemetrySampler] = None
    signal_handler = SignalHandler().install()

    exit_code = int(ExitCode.COMPLETED)
    current_phase: Optional[str] = "setup"

    try:
        record = registry.transition(
            run_id,
            to_state=RunState.RUNNING,
            expected_state_version=record.state_version,
            actor="worker",
            reason="worker started",
            fields={
                "pid": os.getpid(),
                "process_group_id": os.getpgrp(),
                "host": socket.gethostname(),
                "phase": "setup",
            },
        )
        # Record the first heartbeat synchronously so a run is never RUNNING with
        # a null heartbeat, which the orphan detector would flag immediately.
        registry.record_heartbeat(run_id, worker_instance_id=worker_instance_id)
        heartbeat.start()

        writer.status(
            "RUNNING",
            phase="setup",
            message=f"worker pid={os.getpid()} pgid={os.getpgrp()} host={socket.gethostname()}",
        )

        if spec.draft.telemetry.enabled:
            sampler = TelemetrySampler(
                run_id=run_id,
                collectors=default_collectors(
                    pid=os.getpid(), run_dir=location.root, device=spec.draft.runtime.device
                ),
                sink=lambda sample: observer.resource(sample.as_dict()),
                interval_seconds=spec.draft.telemetry.interval_seconds,
                pid=os.getpid(),
            )
            sampler.start()

        executor_name = str(
            args.executor or spec.draft.bench_context.get("executor") or "suite"
        )
        executor_kwargs: dict[str, Any] = {}
        if executor_name == "synthetic":
            if args.fail_at_step is not None:
                executor_kwargs["fail_at_step"] = int(args.fail_at_step)
            if args.step_sleep and float(args.step_sleep) > 0:
                executor_kwargs["step_sleep_seconds"] = float(args.step_sleep)
        if executor_name == "suite":
            # A resumed child carries its lineage on the run row. The worker
            # reads it here and hands the checkpoint to the runner, so the
            # child restores before its first new optimizer update. A fresh
            # run has no lineage and this is empty (continuation gate B1).
            record = registry.get_run(spec.run_id.value)
            extra: dict[str, Any] = {}
            if record is not None and record.resumed_from_checkpoint_id:
                parent = registry.get_run(str(record.resumed_from_run_id))
                if parent is None:
                    raise ExecutionError(
                        f"run {spec.run_id.value} claims to resume from "
                        f"{record.resumed_from_run_id} which does not exist"
                    )
                extra = {
                    "resume_checkpoint_id": record.resumed_from_checkpoint_id,
                    "parent_run_dir": str(parent.run_dir),
                    "control_root": str(Path(args.registry).resolve().parent),
                }
                observer.status(
                    "RESUME_RESTORE",
                    phase="setup",
                    message=(
                        f"resuming from checkpoint {record.resumed_from_checkpoint_id} "
                        f"of parent run {record.resumed_from_run_id}"
                    ),
                    resumed_from_run_id=record.resumed_from_run_id,
                    resumed_from_checkpoint_id=record.resumed_from_checkpoint_id,
                    training_path_id=spec.draft.execution.training_path_id,
                )
            # Graceful stop is wired only for the certified resumable path.
            # legacy_train_v1 and not_applicable get no stop callback at all,
            # because their loops have no safe boundary to honour one at.
            training_path_id = str(spec.draft.execution.training_path_id)
            stop_coordinator = None
            if training_path_id == "control_resumable_v1":
                stop_coordinator = StopCoordinator(
                    run_id=spec.run_id.value,
                    registry=registry,
                    install_signal_handlers=False,
                )
                extra["stop_requested"] = stop_coordinator
                extra["on_interrupt"] = _build_interrupt_settlement(
                    spec=spec,
                    location=location,
                    registry=registry,
                    writer=writer,
                    control_root=Path(args.registry).resolve().parent,
                    stop_coordinator=stop_coordinator,
                )
            executor_kwargs["extra_contract"] = extra
        executor = build_executor(executor_name, **executor_kwargs)

        current_phase = "execute"
        result = executor.execute(spec, location=location, observer=observer)

        if signal_handler.triggered:
            raise KeyboardInterrupt(
                f"termination signal {signal_handler.received} received during execution"
            )

        observer.flush_progress()

        # A graceful stop settles its own terminal state (INTERRUPTED, or
        # FAILED if the checkpoint could not be written). Falling through to
        # the COMPLETED handler would overwrite it and report an interrupted
        # run as finished.
        settled = registry.get_run(run_id)
        if settled is not None and settled.state in _RUNNER_SETTLED_STATES:
            exit_code = (
                int(ExitCode.INTERRUPTED_WITH_CHECKPOINT)
                if settled.state is RunState.INTERRUPTED
                else int(ExitCode.CHECKPOINT_WRITE_FAILURE)
            )
            print(
                f"[worker] run {run_id} settled by graceful stop: "
                f"{settled.state.value} exit={exit_code}"
            )
            return exit_code

        writer.status("COMPLETED", phase="report", message="workload finished")
        record = registry.get_run(run_id) or record
        registry.transition(
            run_id,
            to_state=RunState.COMPLETED,
            expected_state_version=record.state_version,
            actor="worker",
            reason="workload completed",
            fields={
                "exit_code": int(ExitCode.COMPLETED),
                "terminal_reason": "completed",
                "phase": "report",
                "last_event_id": writer.last_event_id,
            },
        )
        print(f"[worker] run {run_id} completed: {json.dumps(dict(result), default=str)}")
        exit_code = int(ExitCode.COMPLETED)

    except (KeyboardInterrupt, SystemExit) as exc:
        # Deliberate external termination. This tranche does NOT checkpoint here.
        exit_code = int(ExitCode.EXTERNAL_TERMINATION)
        _write_failure(location, run_id=run_id, phase=current_phase, exc=exc)
        try:
            writer.status(
                "TERMINATED",
                phase=current_phase,
                message=(
                    "worker terminated by signal; no checkpoint was written because "
                    "graceful stop is not implemented in this build"
                ),
            )
            record = registry.get_run(run_id) or record
            if record.state not in (RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED):
                # The state machine allows RUNNING → FAILED but not RUNNING →
                # CANCELLED: CANCELLED means "never executed". A signalled run
                # did execute, so FAILED with an explicit terminal_reason is the
                # honest classification.
                registry.transition(
                    run_id,
                    to_state=RunState.FAILED,
                    expected_state_version=record.state_version,
                    actor="worker",
                    reason="terminated by signal",
                    fields={
                        "exit_code": exit_code,
                        "terminal_reason": "external_termination",
                        "error_summary": str(exc) or "terminated by signal",
                        "last_event_id": writer.last_event_id,
                    },
                )
        except Exception:
            pass

    except BaseException as exc:  # noqa: BLE001 - ordinary workload failure
        exit_code = int(ExitCode.EXECUTION_FAILURE)
        failure_path, traceback_path = _write_failure(
            location, run_id=run_id, phase=current_phase, exc=exc
        )
        summary = f"{type(exc).__name__}: {exc}"
        try:
            observer.failure(summary, phase=current_phase)
            writer.artifact(
                kind="failure",
                uri=str(traceback_path.relative_to(location.root)),
                bytes_=traceback_path.stat().st_size,
            )
            writer.status("FAILED", phase=current_phase, message=summary)
            record = registry.get_run(run_id) or record
            registry.transition(
                run_id,
                to_state=RunState.FAILED,
                expected_state_version=record.state_version,
                actor="worker",
                reason="workload raised",
                fields={
                    "exit_code": exit_code,
                    "terminal_reason": "execution_failure",
                    "error_summary": summary[:2000],
                    "last_event_id": writer.last_event_id,
                },
            )
        except Exception:
            exit_code = int(ExitCode.WORKER_PROTOCOL_FAILURE)
        print(f"[worker] run {run_id} FAILED: {summary}", file=sys.stderr)
        traceback.print_exc()

    finally:
        if sampler is not None:
            sampler.stop()
        heartbeat.stop()
        signal_handler.restore()
        set_active_observer(None)
        if worker_instance_id:
            try:
                registry.finish_worker(
                    worker_instance_id, state="EXITED", exit_code=int(exit_code)
                )
            except Exception:
                pass
        try:
            writer.close()
        except Exception:
            pass
        try:
            registry.close()
        except Exception:
            pass

    return int(exit_code)


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
