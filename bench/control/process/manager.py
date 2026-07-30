"""Worker process supervision.

The manager launches each run as an independent, detached OS process and then
*forgets about it*. It holds no handle the run depends on:

* ``start_new_session=True`` puts the worker in its own session and process
  group, so it survives the death of whatever launched it (acceptance P-01) and
  its DataLoader children are one killable unit (P-02);
* stdout/stderr go to **files opened by the child**, not to pipes. A pipe would
  tie the worker's lifetime to a reader — if the API process exited, the worker
  would eventually block on a full pipe buffer and hang. This is the single most
  important detail in this module;
* the command is an **argv list**, never a shell string (design doc 03 §17).

The manager never becomes the training process's parent in any meaningful sense:
after ``launch`` returns, the registry is the only link between the two.

What this module is *not*: a scheduler. There is no queue, no multi-GPU
placement, no retry policy. Those are later tranches.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..allocation import RunLocation, allocate_run_directory, write_run_spec
from ..config.schema import ResolvedRunSpec
from ..identity import ExperimentId, WorkerInstanceId, uuid7
from ..paths import control_root, registry_path
from ..registry.schema import ExperimentRecord, RunRecord, RunState, WorkerRecord
from ..registry.sqlite import SqliteRegistry, utc_now
from ..telemetry.cpu import process_alive, process_start_time
from .signals import describe_exit_code

#: A worker is considered *possibly* dead once its heartbeat is this stale.
#: Staleness alone never kills anything — it only triggers PID verification
#: (design doc 05 §9: "stale heartbeat 후 자동 kill 금지").
DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 90.0

#: How often the worker refreshes its heartbeat.
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0


@dataclass(frozen=True)
class LaunchResult:
    """Outcome of launching one worker."""

    run_id: str
    worker_instance_id: str
    pid: int
    process_group_id: int
    location: RunLocation
    command: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "worker_instance_id": self.worker_instance_id,
            "pid": self.pid,
            "process_group_id": self.process_group_id,
            "run_dir": str(self.location.root),
            "command": list(self.command),
        }


@dataclass(frozen=True)
class OrphanCandidate:
    """A run whose worker appears to have died without recording an outcome."""

    run_id: str
    state: str
    reason: str
    pid: Optional[int]
    heartbeat_at: Optional[str]
    heartbeat_age_seconds: Optional[float]
    pid_alive: bool
    pid_identity_matches: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.state,
            "reason": self.reason,
            "pid": self.pid,
            "heartbeat_at": self.heartbeat_at,
            "heartbeat_age_seconds": self.heartbeat_age_seconds,
            "pid_alive": self.pid_alive,
            "pid_identity_matches": self.pid_identity_matches,
        }


class WorkerManager:
    """Launches and adjudicates run workers."""

    def __init__(
        self,
        registry: SqliteRegistry,
        *,
        control_root_path: Optional[str | os.PathLike[str]] = None,
        python_executable: Optional[str] = None,
        repo_root: Optional[Path] = None,
    ):
        self.registry = registry
        self.control_root = control_root(control_root_path)
        self.python_executable = python_executable or sys.executable
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        # Popen handles for workers launched by *this* manager instance, kept
        # only so they can be reaped. A worker's lifetime does not depend on
        # this dict — a manager restart simply forgets the handles and the
        # workers keep running (which is the whole point of P-01).
        self._launched: dict[str, subprocess.Popen[bytes]] = {}

    def reap(self) -> dict[str, int]:
        """Reap exited child workers, returning ``{run_id: exit_status}``.

        Without this, a worker that dies while the manager lives on becomes a
        **zombie**: still listed in the process table, so liveness checks would
        report it as alive and its run would never be classified as ORPHANED.
        Called at the start of orphan detection.
        """
        reaped: dict[str, int] = {}
        for run_id, process in list(self._launched.items()):
            status = process.poll()
            if status is not None:
                reaped[run_id] = int(status)
                self._launched.pop(run_id, None)
        return reaped

    # -- launch --------------------------------------------------------------

    def prepare_run(
        self,
        spec: ResolvedRunSpec,
        *,
        experiment_name: Optional[str] = None,
    ) -> RunLocation:
        """Allocate the run directory and register the run in state CREATED.

        Allocation happens before registration on purpose: if the filesystem
        cannot accept a new run, no half-registered row is left behind.
        """
        experiment_id = ExperimentId(spec.draft.experiment.experiment_id)
        self.registry.upsert_experiment(
            ExperimentRecord(
                experiment_id=experiment_id.value,
                name=experiment_name or spec.draft.experiment.name,
                description=spec.draft.experiment.description,
                tags=tuple(spec.draft.experiment.tags),
            )
        )
        location = allocate_run_directory(
            run_id=spec.run_id, experiment_id=experiment_id, control_root=self.control_root
        )
        write_run_spec(location, spec)
        if spec.draft.original_config is not None:
            from ..allocation import atomic_write_text
            import json as _json

            atomic_write_text(
                location.original_config_path,
                _json.dumps(dict(spec.draft.original_config), indent=2, default=str),
                tmp_dir=location.tmp_dir,
            )

        self.registry.create_run(
            RunRecord(
                run_id=spec.run_id.value,
                experiment_id=experiment_id.value,
                state=RunState.CREATED,
                state_version=0,
                created_at=utc_now(),
                updated_at=utc_now(),
                model_id=spec.model_id.value,
                implementation_id=spec.implementation_id.value,
                init_id=spec.init_id.mode,
                variant_id=spec.variant_id.value,
                task_id=spec.draft.system.task_id,
                scenario_id=spec.draft.system.scenario_id,
                seed=spec.draft.runtime.seed,
                device=spec.draft.runtime.device,
                run_dir=str(location.root),
                structural_config_hash=spec.structural_config_hash,
                operational_config_hash=spec.operational_config_hash,
                resolved_spec_hash=spec.resolved_spec_hash,
            )
        )
        return location

    def build_command(self, *, run_id: str, location: RunLocation) -> list[str]:
        """Worker argv (design doc 03 §14)."""
        return [
            self.python_executable,
            "-m",
            "bench.control.process.worker_cli",
            "--run-id",
            run_id,
            "--registry",
            str(registry_path(self.control_root)),
            "--run-spec",
            str(location.resolved_spec_path),
        ]

    def launch(
        self,
        spec: ResolvedRunSpec,
        *,
        location: Optional[RunLocation] = None,
        extra_env: Optional[Mapping[str, str]] = None,
        worker_token: Optional[str] = None,
    ) -> LaunchResult:
        """Start a detached worker process for *spec*.

        The run must already be registered (via :meth:`prepare_run`). Moves the
        run through VALIDATING → QUEUED → STARTING before spawning, so a crash
        between those points is visible in the transition log rather than
        looking like a run that never existed.
        """
        run_id = spec.run_id.value
        if location is None:
            record = self.registry.get_run(run_id)
            if record is None:
                raise RuntimeError(f"run {run_id} is not registered; call prepare_run first")
            location = RunLocation(
                run_id=spec.run_id,
                experiment_id=ExperimentId(spec.draft.experiment.experiment_id),
                root=Path(record.run_dir),
            )

        record = self.registry.get_run(run_id)
        assert record is not None
        for state in (RunState.VALIDATING, RunState.QUEUED, RunState.STARTING):
            record = self.registry.transition(
                run_id,
                to_state=state,
                expected_state_version=record.state_version,
                actor="manager",
                reason="manager launch sequence",
            )

        command = self.build_command(run_id=run_id, location=location)
        token = worker_token or uuid7()
        worker_instance_id = WorkerInstanceId.new().value

        environment = dict(os.environ)
        environment.update(
            {
                "BENCH_CONTROL_ROOT": str(self.control_root),
                "BENCH_CONTROL_WORKER_TOKEN": token,
                "BENCH_CONTROL_WORKER_INSTANCE_ID": worker_instance_id,
                # Unbuffered stdio so the dashboard's log tail is not held
                # hostage by a 4 KB pipe buffer that never fills on a quiet run.
                "PYTHONUNBUFFERED": "1",
            }
        )
        if extra_env:
            environment.update({str(k): str(v) for k, v in extra_env.items()})

        # Open the log files here but let the child own them. Using files (not
        # PIPEs) is what makes the worker independent of this process.
        stdout_handle = location.stdout_path.open("ab")
        stderr_handle = location.stderr_path.open("ab")
        try:
            process = subprocess.Popen(  # noqa: S603 - argv list, never shell
                command,
                cwd=str(self.repo_root),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()

        self._launched[run_id] = process
        pid = int(process.pid)
        try:
            pgid = os.getpgid(pid)
        except OSError:
            # The child may already have exited; its own pid is its group leader.
            pgid = pid

        self.registry.register_worker(
            WorkerRecord(
                worker_instance_id=worker_instance_id,
                run_id=run_id,
                host=socket.gethostname(),
                pid=pid,
                process_group_id=pgid,
                process_start_time=float(process_start_time(pid) or 0.0),
                worker_token=token,
                started_at=utc_now(),
                state="STARTING",
            )
        )
        return LaunchResult(
            run_id=run_id,
            worker_instance_id=worker_instance_id,
            pid=pid,
            process_group_id=pgid,
            location=location,
            command=tuple(command),
        )

    # -- orphan adjudication -------------------------------------------------

    def find_orphan_candidates(
        self, *, heartbeat_timeout_seconds: float = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS
    ) -> list[OrphanCandidate]:
        """Identify non-terminal runs whose worker appears to be gone.

        Three signals are combined, and all three are reported so a researcher
        can judge:

        1. heartbeat age beyond the timeout,
        2. PID absence,
        3. PID **identity** — a live PID whose recorded process start time no
           longer matches is a *recycled* PID belonging to an unrelated process,
           not our worker (acceptance P-06). Treating it as alive would leave a
           dead run marked RUNNING forever; treating it as ours would risk
           signalling a stranger's process.

        This method only *reports*. It never kills and never transitions.
        """
        # Clear zombies first, so a dead-but-unreaped child is not read as alive.
        self.reap()
        candidates: list[OrphanCandidate] = []
        now = datetime.now(timezone.utc)
        cutoff = timedelta(seconds=float(heartbeat_timeout_seconds))

        for record in self.registry.list_runs(active_only=True, limit=10_000):
            if record.state in (RunState.CREATED, RunState.VALIDATING, RunState.QUEUED):
                # Not started yet — absence of a heartbeat is expected.
                continue
            if record.legacy:
                continue

            heartbeat_age: Optional[float] = None
            if record.heartbeat_at:
                try:
                    beat = datetime.fromisoformat(record.heartbeat_at.replace("Z", "+00:00"))
                    heartbeat_age = (now - beat).total_seconds()
                except ValueError:
                    heartbeat_age = None

            pid = record.pid
            alive = process_alive(int(pid)) if pid else False
            identity_matches = False
            worker = self.registry.worker_for_run(record.run_id)
            if alive and pid and worker is not None:
                observed = process_start_time(int(pid))
                recorded = float(worker.process_start_time or 0.0)
                if observed is not None and recorded > 0:
                    # Filesystem/clock granularity makes exact equality unsafe.
                    identity_matches = abs(observed - recorded) < 1.0
                else:
                    identity_matches = True  # cannot verify; assume ours

            stale = heartbeat_age is not None and heartbeat_age > cutoff.total_seconds()
            never_beat = record.heartbeat_at is None

            reason: Optional[str] = None
            if not alive:
                reason = f"PID {pid} is not present; worker recorded no terminal state"
            elif not identity_matches:
                reason = (
                    f"PID {pid} exists but its process start time does not match the "
                    "recorded worker — the PID was recycled by an unrelated process"
                )
            elif stale:
                reason = (
                    f"heartbeat is {heartbeat_age:.0f}s old (timeout {heartbeat_timeout_seconds:.0f}s) "
                    "while the process is still alive; the worker may be hung"
                )
            elif never_beat and record.state == RunState.RUNNING:
                reason = "run is RUNNING but has never recorded a heartbeat"

            if reason is None:
                continue
            candidates.append(
                OrphanCandidate(
                    run_id=record.run_id,
                    state=record.state.value,
                    reason=reason,
                    pid=pid,
                    heartbeat_at=record.heartbeat_at,
                    heartbeat_age_seconds=heartbeat_age,
                    pid_alive=alive,
                    pid_identity_matches=identity_matches,
                )
            )
        return candidates

    def mark_orphaned(self, candidate: OrphanCandidate, *, actor: str = "orphan-detector") -> None:
        """Record an ORPHANED transition for a verified-dead worker.

        Refuses to act while the process is still alive: a hung-but-live worker
        is a different problem from a vanished one, and marking it ORPHANED
        would let a second run start against the same GPU.

        ORPHANED is **not** FAILED. It means "we do not know how this ended" —
        adjudicating it is a researcher decision (design doc 03 §6).
        """
        if candidate.pid_alive and candidate.pid_identity_matches:
            raise RuntimeError(
                f"run {candidate.run_id} still has a live, identity-matched worker "
                f"(pid {candidate.pid}); refusing to mark it ORPHANED"
            )
        record = self.registry.get_run(candidate.run_id)
        if record is None or record.state in (
            RunState.COMPLETED,
            RunState.FAILED,
            RunState.CANCELLED,
        ):
            return
        self.registry.transition(
            candidate.run_id,
            to_state=RunState.ORPHANED,
            expected_state_version=record.state_version,
            actor=actor,
            reason=candidate.reason,
            fields={
                "terminal_reason": "worker_vanished",
                "error_summary": candidate.reason,
                "status_confidence": "unknown",
            },
        )
        worker = self.registry.worker_for_run(candidate.run_id)
        if worker is not None:
            self.registry.finish_worker(
                worker.worker_instance_id, state="ORPHANED", exit_code=None
            )

    def reconcile(
        self, *, heartbeat_timeout_seconds: float = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS
    ) -> list[OrphanCandidate]:
        """Find and mark verified-dead workers. Returns what was marked.

        Called on manager start-up (acceptance P-08): after a manager restart,
        runs whose workers are still alive keep running untouched, and runs whose
        workers vanished are classified rather than left in a lying RUNNING state.
        """
        marked: list[OrphanCandidate] = []
        for candidate in self.find_orphan_candidates(
            heartbeat_timeout_seconds=heartbeat_timeout_seconds
        ):
            if candidate.pid_alive and candidate.pid_identity_matches:
                continue
            try:
                self.mark_orphaned(candidate)
                marked.append(candidate)
            except Exception:
                continue
        return marked

    # -- diagnostics ---------------------------------------------------------

    def describe_worker(self, run_id: str) -> dict[str, Any]:
        """Worker liveness summary for the dashboard."""
        record = self.registry.get_run(run_id)
        worker = self.registry.worker_for_run(run_id)
        if record is None:
            return {"run_id": run_id, "known": False}
        alive = process_alive(int(record.pid)) if record.pid else False
        return {
            "run_id": run_id,
            "known": True,
            "state": record.state.value,
            "pid": record.pid,
            "process_group_id": record.process_group_id,
            "host": record.host,
            "pid_alive": alive,
            "heartbeat_at": record.heartbeat_at,
            "worker_instance_id": record.worker_instance_id,
            "worker_state": worker.state if worker else None,
            "exit_code": record.exit_code,
            "exit_code_description": describe_exit_code(record.exit_code),
        }
