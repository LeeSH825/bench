"""Worker process lifecycle and telemetry acceptance tests (design doc 06, P-01 … P-06, T-01 … T-03).

These tests spawn **real** subprocesses. They use the synthetic executor so a
full lifecycle costs a couple of seconds and needs no dataset or GPU, per the
"tiny/synthetic smoke run" mandate. Every test gets its own temporary control
root, so nothing touches `runs/`, `reports/`, or the real registry.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from bench.control.config.resolver import resolve_run_spec
from bench.control.config.schema import (
    DatasetSection,
    ExperimentSection,
    RunSpecDraft,
    RuntimeSection,
    SystemSection,
    TelemetrySection,
    TrainingSection,
)
from bench.control.events.reader import EventReader
from bench.control.events.schema import EventType
from bench.control.identity import ExperimentId, ImplementationId, ModelId
from bench.control.process.manager import WorkerManager
from bench.control.process.signals import ExitCode, describe_exit_code
from bench.control.registry.schema import RunState
from bench.control.registry.sqlite import SqliteRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Generous ceiling for a synthetic run to reach a terminal state.
TERMINAL_TIMEOUT_SECONDS = 120.0


def make_spec(*, updates: int = 6, seed: int = 0, telemetry_interval: float = 0.3):
    draft = RunSpecDraft(
        experiment=ExperimentSection(experiment_id=ExperimentId.new().value, name="process-test"),
        model_id=ModelId("kalmannet_tsp"),
        implementation_id=ImplementationId("bench_kalmannet_tsp_adapter_v1"),
        system=SystemSection(task_id="t", scenario_id="s", state_dim=2, observation_dim=2),
        dataset=DatasetSection(dataset_id="ds"),
        training=TrainingSection(
            enabled=True, max_updates=updates, batch_size=2, validation_interval_updates=2
        ),
        runtime=RuntimeSection(device="cpu", seed=seed),
        telemetry=TelemetrySection(enabled=True, interval_seconds=telemetry_interval),
        bench_context={"executor": "synthetic"},
    )
    return resolve_run_spec(draft)


class WorkerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry = SqliteRegistry(self.root / "registry.sqlite3")
        self.manager = WorkerManager(self.registry, control_root_path=self.root, repo_root=REPO_ROOT)
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        # Never leave a stray worker behind, whatever the test did.
        for record in self.registry.list_runs(limit=1000):
            if record.pid:
                try:
                    os.kill(int(record.pid), signal.SIGKILL)
                except OSError:
                    pass
        try:
            self.registry.close()
        except Exception:
            pass
        self._tmp.cleanup()

    def wait_for_state(self, run_id: str, states, timeout: float = TERMINAL_TIMEOUT_SECONDS):
        deadline = time.time() + timeout
        while time.time() < deadline:
            record = self.registry.get_run(run_id)
            if record is not None and record.state in states:
                return record
            time.sleep(0.15)
        record = self.registry.get_run(run_id)
        raise AssertionError(
            f"run {run_id} did not reach {states} within {timeout}s (last state {record.state if record else None})"
        )


class WorkerLifecycleTests(WorkerTestCase):
    def test_successful_run_reaches_completed_with_exit_code_zero(self) -> None:
        spec = make_spec(updates=6)
        location = self.manager.prepare_run(spec)
        self.manager.launch(spec, location=location)
        record = self.wait_for_state(spec.run_id.value, {RunState.COMPLETED, RunState.FAILED})
        self.assertEqual(record.state, RunState.COMPLETED)
        self.assertEqual(record.exit_code, int(ExitCode.COMPLETED))
        self.assertTrue(record.started_at)
        self.assertTrue(record.ended_at)
        self.assertEqual(record.global_step, 6)

    def test_worker_runs_in_its_own_process_group(self) -> None:
        """P-02: the worker and its children form one supervisable unit."""
        spec = make_spec(updates=400)
        location = self.manager.prepare_run(spec)
        result = self.manager.launch(spec, location=location)
        self.assertEqual(result.pid, result.process_group_id)
        self.assertNotEqual(result.process_group_id, os.getpgrp())
        self.wait_for_state(spec.run_id.value, {RunState.COMPLETED, RunState.FAILED})

    def test_command_is_an_argv_list_never_a_shell_string(self) -> None:
        spec = make_spec()
        location = self.manager.prepare_run(spec)
        command = self.manager.build_command(run_id=spec.run_id.value, location=location)
        self.assertIsInstance(command, list)
        self.assertTrue(all(isinstance(part, str) for part in command))
        self.assertIn("bench.control.process.worker_cli", command)

    def test_heartbeat_is_recorded_and_stops_after_exit(self) -> None:
        """P-04: heartbeat advances while running and freezes at exit."""
        spec = make_spec(updates=8)
        location = self.manager.prepare_run(spec)
        self.manager.launch(spec, location=location)
        record = self.wait_for_state(spec.run_id.value, {RunState.COMPLETED, RunState.FAILED})
        self.assertIsNotNone(record.heartbeat_at)
        frozen = record.heartbeat_at
        time.sleep(1.5)
        self.assertEqual(self.registry.get_run(spec.run_id.value).heartbeat_at, frozen)

    def test_worker_registry_state_matches_running_transition(self) -> None:
        """A recovered API must not render a live worker as STARTING."""
        spec = make_spec(updates=80, telemetry_interval=0.2)
        location = self.manager.prepare_run(spec)
        self.manager.launch(spec, location=location, extra_env={"BENCH_CONTROL_STEP_SLEEP": "0.02"})
        self.wait_for_state(spec.run_id.value, {RunState.RUNNING}, timeout=30)
        record = self.registry.get_run(spec.run_id.value)
        worker = self.registry.worker_for_run(spec.run_id.value)
        self.assertEqual(record.state, RunState.RUNNING)
        self.assertEqual(worker.state, "RUNNING")
        self.assertEqual(worker.last_heartbeat_at, record.heartbeat_at)

    def test_stdout_and_stderr_are_captured_to_files(self) -> None:
        """P-03: print/logging/traceback all land in per-run log files."""
        spec = make_spec(updates=4)
        location = self.manager.prepare_run(spec)
        self.manager.launch(spec, location=location)
        self.wait_for_state(spec.run_id.value, {RunState.COMPLETED, RunState.FAILED})
        self.assertTrue(location.stdout_path.exists())
        self.assertTrue(location.stderr_path.exists())
        self.assertIn("completed", location.stdout_path.read_text(encoding="utf-8"))

    def test_worker_survives_the_death_of_its_launcher(self) -> None:
        """P-01: UI/CLI independence.

        The worker is launched by a *separate short-lived process* which then
        exits. If the worker were tied to its launcher (pipes, same process
        group, same session) it would die with it.
        """
        spec = make_spec(updates=60, telemetry_interval=0.2)
        location = self.manager.prepare_run(spec)
        script = (
            "import sys;"
            f"sys.path.insert(0, {str(REPO_ROOT)!r});"
            "from pathlib import Path;"
            "from bench.control.registry.sqlite import SqliteRegistry;"
            "from bench.control.process.manager import WorkerManager;"
            "from bench.control.config.resolver import resolved_from_json;"
            f"registry = SqliteRegistry({str(self.root / 'registry.sqlite3')!r}, migrate=False);"
            f"spec = resolved_from_json(Path({str(location.resolved_spec_path)!r}).read_text());"
            f"manager = WorkerManager(registry, control_root_path={str(self.root)!r}, repo_root={str(REPO_ROOT)!r});"
            "result = manager.launch(spec);"
            "print(result.pid)"
        )
        launcher = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(REPO_ROOT),
            env={**os.environ, "BENCH_CONTROL_ROOT": str(self.root), "BENCH_CONTROL_STEP_SLEEP": "0.05"},
        )
        launcher_pid = launcher.pid
        stdout, stderr = launcher.communicate(timeout=60)
        self.assertEqual(launcher.returncode, 0, f"launcher failed: {stderr}")
        worker_pid = int(stdout.strip().splitlines()[-1])

        # The launcher process is gone; only the worker should remain.
        self.assertFalse(_pid_is_running(launcher_pid))
        self.assertNotEqual(worker_pid, launcher_pid)
        # The worker must still be alive and making progress.
        deadline = time.time() + 10
        saw_running = False
        while time.time() < deadline:
            record = self.registry.get_run(spec.run_id.value)
            if record.state == RunState.RUNNING:
                saw_running = True
                break
            if record.state in (RunState.COMPLETED, RunState.FAILED):
                saw_running = True
                break
            time.sleep(0.2)
        self.assertTrue(saw_running, "worker never reached RUNNING after its launcher exited")

        record = self.wait_for_state(spec.run_id.value, {RunState.COMPLETED, RunState.FAILED})
        self.assertEqual(record.state, RunState.COMPLETED, "worker did not survive its launcher")
        self.assertEqual(record.pid, worker_pid)


def _pid_is_running(pid: int) -> bool:
    from bench.control.telemetry.cpu import process_alive

    return process_alive(pid)


class FailurePathTests(WorkerTestCase):
    def test_ordinary_failure_is_recorded_with_traceback(self) -> None:
        spec = make_spec(updates=10)
        location = self.manager.prepare_run(spec)
        record = self.registry.get_run(spec.run_id.value)
        for state in (RunState.VALIDATING, RunState.QUEUED, RunState.STARTING):
            record = self.registry.transition(record.run_id, to_state=state)

        command = self.manager.build_command(run_id=spec.run_id.value, location=location)
        result = subprocess.run(
            command + ["--fail-at-step", "4"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env={**os.environ, "BENCH_CONTROL_ROOT": str(self.root)},
            timeout=TERMINAL_TIMEOUT_SECONDS,
        )
        self.assertEqual(result.returncode, int(ExitCode.EXECUTION_FAILURE))

        record = self.registry.get_run(spec.run_id.value)
        self.assertEqual(record.state, RunState.FAILED)
        self.assertEqual(record.exit_code, int(ExitCode.EXECUTION_FAILURE))
        self.assertEqual(record.terminal_reason, "execution_failure")
        self.assertIn("synthetic failure injected", record.error_summary)

        self.assertTrue(location.failure_path.exists())
        traceback_path = location.artifacts_dir / "traceback.txt"
        self.assertTrue(traceback_path.exists())
        self.assertIn("Traceback", traceback_path.read_text(encoding="utf-8"))

        page = EventReader(location.events_path).scan(limit=10**6)
        kinds = {event.event_type for event in page.events}
        self.assertIn(EventType.FAILURE, kinds)

    def test_abrupt_death_is_never_reported_as_completed(self) -> None:
        """P-05: SIGKILL leaves an ORPHANED candidate, not a COMPLETED run."""
        spec = make_spec(updates=100_000, telemetry_interval=0.2)
        location = self.manager.prepare_run(spec)
        result = self.manager.launch(spec, location=location)
        self.wait_for_state(spec.run_id.value, {RunState.RUNNING}, timeout=30)
        time.sleep(0.6)

        os.kill(result.pid, signal.SIGKILL)
        time.sleep(1.2)

        record = self.registry.get_run(spec.run_id.value)
        self.assertNotEqual(record.state, RunState.COMPLETED)
        self.assertEqual(record.state, RunState.RUNNING, "state should be stale, not falsified")
        self.assertIsNone(record.exit_code)

        candidates = self.manager.find_orphan_candidates(heartbeat_timeout_seconds=1.0)
        self.assertTrue(candidates)
        candidate = next(c for c in candidates if c.run_id == spec.run_id.value)
        self.assertFalse(candidate.pid_alive, "a SIGKILLed zombie must not count as alive")

        self.manager.reconcile(heartbeat_timeout_seconds=1.0)
        record = self.registry.get_run(spec.run_id.value)
        self.assertEqual(record.state, RunState.ORPHANED)
        self.assertEqual(record.status_confidence, "unknown")
        self.assertEqual(record.terminal_reason, "worker_vanished")

    def test_live_worker_is_never_marked_orphaned(self) -> None:
        spec = make_spec(updates=100_000, telemetry_interval=0.2)
        location = self.manager.prepare_run(spec)
        self.manager.launch(spec, location=location)
        self.wait_for_state(spec.run_id.value, {RunState.RUNNING}, timeout=30)

        marked = self.manager.reconcile(heartbeat_timeout_seconds=0.001)
        self.assertEqual([c.run_id for c in marked], [])
        self.assertEqual(self.registry.get_run(spec.run_id.value).state, RunState.RUNNING)

    def test_pid_reuse_is_not_mistaken_for_the_original_worker(self) -> None:
        """P-06: identity is (pid, process start time), not pid alone."""
        spec = make_spec(updates=100_000, telemetry_interval=0.2)
        location = self.manager.prepare_run(spec)
        result = self.manager.launch(spec, location=location)
        self.wait_for_state(spec.run_id.value, {RunState.RUNNING}, timeout=30)

        worker = self.registry.worker_for_run(spec.run_id.value)
        self.assertIsNotNone(worker)
        self.assertGreater(worker.process_start_time, 0)

        # Rewrite the recorded start time to simulate a recycled PID: the process
        # at this pid is alive, but it is not the process we launched.
        with self.registry.transaction() as connection:
            connection.execute(
                "UPDATE workers SET process_start_time = ? WHERE worker_instance_id = ?",
                (worker.process_start_time - 10_000.0, worker.worker_instance_id),
            )

        candidates = self.manager.find_orphan_candidates(heartbeat_timeout_seconds=600)
        candidate = next(c for c in candidates if c.run_id == spec.run_id.value)
        self.assertTrue(candidate.pid_alive)
        self.assertFalse(candidate.pid_identity_matches)
        self.assertIn("recycled", candidate.reason)

        os.kill(result.pid, signal.SIGKILL)

    def test_exit_code_descriptions_cover_the_contract(self) -> None:
        for code in ExitCode:
            self.assertTrue(describe_exit_code(int(code)))
        self.assertIn("SIGKILL", describe_exit_code(-int(signal.SIGKILL)))
        self.assertIn("No exit code", describe_exit_code(None))


class EventsFromWorkerTests(WorkerTestCase):
    def test_metrics_logs_and_resources_appear_before_the_terminal_state(self) -> None:
        """U-02: live observability, not a post-mortem dump."""
        spec = make_spec(updates=200, telemetry_interval=0.2)
        location = self.manager.prepare_run(spec)
        # Pace the synthetic steps so the run is genuinely observable in flight.
        # Without this it finishes faster than any poll interval, which would
        # make the test pass or fail on timing rather than on behaviour.
        self.manager.launch(spec, location=location, extra_env={"BENCH_CONTROL_STEP_SLEEP": "0.05"})
        self.wait_for_state(spec.run_id.value, {RunState.RUNNING}, timeout=30)

        reader = EventReader(location.events_path)
        deadline = time.time() + 25
        seen_metric = seen_resource = seen_status = False
        while time.time() < deadline:
            record = self.registry.get_run(spec.run_id.value)
            page = reader.scan(limit=10**6)
            kinds = {event.event_type for event in page.events}
            seen_metric = seen_metric or EventType.METRIC in kinds
            seen_resource = seen_resource or EventType.RESOURCE in kinds
            seen_status = seen_status or EventType.STATUS in kinds
            if seen_metric and seen_resource and seen_status:
                self.assertEqual(
                    record.state,
                    RunState.RUNNING,
                    "events must be visible while the run is still RUNNING",
                )
                break
            time.sleep(0.2)

        self.assertTrue(seen_metric, "no metric events observed during the run")
        self.assertTrue(seen_status, "no status events observed during the run")
        self.assertTrue(seen_resource, "no resource events observed during the run")

    def test_registry_progress_mirrors_journal_metrics(self) -> None:
        spec = make_spec(updates=30)
        location = self.manager.prepare_run(spec)
        self.manager.launch(spec, location=location)
        record = self.wait_for_state(spec.run_id.value, {RunState.COMPLETED, RunState.FAILED})
        self.assertEqual(record.state, RunState.COMPLETED)
        self.assertEqual(record.global_step, 30)
        self.assertGreater(record.last_event_id, 0)

    def test_state_survives_a_fresh_registry_connection(self) -> None:
        """R-07: nothing lives in process memory; a new reader recovers everything."""
        spec = make_spec(updates=8)
        location = self.manager.prepare_run(spec)
        self.manager.launch(spec, location=location)
        self.wait_for_state(spec.run_id.value, {RunState.COMPLETED, RunState.FAILED})

        reopened = SqliteRegistry(self.root / "registry.sqlite3", migrate=False)
        try:
            record = reopened.get_run(spec.run_id.value)
            self.assertEqual(record.state, RunState.COMPLETED)
            self.assertTrue(reopened.list_transitions(spec.run_id.value))
            page = EventReader(location.events_path).scan(limit=10**6)
            self.assertTrue(page.events)
        finally:
            reopened.close()


class TelemetryTests(unittest.TestCase):
    """T-01 NVIDIA, T-02 CPU-only fallback, T-03 process-tree aggregation."""

    def test_cpu_collector_reports_process_tree(self) -> None:
        from bench.control.telemetry.cpu import CpuCollector, psutil_available

        collector = CpuCollector(pid=os.getpid())
        if not psutil_available():
            self.assertFalse(collector.available())
            self.skipTest("psutil is not installed; CPU telemetry is unavailable by design")
        self.assertTrue(collector.available())
        sample = collector.collect()
        self.assertIsNotNone(sample["process_tree_rss_bytes"])
        self.assertGreater(sample["process_tree_rss_bytes"], 0)
        self.assertGreaterEqual(sample["process_count"], 1)
        self.assertGreater(sample["system_ram_total_bytes"], 0)

    def test_sampler_never_raises_when_a_collector_fails(self) -> None:
        from bench.control.telemetry.base import TelemetrySampler

        class BrokenCollector:
            name = "broken"

            def available(self) -> bool:
                return True

            def collect(self):
                raise RuntimeError("collector exploded")

        collected = []
        sampler = TelemetrySampler(
            run_id="r", collectors=[BrokenCollector()], sink=collected.append, interval_seconds=0.1
        )
        sample = sampler.sample_once()
        self.assertEqual(len(sample.collector_errors), 1)
        self.assertIn("collector exploded", sample.collector_errors[0])
        # a failing collector produces a sample with null fields, not an exception
        self.assertIsNone(sample.gpu)

    def test_cpu_only_sample_serializes_with_explicit_nulls(self) -> None:
        """T-02: absence must be null/absent, never zero."""
        from bench.control.telemetry.base import TelemetrySampler
        from bench.control.telemetry.cpu import CpuCollector

        sampler = TelemetrySampler(
            run_id="r", collectors=[CpuCollector(pid=os.getpid())], sink=lambda s: None, interval_seconds=1.0
        )
        document = sampler.sample_once().as_dict()
        self.assertIsNone(document["gpu"], "GPU must be null on a CPU-only sample, not zeros")
        self.assertIn("disk", document)
        self.assertEqual(document["collector_errors"], [])

    def test_sampler_thread_starts_and_stops(self) -> None:
        from bench.control.telemetry.base import TelemetrySampler
        from bench.control.telemetry.cpu import CpuCollector

        collected = []
        with TelemetrySampler(
            run_id="r",
            collectors=[CpuCollector(pid=os.getpid())],
            sink=collected.append,
            interval_seconds=0.1,
        ):
            time.sleep(0.5)
        self.assertGreaterEqual(len(collected), 2)

    def test_default_collectors_skip_gpu_for_cpu_runs(self) -> None:
        from bench.control.telemetry import default_collectors

        names = {collector.name for collector in default_collectors(pid=os.getpid(), device="cpu")}
        self.assertNotIn("nvidia", names)

    def test_gpu_inventory_is_a_list_even_without_a_gpu(self) -> None:
        from bench.control.telemetry import gpu_inventory

        self.assertIsInstance(gpu_inventory(), list)


class NvidiaTelemetryTests(unittest.TestCase):
    """T-01, executed only where an NVIDIA device is actually visible."""

    def setUp(self) -> None:
        from bench.control.telemetry.nvidia import NvidiaCollector

        self.collector = NvidiaCollector(device_index=0, pid=os.getpid())
        if not self.collector.available():
            self.skipTest("no NVIDIA GPU visible on this host")

    def test_whole_device_readings_are_present(self) -> None:
        sample = self.collector.collect()["gpu"]
        self.assertIsNotNone(sample)
        self.assertIsNotNone(sample.device_memory_total_bytes)
        self.assertGreater(sample.device_memory_total_bytes, 0)
        self.assertIn(sample.backend, ("nvml", "nvidia_smi"))

    def test_attribution_quality_is_declared(self) -> None:
        """Whole-device utilization and per-process memory must not be conflated."""
        sample = self.collector.collect()["gpu"]
        self.assertIn(sample.attribution_quality, ("memory_only", "device_only", "unavailable"))

    def test_inventory_lists_the_device(self) -> None:
        from bench.control.telemetry import gpu_inventory

        devices = gpu_inventory()
        self.assertTrue(devices)
        self.assertIn("device_index", devices[0])
        self.assertIn("attribution_quality", devices[0])

    def test_gpu_collector_is_selected_for_cuda_runs(self) -> None:
        from bench.control.telemetry import default_collectors

        names = {collector.name for collector in default_collectors(pid=os.getpid(), device="cuda:0")}
        self.assertIn("nvidia", names)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
