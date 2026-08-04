"""Registry, allocation, and event-journal acceptance tests (design doc 06, R-01 … R-06)."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from bench.control.allocation import AllocationError, allocate_run_directory, write_run_spec
from bench.control.config.resolver import resolve_run_spec
from bench.control.config.schema import (
    DatasetSection,
    ExperimentSection,
    RunSpecDraft,
    RuntimeSection,
    SystemSection,
    TrainingSection,
)
from bench.control.events.reader import EventReader
from bench.control.events.schema import EventType, EventValidationError
from bench.control.events.writer import EventWriter
from bench.control.identity import ExperimentId, ImplementationId, ModelId, RunId
from bench.control.registry.schema import (
    ACTIVE_STATES_THIS_TRANCHE,
    ExperimentRecord,
    InvalidTransitionError,
    RunRecord,
    RunState,
    is_terminal,
)
from bench.control.registry.sqlite import ConcurrencyError, RegistryError, SqliteRegistry


def make_spec(seed: int = 0):
    draft = RunSpecDraft(
        experiment=ExperimentSection(experiment_id=ExperimentId.new().value, name="registry-test"),
        model_id=ModelId("kalmannet_tsp"),
        implementation_id=ImplementationId("bench_kalmannet_tsp_adapter_v1"),
        system=SystemSection(task_id="t", scenario_id="s", state_dim=2, observation_dim=2),
        dataset=DatasetSection(dataset_id="ds"),
        training=TrainingSection(enabled=True, max_updates=5, batch_size=2, validation_interval_updates=1),
        runtime=RuntimeSection(device="cpu", seed=seed),
    )
    return resolve_run_spec(draft)


class RegistryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry = SqliteRegistry(self.root / "registry.sqlite3")
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(self.registry.close)

    def create_run(self, spec=None, state: RunState = RunState.CREATED) -> RunRecord:
        spec = spec or make_spec()
        experiment_id = spec.draft.experiment.experiment_id
        self.registry.upsert_experiment(ExperimentRecord(experiment_id=experiment_id, name="e"))
        return self.registry.create_run(
            RunRecord(
                run_id=spec.run_id.value,
                experiment_id=experiment_id,
                state=state,
                state_version=0,
                created_at="",
                updated_at="",
                model_id=spec.model_id.value,
                variant_id=spec.variant_id.value,
                run_dir=str(self.root / spec.run_id.value),
            )
        )


class MigrationTests(RegistryTestCase):
    def test_schema_version_is_recorded(self) -> None:
        # 2 made the checkpoint/action tables live; 3 persists the training
        # path on runs, checkpoints and certifications; 4 makes
        # run_actions.run_id nullable so a LAUNCH_RUN action can be recorded
        # before the run it creates exists.
        self.assertEqual(self.registry.schema_version, 4)
        rows = self.registry.connection.execute("SELECT version, name FROM schema_migrations").fetchall()
        self.assertEqual([row[0] for row in rows], [1, 2, 3, 4])

    def test_migration_is_idempotent(self) -> None:
        self.assertEqual(self.registry.migrate(), [])

    def test_wal_and_foreign_keys_are_enabled(self) -> None:
        mode = self.registry.connection.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(str(mode).lower(), "wal")
        self.assertEqual(self.registry.connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_newer_schema_is_refused(self) -> None:
        from bench.control.registry.sqlite import SchemaVersionError

        self.registry.connection.execute("PRAGMA user_version=999")
        with self.assertRaises(SchemaVersionError):
            SqliteRegistry(self.root / "registry.sqlite3")

    def test_foreign_key_is_enforced(self) -> None:
        import sqlite3

        with self.assertRaises(sqlite3.IntegrityError):
            self.registry.create_run(
                RunRecord(
                    run_id=RunId.new().value,
                    experiment_id=ExperimentId.new().value,  # never inserted
                    state=RunState.CREATED,
                    state_version=0,
                    created_at="",
                    updated_at="",
                )
            )


class StateTransitionTests(RegistryTestCase):
    """R-01 state transition validation, R-02 optimistic concurrency."""

    def test_happy_path_transitions(self) -> None:
        record = self.create_run()
        for state in (RunState.VALIDATING, RunState.QUEUED, RunState.STARTING, RunState.RUNNING, RunState.COMPLETED):
            record = self.registry.transition(record.run_id, to_state=state, expected_state_version=record.state_version)
        self.assertEqual(record.state, RunState.COMPLETED)
        self.assertEqual(record.state_version, 5)
        self.assertTrue(record.started_at)
        self.assertTrue(record.ended_at)

    def test_illegal_transition_is_rejected(self) -> None:
        record = self.create_run()
        with self.assertRaises(InvalidTransitionError):
            self.registry.transition(record.run_id, to_state=RunState.COMPLETED)

    def test_terminal_state_is_final(self) -> None:
        record = self.create_run()
        for state in (RunState.VALIDATING, RunState.QUEUED, RunState.STARTING, RunState.RUNNING, RunState.COMPLETED):
            record = self.registry.transition(record.run_id, to_state=state)
        for target in (RunState.RUNNING, RunState.FAILED, RunState.QUEUED):
            with self.assertRaises(InvalidTransitionError):
                self.registry.transition(record.run_id, to_state=target)

    def test_orphaned_is_not_terminal_and_needs_adjudication(self) -> None:
        """ORPHANED means 'unknown outcome', not 'failed' (design doc 03 §6)."""
        self.assertFalse(is_terminal(RunState.ORPHANED))
        record = self.create_run()
        for state in (RunState.VALIDATING, RunState.QUEUED, RunState.STARTING):
            record = self.registry.transition(record.run_id, to_state=state)
        record = self.registry.transition(record.run_id, to_state=RunState.ORPHANED)
        # a researcher may adjudicate it, but it cannot silently resume
        with self.assertRaises(InvalidTransitionError):
            self.registry.transition(record.run_id, to_state=RunState.RUNNING)
        self.registry.transition(record.run_id, to_state=RunState.FAILED)

    def test_stale_state_version_is_rejected(self) -> None:
        record = self.create_run()
        stale = record.state_version
        self.registry.transition(record.run_id, to_state=RunState.VALIDATING, expected_state_version=stale)
        with self.assertRaises(ConcurrencyError):
            self.registry.transition(record.run_id, to_state=RunState.QUEUED, expected_state_version=stale)

    def test_unknown_run_raises(self) -> None:
        with self.assertRaises(RegistryError):
            self.registry.transition(RunId.new().value, to_state=RunState.VALIDATING)

    def test_transitions_are_logged_with_actor_and_reason(self) -> None:
        record = self.create_run()
        self.registry.transition(record.run_id, to_state=RunState.VALIDATING, actor="tester", reason="because")
        rows = self.registry.list_transitions(record.run_id)
        self.assertEqual(rows[-1]["to_state"], "VALIDATING")
        self.assertEqual(rows[-1]["actor"], "tester")
        self.assertEqual(rows[-1]["reason"], "because")

    def test_progress_update_does_not_bump_state_version(self) -> None:
        """Optimistic concurrency guards control actions, not training steps."""
        record = self.create_run()
        self.registry.update_progress(record.run_id, global_step=42, phase="train")
        again = self.registry.get_run(record.run_id)
        self.assertEqual(again.state_version, record.state_version)
        self.assertEqual(again.global_step, 42)

    def test_schema_only_states_are_not_produced_by_this_build(self) -> None:
        # The checkpoint/stop tranche produces STOP_REQUESTED, CHECKPOINTING and
        # INTERRUPTED for real (see tests/test_control_graceful_stop.py), so they
        # are no longer schema-only. RESUMING stays schema-only: a resume creates
        # a *child* run rather than moving the parent back to running
        # (ADR-CSR-003), so nothing in this build ever enters that state.
        schema_only = {s for s in RunState} - ACTIVE_STATES_THIS_TRANCHE
        self.assertEqual({s.value for s in schema_only}, {"RESUMING"})

    def test_transition_rejects_unknown_columns(self) -> None:
        record = self.create_run()
        with self.assertRaises(RegistryError):
            self.registry.transition(
                record.run_id, to_state=RunState.VALIDATING, fields={"state": "HACKED"}
            )


class ConcurrencyTests(RegistryTestCase):
    """R-03 SQLite concurrency smoke."""

    def test_concurrent_heartbeats_and_reads_do_not_corrupt(self) -> None:
        records = [self.create_run(make_spec(seed=i)) for i in range(4)]
        errors: list[str] = []

        def beat(run_id: str) -> None:
            try:
                registry = SqliteRegistry(self.root / "registry.sqlite3", migrate=False)
                for step in range(40):
                    registry.record_heartbeat(run_id)
                    registry.update_progress(run_id, global_step=step)
                    registry.list_runs(limit=10)
                registry.close()
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=beat, args=(record.run_id,)) for record in records]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        self.assertEqual(errors, [])
        self.assertEqual(
            self.registry.connection.execute("PRAGMA integrity_check").fetchone()[0], "ok"
        )
        for record in records:
            refreshed = self.registry.get_run(record.run_id)
            self.assertEqual(refreshed.global_step, 39)
            self.assertIsNotNone(refreshed.heartbeat_at)

    def test_only_one_thread_wins_a_contested_transition(self) -> None:
        record = self.create_run()
        outcomes: list[str] = []
        barrier = threading.Barrier(2)

        def attempt() -> None:
            registry = SqliteRegistry(self.root / "registry.sqlite3", migrate=False)
            barrier.wait()
            try:
                registry.transition(record.run_id, to_state=RunState.VALIDATING, expected_state_version=0)
                outcomes.append("won")
            except (ConcurrencyError, InvalidTransitionError):
                outcomes.append("lost")
            finally:
                registry.close()

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual(sorted(outcomes), ["lost", "won"])

    def test_gpu_lease_is_exclusive(self) -> None:
        """P-07: two runs cannot hold the same device."""
        first = self.create_run(make_spec(seed=1))
        second = self.create_run(make_spec(seed=2))
        lease = self.registry.acquire_gpu_lease(device_index=0, run_id=first.run_id)
        self.assertIsNotNone(lease)
        self.assertIsNone(self.registry.acquire_gpu_lease(device_index=0, run_id=second.run_id))
        self.registry.release_gpu_lease(lease)
        self.assertIsNotNone(self.registry.acquire_gpu_lease(device_index=0, run_id=second.run_id))


class AllocationTests(unittest.TestCase):
    """C-02 immutable run allocation."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_identical_config_gets_distinct_directories(self) -> None:
        first, second = make_spec(), make_spec()
        # same structural identity ...
        self.assertEqual(first.structural_config_hash, second.structural_config_hash)
        location_a = allocate_run_directory(
            run_id=first.run_id, experiment_id=ExperimentId(first.draft.experiment.experiment_id), control_root=self.root
        )
        location_b = allocate_run_directory(
            run_id=second.run_id, experiment_id=ExperimentId(second.draft.experiment.experiment_id), control_root=self.root
        )
        # ... but different directories, so neither can overwrite the other.
        self.assertNotEqual(location_a.root, location_b.root)
        self.assertTrue(location_a.root.is_dir())
        self.assertTrue(location_b.root.is_dir())

    def test_reallocating_the_same_run_id_fails_loudly(self) -> None:
        spec = make_spec()
        experiment_id = ExperimentId(spec.draft.experiment.experiment_id)
        allocate_run_directory(run_id=spec.run_id, experiment_id=experiment_id, control_root=self.root)
        with self.assertRaises(AllocationError):
            allocate_run_directory(run_id=spec.run_id, experiment_id=experiment_id, control_root=self.root)

    def test_concurrent_allocation_of_identical_configs(self) -> None:
        """Twenty threads launching the same config get twenty directories."""
        results: list[Path] = []
        errors: list[str] = []
        lock = threading.Lock()

        def allocate() -> None:
            try:
                spec = make_spec()
                location = allocate_run_directory(
                    run_id=spec.run_id,
                    experiment_id=ExperimentId(spec.draft.experiment.experiment_id),
                    control_root=self.root,
                )
                with lock:
                    results.append(location.root)
            except Exception as exc:  # pragma: no cover
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=allocate) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 20)
        self.assertEqual(len(set(results)), 20)

    def test_subdirectories_and_spec_are_created(self) -> None:
        spec = make_spec()
        location = allocate_run_directory(
            run_id=spec.run_id, experiment_id=ExperimentId(spec.draft.experiment.experiment_id), control_root=self.root
        )
        for name in ("checkpoints", "artifacts", "provenance", "tmp"):
            self.assertTrue((location.root / name).is_dir(), name)
        write_run_spec(location, spec)
        document = json.loads(location.resolved_spec_path.read_text(encoding="utf-8"))
        self.assertEqual(document["identity"]["run_id"], spec.run_id.value)


class EventJournalTests(unittest.TestCase):
    """R-04 crash tail, R-05 monotonicity."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "events.jsonl"
        self.run_id = RunId.new().value
        self.addCleanup(self._tmp.cleanup)

    def test_event_ids_are_monotonic_and_gap_free(self) -> None:
        with EventWriter(self.path, self.run_id) as writer:
            for step in range(50):
                writer.metric("loss/train_total", 1.0 / (step + 1), step=step)
        events = EventReader(self.path).scan(limit=10**6).events
        self.assertEqual([event.event_id for event in events], list(range(1, 51)))

    def test_ids_stay_monotonic_across_concurrent_writers(self) -> None:
        writer = EventWriter(self.path, self.run_id)

        def emit() -> None:
            for step in range(100):
                writer.metric("loss/train_total", float(step), step=step)

        threads = [threading.Thread(target=emit) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        writer.close()
        ids = [event.event_id for event in EventReader(self.path).scan(limit=10**6).events]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 400)

    def test_partial_last_line_is_recovered_with_a_warning(self) -> None:
        with EventWriter(self.path, self.run_id) as writer:
            for step in range(5):
                writer.metric("loss/train_total", float(step), step=step)
        # simulate a worker SIGKILLed mid-write
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"schema_version": 1, "event_id": 6, "run_i')

        page = EventReader(self.path).scan(limit=10**6)
        self.assertEqual(len(page.events), 5)
        self.assertEqual(len(page.warnings), 1)
        self.assertTrue(page.warnings[0].is_tail)

    def test_midfile_corruption_is_reported_as_not_a_tail(self) -> None:
        with EventWriter(self.path, self.run_id) as writer:
            writer.metric("a", 1.0, step=1)
            writer.metric("b", 2.0, step=2)
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.path.write_text("\n".join([lines[0], "{ not json", lines[1]]) + "\n", encoding="utf-8")

        page = EventReader(self.path).scan(limit=10**6)
        self.assertEqual(len(page.events), 2)
        self.assertEqual(len(page.warnings), 1)
        self.assertFalse(page.warnings[0].is_tail)

    def test_writer_continues_the_sequence_after_a_crash_tail(self) -> None:
        with EventWriter(self.path, self.run_id) as writer:
            writer.metric("a", 1.0, step=1)
            writer.metric("b", 2.0, step=2)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"event_id": 3, "trunc')
        with EventWriter(self.path, self.run_id) as writer:
            event = writer.metric("c", 3.0, step=3)
        self.assertEqual(event.event_id, 3)

    def test_cursor_pagination(self) -> None:
        with EventWriter(self.path, self.run_id) as writer:
            for step in range(25):
                writer.metric("loss/train_total", float(step), step=step)
        reader = EventReader(self.path)
        collected: list[int] = []
        cursor = 0
        for _ in range(10):
            page = reader.scan(after_event_id=cursor, limit=7)
            collected.extend(event.event_id for event in page.events)
            cursor = page.next_cursor
            if not page.has_more:
                break
        self.assertEqual(collected, list(range(1, 26)))

    def test_event_type_filter(self) -> None:
        with EventWriter(self.path, self.run_id) as writer:
            writer.metric("m", 1.0, step=1)
            writer.log("hello")
            writer.status("RUNNING")
        page = EventReader(self.path).scan(event_types=[EventType.LOG])
        self.assertEqual([event.event_type for event in page.events], [EventType.LOG])

    def test_oversized_payload_is_refused(self) -> None:
        """Large arrays belong in artifacts, referenced by URI and hash."""
        with EventWriter(self.path, self.run_id) as writer:
            with self.assertRaises(EventValidationError):
                writer.metric("big", 1.0, step=1, blob="x" * 40_000)

    def test_long_message_is_truncated_not_rejected(self) -> None:
        with EventWriter(self.path, self.run_id) as writer:
            event = writer.log("y" * 50_000)
        self.assertLess(len(event.message), 50_000)
        self.assertIn("truncated", event.message)

    def test_metric_series_and_resource_helpers(self) -> None:
        with EventWriter(self.path, self.run_id) as writer:
            for step in range(10):
                writer.metric("loss/train_total", float(step), step=step, phase="train")
            writer.resource({"process_tree_cpu_percent": 12.5})
        reader = EventReader(self.path)
        series = reader.metric_series()
        self.assertIn("loss/train_total", series)
        self.assertEqual(len(series["loss/train_total"]), 10)
        self.assertEqual(len(reader.resource_samples()), 1)

    def test_tail_returns_the_most_recent_events(self) -> None:
        with EventWriter(self.path, self.run_id) as writer:
            for step in range(100):
                writer.metric("m", float(step), step=step)
        page = EventReader(self.path).tail(limit=5)
        self.assertEqual([event.event_id for event in page.events], [96, 97, 98, 99, 100])

    def test_missing_journal_reads_as_empty(self) -> None:
        reader = EventReader(Path(self._tmp.name) / "absent.jsonl")
        self.assertFalse(reader.exists)
        self.assertEqual(reader.scan().events, ())
        self.assertEqual(reader.last_event_id(), 0)


class ObserverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "events.jsonl"
        self.addCleanup(self._tmp.cleanup)

    def test_null_observer_is_the_default(self) -> None:
        from bench.control.events.observer import NullObserver, active_observer, set_active_observer

        set_active_observer(None)
        self.assertIsInstance(active_observer(), NullObserver)
        # calling it must be safe and silent
        active_observer().metric("x", 1.0, step=1)
        active_observer().status("RUNNING")

    def test_observer_failure_does_not_propagate_into_training(self) -> None:
        """A broken journal must degrade observability, never kill a run."""
        from bench.control.events.observer import JournalObserver

        writer = EventWriter(self.path, RunId.new().value)
        observer = JournalObserver(writer)
        writer.close()  # journal is now unusable

        observer.metric("loss/train_total", 1.0, step=1)
        observer.status("RUNNING")
        observer.log("still alive")
        self.assertGreater(observer.dropped_events, 0)

    def test_journal_observer_writes_metrics(self) -> None:
        from bench.control.events.observer import JournalObserver

        with EventWriter(self.path, RunId.new().value) as writer:
            observer = JournalObserver(writer)
            observer.status("PHASE_START", phase="train")
            for step in range(5):
                observer.metric("loss/train_total", float(step), step=step, phase="train")
        series = EventReader(self.path).metric_series()
        self.assertEqual(len(series["loss/train_total"]), 5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
