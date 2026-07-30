"""Legacy run import acceptance tests (design doc 03 §16, doc 05 §5.2).

The non-negotiable property: the importer **never writes to the legacy tree**.
`test_import_does_not_modify_the_source_tree` enforces that by fingerprinting
every file before and after an import.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from bench.control.legacy.importer import (
    discover_legacy_runs,
    import_legacy_runs,
    inspect_legacy_run,
    legacy_path_hash,
    legacy_run_id,
)
from bench.control.registry.schema import RunState
from bench.control.registry.sqlite import SqliteRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_RUNS_ROOT = REPO_ROOT / "runs"


def write_legacy_run(
    root: Path,
    name: str,
    *,
    model_id: str = "kalmannet_tsp",
    status: str | None = "ok",
    failure: bool = False,
    with_meta: bool = False,
    with_checkpoint: bool = False,
    plan_only: bool = False,
) -> Path:
    """Create a synthetic legacy run directory mirroring the real layout."""
    directory = root / name / "task_x" / model_id / "frozen" / "seed_0" / "scenario_abc123"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_plan.json").write_text(
        json.dumps(
            {
                "plan_id": "trained__frozen",
                "init_id": "trained",
                "track_id": "frozen",
                "suite_name": name,
                "task_id": "task_x",
                "scenario_id": "scenario_abc123",
                "seed": 0,
                "model_id": model_id,
            }
        ),
        encoding="utf-8",
    )
    if not plan_only:
        if failure:
            (directory / "failure.json").write_text(
                json.dumps({"status": "failed", "error": "SomeError: boom", "failure_type": "adapter_error"}),
                encoding="utf-8",
            )
        else:
            document = {
                "suite_name": name,
                "task_id": "task_x",
                "scenario_id": "scenario_abc123",
                "seed": 0,
                "model_id": model_id,
                "track_id": "frozen",
                "accuracy": {"mse": 0.01, "mse_db": -20.0},
            }
            if status is not None:
                document["status"] = status
            (directory / "metrics.json").write_text(json.dumps(document), encoding="utf-8")
    if with_meta:
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "artifact_version": "1.1",
                    "created_at": "2026-07-27T21:46:28.462800+09:00",
                    "model_id": model_id,
                    "suite": name,
                    "task": "task_x",
                    "scenario_id": "scenario_abc123",
                    "seed": 0,
                    "track_id": "frozen",
                    "init_id": "trained",
                }
            ),
            encoding="utf-8",
        )
    if with_checkpoint:
        (directory / "checkpoints").mkdir(exist_ok=True)
        (directory / "checkpoints" / "model.pt").write_bytes(b"not-a-real-checkpoint")
    (directory / "config_snapshot.yaml").write_text(yaml.safe_dump({"suite": {"name": name}}), encoding="utf-8")
    return directory


def tree_fingerprint(root: Path) -> dict[str, tuple[int, str]]:
    """Path → (size, sha256) for every file under *root*."""
    fingerprint: dict[str, tuple[int, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            fingerprint[str(path.relative_to(root))] = (len(data), hashlib.sha256(data).hexdigest())
    return fingerprint


class LegacyImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.legacy_root = self.base / "runs"
        self.legacy_root.mkdir()
        self.registry = SqliteRegistry(self.base / "registry.sqlite3")
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        try:
            self.registry.close()
        except Exception:
            pass
        self._tmp.cleanup()

    # -- read-only guarantee ------------------------------------------------

    def test_import_does_not_modify_the_source_tree(self) -> None:
        """The single most important property of the importer."""
        write_legacy_run(self.legacy_root, "suite_a", with_meta=True, with_checkpoint=True)
        write_legacy_run(self.legacy_root, "suite_b", failure=True)
        before = tree_fingerprint(self.legacy_root)

        report = import_legacy_runs(self.registry, root=self.legacy_root)
        self.assertEqual(report.imported, 2)

        after = tree_fingerprint(self.legacy_root)
        self.assertEqual(before, after, "the legacy tree must be byte-identical after an import")

    # -- identity -----------------------------------------------------------

    def test_synthetic_run_id_is_deterministic(self) -> None:
        directory = write_legacy_run(self.legacy_root, "suite_a")
        self.assertEqual(legacy_run_id(directory), legacy_run_id(directory))
        other = write_legacy_run(self.legacy_root, "suite_b")
        self.assertNotEqual(legacy_run_id(directory), legacy_run_id(other))

    def test_legacy_ids_are_uuid5_not_uuid7(self) -> None:
        """Derived vs allocated ids stay distinguishable by version digit."""
        import uuid as uuid_module

        directory = write_legacy_run(self.legacy_root, "suite_a")
        self.assertEqual(uuid_module.UUID(legacy_run_id(directory)).version, 5)

    def test_import_is_idempotent(self) -> None:
        write_legacy_run(self.legacy_root, "suite_a")
        first = import_legacy_runs(self.registry, root=self.legacy_root)
        second = import_legacy_runs(self.registry, root=self.legacy_root)
        self.assertEqual(first.imported, 1)
        self.assertEqual(second.imported, 0)
        self.assertEqual(second.already_present, 1)
        self.assertEqual(self.registry.count_runs(), 1)

    def test_each_directory_is_yielded_once_however_many_markers(self) -> None:
        write_legacy_run(self.legacy_root, "suite_a", with_meta=True)  # run_plan + metrics + meta
        directories = list(discover_legacy_runs(self.legacy_root))
        self.assertEqual(len(directories), 1)

    def test_model_cache_and_quarantine_are_skipped(self) -> None:
        write_legacy_run(self.legacy_root, "_model_cache")
        write_legacy_run(self.legacy_root, "_quarantine_bad_thing")
        write_legacy_run(self.legacy_root, "real_suite")
        directories = [str(path) for path in discover_legacy_runs(self.legacy_root)]
        self.assertEqual(len(directories), 1)
        self.assertIn("real_suite", directories[0])

    # -- status inference ---------------------------------------------------

    def test_completed_status_with_high_confidence(self) -> None:
        directory = write_legacy_run(self.legacy_root, "suite_a", status="ok")
        candidate = inspect_legacy_run(directory)
        self.assertEqual(candidate.state, RunState.COMPLETED)
        self.assertEqual(candidate.status_confidence, "high")
        self.assertIn("status='ok'", candidate.status_reason)

    def test_failed_status_from_failure_json(self) -> None:
        directory = write_legacy_run(self.legacy_root, "suite_a", failure=True)
        candidate = inspect_legacy_run(directory)
        self.assertEqual(candidate.state, RunState.FAILED)
        self.assertEqual(candidate.status_confidence, "high")
        self.assertIn("SomeError", candidate.error_summary)

    def test_metrics_without_status_is_medium_confidence(self) -> None:
        directory = write_legacy_run(self.legacy_root, "suite_a", status=None)
        candidate = inspect_legacy_run(directory)
        self.assertEqual(candidate.state, RunState.COMPLETED)
        self.assertEqual(candidate.status_confidence, "medium")

    def test_plan_only_run_is_orphaned_with_low_confidence(self) -> None:
        """An incomplete legacy directory must not be presented as successful."""
        directory = write_legacy_run(self.legacy_root, "suite_a", plan_only=True)
        candidate = inspect_legacy_run(directory)
        self.assertEqual(candidate.state, RunState.ORPHANED)
        self.assertEqual(candidate.status_confidence, "low")

    def test_unknown_fields_are_reported_not_invented(self) -> None:
        directory = self.legacy_root / "sparse" / "leaf"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        candidate = inspect_legacy_run(directory)
        self.assertIn("model_id", candidate.unknown_fields)
        self.assertIn("task_id", candidate.unknown_fields)
        self.assertEqual(candidate.model_id, "unknown")

    # -- registry projection ------------------------------------------------

    def test_imported_records_are_marked_legacy(self) -> None:
        write_legacy_run(self.legacy_root, "suite_a", with_checkpoint=True)
        import_legacy_runs(self.registry, root=self.legacy_root)
        record = self.registry.list_runs()[0]
        self.assertTrue(record.legacy)
        self.assertEqual(record.status_confidence, "high")
        self.assertEqual(record.state, RunState.COMPLETED)
        self.assertEqual(record.terminal_reason, "legacy_import")

    def test_transition_history_is_reconstructed_legally(self) -> None:
        write_legacy_run(self.legacy_root, "suite_a")
        import_legacy_runs(self.registry, root=self.legacy_root)
        record = self.registry.list_runs()[0]
        transitions = self.registry.list_transitions(record.run_id)
        self.assertEqual(transitions[0]["to_state"], "CREATED")
        self.assertEqual(transitions[-1]["to_state"], "COMPLETED")
        self.assertEqual(transitions[-1]["actor"], "legacy-importer")

    def test_mapping_table_records_the_source_path(self) -> None:
        directory = write_legacy_run(self.legacy_root, "suite_a")
        import_legacy_runs(self.registry, root=self.legacy_root)
        run_id = legacy_run_id(directory)
        mapping = self.registry.legacy_mapping(run_id)
        self.assertEqual(mapping["legacy_path"], str(directory.resolve()))
        self.assertEqual(
            self.registry.legacy_run_for_path_hash(legacy_path_hash(directory)), run_id
        )

    def test_legacy_checkpoints_are_not_registered_as_resumable(self) -> None:
        """DND-003: a legacy model.pt is a warm-start source, not a resume point."""
        write_legacy_run(self.legacy_root, "suite_a", with_checkpoint=True)
        import_legacy_runs(self.registry, root=self.legacy_root)
        record = self.registry.list_runs()[0]
        self.assertEqual(self.registry.list_checkpoints(record.run_id), [])
        self.assertIsNone(record.latest_checkpoint_id)
        self.assertIsNone(record.best_checkpoint_id)

    def test_legacy_runs_can_be_filtered_out(self) -> None:
        write_legacy_run(self.legacy_root, "suite_a")
        import_legacy_runs(self.registry, root=self.legacy_root)
        self.assertEqual(len(self.registry.list_runs(include_legacy=True)), 1)
        self.assertEqual(len(self.registry.list_runs(include_legacy=False)), 0)

    def test_variant_id_is_assigned_to_legacy_runs(self) -> None:
        write_legacy_run(self.legacy_root, "suite_a", model_id="split_knet")
        import_legacy_runs(self.registry, root=self.legacy_root)
        record = self.registry.list_runs()[0]
        self.assertTrue(record.variant_id.startswith("sha256:"))
        self.assertEqual(record.implementation_id, "bench_split_adapter_v1")

    def test_same_model_different_init_do_not_collide(self) -> None:
        """C-03 for imported runs."""
        trained = write_legacy_run(self.legacy_root, "suite_t", model_id="split_knet")
        untrained = write_legacy_run(self.legacy_root, "suite_u", model_id="split_knet")
        plan = json.loads((untrained / "run_plan.json").read_text())
        plan["init_id"] = "untrained"
        (untrained / "run_plan.json").write_text(json.dumps(plan), encoding="utf-8")
        metrics = json.loads((untrained / "metrics.json").read_text())
        (untrained / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

        import_legacy_runs(self.registry, root=self.legacy_root)
        records = self.registry.list_runs()
        self.assertEqual(len(records), 2)
        self.assertEqual(len({record.run_id for record in records}), 2)
        self.assertEqual(len({record.variant_id for record in records}), 2)

    def test_broken_directory_is_reported_not_fatal(self) -> None:
        write_legacy_run(self.legacy_root, "good")
        broken = self.legacy_root / "broken" / "leaf"
        broken.mkdir(parents=True)
        (broken / "metrics.json").write_text("{ this is not json", encoding="utf-8")
        report = import_legacy_runs(self.registry, root=self.legacy_root)
        # the good run still imports; the broken one is skipped, not fatal
        self.assertGreaterEqual(report.imported, 1)
        self.assertEqual(self.registry.count_runs(include_legacy=False), 0)

    def test_missing_root_yields_nothing(self) -> None:
        report = import_legacy_runs(self.registry, root=self.base / "does-not-exist")
        self.assertEqual(report.scanned, 0)
        self.assertEqual(report.imported, 0)


@unittest.skipUnless(REAL_RUNS_ROOT.is_dir(), "repository runs/ tree not present")
class RealRunsTreeTests(unittest.TestCase):
    """Import a bounded slice of the repository's actual runs/ tree, read-only."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.registry = SqliteRegistry(Path(self._tmp.name) / "registry.sqlite3")
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        try:
            self.registry.close()
        except Exception:
            pass
        self._tmp.cleanup()

    def test_import_a_bounded_slice_of_the_real_tree(self) -> None:
        report = import_legacy_runs(self.registry, root=REAL_RUNS_ROOT, limit=25)
        self.assertGreater(report.imported, 0, "no legacy runs were importable from runs/")
        self.assertEqual(report.errors, ())
        for record in self.registry.list_runs(limit=100):
            self.assertTrue(record.legacy)
            self.assertIn(
                record.status_confidence, ("high", "medium", "low", "unknown")
            )
            self.assertTrue(Path(record.run_dir).exists())
            self.assertTrue(record.implementation_id)

    def test_real_tree_is_not_written_to(self) -> None:
        """Belt-and-braces: nothing under runs/ may change mtime during import."""
        directories = list(discover_legacy_runs(REAL_RUNS_ROOT, limit=10))
        self.assertTrue(directories)
        before = {path: path.stat().st_mtime_ns for path in directories}
        import_legacy_runs(self.registry, directories=directories)
        after = {path: path.stat().st_mtime_ns for path in directories}
        self.assertEqual(before, after)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
