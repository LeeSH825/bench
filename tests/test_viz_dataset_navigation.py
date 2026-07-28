from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from viz.app.components.overlay_picker import filter_run_index, scenario_label, split_label
from viz.app.components.regime_strip import build_regime_strip
from viz.app.views.run_inspector import (
    build_run_inspector_bundle,
    preferred_trajectory_info,
    trajectory_option_label,
)
from viz.contract import ContractError, deterministic_traj_index
from viz.io import loader as loader_module
from viz.io.loader import assert_overlay_compatible, load_run
from viz.io.writer import write_viz_artifacts


REPO_ROOT = Path(__file__).resolve().parents[1]


def _diagnostics(*, n_seq: int, n_step: int, full: bool) -> dict[str, np.ndarray]:
    gain = np.full((n_seq, n_step, 2, 1), 0.25, dtype=np.float32)
    innov = np.full((n_seq, n_step, 1), 0.01, dtype=np.float32)
    if not full:
        return {"gain": gain, "innov": innov}
    p = np.broadcast_to(np.eye(2, dtype=np.float32), (n_seq, n_step, 2, 2)).copy()
    s = np.ones((n_seq, n_step, 1, 1), dtype=np.float32)
    return {"P": p, "S": s, "gain": gain, "innov": innov}


def _write_run(
    root: Path,
    name: str,
    *,
    data_split: str = "test",
    model_id: str = "oracle_kf",
    n_seq: int = 16,
    n_step: int = 20,
    k_traj: int = 4,
    trajectory_ids: np.ndarray | None = None,
    task_id: str = "navigation_linear_v0",
    scenario_id: str = "scenario_navigation",
    full_diagnostics: bool = True,
) -> Path:
    t = np.arange(n_step, dtype=np.float32)
    x_true = np.zeros((n_seq, n_step, 2), dtype=np.float32)
    offsets = np.arange(n_seq, dtype=np.float32).reshape(n_seq, 1, 1) * np.float32(0.01)
    x_hat = x_true + offsets
    y_obs = np.zeros((n_seq, n_step, 1), dtype=np.float32)
    event = np.zeros((n_seq, n_step), dtype=bool)
    eclipse = np.zeros((n_seq, n_step), dtype=bool)
    if n_seq:
        event[0, 2:5] = True
    if n_seq > 4:
        eclipse[4, 7:11] = True
    run_dir = root / name
    write_viz_artifacts(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        suite_name="navigation_suite",
        task_id=task_id,
        task_family="linear_gaussian_v0",
        scenario_id=scenario_id,
        model_id=model_id,
        seed=3,
        track_id="frozen",
        init_id="checkpoint_a",
        run_status="ok",
        time_s=t,
        time_meta={"time_source": "fixture", "time_unit": "s", "dt_s": 1.0},
        x_true=x_true,
        y_obs=y_obs,
        x_hat=x_hat,
        split_extras={"event_flag_seq": event, "eclipse_flag_seq": eclipse},
        diagnostics=_diagnostics(n_seq=n_seq, n_step=n_step, full=full_diagnostics),
        adapter_meta={"adapter_id": model_id},
        k_traj=k_traj,
        data_split=data_split,
        split_source="explicit",
        trajectory_ids=trajectory_ids,
        trajectory_id_source="fixture:trajectory_id",
        scenario_meta={"display_name": "Navigation fixture", "parameters": {"dt": 1.0}},
    )
    return run_dir


def _fixture_ids() -> np.ndarray:
    ids = np.arange(100, 116, dtype=np.int64)
    ids[[0, 4, 8, 12]] = np.asarray([0, 5, 10, 15], dtype=np.int64)
    return ids


class VizDatasetNavigationTest(unittest.TestCase):
    def test_data_split_and_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "test", trajectory_ids=_fixture_ids()))
            self.assertEqual(run.meta["data_spec"]["split"], "test")
            self.assertEqual(run.meta["data_spec"]["split_source"], "explicit")
            self.assertEqual(run.meta["data_spec"]["num_trajectories"], 16)
            self.assertEqual(run.meta["data_spec"]["num_stored_trajectories"], 4)
            self.assertEqual([item.stored_index for item in run.trajectories], [0, 1, 2, 3])
            self.assertEqual([item.source_trajectory_id for item in run.trajectories], [0, 5, 10, 15])
            self.assertEqual([item.file for item in run.trajectories], [f"series/traj_{i:04d}.npz" for i in range(4)])

    def test_stored_index_and_source_id_load_same_selected_file(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "selectors", trajectory_ids=_fixture_ids()))
            stored = run.load_trajectory(stored_index=1)
            source = run.load_trajectory(source_trajectory_id=5)
            for key in stored:
                np.testing.assert_array_equal(stored[key], source[key])
            with self.assertRaisesRegex(KeyError, "stored trajectory index 8 is unavailable"):
                run.load_trajectory(stored_index=8)
            with self.assertRaisesRegex(KeyError, "source trajectory ID 8 is unavailable"):
                run.load_trajectory(source_trajectory_id=8)

    def test_legacy_split_and_source_id_are_unknown(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_dir = _write_run(Path(tmp), "legacy", trajectory_ids=_fixture_ids())
            meta_path = run_dir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.pop("data_spec")
            meta.pop("trajectories")
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            run = load_run(run_dir)
            self.assertEqual(run.meta["data_spec"]["split"], "unknown")
            self.assertEqual(run.meta["data_spec"]["split_source"], "legacy_unknown")
            self.assertTrue(all(item.source_trajectory_id is None for item in run.trajectories))
            self.assertEqual(split_label(run), "Unknown (legacy artifact)")

    def test_legacy_without_counts_uses_scanned_file_count_as_minimum_total(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_dir = _write_run(Path(tmp), "legacy_no_counts", trajectory_ids=_fixture_ids())
            meta_path = run_dir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for key in ("data_spec", "trajectories", "N_test", "traj_index"):
                meta.pop(key)
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            run = load_run(run_dir)
            self.assertEqual(run.meta["data_spec"]["num_trajectories"], 4)
            self.assertEqual(run.meta["data_spec"]["num_stored_trajectories"], 4)

    def test_duplicate_source_id_is_rejected_before_write(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            ids = np.zeros(16, dtype=np.int64)
            with self.assertRaisesRegex(ValueError, "source trajectory IDs must be unique"):
                _write_run(Path(tmp), "duplicate", trajectory_ids=ids)

    def test_missing_manifest_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_dir = _write_run(Path(tmp), "missing", trajectory_ids=_fixture_ids())
            (run_dir / "series" / "traj_0002.npz").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "manifest entry is missing"):
                load_run(run_dir)

    def test_extra_trajectory_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_dir = _write_run(Path(tmp), "extra", trajectory_ids=_fixture_ids())
            source = run_dir / "series" / "traj_0000.npz"
            extra = run_dir / "series" / "traj_0004.npz"
            extra.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ContractError, "manifest/file mismatch"):
                load_run(run_dir)

    def test_aggregate_only_run_has_no_selectable_trajectory(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "aggregate_only", k_traj=0))
            self.assertEqual(run.trajectories, ())
            self.assertEqual(run.meta["data_spec"]["num_stored_trajectories"], 0)
            self.assertIsNotNone(run.aggregate)

    def test_deterministic_selection_and_cross_model_source_ids(self) -> None:
        self.assertEqual(deterministic_traj_index(32, 8), [0, 4, 8, 12, 16, 20, 24, 28])
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            ids = np.arange(32, dtype=np.int64) * 2
            mb = load_run(_write_run(Path(tmp), "mb", model_id="oracle_kf", n_seq=32, k_traj=8, trajectory_ids=ids))
            knet = load_run(
                _write_run(
                    Path(tmp),
                    "knet",
                    model_id="kalmannet_tsp",
                    n_seq=32,
                    k_traj=8,
                    trajectory_ids=ids,
                    full_diagnostics=False,
                )
            )
            self.assertEqual(
                [item.source_trajectory_id for item in mb.trajectories],
                [item.source_trajectory_id for item in knet.trajectories],
            )

    def test_split_mismatch_blocks_overlay(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            test_run = load_run(_write_run(Path(tmp), "test", data_split="test", trajectory_ids=_fixture_ids()))
            val_run = load_run(_write_run(Path(tmp), "validation", data_split="validation", trajectory_ids=_fixture_ids()))
            with self.assertRaisesRegex(ContractError, "data split mismatch"):
                assert_overlay_compatible(test_run, val_run)

    def test_matching_source_overlay_succeeds_and_mismatch_blocks(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", model_id="oracle_kf", trajectory_ids=_fixture_ids()))
            matched = load_run(
                _write_run(
                    Path(tmp), "matched", model_id="kalmannet_tsp", trajectory_ids=_fixture_ids(), full_diagnostics=False
                )
            )
            assert_overlay_compatible(base, matched, source_trajectory_id=10)
            other_ids = _fixture_ids()
            other_ids[8] = 11
            mismatched = load_run(
                _write_run(
                    Path(tmp), "mismatched", model_id="kalmannet_tsp", trajectory_ids=other_ids, full_diagnostics=False
                )
            )
            with self.assertRaisesRegex(ContractError, "did not store this trajectory"):
                assert_overlay_compatible(base, mismatched, source_trajectory_id=10)

    def test_bundle_selection_changes_trajectory_not_dataset_summary(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "bundle", trajectory_ids=_fixture_ids()))
            first = build_run_inspector_bundle(run, traj_idx=0, axis_mode="split")
            second = build_run_inspector_bundle(run, traj_idx=1, axis_mode="split")
            self.assertEqual(first["dataset_summary"], second["dataset_summary"])
            self.assertNotEqual(first["trajectory_summary"]["generic_state_rmse"], second["trajectory_summary"]["generic_state_rmse"])
            self.assertFalse(np.array_equal(first["traj"]["x_hat"], second["traj"]["x_hat"]))
            first_regime = build_regime_strip(run.meta, first["traj"])
            second_regime = build_regime_strip(run.meta, second["traj"])
            self.assertNotEqual(first_regime.intervals, second_regime.intervals)

    def test_selector_label_and_scenario_provenance(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "label", trajectory_ids=_fixture_ids()))
            label = trajectory_option_label(run.trajectories[1])
            self.assertEqual(label, "Stored #1 · Source ID 5 · T=20 · Event=No · Eclipse=Yes")
            self.assertEqual(scenario_label(run), "Navigation fixture · scenario_navigation")

    def test_source_id_selection_is_preserved_across_models(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", model_id="oracle_kf", trajectory_ids=_fixture_ids()))
            overlay = load_run(
                _write_run(
                    Path(tmp), "overlay", model_id="kalmannet_tsp", trajectory_ids=_fixture_ids(), full_diagnostics=False
                )
            )
            selected = preferred_trajectory_info(
                overlay,
                query_stored_index=0,
                preserve_previous=True,
                previous_source_trajectory_id=base.trajectories[2].source_trajectory_id,
            )
            self.assertIsNotNone(selected)
            self.assertEqual(selected.source_trajectory_id, 10)
            self.assertEqual(selected.stored_index, 2)

            mismatched_ids = _fixture_ids()
            mismatched_ids[8] = 11
            mismatched = load_run(
                _write_run(
                    Path(tmp),
                    "mismatched_selection",
                    model_id="kalmannet_tsp",
                    trajectory_ids=mismatched_ids,
                    full_diagnostics=False,
                )
            )
            with self.assertRaisesRegex(KeyError, "Source ID 10 is not stored"):
                preferred_trajectory_info(
                    mismatched,
                    query_stored_index=2,
                    preserve_previous=True,
                    previous_source_trajectory_id=10,
                )

    def test_filter_index_exposes_only_existing_combinations(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            test_run = load_run(_write_run(Path(tmp), "test", data_split="test"))
            validation_run = load_run(_write_run(Path(tmp), "validation", data_split="validation"))
            filtered = filter_run_index([test_run, validation_run], {"data_split": "validation"})
            self.assertEqual([run.run_dir for run in filtered], [validation_run.run_dir])

    def test_only_selected_trajectory_npz_is_loaded_after_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "lazy", trajectory_ids=_fixture_ids()))
            original = loader_module._load_npz
            paths: list[Path] = []

            def recording_load(path: Path) -> dict[str, np.ndarray]:
                paths.append(path)
                return original(path)

            with patch("viz.io.loader._load_npz", side_effect=recording_load):
                run.load_trajectory(stored_index=2)
            self.assertEqual([path.name for path in paths], ["traj_0002.npz"])

    def test_kalmannet_keeps_p_s_absent_and_mb_kf_keeps_them(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            mb = load_run(_write_run(Path(tmp), "mb", model_id="oracle_kf"))
            knet = load_run(
                _write_run(Path(tmp), "knet", model_id="kalmannet_tsp", full_diagnostics=False)
            )
            mb_traj = mb.load_trajectory(stored_index=0)
            knet_traj = knet.load_trajectory(stored_index=0)
            self.assertIn("P", mb_traj)
            self.assertIn("S", mb_traj)
            self.assertNotIn("P", knet_traj)
            self.assertNotIn("S", knet_traj)
            self.assertFalse(knet.meta["capabilities"]["covariance"])
            self.assertFalse(knet.meta["capabilities"]["innovation_cov"])


if __name__ == "__main__":
    unittest.main()
