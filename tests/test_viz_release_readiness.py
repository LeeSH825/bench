from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tests.test_viz_dataset_navigation import _fixture_ids, _write_run
from viz.app.components.overlay_picker import discover_run_index, discover_runs
from viz.app.views.run_inspector import available_gain_sources
from viz.contract import ContractError, UnsupportedArtifactVersion
from viz.io.loader import assert_overlay_compatible, load_run


REPO_ROOT = Path(__file__).resolve().parents[1]


class VizReleaseReadinessTests(unittest.TestCase):
    def test_empty_artifact_root_is_an_empty_navigation_state(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp) / "empty"
            root.mkdir()
            self.assertEqual(discover_runs(root), [])
            runs, errors, _ = discover_run_index(root)
            self.assertEqual(runs, [])
            self.assertEqual(errors, [])

    def test_missing_meta_and_unsupported_version_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            missing = Path(tmp) / "missing"
            missing.mkdir()
            with self.assertRaises(FileNotFoundError):
                load_run(missing)

            run_dir = _write_run(Path(tmp), "unsupported", trajectory_ids=_fixture_ids())
            meta_path = run_dir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["artifact_version"] = "9.9"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(UnsupportedArtifactVersion):
                load_run(run_dir)

    def test_legacy_artifact_is_unknown_and_aggregate_only_is_selectable_free(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            legacy = _write_run(Path(tmp), "legacy", trajectory_ids=_fixture_ids())
            meta_path = legacy / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.pop("data_spec")
            meta.pop("trajectories")
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            loaded_legacy = load_run(legacy)
            self.assertEqual(loaded_legacy.meta["data_spec"]["split"], "unknown")
            self.assertEqual(loaded_legacy.meta["data_spec"]["split_source"], "legacy_unknown")
            self.assertTrue(all(item.source_trajectory_id is None for item in loaded_legacy.trajectories))

            aggregate_only = load_run(_write_run(Path(tmp), "aggregate_only", k_traj=0))
            self.assertEqual(aggregate_only.trajectories, ())
            self.assertIsNotNone(aggregate_only.aggregate)
            self.assertFalse(aggregate_only.meta["data_spec"]["is_live"])

    def test_manifest_scan_is_lazy_and_selected_file_is_loaded_later(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_dir = _write_run(Path(tmp), "lazy", trajectory_ids=_fixture_ids())
            with patch("viz.io.loader._load_npz", side_effect=AssertionError("eager NPZ load")):
                run = load_run(run_dir, load_aggregate=False, load_metrics=False)
            self.assertEqual(len(run.trajectories), 4)
            selected = run.load_trajectory(stored_index=0)
            self.assertIn("x_hat", selected)

    def test_malformed_selected_trajectory_is_rejected_without_fallback(self) -> None:
        source = REPO_ROOT / "runs/viz_v4c_fixtures/combined_only"
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_dir = Path(tmp) / "malformed"
            shutil.copytree(source, run_dir)
            target = run_dir / "series/traj_0000.npz"
            with np.load(target, allow_pickle=False) as data:
                arrays = {key: np.array(data[key], copy=True) for key in data.files}
            arrays["gain"][:, 0, 0] = np.nan
            np.savez_compressed(target, **arrays)
            run = load_run(run_dir)
            with self.assertRaisesRegex(ContractError, "NaN/Inf at a valid"):
                run.load_trajectory(stored_index=0)

    def test_learned_filter_has_no_physical_covariance_and_split_has_component_labels(self) -> None:
        knet_dir = REPO_ROOT / "runs/viz_v4c_cross_models/A_linear_split_train_smoke_v0/kalmannet_tsp/frozen/seed_0/scenario_1a547ae6bce5"
        split_dir = REPO_ROOT / "runs/viz_v4c_cross_models/A_linear_split_train_smoke_v0/split_knet/frozen/seed_0/scenario_1a547ae6bce5"
        knet = load_run(knet_dir)
        knet_traj = knet.load_trajectory(stored_index=0)
        self.assertNotIn("P", knet_traj)
        self.assertNotIn("S", knet_traj)
        self.assertEqual(available_gain_sources(knet.meta, knet_traj), [("gain", "Learned Kalman gain")])

        split = load_run(split_dir)
        split_traj = split.load_trajectory(stored_index=0)
        labels = [label for _key, label in available_gain_sources(split.meta, split_traj)]
        self.assertEqual(labels, ["Learned combined Kalman gain", "Learned G1 factor", "Learned G2 factor"])
        self.assertNotIn("P", split_traj)
        self.assertNotIn("S", split_traj)

    def test_split_source_overlay_mismatch_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", trajectory_ids=_fixture_ids()))
            ids = _fixture_ids()
            ids[8] = 11
            other = load_run(_write_run(Path(tmp), "other", model_id="kalmannet_tsp", trajectory_ids=ids, full_diagnostics=False))
            with self.assertRaisesRegex(ContractError, "selected run did not store this trajectory"):
                assert_overlay_compatible(base, other, source_trajectory_id=10)

    def test_failed_run_remains_visible_as_failed_status(self) -> None:
        path = REPO_ROOT / "runs/viz_v4a_fixtures/failed_train_nan"
        run = load_run(path)
        self.assertEqual(run.meta["run_status"], "train_nan")

    def test_gain_capability_is_key_driven(self) -> None:
        meta = {"diagnostic_semantics": {"gain": "learned_combined_kalman_gain"}}
        self.assertEqual(available_gain_sources(meta, {"innov": np.zeros((2, 1))}), [])
        self.assertEqual(
            available_gain_sources(meta, {"gain": np.zeros((2, 1, 1))}),
            [("gain", "Learned combined Kalman gain")],
        )


if __name__ == "__main__":
    unittest.main()
