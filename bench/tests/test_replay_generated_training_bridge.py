from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bench.tasks.replay_generated_data import (
    build_replay_generated_dataset,
    build_replay_generated_system_model,
)
from bench.visualization.replay_suite_scenario import load_suite_yaml, select_task_from_suite


class ReplayGeneratedTrainingBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        suite_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "suite_phase6f_kalmannet_adcs_tiny.yaml"
        )
        self.suite = load_suite_yaml(suite_path)
        self.task = select_task_from_suite(self.suite, "phase6f_adcs_bias_tiny_v0")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_build_replay_generated_dataset_shapes_and_determinism(self) -> None:
        dataset_a = build_replay_generated_dataset(self.suite, self.task, seed=0)
        dataset_b = build_replay_generated_dataset(self.suite, self.task, seed=0)

        self.assertEqual(dataset_a.x_train.shape[2], 9)
        self.assertEqual(dataset_a.y_train.shape[2], 6)
        self.assertEqual(dataset_a.x_val.shape[2], 9)
        self.assertEqual(dataset_a.y_val.shape[2], 6)
        self.assertEqual(dataset_a.x_test.shape[2], 9)
        self.assertEqual(dataset_a.y_test.shape[2], 6)
        self.assertEqual(dataset_a.time_s.shape, (25,))
        self.assertEqual(dataset_a.trajectory_id_train.shape, (32,))
        self.assertEqual(dataset_a.trajectory_id_val.shape, (8,))
        self.assertEqual(dataset_a.trajectory_id_test.shape, (8,))

        np.testing.assert_allclose(dataset_a.x_train, dataset_b.x_train)
        np.testing.assert_allclose(dataset_a.y_train, dataset_b.y_train)
        np.testing.assert_allclose(dataset_a.x_val, dataset_b.x_val)
        np.testing.assert_allclose(dataset_a.y_val, dataset_b.y_val)
        np.testing.assert_allclose(dataset_a.x_test, dataset_b.x_test)
        np.testing.assert_allclose(dataset_a.y_test, dataset_b.y_test)
        np.testing.assert_allclose(dataset_a.time_s, dataset_b.time_s)

        self.assertTrue(np.isfinite(dataset_a.x_train).all())
        self.assertTrue(np.isfinite(dataset_a.y_train).all())
        self.assertTrue(np.isfinite(dataset_a.x_val).all())
        self.assertTrue(np.isfinite(dataset_a.y_val).all())
        self.assertTrue(np.isfinite(dataset_a.x_test).all())
        self.assertTrue(np.isfinite(dataset_a.y_test).all())
        self.assertEqual(
            dataset_a.meta["replay_generated"]["observed_state"],
            [0, 1, 2, 3, 4, 5],
        )
        self.assertEqual(dataset_a.meta["dims"], {"x_dim": 9, "y_dim": 6, "T": 25})
        self.assertEqual(dataset_a.meta["seed"], 0)
        self.assertEqual(dataset_a.system_model["state_dim"], 9)
        self.assertEqual(dataset_a.system_model["measurement_dim"], 6)

    def test_system_model_generation(self) -> None:
        system_model = build_replay_generated_system_model(
            task_cfg=self.task,
            dt_s=0.5,
        )
        F = np.asarray(system_model["F"], dtype=np.float64)
        H = np.asarray(system_model["H"], dtype=np.float64)
        self.assertEqual(F.shape, (9, 9))
        self.assertEqual(H.shape, (6, 9))
        self.assertTrue(np.isfinite(F).all())
        self.assertTrue(np.isfinite(H).all())
        self.assertTrue(np.allclose(H[:, :6], np.eye(6)))

    def test_invalid_observed_state_raises(self) -> None:
        bad_task = copy.deepcopy(self.task)
        bad_task["observation"] = dict(bad_task["observation"])
        bad_task["observation"]["observed_state"] = [0, 1, 2, 3, 4, 9]
        with self.assertRaises(ValueError):
            build_replay_generated_dataset(self.suite, bad_task, seed=0)

    def test_invalid_dimensions_raises(self) -> None:
        bad_task = copy.deepcopy(self.task)
        bad_task["x_dim"] = 8
        with self.assertRaises(ValueError):
            build_replay_generated_dataset(self.suite, bad_task, seed=0)


@dataclass
class ReplayGeneratedTrainingBridgeResult:
    ok: bool
    note: str


def run_replay_generated_training_bridge_tests() -> ReplayGeneratedTrainingBridgeResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        ReplayGeneratedTrainingBridgeTests
    )
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return ReplayGeneratedTrainingBridgeResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Replay-generated training bridge tests passed"
            if result.wasSuccessful()
            else "Replay-generated training bridge tests failed"
        ),
    )


if __name__ == "__main__":
    unittest.main()
