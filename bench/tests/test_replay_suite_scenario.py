from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bench.visualization.replay_suite_scenario import (
    REPLAY_SCENARIO_FILENAME,
    REPLAY_SCENARIO_META_FILENAME,
    load_suite_yaml,
    materialize_adcs_replay_task,
    save_replay_input_npz,
    select_task_from_suite,
    validate_adcs_replay_task,
)


def _short_suite_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "suite_phase6a_replay_short.yaml"
    )


class ReplaySuiteScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.suite = load_suite_yaml(_short_suite_path())
        self.task = select_task_from_suite(
            self.suite,
            "phase6a_adcs_bias_short_v0",
        )

    def test_load_select_validate_and_materialize(self) -> None:
        resolved = validate_adcs_replay_task(
            self.suite,
            self.task,
            seed=0,
        )
        self.assertEqual(resolved["suite_name"], "phase6a_replay_short")
        self.assertEqual(resolved["sequence_length_T"], 25)
        self.assertEqual(resolved["dt_s"], 0.5)

        scenario = materialize_adcs_replay_task(
            self.suite,
            self.task,
            seed=0,
        )
        self.assertEqual(scenario.x_true.shape, (1, 25, 9))
        self.assertEqual(scenario.y_obs.shape, (1, 25, 6))
        self.assertEqual(scenario.time_s.shape, (25,))
        self.assertEqual(scenario.trajectory_id.shape, (1,))
        self.assertAlmostEqual(float(scenario.time_s[1]), 0.5)
        self.assertAlmostEqual(float(scenario.time_s[-1]), 12.0)
        self.assertEqual(
            scenario.meta["time"]["sequence_length_T"],
            self.task["sequence_length_T"],
        )
        self.assertEqual(
            scenario.meta["time"]["dt_s"],
            self.task["time"]["dt_s"],
        )

    def test_noise_and_scenario_id_are_deterministic(self) -> None:
        first = materialize_adcs_replay_task(
            self.suite,
            self.task,
            seed=0,
        )
        second = materialize_adcs_replay_task(
            self.suite,
            self.task,
            seed=0,
        )
        self.assertTrue(np.array_equal(first.x_true, second.x_true))
        self.assertTrue(np.array_equal(first.y_obs, second.y_obs))
        self.assertEqual(first.scenario_id, second.scenario_id)
        self.assertRegex(first.scenario_id, r"^scenario_[0-9a-f]{8}$")

    def test_invalid_dt_observation_dynamics_and_system_raise(self) -> None:
        cases = []
        invalid_dt = copy.deepcopy(self.task)
        invalid_dt["time"]["dt_s"] = 0.0
        cases.append((invalid_dt, "task.time.dt_s"))

        invalid_observation = copy.deepcopy(self.task)
        invalid_observation["observation"]["observed_state"][-1] = 99
        cases.append((invalid_observation, "observed_state"))

        invalid_dynamics = copy.deepcopy(self.task)
        invalid_dynamics["dynamics"]["type"] = "rigid_body_truth"
        cases.append((invalid_dynamics, "simple_attitude_bias"))

        invalid_system = copy.deepcopy(self.task)
        invalid_system["system_type"] = "linear"
        cases.append((invalid_system, "task.system_type"))

        for task, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    materialize_adcs_replay_task(
                        self.suite,
                        task,
                        seed=0,
                    )

    def test_save_outputs_exist_and_have_required_keys(self) -> None:
        scenario = materialize_adcs_replay_task(
            self.suite,
            self.task,
            seed=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            npz_path, meta_path = save_replay_input_npz(scenario, tmp)
            self.assertEqual(npz_path.name, REPLAY_SCENARIO_FILENAME)
            self.assertEqual(meta_path.name, REPLAY_SCENARIO_META_FILENAME)
            self.assertTrue(npz_path.exists())
            self.assertTrue(meta_path.exists())
            with np.load(npz_path, allow_pickle=False) as data:
                self.assertEqual(
                    set(data.files),
                    {"time_s", "x_true", "y_obs", "trajectory_id"},
                )
                self.assertEqual(data["x_true"].shape, (1, 25, 9))
                self.assertEqual(data["y_obs"].shape, (1, 25, 6))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(
                meta["schema_version"],
                "phase6a_replay_input_v1",
            )
            self.assertEqual(meta["vizard"]["position_source"], "dummy_circular_orbit")


@dataclass
class ReplaySuiteScenarioResult:
    ok: bool
    note: str


def run_replay_suite_scenario_tests() -> ReplaySuiteScenarioResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        ReplaySuiteScenarioTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return ReplaySuiteScenarioResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Phase 6A replay suite scenario tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
