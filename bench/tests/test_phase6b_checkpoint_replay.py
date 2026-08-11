from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bench.visualization.phase6b_checkpoint_replay import (
    IDENTITY_MODEL_ID,
    main,
    run_phase6b_replay,
)
from bench.visualization.pred_artifact import load_pred_artifact
from bench.visualization.replay_suite_scenario import (
    REPLAY_SCENARIO_FILENAME,
    REPLAY_SCENARIO_META_FILENAME,
)


class Phase6BCheckpointReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.replay_dir = self.root / "replay"
        self.replay_dir.mkdir()
        self.n_trajectory = 1
        self.n_step = 5
        self.x_dim = 9
        self.y_dim = 6
        self.time_s = np.arange(self.n_step, dtype=np.float32) * 0.5
        self.x_true = np.arange(
            self.n_trajectory * self.n_step * self.x_dim,
            dtype=np.float32,
        ).reshape(self.n_trajectory, self.n_step, self.x_dim)
        self.y_obs = (
            self.x_true[..., : self.y_dim]
            + np.float32(0.25)
        )
        self.trajectory_id = np.array([7], dtype=np.int64)
        self.meta = {
            "schema_version": "phase6a_replay_input_v1",
            "suite_name": "phase6b_test_suite",
            "suite_version": "0.1.0",
            "task_id": "phase6b_test_task",
            "task_name": "Phase 6B test task",
            "seed": 3,
            "scenario_id": "scenario_1234abcd",
            "time": {
                "dt_s": 0.5,
                "sequence_length_T": self.n_step,
                "duration_s": 2.0,
            },
            "state_dim": self.x_dim,
            "measurement_dim": self.y_dim,
            "state_schema": {
                "attitude": {
                    "type": "mrp",
                    "name": "sigma_BN",
                    "indices": [0, 1, 2],
                },
                "angular_rate": {
                    "type": "rad_s",
                    "name": "omega_BN_B",
                    "indices": [3, 4, 5],
                },
                "gyro_bias": {
                    "type": "rad_s",
                    "name": "gyro_bias",
                    "indices": [6, 7, 8],
                    "optional": True,
                },
            },
            "observation": {
                "type": "partial",
                "observed_state": [0, 1, 2, 3, 4, 5],
            },
            "replay": {
                "enabled": True,
                "expected_model_input": "y_obs",
            },
            "vizard": {"position_source": "dummy_circular_orbit"},
        }
        self._write_replay_input()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_replay_input(
        self,
        *,
        meta: dict | None = None,
        x_true: np.ndarray | None = None,
        y_obs: np.ndarray | None = None,
    ) -> None:
        np.savez_compressed(
            self.replay_dir / REPLAY_SCENARIO_FILENAME,
            time_s=self.time_s,
            x_true=self.x_true if x_true is None else x_true,
            y_obs=self.y_obs if y_obs is None else y_obs,
            trajectory_id=self.trajectory_id,
        )
        (self.replay_dir / REPLAY_SCENARIO_META_FILENAME).write_text(
            json.dumps(self.meta if meta is None else meta, indent=2) + "\n",
            encoding="utf-8",
        )

    def test_identity_baseline_zero_fills_unobserved_state(self) -> None:
        output_dir = self.root / "identity"
        artifact_path, meta_path = run_phase6b_replay(
            self.replay_dir,
            out_dir=output_dir,
        )
        self.assertTrue(artifact_path.exists())
        self.assertTrue(meta_path.exists())

        loaded = load_pred_artifact(artifact_path)
        self.assertEqual(loaded["x_hat"].shape, self.x_true.shape)
        np.testing.assert_array_equal(loaded["x_hat"][..., :6], self.y_obs)
        np.testing.assert_array_equal(
            loaded["x_hat"][..., 6:],
            np.zeros_like(self.x_true[..., 6:]),
        )
        np.testing.assert_array_equal(loaded["time_s"], self.time_s)
        np.testing.assert_array_equal(
            loaded["trajectory_id"],
            self.trajectory_id,
        )

    def test_identity_baseline_true_fills_unobserved_state(self) -> None:
        loaded = load_pred_artifact(
            run_phase6b_replay(
                self.replay_dir,
                out_dir=self.root / "true_fill",
                allow_true_fill=True,
            )[0]
        )
        np.testing.assert_array_equal(loaded["x_hat"][..., :6], self.y_obs)
        np.testing.assert_array_equal(
            loaded["x_hat"][..., 6:],
            self.x_true[..., 6:],
        )
        self.assertEqual(
            loaded["meta"]["identity_fill_policy"],
            "observed_from_y_obs_unobserved_from_x_true",
        )

    def test_metadata_records_replay_provenance(self) -> None:
        loaded = load_pred_artifact(
            run_phase6b_replay(
                self.replay_dir,
                out_dir=self.root / "metadata",
            )[0]
        )
        meta = loaded["meta"]
        self.assertEqual(meta["phase"], "phase6b_checkpoint_replay")
        self.assertEqual(meta["model_id"], IDENTITY_MODEL_ID)
        self.assertFalse(meta["is_trained_checkpoint"])
        self.assertFalse(meta["used_fallback"])
        self.assertEqual(meta["scenario_id"], "scenario_1234abcd")
        self.assertEqual(meta["task_id"], "phase6b_test_task")
        self.assertEqual(meta["suite_name"], "phase6b_test_suite")
        self.assertTrue(meta["not_for_benchmark_reporting"])
        self.assertIn("not a trained model result", meta["notes"])

    def test_missing_replay_input_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_phase6b_replay(
                self.root / "missing",
                out_dir=self.root / "missing_output",
            )

    def test_invalid_observed_state_raises(self) -> None:
        cases = (
            ([0, 1, 2, 3, 4, 99], "out of bounds"),
            ([0, 1, 2, 3, 4, 4], "duplicate"),
            ([0, 1, 2, 3, 4], "must equal y_obs Dy"),
        )
        for observed_state, pattern in cases:
            with self.subTest(observed_state=observed_state):
                bad_meta = json.loads(json.dumps(self.meta))
                bad_meta["observation"]["observed_state"] = observed_state
                self._write_replay_input(meta=bad_meta)
                with self.assertRaisesRegex(ValueError, pattern):
                    run_phase6b_replay(
                        self.replay_dir,
                        out_dir=self.root / f"invalid_{len(observed_state)}",
                    )
        self._write_replay_input()

    def test_nonfinite_input_raises_in_strict_mode(self) -> None:
        cases = (
            ("x_true", np.nan),
            ("y_obs", np.inf),
        )
        for name, value in cases:
            with self.subTest(name=name):
                bad_x = self.x_true.copy()
                bad_y = self.y_obs.copy()
                if name == "x_true":
                    bad_x[0, 0, 0] = value
                else:
                    bad_y[0, 0, 0] = value
                self._write_replay_input(x_true=bad_x, y_obs=bad_y)
                with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                    run_phase6b_replay(
                        self.replay_dir,
                        out_dir=self.root / f"nonfinite_{name}",
                        strict=True,
                    )
        self._write_replay_input()

    def test_checkpoint_scaffold_and_explicit_fallback(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_phase6b_replay(
                self.replay_dir,
                out_dir=self.root / "omitted_checkpoint",
                model_id="kalmannet_tsp",
            )
        with self.assertRaises(FileNotFoundError):
            run_phase6b_replay(
                self.replay_dir,
                out_dir=self.root / "missing_checkpoint",
                model_id="kalmannet_tsp",
                checkpoint=self.root / "missing.pt",
            )

        checkpoint = self.root / "dummy.pt"
        torch.save({"state_dict": {}}, checkpoint)
        with self.assertRaisesRegex(ValueError, "requires replay_contract.json"):
            run_phase6b_replay(
                self.replay_dir,
                out_dir=self.root / "unsupported_checkpoint",
                model_id="kalmannet_tsp",
                checkpoint=checkpoint,
            )

        loaded = load_pred_artifact(
            run_phase6b_replay(
                self.replay_dir,
                out_dir=self.root / "fallback",
                model_id="kalmannet_tsp",
                checkpoint=checkpoint,
                allow_fallback=True,
            )[0]
        )
        self.assertTrue(loaded["meta"]["used_fallback"])
        self.assertEqual(loaded["meta"]["model_id"], IDENTITY_MODEL_ID)
        self.assertEqual(
            loaded["meta"]["requested_model_id"],
            "kalmannet_tsp",
        )
        self.assertIn("ValueError", loaded["meta"]["fallback_reason"])
        self.assertIn("replay_contract.json", loaded["meta"]["fallback_reason"])

    def test_cli_smoke_and_phase1_compatibility(self) -> None:
        output_dir = self.root / "cli"
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--replay-input-dir",
                    str(self.replay_dir),
                    "--model-id",
                    IDENTITY_MODEL_ID,
                    "--out-dir",
                    str(output_dir),
                    "--device",
                    "cpu",
                ]
            )
        self.assertEqual(result, 0)
        loaded = load_pred_artifact(output_dir / "artifacts")
        self.assertEqual(loaded["x_true"].shape, (1, 5, 9))
        self.assertEqual(loaded["y_obs"].shape, (1, 5, 6))
        self.assertEqual(loaded["x_hat"].shape, (1, 5, 9))
        self.assertEqual(
            set(
                key
                for key in loaded
                if key
                in {
                    "time_s",
                    "x_true",
                    "y_obs",
                    "x_hat",
                    "trajectory_id",
                }
            ),
            {
                "time_s",
                "x_true",
                "y_obs",
                "x_hat",
                "trajectory_id",
            },
        )


@dataclass
class Phase6BCheckpointReplayResult:
    ok: bool
    note: str


def run_phase6b_checkpoint_replay_tests() -> Phase6BCheckpointReplayResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        Phase6BCheckpointReplayTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return Phase6BCheckpointReplayResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Phase 6B checkpoint replay tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
