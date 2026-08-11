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

from bench.visualization.checkpoint_replay_adapters import (
    MOCK_CHECKPOINT_MODEL_ID,
)
from bench.visualization.phase6b_checkpoint_replay import (
    main,
    run_phase6b_replay,
)
from bench.visualization.pred_artifact import load_pred_artifact
from bench.visualization.replay_suite_scenario import (
    REPLAY_SCENARIO_FILENAME,
    REPLAY_SCENARIO_META_FILENAME,
)


class Phase6DCheckpointReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.replay_dir = self.root / "replay"
        self.replay_dir.mkdir()
        self.n_step = 5
        self.x_true = np.arange(
            self.n_step * 9,
            dtype=np.float32,
        ).reshape(1, self.n_step, 9)
        self.y_obs = self.x_true[..., :6] + np.float32(0.125)
        np.savez_compressed(
            self.replay_dir / REPLAY_SCENARIO_FILENAME,
            time_s=np.arange(self.n_step, dtype=np.float32) * 0.5,
            x_true=self.x_true,
            y_obs=self.y_obs,
            trajectory_id=np.array([0], dtype=np.int64),
        )
        self.meta = {
            "schema_version": "phase6a_replay_input_v1",
            "suite_name": "phase6d_test_suite",
            "task_id": "phase6d_test_task",
            "task_name": "Phase 6D test task",
            "seed": 0,
            "scenario_id": "scenario_phase6d",
            "state_dim": 9,
            "measurement_dim": 6,
            "state_schema": {
                "attitude": {"type": "mrp", "indices": [0, 1, 2]},
                "angular_rate": {
                    "type": "rad_s",
                    "indices": [3, 4, 5],
                },
                "gyro_bias": {
                    "type": "rad_s",
                    "indices": [6, 7, 8],
                    "optional": True,
                },
            },
            "observation": {
                "type": "partial",
                "observed_state": [0, 1, 2, 3, 4, 5],
            },
            "time": {
                "dt_s": 0.5,
                "sequence_length_T": self.n_step,
                "duration_s": 2.0,
            },
        }
        (self.replay_dir / REPLAY_SCENARIO_META_FILENAME).write_text(
            json.dumps(self.meta, indent=2) + "\n",
            encoding="utf-8",
        )
        self.checkpoint = self.root / "mock_checkpoint.pt"
        torch.save(
            {
                "model_id": MOCK_CHECKPOINT_MODEL_ID,
                "gain": 1.0,
                "bias": 0.0,
            },
            self.checkpoint,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_mock_checkpoint_replay_writes_phase1_artifact(self) -> None:
        artifact, meta_path = run_phase6b_replay(
            self.replay_dir,
            out_dir=self.root / "output",
            model_id=MOCK_CHECKPOINT_MODEL_ID,
            checkpoint=self.checkpoint,
        )
        self.assertTrue(artifact.exists())
        self.assertTrue(meta_path.exists())
        loaded = load_pred_artifact(artifact)
        self.assertEqual(loaded["x_hat"].shape, self.x_true.shape)
        np.testing.assert_array_equal(loaded["x_hat"][..., :6], self.y_obs)
        np.testing.assert_array_equal(
            loaded["x_hat"][..., 6:],
            np.zeros_like(self.x_true[..., 6:]),
        )
        meta = loaded["meta"]
        self.assertEqual(meta["phase"], "phase6d_checkpoint_replay")
        self.assertEqual(meta["model_id"], MOCK_CHECKPOINT_MODEL_ID)
        self.assertEqual(meta["checkpoint_path"], str(self.checkpoint))
        self.assertTrue(meta["is_mock_adapter"])
        self.assertFalse(meta["is_trained_checkpoint"])
        self.assertTrue(meta["not_for_benchmark_reporting"])
        self.assertIsNotNone(meta["checkpoint_contract_probe_summary"])
        self.assertEqual(
            meta["output_shape_summary"]["x_hat"],
            [1, self.n_step, 9],
        )

    def test_missing_checkpoint_for_nonbaseline_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_phase6b_replay(
                self.replay_dir,
                out_dir=self.root / "missing",
                model_id=MOCK_CHECKPOINT_MODEL_ID,
                checkpoint=self.root / "missing.pt",
            )

    def test_real_model_without_contract_raises(self) -> None:
        unsupported = self.root / "kalmannet.pt"
        torch.save({"state_dict": {}}, unsupported)
        with self.assertRaisesRegex(ValueError, "requires replay_contract.json"):
            run_phase6b_replay(
                self.replay_dir,
                out_dir=self.root / "unsupported",
                model_id="kalmannet_tsp",
                checkpoint=unsupported,
            )

    def test_cli_smoke(self) -> None:
        output_dir = self.root / "cli"
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--replay-input-dir",
                    str(self.replay_dir),
                    "--model-id",
                    MOCK_CHECKPOINT_MODEL_ID,
                    "--checkpoint",
                    str(self.checkpoint),
                    "--out-dir",
                    str(output_dir),
                    "--device",
                    "cpu",
                ]
            )
        self.assertEqual(result, 0)
        loaded = load_pred_artifact(output_dir / "artifacts")
        self.assertEqual(loaded["x_hat"].shape, (1, self.n_step, 9))


@dataclass
class Phase6DCheckpointReplayResult:
    ok: bool
    note: str


def run_phase6d_checkpoint_replay_tests() -> Phase6DCheckpointReplayResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        Phase6DCheckpointReplayTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return Phase6DCheckpointReplayResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "Phase 6D checkpoint replay tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
