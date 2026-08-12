from __future__ import annotations

import io
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bench.visualization.checkpoint_replay_adapters import (
    MOCK_CHECKPOINT_MODEL_ID,
    get_supported_checkpoint_replay_model_ids,
    run_checkpoint_replay_adapter,
    validate_checkpoint_replay_output,
)


class CheckpointReplayAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.checkpoint = self.root / "mock.pt"
        torch.save(
            {
                "model_id": MOCK_CHECKPOINT_MODEL_ID,
                "gain": 2.0,
                "bias": -1.0,
            },
            self.checkpoint,
        )
        self.y_obs = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
        self.replay_meta = {
            "state_dim": 9,
            "observation": {
                "observed_state": [0, 1, 2, 3, 4, 5],
            },
        }

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_unknown_model_raises(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "not supported"):
            run_checkpoint_replay_adapter(
                model_id="unknown_model",
                checkpoint=self.checkpoint,
                model_config=None,
                y_obs=self.y_obs,
                replay_meta=self.replay_meta,
            )

    def test_mock_adapter_output_and_metadata(self) -> None:
        result = run_checkpoint_replay_adapter(
            model_id=MOCK_CHECKPOINT_MODEL_ID,
            checkpoint=self.checkpoint,
            model_config=None,
            y_obs=self.y_obs,
            replay_meta=self.replay_meta,
        )
        self.assertEqual(result.x_hat.shape, (1, 4, 9))
        np.testing.assert_array_equal(
            result.x_hat[..., :6],
            self.y_obs * 2.0 - 1.0,
        )
        np.testing.assert_array_equal(
            result.x_hat[..., 6:],
            np.zeros((1, 4, 3), dtype=np.float32),
        )
        self.assertTrue(result.metadata["is_mock_adapter"])
        self.assertTrue(result.metadata["not_for_benchmark_reporting"])
        self.assertFalse(result.metadata["normalization_applied"])
        self.assertIn(
            MOCK_CHECKPOINT_MODEL_ID,
            get_supported_checkpoint_replay_model_ids(include_test=True),
        )

    def test_output_validation_rejects_wrong_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "match x_true shape"):
            validate_checkpoint_replay_output(
                x_hat=np.zeros((1, 4, 8), dtype=np.float32),
                x_true=np.zeros((1, 4, 9), dtype=np.float32),
                model_id=MOCK_CHECKPOINT_MODEL_ID,
            )

    def test_output_validation_rejects_nonfinite(self) -> None:
        values = np.zeros((1, 4, 9), dtype=np.float32)
        values[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            validate_checkpoint_replay_output(
                x_hat=values,
                x_true=np.zeros_like(values),
                model_id=MOCK_CHECKPOINT_MODEL_ID,
            )


@dataclass
class CheckpointReplayAdapterResult:
    ok: bool
    note: str


def run_checkpoint_replay_adapter_tests() -> CheckpointReplayAdapterResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        CheckpointReplayAdapterTests
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return CheckpointReplayAdapterResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "checkpoint replay adapter tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
