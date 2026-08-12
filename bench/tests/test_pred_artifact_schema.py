from __future__ import annotations

import io
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bench.visualization.pred_artifact import (
    PRED_ARTIFACT_FILENAME,
    PRED_META_FILENAME,
    load_pred_artifact,
    save_pred_artifact,
    validate_pred_artifact,
)


class PredictionArtifactSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name) / "artifacts"
        self.n_seq, self.n_step, self.x_dim, self.y_dim = 3, 5, 4, 2
        rng = np.random.default_rng(7)
        self.x_true = rng.normal(size=(self.n_seq, self.n_step, self.x_dim))
        self.y_obs = rng.normal(size=(self.n_seq, self.n_step, self.y_dim))
        self.x_hat = self.x_true + 0.01
        self.time_s = np.arange(self.n_step, dtype=np.float64) * 0.1

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_normal_save_load(self) -> None:
        npz_path, meta_path = save_pred_artifact(
            self.out_dir,
            time_s=self.time_s,
            x_true=self.x_true,
            y_obs=self.y_obs,
            x_hat=self.x_hat,
            meta={"task_id": "dummy", "schema_version": "must_not_override"},
        )

        self.assertEqual(npz_path.name, PRED_ARTIFACT_FILENAME)
        self.assertEqual(meta_path.name, PRED_META_FILENAME)
        loaded = load_pred_artifact(npz_path)
        self.assertEqual(
            {"time_s", "x_true", "y_obs", "x_hat", "trajectory_id"},
            {key for key in loaded if key in {"time_s", "x_true", "y_obs", "x_hat", "trajectory_id"}},
        )
        self.assertEqual(loaded["x_true"].shape, self.x_true.shape)
        self.assertEqual(loaded["y_obs"].shape, self.y_obs.shape)
        self.assertEqual(loaded["x_hat"].shape, self.x_hat.shape)
        self.assertEqual(loaded["time_s"].shape, self.time_s.shape)
        self.assertEqual(loaded["x_true"].dtype, np.float32)
        self.assertEqual(loaded["trajectory_id"].dtype, np.int64)
        self.assertEqual(loaded["meta"]["schema_version"], "pred_artifact_v1")
        self.assertEqual(loaded["meta"]["layout"], "NTD")
        self.assertEqual(loaded["meta"]["x_shape"], list(self.x_true.shape))
        self.assertEqual(loaded["meta"]["task_id"], "dummy")

    def test_accepts_per_trajectory_time(self) -> None:
        time_s = np.repeat(self.time_s[None, :], self.n_seq, axis=0)
        save_pred_artifact(
            self.out_dir,
            time_s=time_s,
            x_true=self.x_true,
            y_obs=self.y_obs,
            x_hat=self.x_hat,
        )
        loaded = load_pred_artifact(self.out_dir)
        self.assertEqual(loaded["time_s"].shape, (self.n_seq, self.n_step))

    def test_rejects_x_hat_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "x_hat shape must match x_true"):
            validate_pred_artifact(
                time_s=self.time_s,
                x_true=self.x_true,
                y_obs=self.y_obs,
                x_hat=self.x_hat[:, :-1, :],
            )

    def test_rejects_nonfinite_in_strict_mode(self) -> None:
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                bad = self.x_hat.copy()
                bad[0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "x_hat contains NaN or Inf"):
                    validate_pred_artifact(
                        time_s=self.time_s,
                        x_true=self.x_true,
                        y_obs=self.y_obs,
                        x_hat=bad,
                        strict=True,
                    )

    def test_auto_generates_trajectory_id(self) -> None:
        save_pred_artifact(
            self.out_dir,
            time_s=self.time_s,
            x_true=self.x_true,
            y_obs=self.y_obs,
            x_hat=self.x_hat,
        )
        loaded = load_pred_artifact(self.out_dir)
        np.testing.assert_array_equal(
            loaded["trajectory_id"],
            np.arange(self.n_seq, dtype=np.int64),
        )


@dataclass
class PredictionArtifactSchemaResult:
    ok: bool
    note: str


def run_pred_artifact_schema_tests() -> PredictionArtifactSchemaResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PredictionArtifactSchemaTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return PredictionArtifactSchemaResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "prediction artifact schema tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
