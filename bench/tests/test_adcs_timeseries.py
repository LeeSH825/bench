from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bench.visualization.adcs_timeseries import build_adcs_timeseries
from bench.visualization.pred_artifact import save_pred_artifact


def _state_meta() -> dict:
    return {
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
        "attitude_convention": "MRP sigma_BN",
        "time_unit": "s",
    }


class ADCSTimeseriesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.artifacts = self.root / "artifacts"
        self.n_seq, self.n_step, self.x_dim, self.y_dim = 2, 5, 9, 6
        rng = np.random.default_rng(11)
        self.x_true = rng.normal(
            scale=0.01,
            size=(self.n_seq, self.n_step, self.x_dim),
        )
        self.x_hat = self.x_true + 0.001
        self.y_obs = rng.normal(size=(self.n_seq, self.n_step, self.y_dim))
        self.time_s = np.arange(self.n_step, dtype=np.float64) * 0.1
        self.pred_path, self.pred_meta_path = save_pred_artifact(
            self.artifacts,
            time_s=self.time_s,
            x_true=self.x_true,
            y_obs=self.y_obs,
            x_hat=self.x_hat,
            trajectory_id=np.arange(self.n_seq, dtype=np.int64),
            meta=_state_meta(),
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)

    def test_single_trajectory_outputs_required_and_bias_columns(self) -> None:
        csv_path, meta_path = build_adcs_timeseries(
            self.pred_path,
            trajectory_id=0,
        )
        self.assertTrue(csv_path.exists())
        self.assertTrue(meta_path.exists())
        columns, rows = self._read_csv(csv_path)
        self.assertEqual(len(rows), self.n_step)

        required = {
            "traj_id",
            "t_idx",
            "time_s",
            "sigma1_true",
            "sigma2_true",
            "sigma3_true",
            "sigma1_hat",
            "sigma2_hat",
            "sigma3_hat",
            "mrp_err_norm",
            "roll_true_rad",
            "pitch_true_rad",
            "yaw_true_rad",
            "roll_hat_rad",
            "pitch_hat_rad",
            "yaw_hat_rad",
            "roll_err_rad",
            "pitch_err_rad",
            "yaw_err_rad",
            "omega_x_true_rad_s",
            "omega_y_true_rad_s",
            "omega_z_true_rad_s",
            "omega_x_hat_rad_s",
            "omega_y_hat_rad_s",
            "omega_z_hat_rad_s",
            "omega_err_norm_rad_s",
            "bias_x_true_rad_s",
            "bias_y_true_rad_s",
            "bias_z_true_rad_s",
            "bias_x_hat_rad_s",
            "bias_y_hat_rad_s",
            "bias_z_hat_rad_s",
            "bias_err_norm_rad_s",
        }
        self.assertTrue(required.issubset(set(columns)))

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["schema_version"], "adcs_timeseries_v1")
        self.assertEqual(meta["selected_trajectory_ids"], [0])
        self.assertEqual(meta["num_rows"], self.n_step)
        self.assertEqual(meta["num_trajectories"], 1)
        self.assertEqual(meta["schema_source"], "explicit")

    def test_all_trajectories_outputs_n_times_t_rows(self) -> None:
        csv_path, meta_path = build_adcs_timeseries(
            self.pred_path,
            all_trajectories=True,
        )
        _, rows = self._read_csv(csv_path)
        self.assertEqual(len(rows), self.n_seq * self.n_step)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["selected_trajectory_ids"], [0, 1])
        self.assertEqual(meta["num_trajectories"], self.n_seq)

    def test_fallback_schema_for_six_or_more_states(self) -> None:
        fallback_dir = self.root / "fallback"
        pred_path, _ = save_pred_artifact(
            fallback_dir,
            time_s=self.time_s,
            x_true=self.x_true[:, :, :6],
            y_obs=self.y_obs,
            x_hat=self.x_hat[:, :, :6],
            meta={"time_unit": "sample_index", "time_source": "sample_index_fallback"},
        )
        csv_path, meta_path = build_adcs_timeseries(pred_path, trajectory_id=0)
        columns, _ = self._read_csv(csv_path)
        self.assertNotIn("bias_x_true_rad_s", columns)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["schema_source"], "fallback_default_adcs")
        self.assertEqual(meta["time_unit"], "sample_index")
        self.assertEqual(meta["time_source"], "sample_index_fallback")

    def test_per_trajectory_time_is_preserved(self) -> None:
        per_time_dir = self.root / "per_time"
        time_s = np.stack([self.time_s, self.time_s + 10.0], axis=0)
        pred_path, _ = save_pred_artifact(
            per_time_dir,
            time_s=time_s,
            x_true=self.x_true,
            y_obs=self.y_obs,
            x_hat=self.x_hat,
            meta=_state_meta(),
        )
        csv_path, _ = build_adcs_timeseries(pred_path, trajectory_id=1)
        _, rows = self._read_csv(csv_path)
        self.assertAlmostEqual(float(rows[0]["time_s"]), 10.0)

    def test_invalid_schema_raises(self) -> None:
        invalid_dir = self.root / "invalid"
        bad_meta = _state_meta()
        bad_meta["state_schema"]["attitude"]["indices"] = [0, 0, 2]
        pred_path, _ = save_pred_artifact(
            invalid_dir,
            time_s=self.time_s,
            x_true=self.x_true,
            y_obs=self.y_obs,
            x_hat=self.x_hat,
            meta=bad_meta,
        )
        with self.assertRaisesRegex(ValueError, "state_schema.attitude.indices"):
            build_adcs_timeseries(pred_path, trajectory_id=0)

    def test_missing_prediction_artifact_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            build_adcs_timeseries(self.root / "missing" / "preds_test.npz")

    def test_unknown_trajectory_id_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            build_adcs_timeseries(self.pred_path, trajectory_id=99)

    def test_conflicting_selection_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_adcs_timeseries(
                self.pred_path,
                trajectory_id=0,
                all_trajectories=True,
            )


@dataclass
class ADCSTimeseriesResult:
    ok: bool
    note: str


def run_adcs_timeseries_tests() -> ADCSTimeseriesResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ADCSTimeseriesTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return ADCSTimeseriesResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "ADCS timeseries tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
