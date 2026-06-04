from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bench.diagnostics.imu_measurement_model_audit import (
    CANDIDATES,
    analytic_H_gyro_delta_simple,
    finite_difference_H,
    h_gyro_delta_simple,
    run_audit,
)
from bench.tasks.data_format import dump_meta_json_v0


class BasiliskImuMeasurementModelAuditTest(unittest.TestCase):
    def _x(self) -> np.ndarray:
        x = np.zeros((2, 5, 6), dtype=np.float32)
        x[:, :, 0] = np.linspace(0.0, 0.02, 5, dtype=np.float32)
        x[:, :, 1] = np.linspace(0.0, -0.01, 5, dtype=np.float32)
        x[:, :, 3] = 0.01
        x[:, :, 4] = -0.02
        x[:, :, 5] = 0.03
        return x

    def _meta(self, split: str) -> dict:
        h = analytic_H_gyro_delta_simple(0.1).astype(float).tolist()
        return {
            "split": split,
            "scenario_id": "unit",
            "task_family": "basilisk_imu_adcs_v0",
            "observation": {
                "measurement_mode": "gyro_delta_angle",
                "field_mapping": [
                    {"field": "AngVelPlatform", "alias": "gyro", "columns": [0, 1, 2], "units": "rad/s"},
                    {"field": "DRFramePlatform", "alias": "delta_theta", "columns": [3, 4, 5], "units": "rad"},
                ],
            },
            "imu_config": {"profile_id": "clean_imu", "body_to_platform_euler321_rad": [0.0, 0.0, 0.0]},
            "ssm": {"true": {"dt": 0.1}, "assumed": {"H": h}},
        }

    def _write_split(self, path: Path, split: str) -> None:
        x = self._x()
        y_clean = h_gyro_delta_simple(x, dt=0.1).astype(np.float32)
        y = y_clean.copy()
        h = analytic_H_gyro_delta_simple(0.1).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=x,
            y=y,
            H=h,
            meta_json=dump_meta_json_v0(self._meta(split)),
            imu_clean_y_seq=y_clean,
            imu_error_seq=np.zeros_like(y_clean, dtype=np.float32),
        )

    def test_candidate_shape(self) -> None:
        x = self._x()
        y = h_gyro_delta_simple(x, dt=0.1)
        self.assertEqual(y.shape, (2, 5, 6))

    def test_candidate1_H_shape_and_finite_difference(self) -> None:
        h = analytic_H_gyro_delta_simple(0.1)
        h_fd = finite_difference_H("gyro_delta_simple", self._x()[0, 1], dt=0.1)
        self.assertEqual(h.shape, (6, 6))
        self.assertLess(float(np.max(np.abs(h - h_fd))), 1.0e-8)

    def test_direct_candidate_is_labeled_invalid(self) -> None:
        direct = [c for c in CANDIDATES if c.name == "direct_identity_invalid"][0]
        self.assertFalse(direct.valid_for_acceptance)

    def test_audit_writes_outputs_and_handles_zero_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache = root / "cache"
            for split in ("train", "val", "test"):
                self._write_split(cache / "suite" / "task" / "scenario_unit" / "seed_0" / f"{split}.npz", split)
            reports = root / "reports"
            plots = root / "plots"
            result = run_audit(
                root=root,
                cache_root=cache,
                suite_name="suite",
                task_id="task",
                seed=0,
                reports_dir=reports,
                plots_dir=plots,
                write_plots=False,
            )
            self.assertEqual(result["accepted_candidate"], "gyro_delta_simple")
            self.assertTrue((reports / "basilisk_imu_h_model_comparison.csv").exists())
            self.assertTrue((reports / "basilisk_imu_H_audit.csv").exists())
            self.assertTrue((reports / "basilisk_imu_innovation_diagnostic_audit.csv").exists())
            self.assertTrue((reports / "basilisk_imu_measurement_model_audit_final.md").exists())


if __name__ == "__main__":
    unittest.main()

