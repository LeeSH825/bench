from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from bench.tasks.generator.basilisk_imu_adcs import generate_basilisk_imu_adcs_v0


HAS_BASILISK_IMU = importlib.util.find_spec("Basilisk") is not None


def _task_cfg(profile_id: str = "clean_imu", *, mode: str = "gyro_delta_angle", n: int = 4, t: int = 8) -> dict:
    y_dim = {"gyro_only": 3, "gyro_accel": 6, "gyro_delta_angle": 6, "full_imu": 12}[mode]
    return {
        "task_id": "Basilisk_IMU_ADCS_unit_v0",
        "task_family": "basilisk_imu_adcs_v0",
        "system_type": "nonlinear",
        "x_dim": 6,
        "y_dim": y_dim,
        "sequence_length_T": int(t),
        "dataset_sizes": {"N_train": int(n), "N_val": 0, "N_test": 0},
        "simulation": {
            "dt": 0.1,
            "inertia": [10.0, 8.0, 6.0],
            "disturbance_torque": [0.0, 0.0, 0.0],
            "sigma0_std": 0.05,
            "sigma0_max_norm": 0.25,
            "omega0_std": 0.01,
        },
        "noise": {"Q": {"type": "scaled_identity", "q2": 1.0e-8}},
        "observation": {"type": "basilisk_imu_sensor", "measurement_mode": mode},
        "imu": {
            "profile_id": profile_id,
            "measurement_mode": mode,
            "sensor_pos_B": [0.0, 0.0, 0.0],
            "body_to_platform_euler321_rad": [0.0, 0.0, 0.0],
            "profiles": {
                "clean_imu": {
                    "severity": "clean_imu",
                    "gyro_noise_std": 0.0,
                    "accel_noise_std": 0.0,
                    "gyro_bias_std": 0.0,
                    "accel_bias_std": 0.0,
                },
                "low_cost_imu": {
                    "severity": "low_cost_imu",
                    "gyro_noise_std": 5.0e-4,
                    "accel_noise_std": 5.0e-3,
                    "gyro_bias_std": 2.0e-3,
                    "accel_bias_std": 2.0e-2,
                    "gyro_walk_bound": 2.0e-5,
                    "accel_walk_bound": 2.0e-4,
                },
            },
        },
        "control_input_u": False,
        "ground_truth": {"has_gt": True},
    }


@unittest.skipUnless(HAS_BASILISK_IMU, "AVS Basilisk is not installed")
class BasiliskImuGeneratorTests(unittest.TestCase):
    def test_import_and_payload_fields(self) -> None:
        from Basilisk.architecture import messaging
        from Basilisk.simulation import imuSensor

        obj = imuSensor.ImuSensor()
        self.assertTrue(hasattr(obj, "scStateInMsg"))
        self.assertTrue(hasattr(obj, "sensorOutMsg"))

        payload = messaging.IMUSensorMsgPayload()
        for name in ("AngVelPlatform", "AccelPlatform", "DRFramePlatform", "DVFramePlatform"):
            self.assertTrue(hasattr(payload, name))

    def test_generator_shape_dtype_finite_and_metadata(self) -> None:
        out, f, h = generate_basilisk_imu_adcs_v0(
            suite_name="unit_basilisk_imu",
            task_cfg_dict=_task_cfg("clean_imu", n=3, t=7),
            scenario_cfg={},
            seed=0,
            scenario_id="clean",
        )
        self.assertEqual(out.x.shape, (3, 7, 6))
        self.assertEqual(out.y.shape, (3, 7, 6))
        self.assertEqual(out.x.dtype, np.float32)
        self.assertEqual(out.y.dtype, np.float32)
        self.assertTrue(np.isfinite(out.x).all())
        self.assertTrue(np.isfinite(out.y).all())
        self.assertEqual(out.meta["task_family"], "basilisk_imu_adcs_v0")
        self.assertFalse(bool(out.meta["fake_marker"]))
        self.assertEqual(out.meta["observation"]["measurement_mode"], "gyro_delta_angle")
        self.assertEqual(f.shape, (6, 6))
        self.assertEqual(h.shape, (6, 6))
        self.assertIn("imu_clean_y_seq", out.extras)
        self.assertIn("imu_error_seq", out.extras)
        self.assertGreater(float(np.linalg.norm(out.extras["imu_gyro_seq"])), 0.0)
        self.assertGreater(float(np.linalg.norm(out.extras["imu_delta_theta_seq"])), 0.0)

    def test_determinism(self) -> None:
        kwargs = dict(
            suite_name="unit_basilisk_imu",
            task_cfg_dict=_task_cfg("low_cost_imu", n=2, t=6),
            scenario_cfg={},
            seed=11,
            scenario_id="det",
        )
        out1, _, _ = generate_basilisk_imu_adcs_v0(**kwargs)
        out2, _, _ = generate_basilisk_imu_adcs_v0(**kwargs)
        self.assertTrue(np.array_equal(out1.x, out2.x))
        self.assertTrue(np.array_equal(out1.y, out2.y))
        self.assertTrue(np.array_equal(out1.extras["imu_error_seq"], out2.extras["imu_error_seq"]))

    def test_severity_effect(self) -> None:
        clean, _, _ = generate_basilisk_imu_adcs_v0(
            suite_name="unit_basilisk_imu",
            task_cfg_dict=_task_cfg("clean_imu", n=2, t=6),
            scenario_cfg={},
            seed=3,
            scenario_id="same",
        )
        low_cost, _, _ = generate_basilisk_imu_adcs_v0(
            suite_name="unit_basilisk_imu",
            task_cfg_dict=_task_cfg("low_cost_imu", n=2, t=6),
            scenario_cfg={},
            seed=3,
            scenario_id="same",
        )
        clean_ratio = float(clean.meta["imu_config"]["stats"]["imu_error_to_clean_ratio"])
        low_cost_ratio = float(low_cost.meta["imu_config"]["stats"]["imu_error_to_clean_ratio"])
        self.assertLess(clean_ratio, low_cost_ratio)
        self.assertGreater(low_cost_ratio, 0.0)

    def test_supported_modes_have_expected_y_dim(self) -> None:
        for mode, y_dim in (("gyro_only", 3), ("gyro_accel", 6), ("gyro_delta_angle", 6), ("full_imu", 12)):
            out, _, h = generate_basilisk_imu_adcs_v0(
                suite_name="unit_basilisk_imu",
                task_cfg_dict=_task_cfg("clean_imu", mode=mode, n=1, t=4),
                scenario_cfg={},
                seed=5,
                scenario_id=mode,
            )
            self.assertEqual(out.y.shape, (1, 4, y_dim))
            self.assertEqual(h.shape, (y_dim, 6))


if __name__ == "__main__":
    unittest.main()

