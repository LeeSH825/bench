from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from bench.tasks.generator.basilisk_imu_adcs import generate_basilisk_imu_sparse_ref_adcs_v0


HAS_BASILISK_IMU = importlib.util.find_spec("Basilisk") is not None


def _task_cfg(profile_id: str = "mild_bias_ref", *, n: int = 4, t: int = 12, dropout: float = 0.0) -> dict:
    return {
        "task_id": "Basilisk_IMU_SparseRef_ADCS_unit_v0",
        "task_family": "basilisk_imu_sparse_ref_adcs_v0",
        "system_type": "nonlinear",
        "x_dim": 9,
        "y_dim": 9,
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
        "observation": {"type": "basilisk_imu_sensor_with_controlled_gyro_bias_and_sparse_attitude_reference"},
        "imu": {
            "profile_id": "clean_imu",
            "measurement_mode": "gyro_delta_angle",
            "sensor_pos_B": [0.0, 0.0, 0.0],
            "body_to_platform_euler321_rad": [0.0, 0.0, 0.0],
            "profiles": {
                "clean_imu": {
                    "severity": "clean_imu",
                    "gyro_noise_std": 0.0,
                    "accel_noise_std": 0.0,
                    "gyro_bias_std": 0.0,
                    "accel_bias_std": 0.0,
                }
            },
        },
        "bias_state": {
            "profile_id": profile_id,
            "profiles": {
                "clean_ref": {
                    "severity": "clean_ref",
                    "bias_init_std": 0.0,
                    "bias_rw_std": 0.0,
                    "gyro_noise_std": 0.0,
                    "delta_noise_std": 0.0,
                },
                "mild_bias_ref": {
                    "severity": "mild_bias_ref",
                    "bias_init_std": 5.0e-4,
                    "bias_rw_std": 1.0e-5,
                    "gyro_noise_std": 5.0e-4,
                    "delta_noise_std": 5.0e-5,
                },
                "low_cost_bias_ref": {
                    "severity": "low_cost_bias_ref",
                    "bias_init_std": 5.0e-3,
                    "bias_rw_std": 1.0e-4,
                    "gyro_noise_std": 2.0e-3,
                    "delta_noise_std": 2.0e-4,
                },
            },
        },
        "sparse_ref": {
            "profile_id": "same_as_bias_state",
            "ref_update_period": 4,
            "ref_noise_std": 0.005,
            "ref_dropout_prob": float(dropout),
            "profiles": {
                "clean_ref": {"severity": "clean_ref"},
                "mild_bias_ref": {"severity": "mild_bias_ref"},
                "low_cost_bias_ref": {"severity": "low_cost_bias_ref"},
            },
        },
        "control_input_u": False,
        "ground_truth": {"has_gt": True},
    }


@unittest.skipUnless(HAS_BASILISK_IMU, "AVS Basilisk is not installed")
class BasiliskImuSparseRefGeneratorTests(unittest.TestCase):
    def test_shape_mask_metadata_and_relation(self) -> None:
        out, f, h = generate_basilisk_imu_sparse_ref_adcs_v0(
            suite_name="unit_basilisk_imu_sparse_ref",
            task_cfg_dict=_task_cfg("mild_bias_ref", n=3, t=12),
            scenario_cfg={},
            seed=0,
            scenario_id="mild",
        )
        self.assertEqual(out.x.shape, (3, 12, 9))
        self.assertEqual(out.y.shape, (3, 12, 9))
        self.assertEqual(out.x.dtype, np.float32)
        self.assertEqual(out.y.dtype, np.float32)
        self.assertTrue(np.isfinite(out.x).all())
        self.assertTrue(np.isfinite(out.y).all())
        self.assertEqual(out.meta["task_family"], "basilisk_imu_sparse_ref_adcs_v0")
        self.assertFalse(bool(out.meta["fake_marker"]))
        self.assertEqual(out.meta["ssm"]["assumed"]["h_type"], "imu_bias_sparse_attitude_ref")
        self.assertEqual(f.shape, (9, 9))
        self.assertEqual(h.shape, (9, 9))
        self.assertIn("ref_mask_seq", out.extras)
        self.assertIn("measurement_mask_seq", out.extras)
        self.assertIn("measurement_clean_y_seq", out.extras)
        ref_mask = out.extras["ref_mask_seq"]
        self.assertEqual(ref_mask.shape, (3, 12, 1))
        self.assertGreater(float(ref_mask.mean()), 0.0)
        self.assertLess(float(ref_mask.mean()), 1.0)
        relation = out.extras["measurement_error_seq"]
        self.assertTrue(np.allclose(out.y - out.extras["measurement_clean_y_seq"], relation, atol=1.0e-6))
        missing = ref_mask[..., 0] < 0.5
        self.assertTrue(np.allclose(out.y[missing, 6:9], 0.0, atol=1.0e-6))

    def test_severity_changes_bias_norm(self) -> None:
        clean, _, _ = generate_basilisk_imu_sparse_ref_adcs_v0(
            suite_name="unit_basilisk_imu_sparse_ref",
            task_cfg_dict=_task_cfg("clean_ref", n=2, t=8),
            scenario_cfg={},
            seed=4,
            scenario_id="clean",
        )
        low, _, _ = generate_basilisk_imu_sparse_ref_adcs_v0(
            suite_name="unit_basilisk_imu_sparse_ref",
            task_cfg_dict=_task_cfg("low_cost_bias_ref", n=2, t=8),
            scenario_cfg={},
            seed=4,
            scenario_id="low",
        )
        self.assertLess(
            float(clean.meta["bias_state"]["stats"]["bias_norm_mean"]),
            float(low.meta["bias_state"]["stats"]["bias_norm_mean"]),
        )


if __name__ == "__main__":
    unittest.main()

