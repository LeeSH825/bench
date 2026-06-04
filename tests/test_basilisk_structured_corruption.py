from __future__ import annotations

import unittest

import numpy as np

from bench.tasks.generator.basilisk_adcs import (
    _resolve_measurement_corruption_cfg,
    apply_structured_measurement_corruption,
)


def _profile(scale: float, *, outlier_prob: float = 0.0) -> dict:
    return {
        "enabled": True,
        "profile_id": f"s{scale}",
        "gaussian": {"enabled": True, "r2": (0.001 * scale) ** 2},
        "bias": {"sigma_std": 0.002 * scale, "omega_std": 0.0005 * scale},
        "random_walk": {"sigma_std": 1e-5 * scale, "omega_std": 1e-6 * scale},
        "scale": {"std": 0.005 * scale},
        "axis_misalignment": {"std": 0.002 * scale},
        "outlier": {"prob": outlier_prob, "sigma_std": 0.02 * scale, "omega_std": 0.005 * scale},
        "vibration": {"sigma_std": 0.002 * scale, "omega_std": 0.0005 * scale, "freq_hz_range": [1.0, 3.0]},
    }


def _x(n: int = 32, t: int = 40) -> np.ndarray:
    tt = np.linspace(0.0, 1.0, t, dtype=np.float64)
    x = np.zeros((n, t, 6), dtype=np.float64)
    for i in range(n):
        x[i, :, 0] = 0.05 * np.sin(2 * np.pi * tt) + 0.001 * i
        x[i, :, 1] = 0.04 * np.cos(2 * np.pi * tt)
        x[i, :, 2] = 0.02
        x[i, :, 3] = 0.01 * np.sin(np.pi * tt)
        x[i, :, 4] = 0.01 * np.cos(np.pi * tt)
        x[i, :, 5] = 0.005
    return x


class BasiliskStructuredCorruptionTests(unittest.TestCase):
    def test_backward_compatibility_absent_config(self) -> None:
        cfg = _resolve_measurement_corruption_cfg({})
        self.assertFalse(bool(cfg["enabled"]))
        self.assertEqual(cfg["profile_id"], "gaussian_only")

    def test_shape_dtype_finite_and_y_dim(self) -> None:
        x = _x()
        y, extras, meta = apply_structured_measurement_corruption(
            x_all=x,
            cfg=_profile(1.0),
            sensor_std=0.01,
            suite_name="suite",
            task_id="task",
            scenario_id="scenario",
            seed=0,
            dt=0.1,
        )
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.float32)
        self.assertEqual(extras["y_clean_seq"].shape, x.shape)
        self.assertEqual(extras["y_clean_seq"].dtype, np.float32)
        self.assertEqual(extras["corruption_total_seq"].shape, x.shape)
        self.assertEqual(int(y.shape[2]), 6)
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue(np.isfinite(extras["corruption_total_seq"]).all())
        self.assertEqual(meta["clean_measurement_reference"], "x")

    def test_determinism(self) -> None:
        x = _x()
        args = dict(
            x_all=x,
            cfg=_profile(1.0),
            sensor_std=0.01,
            suite_name="suite",
            task_id="task",
            scenario_id="scenario",
            seed=7,
            dt=0.1,
        )
        y1, extras1, meta1 = apply_structured_measurement_corruption(**args)
        y2, extras2, meta2 = apply_structured_measurement_corruption(**args)
        self.assertTrue(np.array_equal(y1, y2))
        self.assertTrue(np.array_equal(extras1["corruption_total_seq"], extras2["corruption_total_seq"]))
        self.assertEqual(meta1["stats"]["total_corruption_to_clean_ratio"], meta2["stats"]["total_corruption_to_clean_ratio"])

    def test_severity_monotonicity(self) -> None:
        x = _x(n=96, t=80)
        ratios = []
        for scale in (1.0, 3.0, 8.0):
            _, _, meta = apply_structured_measurement_corruption(
                x_all=x,
                cfg=_profile(scale),
                sensor_std=0.0,
                suite_name="suite",
                task_id="task",
                scenario_id=f"s{scale}",
                seed=11,
                dt=0.1,
            )
            ratios.append(float(meta["stats"]["total_corruption_norm_mean"]))
        self.assertLess(ratios[0], ratios[1])
        self.assertLess(ratios[1], ratios[2])

    def test_outlier_rate(self) -> None:
        x = _x(n=256, t=80)
        prob = 0.05
        _, extras, meta = apply_structured_measurement_corruption(
            x_all=x,
            cfg=_profile(1.0, outlier_prob=prob),
            sensor_std=0.0,
            suite_name="suite",
            task_id="task",
            scenario_id="outlier",
            seed=17,
            dt=0.1,
        )
        observed = float(np.mean(extras["corruption_outlier_mask_seq"]))
        self.assertAlmostEqual(observed, prob, delta=0.01)
        self.assertAlmostEqual(float(meta["stats"]["outlier_rate_observed"]), observed, delta=1e-8)

    def test_profile_resolution(self) -> None:
        cfg = _resolve_measurement_corruption_cfg(
            {
                "measurement_corruption": {
                    "enabled": True,
                    "profile_id": "mild",
                    "profiles": {"mild": {"severity": "mild", "bias": {"sigma_std": 0.1}}},
                }
            }
        )
        self.assertTrue(bool(cfg["enabled"]))
        self.assertEqual(cfg["profile_id"], "mild")
        self.assertEqual(float(cfg["bias"]["sigma_std"]), 0.1)


if __name__ == "__main__":
    unittest.main()
