from __future__ import annotations

import io
import unittest
from dataclasses import dataclass

import numpy as np

from bench.visualization.attitude import (
    euler321_error,
    mrp_to_dcm,
    mrp_to_euler321,
    mrp_to_quat,
    wrap_angle_rad,
)


class ADCSAttitudeConversionTests(unittest.TestCase):
    def test_zero_mrp_maps_to_zero_euler(self) -> None:
        euler = mrp_to_euler321(np.zeros(3, dtype=np.float64))
        np.testing.assert_allclose(euler, np.zeros(3), atol=1.0e-12)

    def test_wrap_angle_range(self) -> None:
        values = np.asarray(
            [-4.0 * np.pi, -np.pi, -0.1, 0.0, np.pi, 4.0 * np.pi],
            dtype=np.float64,
        )
        wrapped = wrap_angle_rad(values)
        self.assertTrue(np.all(wrapped >= -np.pi))
        self.assertTrue(np.all(wrapped < np.pi))
        np.testing.assert_allclose(wrapped[[0, 3, 5]], 0.0, atol=1.0e-12)

    def test_euler_error_wraps_difference(self) -> None:
        true = np.asarray([0.0, 0.0, np.pi - 0.1])
        estimated = np.asarray([0.0, 0.0, -np.pi + 0.1])
        error = euler321_error(estimated, true)
        np.testing.assert_allclose(error, [0.0, 0.0, 0.2], atol=1.0e-12)

    def test_batched_mrp_preserves_leading_shape(self) -> None:
        sigma = np.zeros((2, 5, 3), dtype=np.float64)
        sigma[1, :, 0] = 0.05
        self.assertEqual(mrp_to_quat(sigma).shape, (2, 5, 4))
        self.assertEqual(mrp_to_dcm(sigma).shape, (2, 5, 3, 3))
        self.assertEqual(mrp_to_euler321(sigma).shape, (2, 5, 3))

    def test_normal_finite_mrp_produces_finite_outputs(self) -> None:
        sigma = np.asarray(
            [[0.01, -0.02, 0.03], [0.2, 0.1, -0.05]],
            dtype=np.float64,
        )
        self.assertTrue(np.isfinite(mrp_to_quat(sigma)).all())
        self.assertTrue(np.isfinite(mrp_to_dcm(sigma)).all())
        self.assertTrue(np.isfinite(mrp_to_euler321(sigma)).all())


@dataclass
class ADCSAttitudeConversionResult:
    ok: bool
    note: str


def run_adcs_attitude_conversion_tests() -> ADCSAttitudeConversionResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ADCSAttitudeConversionTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return ADCSAttitudeConversionResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "ADCS attitude conversion tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
