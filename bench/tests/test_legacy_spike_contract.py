from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from bench.models.g1_snn_split_knet import G1SNNSplitKNet
from bench.models.spike_ra_knet import SpikeRAKNet, SpikeRAKNetAdapter
from bench.models.spike_split_knet import SpikeSplitKNetG2SNN
from bench.tasks.generator.basilisk_imu_adcs import (
    _apply_measurement_event_disturbance,
)


class _PredictionProbe(SpikeRAKNetAdapter):
    """Exercise the deployable adapter boundary without an upstream checkout."""

    def __init__(self) -> None:
        super().__init__()
        self._filter_obj = SimpleNamespace(kf_net=object())
        self._x_dim = 2
        self._y_dim = 3
        self._T_setup = 4
        self.last_layout = "BTD"
        self.seen_batches: list[object] = []

    def transform_measurements(
        self,
        y_btd: torch.Tensor,
        *,
        x_btd: torch.Tensor | None = None,
        batch: dict[str, object] | None = None,
        phase: str = "eval",
    ) -> torch.Tensor:
        _ = x_btd, phase
        self.seen_batches.append(batch)
        return y_btd

    def _forward_batch(
        self,
        *,
        y_btd: torch.Tensor,
        x0_batch: object,
    ) -> torch.Tensor:
        _ = x0_batch
        return y_btd[..., :2].clone()


class LegacySpikeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nominal = np.arange(2 * 8 * 6, dtype=np.float64).reshape(2, 8, 6)
        self.mapping = [
            {"alias": "gyro", "columns": [0, 2, 5], "units": "rad/s"},
            {"alias": "delta_angle", "columns": [1, 3, 4], "units": "rad"},
        ]

    def _apply(self, *, enabled: bool):
        return _apply_measurement_event_disturbance(
            y_nominal=self.nominal,
            mapping=self.mapping,
            event_cfg={
                "enabled": enabled,
                "event_start_frac": 0.25,
                "event_duration_frac": 0.5,
                "gyro_noise_scale_event": 3.0,
                "gyro_bias_jump_std": 0.1,
                "event_type": "measurement_gyro_bias_jump",
            },
            nominal_gyro_noise_std=0.2,
            suite_name="legacy_spike_contract",
            task_id="event_contract_v0",
            scenario_id="scenario-fixed",
            seed=7,
        )

    def test_event_disabled_path_is_exact_identity(self) -> None:
        disturbed, extras, meta, gyro_columns = self._apply(enabled=False)
        np.testing.assert_array_equal(disturbed, self.nominal)
        self.assertEqual(extras, {})
        self.assertEqual(meta, {})
        self.assertEqual(gyro_columns, [])

    def test_event_affects_only_configured_gyro_columns_and_window(self) -> None:
        disturbed, extras, meta, gyro_columns = self._apply(enabled=True)
        repeated, repeated_extras, repeated_meta, repeated_columns = self._apply(
            enabled=True
        )

        np.testing.assert_array_equal(disturbed, repeated)
        self.assertEqual(meta, repeated_meta)
        self.assertEqual(gyro_columns, repeated_columns)
        for key in extras:
            np.testing.assert_array_equal(extras[key], repeated_extras[key])

        self.assertEqual(gyro_columns, [0, 2, 5])
        self.assertEqual((meta["event_start"], meta["event_end"]), (2, 6))
        event_mask = extras["event_flag_seq"][..., 0].astype(bool)
        expected_mask = np.zeros((2, 8), dtype=bool)
        expected_mask[:, 2:6] = True
        np.testing.assert_array_equal(event_mask, expected_mask)

        delta = disturbed - self.nominal
        np.testing.assert_array_equal(delta[:, :2], np.zeros_like(delta[:, :2]))
        np.testing.assert_array_equal(delta[:, 6:], np.zeros_like(delta[:, 6:]))
        np.testing.assert_array_equal(
            delta[..., [1, 3, 4]],
            np.zeros_like(delta[..., [1, 3, 4]]),
        )
        self.assertTrue(np.any(delta[:, 2:6, :][..., [0, 2, 5]] != 0.0))
        np.testing.assert_allclose(
            delta,
            extras["event_bias_component_seq"]
            + extras["event_noise_component_seq"],
            rtol=0.0,
            atol=2.0e-7,
        )

    def test_deployable_inference_ignores_event_and_truth_metadata(self) -> None:
        forbidden = {"event", "event_flag", "label", "truth", "x_true", "oracle"}
        for model_class in (SpikeRAKNet, SpikeSplitKNetG2SNN, G1SNNSplitKNet):
            parameters = set(inspect.signature(model_class.forward).parameters)
            self.assertTrue(
                forbidden.isdisjoint(parameters),
                f"{model_class.__name__}.forward accepts forbidden metadata: "
                f"{sorted(forbidden.intersection(parameters))}",
            )

        adapter = _PredictionProbe()
        y = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        state0 = torch.zeros(2, 2)
        event_flag = np.array(
            [[[0.0], [1.0], [0.0], [1.0]], [[1.0], [0.0], [1.0], [0.0]]],
            dtype=np.float32,
        )
        contexts = (
            {},
            {"event_flag_seq": event_flag, "x_true": np.ones((2, 4, 2))},
            {
                "event_flag_seq": event_flag[:, ::-1].copy(),
                "truth": np.full((2, 4, 2), 9.0),
                "oracle": {"event_window": [1, 3]},
            },
        )
        outputs = [
            adapter.predict(y, state0=state0, context=context)
            for context in contexts
        ]
        for output in outputs[1:]:
            torch.testing.assert_close(output, outputs[0], rtol=0.0, atol=0.0)
        self.assertEqual(adapter.seen_batches, [None, None, None])


if __name__ == "__main__":
    unittest.main()
