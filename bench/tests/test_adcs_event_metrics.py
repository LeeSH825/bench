from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from bench.metrics.adcs_event import (
    angular_velocity_rmse,
    attitude_error_deg,
    attitude_rmse_deg,
    compute_adcs_event_metrics,
    event_attitude_rmse_deg,
    event_peak_attitude_error_deg,
)
from bench.models import registry
from bench.models.base import ModelAdapter
from bench.runners.run_suite import (
    _adcs_event_metrics_if_available,
    _build_scenario_cfg_basis,
    _canonicalize_scenario_id,
    run_one,
)
from bench.tasks.data_format import save_npz_split_v0


def _angle_from_zero_mrp_deg(magnitude: np.ndarray) -> np.ndarray:
    return np.rad2deg(4.0 * np.arctan(np.asarray(magnitude, dtype=np.float64)))


class _EventMetricDummyAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.last_layout = "bench_BTD"
        self.last_class = "bench.tests._EventMetricDummyAdapter"

    def setup(
        self,
        cfg: dict,
        system_info: Any,
        run_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        return None

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        budget: Optional[Any] = None,
        ckpt_dir: Optional[Any] = None,
    ) -> Any:
        return {"status": "ok", "ckpt_path": None}

    def eval(
        self,
        test_loader: Any,
        ckpt_path: Optional[str] = None,
        track_cfg: Optional[dict] = None,
    ) -> Any:
        batches = []
        for batch in test_loader:
            x_hat = batch["x"].clone()
            x_hat[:, :, 0] += 0.01
            batches.append(x_hat)
        return {"status": "ok", "x_hat": torch.cat(batches, dim=0)}

    def load(self, ckpt_path: str) -> None:
        return None

    def predict(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[dict] = None,
        return_cov: bool = False,
    ) -> Any:
        raise NotImplementedError

    def adapt(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[dict] = None,
        budget: Optional[Any] = None,
    ) -> None:
        return None

    def save(self, out_dir: str) -> None:
        return None


class ADCSEventMetricTests(unittest.TestCase):
    def test_x_dim_6_exact_attitude_and_omega_metrics(self) -> None:
        x_true = np.zeros((1, 4, 6), dtype=np.float64)
        x_pred = np.zeros_like(x_true)
        magnitudes = np.asarray([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
        x_pred[0, :, 0] = magnitudes
        x_pred[0, :, 3:6] = np.asarray([1.0, 2.0, 2.0])

        expected_angles = _angle_from_zero_mrp_deg(magnitudes)
        np.testing.assert_allclose(
            attitude_error_deg(x_true, x_pred)[0],
            expected_angles,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(
            attitude_rmse_deg(x_true, x_pred),
            float(np.sqrt(np.mean(expected_angles**2))),
            places=12,
        )
        self.assertAlmostEqual(angular_velocity_rmse(x_true, x_pred), 3.0, places=12)

    def test_x_dim_9_ignores_bias_state(self) -> None:
        x_true = np.zeros((5, 9), dtype=np.float64)
        x_pred = np.zeros_like(x_true)
        x_pred[:, 6:9] = 100.0
        event_flag = np.asarray([0.0, 1.0, 1.0, 0.0, 0.0])

        metrics = compute_adcs_event_metrics(
            x_true=x_true,
            x_pred=x_pred,
            event_flag_seq=event_flag,
        )
        self.assertEqual(metrics["attitude_rmse_deg"], 0.0)
        self.assertEqual(metrics["angular_velocity_rmse"], 0.0)
        self.assertEqual(metrics["event_attitude_rmse_deg"], 0.0)
        self.assertEqual(metrics["event_peak_attitude_error_deg"], 0.0)

    def test_event_metrics_use_only_active_samples(self) -> None:
        x_true = np.zeros((1, 4, 6), dtype=np.float64)
        x_pred = np.zeros_like(x_true)
        magnitudes = np.asarray([0.50, 0.02, 0.40, 0.04], dtype=np.float64)
        x_pred[0, :, 1] = magnitudes
        event_flag = np.asarray([[0.0], [1.0], [0.0], [1.0]])

        expected_event_angles = _angle_from_zero_mrp_deg(magnitudes[[1, 3]])
        expected_rmse = float(np.sqrt(np.mean(expected_event_angles**2)))
        expected_peak = float(np.max(expected_event_angles))
        self.assertAlmostEqual(
            event_attitude_rmse_deg(x_true, x_pred, event_flag),
            expected_rmse,
            places=12,
        )
        self.assertAlmostEqual(
            event_peak_attitude_error_deg(x_true, x_pred, event_flag),
            expected_peak,
            places=12,
        )
        self.assertLess(expected_peak, attitude_rmse_deg(x_true, x_pred))

    def test_supported_event_flag_layouts_are_equivalent(self) -> None:
        x_true = np.zeros((2, 4, 6), dtype=np.float64)
        x_pred = np.zeros_like(x_true)
        x_pred[:, :, 2] = np.asarray([0.01, 0.02, 0.03, 0.04])
        flag_t = np.asarray([0.0, 1.0, 1.0, 0.0])
        expected = event_attitude_rmse_deg(x_true, x_pred, flag_t)

        layouts = (
            flag_t,
            flag_t[:, None],
            flag_t[None, :],
            np.broadcast_to(flag_t[None, :], (2, 4)),
            np.broadcast_to(flag_t[None, :, None], (2, 4, 1)),
        )
        for layout in layouts:
            with self.subTest(shape=np.asarray(layout).shape):
                self.assertAlmostEqual(
                    event_attitude_rmse_deg(x_true, x_pred, layout),
                    expected,
                    places=12,
                )

    def test_no_event_returns_nan_without_crashing(self) -> None:
        x_true = np.zeros((2, 3, 6), dtype=np.float64)
        x_pred = np.zeros_like(x_true)
        event_flag = np.zeros((2, 3, 1), dtype=np.float64)
        self.assertTrue(np.isnan(event_attitude_rmse_deg(x_true, x_pred, event_flag)))
        self.assertTrue(np.isnan(event_peak_attitude_error_deg(x_true, x_pred, event_flag)))

    def test_runner_hook_is_conditional_on_event_extra(self) -> None:
        x_true = np.zeros((1, 3, 6), dtype=np.float64)
        x_pred = np.zeros_like(x_true)
        self.assertIsNone(
            _adcs_event_metrics_if_available(
                x_true=x_true,
                x_pred=x_pred,
                split_extras={},
            )
        )

        x_pred[0, 1, 0] = 0.02
        hooked = _adcs_event_metrics_if_available(
            x_true=x_true,
            x_pred=x_pred,
            split_extras={"event_flag_seq": np.asarray([[[0.0], [1.0], [0.0]]])},
        )
        self.assertIsNotNone(hooked)
        assert hooked is not None
        for key in (
            "attitude_rmse_deg",
            "angular_velocity_rmse",
            "event_attitude_rmse_deg",
            "event_peak_attitude_error_deg",
        ):
            self.assertIn(key, hooked)
        self.assertEqual(hooked["event_sample_count"], 1)
        self.assertEqual(hooked["event_flag_source"], "test.npz:event_flag_seq")

    def test_runner_writes_adcs_event_metrics_json(self) -> None:
        with tempfile.TemporaryDirectory(prefix="adcs_event_runner_") as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            suite_name = "adcs_event_runner_smoke"
            task_id = "ADCS_event_metric_runner_smoke_v0"
            model_id = "adcs_event_metric_dummy"
            task = {
                "task_id": task_id,
                "task_family": "basilisk_imu_adcs_v0",
                "system_type": "nonlinear",
                "x_dim": 6,
                "y_dim": 6,
                "sequence_length_T": 4,
                "dataset_sizes": {"N_train": 2, "N_val": 2, "N_test": 2},
                "noise": {"Q": {"type": "scaled_identity", "q2": 1.0e-8}},
                "observation": {"type": "basilisk_imu_sensor"},
                "control_input_u": False,
                "ground_truth": {"has_gt": True},
                "sweep": {},
            }
            scenario_basis = _build_scenario_cfg_basis(task, {})
            scenario_id = _canonicalize_scenario_id(task_id, scenario_basis)
            split_dir = cache_root / suite_name / task_id / f"scenario_{scenario_id}" / "seed_0"

            x = np.zeros((2, 4, 6), dtype=np.float32)
            y = np.zeros((2, 4, 6), dtype=np.float32)
            event_flag = np.zeros((2, 4, 1), dtype=np.float32)
            event_flag[:, 1:3, 0] = 1.0
            meta = {
                "format_version": "0.1",
                "canonical_layout": "NTD",
                "task_family": "basilisk_imu_adcs_v0",
                "suite_name": suite_name,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "seed": 0,
                "x_dim": 6,
                "y_dim": 6,
                "T": 4,
                "attitude_representation": "MRP",
                "measurement_event": {"enabled": True},
            }
            for split_name in ("train", "val", "test"):
                save_npz_split_v0(
                    path=split_dir / f"{split_name}.npz",
                    x=x,
                    y=y,
                    u=None,
                    F=None,
                    H=None,
                    meta={**meta, "split": split_name},
                    extras={"event_flag_seq": event_flag},
                )

            suite = {
                "suite": {"name": suite_name},
                "runner": {
                    "deterministic": True,
                    "artifacts": {"save_predictions": False},
                    "budget": {"train_max_updates": 0, "eval_batch_size": 2},
                    "tracks": [{"track_id": "frozen", "adaptation_enabled": False}],
                },
                "reporting": {
                    "output_dir_template": str(
                        root
                        / "runs"
                        / "{task_id}"
                        / "{model_id}"
                        / "{track_id}"
                        / "seed_{seed}"
                        / "scenario_{scenario_id}"
                    )
                },
            }
            model = {"model_id": model_id, "eval_batch_size": 2}
            old_cache = os.environ.get("BENCH_DATA_CACHE")
            original_adapter = registry._REGISTRY.get(model_id)
            os.environ["BENCH_DATA_CACHE"] = str(cache_root)
            registry._REGISTRY[model_id] = _EventMetricDummyAdapter
            try:
                result = run_one(
                    suite=suite,
                    task=task,
                    model=model,
                    scenario_settings={},
                    seed=0,
                    track_id="frozen",
                    device_str="cpu",
                    precision="fp32",
                    init_id="untrained",
                )
            finally:
                if original_adapter is None:
                    registry._REGISTRY.pop(model_id, None)
                else:
                    registry._REGISTRY[model_id] = original_adapter
                if old_cache is None:
                    os.environ.pop("BENCH_DATA_CACHE", None)
                else:
                    os.environ["BENCH_DATA_CACHE"] = old_cache

            self.assertEqual(result["status"], "ok")
            metrics = json.loads(
                (Path(str(result["run_dir"])) / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertIn("adcs_event", metrics)
            adcs_event = metrics["adcs_event"]
            for key in (
                "attitude_rmse_deg",
                "angular_velocity_rmse",
                "event_attitude_rmse_deg",
                "event_peak_attitude_error_deg",
            ):
                self.assertIn(key, adcs_event)
            self.assertEqual(adcs_event["event_sample_count"], 4)
            self.assertGreater(adcs_event["attitude_rmse_deg"], 0.0)
            self.assertGreater(adcs_event["event_attitude_rmse_deg"], 0.0)


@dataclass
class ADCSEventMetricResult:
    ok: bool
    note: str


def run_adcs_event_metric_tests() -> ADCSEventMetricResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ADCSEventMetricTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return ADCSEventMetricResult(
        ok=bool(result.wasSuccessful()),
        note=(
            "ADCS event metric tests passed"
            if result.wasSuccessful()
            else stream.getvalue().strip()
        ),
    )


if __name__ == "__main__":
    unittest.main()
