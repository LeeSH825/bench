from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from bench.models.split_knet import SplitKNetAdapter
from viz.app.views.run_inspector import (
    available_gain_sources,
    build_run_inspector_bundle,
    clamp_gain_element,
)
from viz.contract import ContractError
from viz.io.loader import load_run
from viz.io.writer import write_viz_artifacts


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixture() -> tuple[dict, dict, np.ndarray, np.ndarray]:
    n_seq, n_step, x_dim, y_dim = 3, 8, 2, 2
    f = np.asarray([[0.95, 0.1], [0.0, 0.9]], dtype=np.float32)
    h = np.asarray([[1.0, 0.2], [-0.15, 0.8]], dtype=np.float32)
    t = np.arange(n_step, dtype=np.float32)
    y = np.empty((n_seq, n_step, y_dim), dtype=np.float32)
    for batch_idx in range(n_seq):
        y[batch_idx, :, 0] = np.sin(np.float32(0.3) * t + np.float32(0.1 * batch_idx))
        y[batch_idx, :, 1] = np.cos(np.float32(0.2) * t - np.float32(0.05 * batch_idx))
    x_true = np.zeros((n_seq, n_step, x_dim), dtype=np.float32)
    cfg = {
        "model_id": "split_knet",
        "repo": {"path": "third_party/Split_KalmanNet"},
        "estimator_class_path": "GSSFiltering.filtering.Split_KalmanNet_Filter",
        "input_layout": "BTD",
        "eval_init_from_gt": False,
    }
    system_info = {
        "x_dim": x_dim,
        "y_dim": y_dim,
        "T": n_step,
        "F": f,
        "H": h,
        "Q": np.eye(x_dim, dtype=np.float32) * np.float32(1e-3),
        "R": np.eye(y_dim, dtype=np.float32) * np.float32(2e-3),
    }
    return cfg, system_info, x_true, y


def _adapter() -> SplitKNetAdapter:
    cfg, system_info, _, _ = _fixture()
    adapter = SplitKNetAdapter()
    adapter.setup(cfg, system_info, {"seed": 23, "deterministic": True, "device": "cpu"})
    return adapter


def _eval(adapter: SplitKNetAdapter, x_true: np.ndarray, y: np.ndarray) -> dict:
    return adapter.eval([{"x": x_true, "y": y}])


def _write_split_run(
    root: Path,
    diagnostics: dict[str, np.ndarray],
    *,
    adapter_meta: dict | None = None,
    x_true: np.ndarray | None = None,
    y: np.ndarray | None = None,
    x_hat: np.ndarray | None = None,
) -> Path:
    _, _, default_x, default_y = _fixture()
    x_true = default_x if x_true is None else x_true
    y = default_y if y is None else y
    x_hat = default_x if x_hat is None else x_hat
    run_dir = root / "split"
    write_viz_artifacts(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        suite_name="viz_split_test",
        task_id="linear_split_v0",
        task_family="linear_gaussian_v0",
        scenario_id="scenario_split",
        model_id="split_knet",
        seed=23,
        track_id="frozen",
        init_id="checkpoint",
        run_status="ok",
        time_s=np.arange(x_true.shape[1], dtype=np.float32),
        time_meta={"time_source": "fixture", "time_unit": "s", "dt_s": 1.0},
        x_true=x_true,
        y_obs=y,
        x_hat=x_hat,
        split_extras={},
        diagnostics=diagnostics,
        adapter_meta=adapter_meta if adapter_meta is not None else _adapter().get_adapter_meta(),
        trajectory_ids=np.arange(x_true.shape[0], dtype=np.int64),
        trajectory_id_source="fixture:trajectory_id",
        data_split="test",
        split_source="explicit",
    )
    return run_dir


class VizSplitKNetAdapterTest(unittest.TestCase):
    def test_diagnostics_off_has_no_history_allocation(self) -> None:
        _, _, x_true, y = _fixture()
        adapter = _adapter()
        with mock.patch(
            "bench.models.split_knet.np.full",
            side_effect=AssertionError("diagnostic history allocation is forbidden while disabled"),
        ):
            result = _eval(adapter, x_true, y)
        self.assertNotIn("diagnostics", result)
        self.assertNotIn("viz_diagnostics", adapter.get_runtime_diagnostics())
        self.assertFalse(adapter._emit_viz_diagnostics)

    def test_diagnostics_capture_actual_update_and_component_formula(self) -> None:
        _, system_info, x_true, y = _fixture()
        adapter = _adapter()
        off = _eval(adapter, x_true, y)
        adapter.set_viz_diagnostics_enabled(True)
        on = _eval(adapter, x_true, y)
        diagnostics = on["diagnostics"]

        self.assertTrue(np.array_equal(off["x_hat"].numpy(), on["x_hat"].numpy()))
        self.assertEqual(
            set(diagnostics),
            {"innov", "innov_valid", "gain", "gain_g1", "gain_g2"},
        )
        self.assertEqual(diagnostics["innov"].shape, (3, 8, 2))
        self.assertEqual(diagnostics["gain"].shape, (3, 8, 2, 2))
        self.assertEqual(diagnostics["gain_g1"].shape, (3, 8, 2, 2))
        self.assertEqual(diagnostics["gain_g2"].shape, (3, 8, 2, 2))
        self.assertEqual(diagnostics["innov"].dtype, np.float32)
        self.assertEqual(diagnostics["gain"].dtype, np.float32)
        self.assertEqual(diagnostics["innov_valid"].dtype, np.bool_)
        self.assertTrue(np.all(~diagnostics["innov_valid"][:, 0]))
        self.assertTrue(np.all(diagnostics["innov_valid"][:, 1:]))
        for key in ("innov", "gain", "gain_g1", "gain_g2"):
            self.assertTrue(np.isnan(diagnostics[key][:, 0]).all())
            self.assertTrue(np.isfinite(diagnostics[key][:, 1:]).all())

        f = np.asarray(system_info["F"], dtype=np.float32)
        h = np.asarray(system_info["H"], dtype=np.float32)
        x_hat = on["x_hat"].numpy()
        prior = np.einsum("ij,btj->bti", f, x_hat[:, :-1])
        expected_innov = y[:, 1:] - np.einsum("ij,btj->bti", h, prior)
        reconstructed_gain = diagnostics["gain_g1"][:, 1:] @ h.T @ diagnostics["gain_g2"][:, 1:]
        reconstructed_state = prior + np.einsum(
            "btij,btj->bti",
            diagnostics["gain"][:, 1:],
            diagnostics["innov"][:, 1:],
        )
        np.testing.assert_array_equal(diagnostics["innov"][:, 1:], expected_innov)
        np.testing.assert_allclose(diagnostics["gain"][:, 1:], reconstructed_gain, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(x_hat[:, 1:], reconstructed_state)

    def test_same_reset_and_fresh_adapter_are_bitwise_deterministic(self) -> None:
        _, _, x_true, y = _fixture()
        first_adapter = _adapter()
        first_adapter.set_viz_diagnostics_enabled(True)
        first = _eval(first_adapter, x_true, y)
        reset = _eval(first_adapter, x_true, y)
        fresh_adapter = _adapter()
        fresh_adapter.set_viz_diagnostics_enabled(True)
        fresh = _eval(fresh_adapter, x_true, y)
        self.assertTrue(np.array_equal(first["x_hat"].numpy(), reset["x_hat"].numpy()))
        self.assertTrue(np.array_equal(first["x_hat"].numpy(), fresh["x_hat"].numpy()))
        for key in ("innov", "innov_valid", "gain", "gain_g1", "gain_g2"):
            self.assertTrue(np.array_equal(first["diagnostics"][key], reset["diagnostics"][key], equal_nan=True))
            self.assertTrue(np.array_equal(first["diagnostics"][key], fresh["diagnostics"][key], equal_nan=True))

    def test_writer_loader_roundtrip_preserves_semantics_and_no_physical_covariance(self) -> None:
        _, _, x_true, y = _fixture()
        adapter = _adapter()
        adapter.set_viz_diagnostics_enabled(True)
        result = _eval(adapter, x_true, y)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(
                _write_split_run(
                    Path(tmp),
                    result["diagnostics"],
                    x_true=x_true,
                    y=y,
                    x_hat=result["x_hat"].numpy(),
                    adapter_meta=adapter.get_adapter_meta(),
                )
            )
            traj = run.load_trajectory(stored_index=0)
            self.assertEqual(run.meta["artifact_version"], "1.1")
            self.assertEqual(
                {key: run.meta["capabilities"][key] for key in ("gain", "innovation", "covariance", "innovation_cov")},
                {"gain": True, "innovation": True, "covariance": False, "innovation_cov": False},
            )
            self.assertEqual(run.meta["diagnostic_semantics"]["gain"], "learned_combined_kalman_gain")
            self.assertEqual(run.meta["diagnostic_semantics"]["gain_g1"], "learned_split_factor_g1")
            self.assertEqual(run.meta["diagnostic_semantics"]["gain_g2"], "learned_split_factor_g2")
            self.assertNotIn("P", traj)
            self.assertNotIn("S", traj)
            self.assertEqual(traj["gain"].dtype, np.float16)
            self.assertEqual(traj["gain_g1"].dtype, np.float16)
            self.assertEqual(traj["gain_g2"].dtype, np.float16)
            self.assertFalse(bool(traj["innov_valid"][0]))
            self.assertTrue(np.isnan(traj["gain"][0]).all())

    def test_malformed_components_shape_semantics_and_nan_are_rejected(self) -> None:
        _, _, x_true, y = _fixture()
        adapter = _adapter()
        adapter.set_viz_diagnostics_enabled(True)
        diagnostics = _eval(adapter, x_true, y)["diagnostics"]
        cases: list[tuple[str, dict[str, np.ndarray], dict, str]] = []
        missing_g2 = {key: value.copy() for key, value in diagnostics.items() if key != "gain_g2"}
        cases.append(("missing_g2", missing_g2, adapter.get_adapter_meta(), "both gain_g1 and gain_g2"))
        short_g1 = {key: value.copy() for key, value in diagnostics.items()}
        short_g1["gain_g1"] = short_g1["gain_g1"][:, :-1]
        cases.append(("short_g1", short_g1, adapter.get_adapter_meta(), "time dimension"))
        nan_g1 = {key: value.copy() for key, value in diagnostics.items()}
        nan_g1["gain_g1"][:, 2] = np.nan
        cases.append(("nan_g1", nan_g1, adapter.get_adapter_meta(), "NaN/Inf at a valid"))
        wrong_meta = adapter.get_adapter_meta()
        wrong_meta["diagnostic_semantics"] = dict(wrong_meta["diagnostic_semantics"])
        wrong_meta["diagnostic_semantics"]["gain_g1"] = "posterior_covariance"
        cases.append(("wrong_semantics", diagnostics, wrong_meta, "must be 'learned_split_factor_g1'"))
        for name, diag, meta, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory(dir="/tmp") as tmp:
                with self.assertRaisesRegex((ContractError, ValueError), message):
                    _write_split_run(Path(tmp), diag, adapter_meta=meta)

    def test_ui_sources_follow_actual_keys_and_shape_changes_are_clamped(self) -> None:
        meta = {
            "diagnostic_semantics": _adapter().get_adapter_meta()["diagnostic_semantics"],
        }
        split_traj = {
            "gain": np.zeros((5, 3, 2)),
            "gain_g1": np.zeros((5, 3, 3)),
            "gain_g2": np.zeros((5, 2, 2)),
        }
        self.assertEqual(
            available_gain_sources(meta, split_traj),
            [
                ("gain", "Learned combined Kalman gain"),
                ("gain_g1", "Learned G1 factor"),
                ("gain_g2", "Learned G2 factor"),
            ],
        )
        self.assertEqual(available_gain_sources({}, {"gain": split_traj["gain"]}), [("gain", "Kalman gain")])
        self.assertEqual(clamp_gain_element(split_traj["gain_g2"].shape, 2, 2), (1, 1))

    def test_trajectory_switch_updates_components_but_keeps_dataset_summary(self) -> None:
        _, _, x_true, y = _fixture()
        diagnostics = {
            "innov": np.stack([np.full((8, 2), value, dtype=np.float32) for value in (0.01, 0.02, 0.03)]),
            "innov_valid": np.ones((3, 8), dtype=bool),
            "gain": np.stack([np.full((8, 2, 2), value, dtype=np.float32) for value in (1.0, 2.0, 3.0)]),
            "gain_g1": np.stack([np.full((8, 2, 2), value, dtype=np.float32) for value in (0.1, 0.2, 0.3)]),
            "gain_g2": np.stack([np.full((8, 2, 2), value, dtype=np.float32) for value in (0.4, 0.5, 0.6)]),
        }
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_split_run(Path(tmp), diagnostics, x_true=x_true, y=y))
            first = build_run_inspector_bundle(run, traj_idx=0, axis_mode="split")
            second = build_run_inspector_bundle(run, traj_idx=1, axis_mode="split")
            self.assertEqual(first["dataset_summary"], second["dataset_summary"])
            for key in ("innov", "gain", "gain_g1", "gain_g2"):
                self.assertFalse(np.array_equal(first["traj"][key], second["traj"][key]))


if __name__ == "__main__":
    unittest.main()
