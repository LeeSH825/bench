from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from viz.app.components.regime_strip import build_regime_strip
from viz.app.views.run_inspector import build_run_inspector_bundle
from viz.figures.panels import _minmax_downsample_indices
from viz.io.loader import load_run
from viz.io.writer import write_viz_artifacts


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_v0_run(root: Path, *, n_seq: int = 8, n_step: int = 24) -> Path:
    rng = np.random.default_rng(44)
    dim = 6
    x_true = rng.normal(scale=0.01, size=(n_seq, n_step, dim)).astype(np.float32)
    x_hat = (x_true + rng.normal(scale=1e-3, size=(n_seq, n_step, dim))).astype(np.float32)
    y_obs = (x_true + rng.normal(scale=1e-3, size=(n_seq, n_step, dim))).astype(np.float32)
    p = np.broadcast_to(np.eye(dim, dtype=np.float32), (n_seq, n_step, dim, dim)).copy()
    s = np.broadcast_to(np.eye(dim, dtype=np.float32), (n_seq, n_step, dim, dim)).copy()
    gain = np.broadcast_to(np.eye(dim, dtype=np.float32), (n_seq, n_step, dim, dim)).copy()
    innov = (y_obs - x_hat).astype(np.float32)
    run_dir = root / "run"
    write_viz_artifacts(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        suite_name="inspector_suite",
        task_id="Basilisk_ADCS_mrp_omega_T24_smoke_v0",
        task_family="basilisk_adcs_v0",
        scenario_id="scenario_inspector",
        model_id="basilisk_mrp_ekf",
        seed=0,
        track_id="frozen",
        init_id="pretrained",
        run_status="ok",
        time_s=np.arange(n_step, dtype=np.float32),
        time_meta={"time_source": "test", "time_unit": "s", "dt_s": 1.0},
        x_true=x_true,
        y_obs=y_obs,
        x_hat=x_hat,
        split_extras={},
        diagnostics={"P": p, "S": s, "gain": gain, "innov": innov},
        adapter_meta={"adapter_id": "test"},
    )
    return run_dir


def _write_linear_run(
    root: Path,
    *,
    name: str,
    diagnostics: dict | None,
    extras: dict | None = None,
    run_status: str = "ok",
    model_id: str = "oracle_kf",
    adapter_meta: dict | None = None,
    error_offsets: np.ndarray | None = None,
) -> Path:
    n_seq, n_step, x_dim, y_dim = 3, 12, 2, 1
    t = np.arange(n_step, dtype=np.float32)
    base = np.stack([0.1 * t, -0.05 * t], axis=1).astype(np.float32)
    x_true = np.broadcast_to(base, (n_seq, n_step, x_dim)).copy()
    if error_offsets is None:
        x_hat = (x_true + np.float32(0.01)).astype(np.float32)
    else:
        offsets = np.asarray(error_offsets, dtype=np.float32).reshape(n_seq, 1, 1)
        x_hat = (x_true + offsets).astype(np.float32)
    y_obs = x_true[:, :, :1].copy()
    run_dir = root / name
    write_viz_artifacts(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        suite_name="inspector_suite",
        task_id=f"{name}_linear_v0",
        task_family="linear_gaussian_v0",
        scenario_id="scenario_linear_inspector",
        model_id=model_id,
        seed=0,
        track_id="frozen",
        init_id="pretrained",
        run_status=run_status,
        time_s=t,
        time_meta={"time_source": "test", "time_unit": "s", "dt_s": 1.0},
        x_true=x_true,
        y_obs=y_obs,
        x_hat=x_hat,
        split_extras=extras or {},
        diagnostics=diagnostics or {},
        adapter_meta=adapter_meta or {"adapter_id": "test"},
    )
    return run_dir


def _full_kf_diagnostics() -> dict:
    n_seq, n_step, x_dim, y_dim = 3, 12, 2, 1
    p = np.broadcast_to(np.eye(x_dim, dtype=np.float32), (n_seq, n_step, x_dim, x_dim)).copy()
    s = np.broadcast_to(np.eye(y_dim, dtype=np.float32), (n_seq, n_step, y_dim, y_dim)).copy()
    gain = np.ones((n_seq, n_step, x_dim, y_dim), dtype=np.float32) * np.float32(0.2)
    innov = np.ones((n_seq, n_step, y_dim), dtype=np.float32) * np.float32(0.01)
    return {"P": p, "S": s, "gain": gain, "innov": innov}


class VizRunInspectorTest(unittest.TestCase):
    def test_v0_artifact_builds_disabled_panels_and_emp_std_warning(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_v0_run(Path(tmp)))
            bundle = build_run_inspector_bundle(run, traj_idx=0, axis_mode="split")
            self.assertIn("capabilities.bias_state=false", bundle["figures"]["bias"].disabled_reason)
            self.assertIn("bias_component", bundle["figures"]["decomposition"].disabled_reason)
            self.assertTrue(bundle["empirical_sigma"]["warning"])
            self.assertEqual(bundle["empirical_sigma"]["n_samples"], 8)
            self.assertEqual(bundle["summary"]["window"], "All (24 steps)")
            self.assertIn("coverage_pct", bundle["summary"])
            self.assertGreater(bundle["build_seconds"], 0.0)

    def test_analysis_window_changes_summary_window(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_v0_run(Path(tmp)))
            bundle = build_run_inspector_bundle(
                run,
                traj_idx=0,
                axis_mode="split",
                analysis_window={"mode": "exclude_20", "label": "Exclude first 20%", "start_idx": 4},
            )
            self.assertEqual(bundle["analysis_window"]["start_idx"], 4)
            self.assertEqual(bundle["summary"]["window"], "Exclude first 20% (20 steps)")

    def test_linear_full_kf_fixture_enables_consistency_and_gain_without_attitude(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_linear_run(Path(tmp), name="full_kf", diagnostics=_full_kf_diagnostics()))
            bundle = build_run_inspector_bundle(run, traj_idx=0, axis_mode="split")
            self.assertIn("state_spec has no attitude state", bundle["figures"]["attitude_rpy"].disabled_reason)
            self.assertIsNone(bundle["figures"]["consistency"].disabled_reason)
            self.assertIsNone(bundle["figures"]["gain"].disabled_reason)
            self.assertEqual(bundle["summary"]["attitude_rmse_deg"], "NA")
            traj = bundle["traj"]
            self.assertEqual(traj["P"].shape, (12, 2, 2))
            self.assertEqual(traj["S"].shape, (12, 1, 1))
            self.assertEqual(traj["gain"].shape, (12, 2, 1))

    def test_gain_only_fixture_keeps_physical_consistency_disabled(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            diag = _full_kf_diagnostics()
            diag = {"gain": diag["gain"], "innov": diag["innov"]}
            run = load_run(
                _write_linear_run(
                    Path(tmp),
                    name="gain_only",
                    diagnostics=diag,
                    model_id="kalmannet_tsp",
                    adapter_meta={"adapter_id": "kalmannet_tsp", "gain_semantics": "learned_kalman_gain"},
                )
            )
            bundle = build_run_inspector_bundle(
                run,
                traj_idx=0,
                axis_mode="split",
                gain_display={"gain_key": "gain", "mode": "element", "row": 1, "col": 0},
            )
            self.assertIsNone(bundle["figures"]["gain"].disabled_reason)
            self.assertIn("NEES unavailable", bundle["figures"]["consistency"].disabled_reason)
            self.assertIn("physical state covariance P is not provided", bundle["figures"]["consistency"].disabled_reason)
            self.assertIn("innovation covariance S is not provided", bundle["figures"]["consistency"].disabled_reason)
            self.assertIn("Learned Kalman gain", bundle["figures"]["gain"].figure.layout.title.text)
            self.assertNotEqual(bundle["summary"]["generic_state_rmse"], "NA")
            self.assertFalse(bundle["empirical_sigma"]["physical_covariance_available"])
            self.assertTrue(run.meta["capabilities"]["gain"])
            self.assertTrue(run.meta["capabilities"]["innovation"])
            self.assertFalse(run.meta["capabilities"]["covariance"])
            self.assertFalse(run.meta["capabilities"]["innovation_cov"])

    def test_empirical_ensemble_uncertainty_uses_sample_std_and_reports_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            offsets = np.asarray([-1.0, 0.0, 2.0], dtype=np.float32)
            run = load_run(
                _write_linear_run(
                    Path(tmp),
                    name="empirical_ensemble",
                    diagnostics={},
                    model_id="kalmannet_tsp",
                    error_offsets=offsets,
                )
            )
            expected = float(np.std(offsets, ddof=1))
            self.assertIsNotNone(run.aggregate)
            np.testing.assert_allclose(run.aggregate["emp_std"], expected, rtol=1e-6, atol=1e-6)
            bundle = build_run_inspector_bundle(run, traj_idx=0, axis_mode="split")
            status = bundle["empirical_sigma"]
            self.assertEqual(status["label"], "Empirical ensemble uncertainty")
            self.assertEqual(status["n_samples"], 3)
            self.assertAlmostEqual(status["emp_std_mean"], expected, places=6)
            self.assertTrue(status["warning"])
            self.assertFalse(status["physical_covariance_available"])
            self.assertEqual(len(status["confidence_interval_mean"]), 2)

    def test_no_diagnostics_fixture_disables_related_panels(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_linear_run(Path(tmp), name="no_diag", diagnostics={}))
            bundle = build_run_inspector_bundle(run, traj_idx=0, axis_mode="split")
            self.assertIn("innovation is not available", bundle["figures"]["innovation"].disabled_reason)
            self.assertIn("Kalman gain is not available", bundle["figures"]["gain"].disabled_reason)
            self.assertFalse(run.meta["capabilities"]["gain"])
            self.assertFalse(run.meta["capabilities"]["covariance"])

    def test_active_regime_fixture_extracts_exact_intervals(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            event = np.zeros((3, 12), dtype=bool)
            eclipse = np.zeros((3, 12), dtype=bool)
            event[:, 2:5] = True
            eclipse[:, 7:11] = True
            run = load_run(
                _write_linear_run(
                    Path(tmp),
                    name="active_regime",
                    diagnostics={},
                    extras={"event_flag_seq": event, "eclipse_flag_seq": eclipse},
                )
            )
            traj = load_run(run.run_dir).aggregate
            self.assertIsNotNone(traj)
            loaded_traj = build_run_inspector_bundle(run, traj_idx=0, axis_mode="split")["traj"]
            result = build_regime_strip(run.meta, loaded_traj)
            self.assertIsNone(result.disabled_reason)
            self.assertEqual(result.intervals["event_flag"], [{"start": 2, "end": 5, "start_time": 2.0, "end_time": 5.0}])
            self.assertEqual(result.intervals["eclipse_flag"], [{"start": 7, "end": 11, "start_time": 7.0, "end_time": 11.0}])

    def test_all_false_regime_fixture_distinguishes_no_event(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            event = np.zeros((3, 12), dtype=bool)
            run = load_run(
                _write_linear_run(
                    Path(tmp),
                    name="inactive_regime",
                    diagnostics={},
                    extras={"event_flag_seq": event},
                )
            )
            loaded_traj = build_run_inspector_bundle(run, traj_idx=0, axis_mode="split")["traj"]
            result = build_regime_strip(run.meta, loaded_traj)
            self.assertIsNone(result.disabled_reason)
            self.assertIn("no eclipse/event interval occurred", result.empty_reason)

    def test_failed_run_fixture_records_status(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_linear_run(Path(tmp), name="failed_run", diagnostics={}, run_status="train_nan"))
            self.assertEqual(run.meta["run_status"], "train_nan")

    def test_minmax_downsample_preserves_extrema(self) -> None:
        values = np.zeros(1000, dtype=np.float64)
        values[123] = 9.0
        values[876] = -7.0
        indices, downsampled = _minmax_downsample_indices(values, max_points=100)
        self.assertTrue(downsampled)
        self.assertIn(123, set(indices.tolist()))
        self.assertIn(876, set(indices.tolist()))


if __name__ == "__main__":
    unittest.main()
