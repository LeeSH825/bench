from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from viz.analysis import attitude, comparison
from viz.app.components.comparison_picker import comparison_candidates
from viz.contract import ContractError
from viz.figures import panels
from viz.io.loader import assert_overlay_compatible, load_run
from viz.io.writer import write_viz_artifacts


REPO_ROOT = Path(__file__).resolve().parents[1]


def _payload(*, n_seq: int = 4, n_step: int = 12) -> tuple[np.ndarray, ...]:
    t = np.arange(n_step, dtype=np.float32) * np.float32(0.5)
    x_true = np.zeros((n_seq, n_step, 9), dtype=np.float32)
    for sequence in range(n_seq):
        x_true[sequence, :, 0] = np.linspace(0.0, 0.04, n_step, dtype=np.float32)
        x_true[sequence, :, 1] = np.float32(sequence) * np.float32(0.002)
        x_true[sequence, :, 3:6] = np.float32(0.001)
        x_true[sequence, :, 6:9] = np.array([1.0e-5, -2.0e-5, 0.5e-5], dtype=np.float32)
    offsets = np.arange(1, n_seq + 1, dtype=np.float32).reshape(n_seq, 1, 1)
    x_hat = x_true.copy()
    x_hat[:, :, 0:3] += offsets * np.float32(2.0e-4)
    x_hat[:, :, 6:9] += offsets * np.float32(1.0e-7)
    y = np.zeros((n_seq, n_step, 6), dtype=np.float32)
    y[:, :, 0:3] = x_true[:, :, 3:6] + x_true[:, :, 6:9]
    y[:, :, 3:6] = x_true[:, :, 3:6] * np.float32(0.5)
    return t, x_true, x_hat, y


def _diagnostics(*, physical: bool, n_seq: int = 4, n_step: int = 12) -> dict[str, np.ndarray]:
    innov = np.full((n_seq, n_step, 6), np.float32(2.0e-4), dtype=np.float32)
    gain = np.zeros((n_seq, n_step, 9, 6), dtype=np.float32)
    gain[:, :, 0:6, :] = np.eye(6, dtype=np.float32) * np.float32(0.1)
    gain[:, :, 6:9, 0:3] = np.eye(3, dtype=np.float32) * np.float32(0.02)
    out = {"innov": innov, "gain": gain}
    if physical:
        p = np.broadcast_to(np.eye(9, dtype=np.float32) * np.float32(1.0e-6), (n_seq, n_step, 9, 9)).copy()
        s = np.broadcast_to(np.eye(6, dtype=np.float32) * np.float32(2.0e-5), (n_seq, n_step, 6, 6)).copy()
        out.update({"P": p, "S": s})
    return out


def _write_run(
    root: Path,
    name: str,
    *,
    physical: bool,
    model_id: str,
    data_split: str = "test",
    source_ids: np.ndarray | None = None,
    time_offset: float = 0.0,
    estimate_scale: float = 1.0,
) -> Path:
    t, x_true, x_hat, y = _payload()
    x_hat = x_true + (x_hat - x_true) * np.float32(estimate_scale)
    t = t + np.float32(time_offset)
    b_true = x_true[:, :, 6:9].copy()
    event_flag = np.zeros((x_true.shape[0], x_true.shape[1]), dtype=bool)
    eclipse_flag = np.zeros_like(event_flag)
    event_flag[:, 3:5] = True
    eclipse_flag[:, 7:10] = True
    run_dir = root / name
    write_viz_artifacts(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        suite_name="cross_model_fixture",
        task_id="attitude_bias_comparison_v0",
        task_family="basilisk_imu_adcs_bias_v0",
        scenario_id="physical_scenario_shared",
        model_id=model_id,
        seed=0,
        track_id="frozen",
        init_id="fixture",
        run_status="ok",
        time_s=t,
        time_meta={"time_source": "fixture", "time_unit": "s", "dt_s": 0.5},
        x_true=x_true,
        y_obs=y,
        x_hat=x_hat,
        split_extras={
            "gyro_bias_seq": b_true,
            "event_flag_seq": event_flag,
            "eclipse_flag_seq": eclipse_flag,
        },
        diagnostics=_diagnostics(physical=physical),
        adapter_meta={
            "adapter_id": "fixture",
            "gain_semantics": "model_based_kalman_gain" if physical else "learned_kalman_gain",
        },
        data_split=data_split,
        split_source="explicit",
        trajectory_ids=(source_ids if source_ids is not None else np.array([0, 5, 10, 15], dtype=np.int64)),
        trajectory_id_source="test_split_row_index_fallback",
        k_traj=4,
    )
    return run_dir


def _read_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))


def _write_meta(run_dir: Path, meta: dict) -> None:
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _as_rotation_vector_formulation(run_dir: Path) -> None:
    meta = _read_meta(run_dir)
    meta["formulation"] = "error_state_mekf_rotation_vector_v1"
    meta["state_spec"]["covariance_space"] = "rotation_vector_rad"
    covariance = meta["comparison_spec"]["covariance"]
    covariance["space"] = "rotation_vector_rad"
    gain = meta["comparison_spec"]["gain"]
    gain["row_state_order"][0:3] = ["delta_theta_x", "delta_theta_y", "delta_theta_z"]
    gain["row_units"][0:3] = ["rad", "rad", "rad"]
    gain["state_scaling"] = "error_state_rotation_vector"
    correction = meta["comparison_spec"]["correction"]
    correction["attitude_coordinate_space"] = "rotation_vector_rad"
    _write_meta(run_dir, meta)


def _compat(
    base,
    candidate,
    metric: str,
    *,
    strict: bool = False,
    source: int = 5,
) -> dict:
    base_traj = base.load_trajectory(source_trajectory_id=source)
    candidate_traj = candidate.load_trajectory(source_trajectory_id=source)
    evaluator = (
        comparison.evaluate_internal_metric_compatibility
        if strict
        else comparison.evaluate_physical_metric_compatibility
    )
    return evaluator(
        base.meta,
        candidate.meta,
        metric=metric,
        base_source_trajectory_id=source,
        candidate_source_trajectory_id=source,
        base_time=base_traj["t"],
        candidate_time=candidate_traj["t"],
    )


class VizCrossModelComparisonTests(unittest.TestCase):
    def test_writer_emits_explicit_canonical_comparison_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "mekf", physical=True, model_id="model_kf"))
            spec = run.meta["comparison_spec"]
            self.assertEqual(spec["comparison_source"], "explicit_writer_v1_1")
            self.assertEqual(spec["attitude"]["quaternion_order"], "wxyz")
            self.assertEqual(spec["attitude"]["rotation_direction"], "body_to_inertial")
            self.assertEqual(spec["attitude"]["rpy_sequence"], "ZYX")
            self.assertEqual(spec["attitude"]["rpy_convention"], "intrinsic")
            self.assertEqual(spec["bias"]["estimate_block"], [6, 9])
            self.assertFalse(spec["empirical_uncertainty"]["physical"])

    def test_physical_attitude_is_compatible_across_formulations(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            additive = load_run(_write_run(Path(tmp), "additive", physical=True, model_id="additive"))
            mekf_dir = _write_run(Path(tmp), "mekf", physical=True, model_id="mekf")
            _as_rotation_vector_formulation(mekf_dir)
            mekf = load_run(mekf_dir)
            result = _compat(additive, mekf, "attitude_geodesic_error")
            self.assertTrue(result["compatible"], result["reasons"])

    def test_rpy_convention_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            other = load_run(_write_run(Path(tmp), "other", physical=True, model_id="other"))
            candidate_meta = json.loads(json.dumps(other.meta))
            candidate_meta["comparison_spec"]["attitude"]["rpy_convention"] = "extrinsic"
            result = comparison.evaluate_physical_metric_compatibility(
                base.meta, candidate_meta, metric="attitude_rpy"
            )
            self.assertFalse(result["compatible"])
            self.assertTrue(any("rpy_convention mismatch" in reason for reason in result["reasons"]))

    def test_frame_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            other = load_run(_write_run(Path(tmp), "other", physical=True, model_id="other"))
            candidate_meta = json.loads(json.dumps(other.meta))
            candidate_meta["comparison_spec"]["attitude"]["frame_to"] = "orbit_L"
            result = comparison.evaluate_physical_metric_compatibility(
                base.meta, candidate_meta, metric="attitude_geodesic_error"
            )
            self.assertFalse(result["compatible"])
            self.assertTrue(any("frame_to mismatch" in reason for reason in result["reasons"]))

    def test_unknown_quaternion_order_is_rejected_by_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_dir = _write_run(Path(tmp), "bad_order", physical=True, model_id="bad")
            meta = _read_meta(run_dir)
            meta["comparison_spec"]["attitude"]["quaternion_order"] = "xyzw"
            _write_meta(run_dir, meta)
            with self.assertRaisesRegex(ContractError, "quaternion_order"):
                load_run(run_dir)

    def test_geodesic_and_error_components_use_canonical_quaternions(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "angles", physical=True, model_id="angles"))
            traj = run.load_trajectory(stored_index=0)
            geodesic = comparison.attitude_geodesic_error_deg(run.meta, traj)
            components = comparison.attitude_error_components_deg(run.meta, traj)
            np.testing.assert_allclose(np.linalg.norm(components, axis=1), geodesic, rtol=1e-8, atol=1e-10)

    def test_mrp_and_rotation_vector_covariance_bands_differ_by_factor(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            mrp = load_run(_write_run(Path(tmp), "mrp", physical=True, model_id="mrp"))
            rot_dir = _write_run(Path(tmp), "rot", physical=True, model_id="rot")
            _as_rotation_vector_formulation(rot_dir)
            rot = load_run(rot_dir)
            mrp_band = comparison.physical_attitude_band_deg(mrp.meta, mrp.load_trajectory(stored_index=0))
            rot_band = comparison.physical_attitude_band_deg(rot.meta, rot.load_trajectory(stored_index=0))
            np.testing.assert_allclose(mrp_band, rot_band * 4, rtol=1e-12, atol=0.0)

    def test_bias_estimate_error_and_physical_band_use_declared_block(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "bias", physical=True, model_id="bias"))
            traj = run.load_trajectory(stored_index=0)
            estimate = comparison.bias_estimate_deg_h(run.meta, traj)
            truth = comparison.bias_truth_deg_h(run.meta, traj)
            error = comparison.bias_error_deg_h(run.meta, traj)
            np.testing.assert_allclose(estimate - truth, error, rtol=0.0, atol=1e-12)
            self.assertEqual(comparison.physical_bias_band_deg_h(run.meta, traj).shape, (12, 3))

    def test_learned_model_has_no_physical_sigma(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "learned", physical=False, model_id="learned"))
            traj = run.load_trajectory(stored_index=0)
            self.assertIsNone(comparison.physical_attitude_band_deg(run.meta, traj))
            self.assertIsNone(comparison.physical_bias_band_deg_h(run.meta, traj))
            self.assertFalse(run.meta["comparison_spec"]["covariance"]["physical"])

    def test_empirical_and_physical_uncertainty_have_distinct_trace_styles(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            physical = load_run(_write_run(Path(tmp), "physical", physical=True, model_id="physical"))
            learned = load_run(_write_run(Path(tmp), "learned", physical=False, model_id="learned"))
            models = [
                {"label": "physical", "meta": physical.meta, "traj": physical.load_trajectory(stored_index=0), "aggregate": physical.aggregate},
                {"label": "learned", "meta": learned.meta, "traj": learned.load_trajectory(stored_index=0), "aggregate": learned.aggregate},
            ]
            result = panels.cross_model_comparison_panel(
                "attitude_uncertainty", models, show_empirical=True
            )
            names = [str(trace.name) for trace in result.figure.data]
            self.assertTrue(any("physical +3 sigma" in name for name in names))
            self.assertTrue(any("empirical +1 sigma (ensemble)" in name for name in names))
            empirical_traces = [trace for trace in result.figure.data if "empirical" in str(trace.name)]
            self.assertTrue(all(getattr(trace, "fill", None) in (None, "none") for trace in empirical_traces))

    def test_physical_band_fill_uses_each_model_color(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            first = load_run(_write_run(Path(tmp), "first", physical=True, model_id="first"))
            second = load_run(_write_run(Path(tmp), "second", physical=True, model_id="second"))
            result = panels.cross_model_comparison_panel(
                "attitude_uncertainty",
                [
                    {"label": "first", "meta": first.meta, "traj": first.load_trajectory(stored_index=0), "aggregate": first.aggregate},
                    {"label": "second", "meta": second.meta, "traj": second.load_trajectory(stored_index=0), "aggregate": second.aggregate},
                ],
            )
            fills = {
                str(trace.fillcolor)
                for trace in result.figure.data
                if getattr(trace, "fill", None) == "tonexty"
            }
            self.assertEqual(len(fills), 2)

    def test_innovation_is_compatible_when_measurement_semantics_match(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            learned = load_run(_write_run(Path(tmp), "learned", physical=False, model_id="learned"))
            result = _compat(base, learned, "innovation", strict=True)
            self.assertTrue(result["compatible"], result["reasons"])

    def test_innovation_measurement_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            other = load_run(_write_run(Path(tmp), "other", physical=True, model_id="other"))
            candidate_meta = json.loads(json.dumps(other.meta))
            candidate_meta["comparison_spec"]["innovation"]["measurement_type"] = "gyro_only"
            result = comparison.evaluate_internal_metric_compatibility(
                base.meta, candidate_meta, metric="innovation"
            )
            self.assertFalse(result["compatible"])
            self.assertTrue(any("measurement_type mismatch" in reason for reason in result["reasons"]))

    def test_innovation_panel_marks_stored_event_starts(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "events", physical=True, model_id="events"))
            traj = dict(run.load_trajectory(source_trajectory_id=5))
            traj["event_flag"] = np.array(
                [False, False, True, True, False, False, True, False, False, False, False, False],
                dtype=bool,
            )
            result = panels.cross_model_comparison_panel(
                "innovation",
                [{"meta": run.meta, "traj": traj, "aggregate": run.aggregate, "label": "events"}],
            )
            event_trace = next(trace for trace in result.figure.data if trace.name == "Event start")
            np.testing.assert_array_equal(np.asarray(event_trace.x), traj["t"][[2, 6]])

    def test_raw_gain_is_compatible_only_for_same_state_semantics(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            same = load_run(_write_run(Path(tmp), "same", physical=False, model_id="same"))
            self.assertTrue(_compat(base, same, "gain_norm", strict=True)["compatible"])
            changed_meta = json.loads(json.dumps(same.meta))
            changed_meta["comparison_spec"]["gain"]["row_state_order"][0:3] = ["bias_x", "bias_y", "bias_z"]
            result = comparison.evaluate_internal_metric_compatibility(
                base.meta, changed_meta, metric="gain_norm"
            )
            self.assertFalse(result["compatible"])
            self.assertTrue(any("row_state_order mismatch" in reason for reason in result["reasons"]))

    def test_correction_comparison_survives_raw_gain_formulation_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            mekf_dir = _write_run(Path(tmp), "mekf", physical=False, model_id="mekf")
            _as_rotation_vector_formulation(mekf_dir)
            mekf = load_run(mekf_dir)
            self.assertFalse(_compat(base, mekf, "gain_norm", strict=True)["compatible"])
            physical_result = _compat(base, mekf, "attitude_correction")
            self.assertTrue(physical_result["compatible"], physical_result["reasons"])
            self.assertTrue(any("reconstructed" in warning for warning in physical_result["warnings"]))

    def test_correction_is_gain_times_innovation_and_invalid_is_nan(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "correction", physical=False, model_id="correction"))
            traj = run.load_trajectory(stored_index=0)
            traj["innov_valid"][3] = False
            got = comparison.reconstructed_state_correction(run.meta, traj)
            expected = np.einsum("tnm,tm->tn", traj["gain"].astype(np.float64), traj["innov"])
            np.testing.assert_allclose(got[np.arange(len(got)) != 3], expected[np.arange(len(got)) != 3])
            self.assertTrue(np.all(np.isnan(got[3])))

    def test_available_metrics_are_key_semantic_driven(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            physical = load_run(_write_run(Path(tmp), "physical", physical=True, model_id="physical"))
            learned = load_run(_write_run(Path(tmp), "learned", physical=False, model_id="learned"))
            self.assertIn("attitude_uncertainty", comparison.available_physical_metrics(learned.meta))
            self.assertNotIn("nees", comparison.available_internal_metrics(learned.meta))
            self.assertIn("nees", comparison.available_internal_metrics(physical.meta))

    def test_raw_state_error_is_truth_subtracted_from_estimate(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "state_error", physical=False, model_id="state_error"))
            traj = run.load_trajectory(stored_index=0)
            got = comparison.strict_metric_series(run.meta, traj, "state_error")
            np.testing.assert_array_equal(got, traj["x_hat"] - traj["x_true"])
            result = panels.cross_model_comparison_panel(
                "state_error",
                [{"label": "state_error", "meta": run.meta, "traj": traj, "aggregate": run.aggregate}],
            )
            self.assertEqual(len(result.figure.data), 9)

    def test_dataset_rmse_is_computed_over_all_trajectories(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "dataset_rmse", physical=False, model_id="dataset_rmse"))
            title, value = panels.dataset_rmse_metric(run.meta, run.aggregate, run.metrics)
            self.assertEqual(title, "Dataset generic-state RMSE [state units]")
            errors = []
            for info in run.list_trajectories():
                traj = run.load_trajectory(stored_index=info.stored_index)
                errors.append(np.asarray(traj["x_hat"], dtype=np.float64) - traj["x_true"])
            expected = float(np.sqrt(np.mean(np.stack(errors) ** 2)))
            self.assertAlmostEqual(float(value), expected, places=4)

    def test_dataset_rmse_prefers_runner_accuracy_over_ambiguous_aggregate_std(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "dataset_rmse_accuracy", physical=False, model_id="dataset_rmse_accuracy"))
            title, value = panels.dataset_rmse_metric(
                run.meta,
                {"emp_std": np.ones((12, 9), dtype=np.float32)},
                {"accuracy": {"rmse": 0.123456}},
            )
            self.assertEqual(title, "Dataset RMSE [native state units]")
            self.assertEqual(value, "0.1235")

    def test_model_toggle_adds_and_removes_traces(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            other = load_run(_write_run(Path(tmp), "other", physical=False, model_id="other"))
            base_data = {"label": "base", "meta": base.meta, "traj": base.load_trajectory(stored_index=0), "aggregate": base.aggregate}
            other_data = {"label": "other", "meta": other.meta, "traj": other.load_trajectory(stored_index=0), "aggregate": other.aggregate}
            one = panels.cross_model_comparison_panel("attitude_geodesic_error", [base_data])
            two = panels.cross_model_comparison_panel("attitude_geodesic_error", [base_data, other_data])
            self.assertEqual(len(one.figure.data), 1)
            self.assertEqual(len(two.figure.data), 2)

    def test_candidate_index_does_not_load_trajectory_npz(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            other = load_run(_write_run(Path(tmp), "other", physical=False, model_id="other"))
            with mock.patch.object(type(other), "load_trajectory", side_effect=AssertionError("eager load")):
                candidates = comparison_candidates(
                    base,
                    [base, other],
                    source_trajectory_id=5,
                    mode="physical",
                    metric="attitude_geodesic_error",
                )
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].trajectory.source_trajectory_id, 5)

    def test_dataset_summary_is_not_changed_by_comparison_selection(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            before = panels.dataset_summary_items(base.meta, base.aggregate, base.metrics)
            panels.cross_model_comparison_panel(
                "attitude_geodesic_error",
                [{"label": "base", "meta": base.meta, "traj": base.load_trajectory(stored_index=0), "aggregate": base.aggregate}],
            )
            after = panels.dataset_summary_items(base.meta, base.aggregate, base.metrics)
            self.assertEqual(before, after)

    def test_trajectory_change_updates_comparison_trace(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run = load_run(_write_run(Path(tmp), "run", physical=True, model_id="run"))
            first = panels.cross_model_comparison_panel(
                "attitude_geodesic_error",
                [{"label": "run", "meta": run.meta, "traj": run.load_trajectory(source_trajectory_id=0), "aggregate": run.aggregate}],
            )
            second = panels.cross_model_comparison_panel(
                "attitude_geodesic_error",
                [{"label": "run", "meta": run.meta, "traj": run.load_trajectory(source_trajectory_id=5), "aggregate": run.aggregate}],
            )
            self.assertFalse(np.array_equal(np.asarray(first.figure.data[0].y), np.asarray(second.figure.data[0].y)))

    def test_source_mismatch_is_rejected_and_candidate_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            other = load_run(
                _write_run(
                    Path(tmp),
                    "other",
                    physical=False,
                    model_id="other",
                    source_ids=np.array([20, 21, 22, 23]),
                )
            )
            result = comparison.evaluate_physical_metric_compatibility(
                base.meta,
                other.meta,
                metric="attitude_geodesic_error",
                base_source_trajectory_id=5,
                candidate_source_trajectory_id=None,
            )
            self.assertFalse(result["compatible"])
            candidates = comparison_candidates(
                base, [base, other], source_trajectory_id=5, mode="physical", metric="attitude_geodesic_error"
            )
            self.assertEqual(len(candidates), 1)
            self.assertFalse(candidates[0].compatibility["compatible"])

    def test_physical_comparison_allows_training_seed_difference_when_truth_matches(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            other = load_run(_write_run(Path(tmp), "other", physical=True, model_id="other"))
            candidate_meta = json.loads(json.dumps(other.meta))
            candidate_meta["seed"] = 7
            physical = comparison.evaluate_physical_metric_compatibility(
                base.meta,
                candidate_meta,
                metric="attitude_geodesic_error",
            )
            strict = comparison.evaluate_internal_metric_compatibility(
                base.meta,
                candidate_meta,
                metric="innovation",
            )
            self.assertTrue(physical["compatible"], physical["reasons"])
            self.assertFalse(strict["compatible"])
            self.assertTrue(any("strict seed mismatch" in reason for reason in strict["reasons"]))

    def test_physical_comparison_requires_truth_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            other = load_run(_write_run(Path(tmp), "other", physical=True, model_id="other"))
            candidate_meta = json.loads(json.dumps(other.meta))
            candidate_meta["comparison_spec"]["identity"]["truth_fingerprints"].pop("attitude_truth")
            result = comparison.evaluate_physical_metric_compatibility(
                base.meta,
                candidate_meta,
                metric="attitude_geodesic_error",
            )
            self.assertFalse(result["compatible"])
            self.assertTrue(any("identity is unavailable" in reason for reason in result["reasons"]))

    def test_time_mismatch_is_rejected_without_interpolation(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            other = load_run(_write_run(Path(tmp), "other", physical=True, model_id="other", time_offset=0.25))
            result = _compat(base, other, "attitude_geodesic_error")
            self.assertFalse(result["compatible"])
            self.assertTrue(any("time axis mismatch" in reason for reason in result["reasons"]))

    def test_split_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base", data_split="test"))
            other = load_run(_write_run(Path(tmp), "other", physical=True, model_id="other", data_split="validation"))
            result = _compat(base, other, "attitude_geodesic_error")
            self.assertFalse(result["compatible"])
            self.assertTrue(any("data split mismatch" in reason for reason in result["reasons"]))

    def test_row_fallback_provenance_produces_warning(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            other = load_run(_write_run(Path(tmp), "other", physical=False, model_id="other"))
            result = _compat(base, other, "attitude_geodesic_error")
            self.assertTrue(result["compatible"])
            self.assertTrue(any("split-row fallback" in warning for warning in result["warnings"]))

    def test_legacy_artifact_loads_but_comparison_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            modern = load_run(_write_run(Path(tmp), "modern", physical=True, model_id="modern"))
            legacy_dir = _write_run(Path(tmp), "legacy", physical=True, model_id="legacy")
            meta = _read_meta(legacy_dir)
            meta.pop("comparison_spec")
            _write_meta(legacy_dir, meta)
            legacy = load_run(legacy_dir)
            self.assertEqual(comparison.available_physical_metrics(legacy.meta), [])
            result = comparison.evaluate_physical_metric_compatibility(
                modern.meta, legacy.meta, metric="attitude_geodesic_error"
            )
            self.assertFalse(result["compatible"])
            self.assertIn("no comparison_spec", result["reasons"][0])

    def test_strict_overlay_regression_still_blocks_formulation_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            base = load_run(_write_run(Path(tmp), "base", physical=True, model_id="base"))
            other_dir = _write_run(Path(tmp), "other", physical=True, model_id="other")
            _as_rotation_vector_formulation(other_dir)
            other = load_run(other_dir)
            with self.assertRaisesRegex(ContractError, "formulation"):
                assert_overlay_compatible(base, other)

    def test_non_normalized_quaternion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_dir = _write_run(Path(tmp), "bad_q", physical=True, model_id="bad_q")
            target = run_dir / "series" / "traj_0000.npz"
            with np.load(target, allow_pickle=False) as data:
                arrays = {key: np.array(data[key], copy=True) for key in data.files}
            arrays["q_hat"][0] *= np.float32(2.0)
            np.savez_compressed(target, **arrays)
            run = load_run(run_dir)
            with self.assertRaisesRegex(ContractError, "not normalized"):
                run.load_trajectory(stored_index=0)

    def test_empirical_uncertainty_cannot_be_declared_physical(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_dir = _write_run(Path(tmp), "bad_empirical", physical=True, model_id="bad")
            meta = _read_meta(run_dir)
            meta["comparison_spec"]["empirical_uncertainty"]["physical"] = True
            _write_meta(run_dir, meta)
            with self.assertRaisesRegex(ContractError, "physical=false"):
                load_run(run_dir)

    def test_declared_attitude_without_truth_fingerprint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            run_dir = _write_run(Path(tmp), "missing_fingerprint", physical=True, model_id="bad")
            meta = _read_meta(run_dir)
            meta["comparison_spec"]["identity"]["truth_fingerprints"].pop("attitude_truth")
            _write_meta(run_dir, meta)
            with self.assertRaisesRegex(ContractError, "attitude_truth fingerprint"):
                load_run(run_dir)


if __name__ == "__main__":
    unittest.main()
