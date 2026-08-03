from __future__ import annotations

import ast
import inspect
import json
import subprocess
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from bench.estimators.mekf import (
    MEKFState,
    quat_exp,
    quat_inverse,
    quat_log,
    quat_multiply,
    quat_to_dcm,
)
from bench.tasks.generator.basilisk_unit_st import (
    CONVENTION_ID,
    FRAME_PROOF_ID,
    GENERATOR_ID,
    SCHEMA_VERSION,
    SEED_POLICY_VERSION,
    SENSOR_MODEL_VERSION,
    SIMULATOR_ADAPTER_VERSION,
    BasiliskUnitSTConfig,
    align_q_NB_time_series,
    basilisk_sigma_BN_to_q_NB,
    generate_basilisk_unit_st,
    run_dynamic_rate_proof,
    run_static_frame_proof,
    simulate_basilisk_truth,
)
from bench.tasks.generator.mekf_events import (
    MEKFEventTable,
    SensorCode,
    load_event_dataset,
    replay_trajectory,
    save_event_dataset,
)


PYTHON = "/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python"


@pytest.fixture(scope="module")
def static_proof() -> dict[str, object]:
    return run_static_frame_proof()


@pytest.fixture(scope="module")
def dynamic_proof() -> dict[str, object]:
    return run_dynamic_rate_proof()


@pytest.fixture(scope="module")
def generated():
    return generate_basilisk_unit_st(
        BasiliskUnitSTConfig(
            num_trajectories=6,
            duration_s=0.2,
            gyro_rate_hz=20,
            star_tracker_rate_hz=5,
            master_seed=410,
        )
    )


def _event_kwargs(table: MEKFEventTable) -> dict[str, np.ndarray]:
    return {field.name: np.array(getattr(table, field.name), copy=True) for field in fields(table)}


def _initial_state(generated, index: int = 0, *, exact_bias: bool = False) -> MEKFState:
    truth = generated.dataset.truth
    start = int(truth.truth_offsets[index])
    bias = truth.gyro_bias_rad_s[start] if exact_bias else np.zeros(3, dtype=np.float64)
    return MEKFState(
        q_NB=truth.q_true_NB[start],
        b_g=bias,
        P=np.eye(6, dtype=np.float64) * 1.0e-5,
    )


def _process_noise() -> np.ndarray:
    return np.diag(np.asarray([1.0e-10] * 3 + [1.0e-12] * 3, dtype=np.float64))


def _attitude_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(quat_log(quat_multiply(quat_inverse(left), right)))
    )


def _assert_replay_equal(left, right) -> None:
    for name in (
        "event_time_s",
        "event_order",
        "sensor_code",
        "q_NB_history",
        "b_g_history",
        "P_history",
        "attitude_step_rad",
        "star_tracker_event_order",
        "star_tracker_residual",
        "star_tracker_S",
    ):
        assert np.array_equal(getattr(left, name), getattr(right, name))


def test_explicit_runtime_identity_and_basilisk_import(generated) -> None:
    runtime = generated.manifest["software_versions"]
    assert Path(PYTHON).is_file()
    assert runtime["python"] == "3.10.13"
    assert runtime["basilisk"] == "2.10.2"
    assert runtime["bsk_distribution"] == "2.10.2"
    assert Path(runtime["basilisk_path"]).is_file()


def test_import_boundary_has_no_runner_model_metric_torch_or_visualization() -> None:
    source_path = Path("bench/tasks/generator/basilisk_unit_st.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = ("torch", "bench.runners", "bench.models", "bench.metrics", "visualization", "viz")
    assert not any(name == item or name.startswith(item + ".") for name in imported for item in forbidden)
    completed = subprocess.run(
        [
            PYTHON,
            "-c",
            (
                "import sys; import bench.tasks.generator.basilisk_unit_st; "
                "print(','.join(sorted(name for name in sys.modules if "
                "name == 'torch' or name.startswith('bench.runners') or "
                "name.startswith('bench.models') or name.startswith('bench.metrics') or "
                "name.startswith('viz'))))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.stdout.strip() == ""


def test_missing_basilisk_fails_without_fallback(monkeypatch) -> None:
    from bench.tasks.generator import basilisk_unit_st as module

    real_import = module.importlib.import_module

    def reject_basilisk(name: str):
        if name == "Basilisk" or name.startswith("Basilisk."):
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(module.importlib, "import_module", reject_basilisk)
    with pytest.raises(RuntimeError, match="no analytic or synthetic truth fallback"):
        module._require_basilisk()


def test_identity_sigma_maps_to_identity_q_NB() -> None:
    assert np.array_equal(
        basilisk_sigma_BN_to_q_NB(np.zeros(3, dtype=np.float64)),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )


@pytest.mark.parametrize("axis", np.eye(3, dtype=np.float64), ids=("x", "y", "z"))
@pytest.mark.parametrize("sign", (1.0, -1.0), ids=("positive", "negative"))
def test_axis_90_body_basis_mapping(axis: np.ndarray, sign: float) -> None:
    sigma = axis * np.tan(sign * np.pi / 8.0)
    observed = quat_to_dcm(basilisk_sigma_BN_to_q_NB(sigma))
    expected = quat_to_dcm(quat_exp(axis * sign * np.pi / 2.0))
    assert np.allclose(observed @ np.eye(3), expected @ np.eye(3), rtol=0.0, atol=5.0e-15)


def test_positive_z_90_maps_body_x_to_navigation_y() -> None:
    sigma = np.asarray([0.0, 0.0, np.tan(np.pi / 8.0)], dtype=np.float64)
    mapped = quat_to_dcm(basilisk_sigma_BN_to_q_NB(sigma)) @ np.asarray([1.0, 0.0, 0.0])
    assert np.allclose(mapped, [0.0, 1.0, 0.0], rtol=0.0, atol=5.0e-15)


def test_ten_arbitrary_recorder_attitudes_pass_closed_form_basis_proof(static_proof) -> None:
    assert static_proof["arbitrary_case_count"] == 10
    assert static_proof["max_basis_error"] <= 5.0e-15


def test_mrp2c_is_C_BN_and_transpose_is_R_NB(static_proof) -> None:
    assert static_proof["max_R_NB_minus_C_BN_transpose_error"] <= 5.0e-15
    assert static_proof["max_inverse_candidate_error"] >= 1.0
    assert "C_BN.T" in static_proof["relation"]


def test_mrp_shadow_set_is_physically_invariant(static_proof) -> None:
    assert static_proof["max_shadow_dcm_error"] <= 1.0e-14
    assert static_proof["min_shadow_abs_quaternion_dot"] == pytest.approx(1.0, abs=2.0e-15)


@pytest.mark.parametrize(
    "bad",
    (
        np.asarray([np.nan, 0.0, 0.0]),
        np.asarray([np.inf, 0.0, 0.0]),
        np.zeros(2),
        np.zeros((3, 1)),
    ),
)
def test_invalid_mrp_fails_loudly(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        basilisk_sigma_BN_to_q_NB(bad)


def test_adapter_is_unit_norm_and_deterministic() -> None:
    sigma = np.asarray([0.17, -0.08, 0.21], dtype=np.float64)
    first = basilisk_sigma_BN_to_q_NB(sigma)
    second = basilisk_sigma_BN_to_q_NB(sigma)
    assert np.array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=2.0e-15)


def test_time_series_alignment_preserves_dcm_and_continuity() -> None:
    q = np.stack((quat_exp([0.1, 0.0, 0.0]), -quat_exp([0.2, 0.0, 0.0]), quat_exp([0.3, 0.0, 0.0])))
    aligned = align_q_NB_time_series(q)
    assert np.all(np.sum(aligned[:-1] * aligned[1:], axis=1) >= 0.0)
    for before, after in zip(q, aligned):
        assert np.allclose(quat_to_dcm(before), quat_to_dcm(after), rtol=0.0, atol=1.0e-15)


def test_zero_rate_preserves_attitude() -> None:
    q_initial = quat_exp(np.asarray([0.3, -0.2, 0.1]))
    _time, q, omega = simulate_basilisk_truth(q_initial, np.zeros(3), duration_s=0.1, task_step_s=0.01)
    assert max(_attitude_error(q_initial, item) for item in q) <= 1.0e-14
    assert np.array_equal(omega, np.zeros_like(omega))


@pytest.mark.parametrize("axis", np.eye(3, dtype=np.float64), ids=("x", "y", "z"))
@pytest.mark.parametrize("sign", (1.0, -1.0), ids=("positive", "negative"))
def test_axis_rates_match_gate_a_right_propagation(axis: np.ndarray, sign: float) -> None:
    q_initial = quat_exp(np.asarray([0.2, -0.1, 0.05]))
    omega = sign * 0.2 * axis
    time_s, q, recorded = simulate_basilisk_truth(q_initial, omega, duration_s=0.1, task_step_s=0.01)
    errors = [
        _attitude_error(quat_multiply(q_initial, quat_exp(omega * time)), actual)
        for time, actual in zip(time_s, q)
    ]
    assert max(errors) <= 1.0e-14
    assert np.array_equal(recorded, np.repeat(omega[None, :], recorded.shape[0], axis=0))


def test_ten_arbitrary_body_rates_pass(dynamic_proof) -> None:
    assert dynamic_proof["arbitrary_rate_case_count"] == 10
    assert dynamic_proof["rate_case_count"] == 17


def test_local_increment_confirms_rate_frame_sign_and_unit(dynamic_proof) -> None:
    assert dynamic_proof["fine"]["max_local_rate_increment_error_rad_s"] <= 5.0e-13
    assert "same sign" in dynamic_proof["omega_semantics"]


def test_spherical_zero_torque_rate_is_constant(dynamic_proof) -> None:
    assert dynamic_proof["coarse"]["max_recorded_rate_component_error_rad_s"] == 0.0
    assert dynamic_proof["fine"]["max_rate_norm_drift_rad_s"] == 0.0


def test_coarse_fine_attitude_convergence(dynamic_proof) -> None:
    assert dynamic_proof["fine_not_increased"] is True
    assert dynamic_proof["fine"]["max_attitude_log_error_rad"] <= dynamic_proof["coarse"]["max_attitude_log_error_rad"]


def test_fine_dynamic_error_meets_predeclared_target(dynamic_proof) -> None:
    assert dynamic_proof["fine_target_rad"] == 1.0e-8
    assert dynamic_proof["fine"]["max_attitude_log_error_rad"] <= 1.0e-8


def test_exact_truth_event_payload_dtypes_and_shapes(generated) -> None:
    events = generated.dataset.events
    truth = generated.dataset.truth
    expected_event_dtypes = {
        "trajectory_id": np.dtype(np.int64),
        "sensor_code": np.dtype(np.int16),
        "measurement_time_s": np.dtype(np.float64),
        "arrival_time_s": np.dtype(np.float64),
        "event_order": np.dtype(np.int64),
        "valid": np.dtype(np.bool_),
        "payload_index": np.dtype(np.int64),
        "gyro_omega_rad_s": np.dtype(np.float64),
        "star_tracker_q_NB": np.dtype(np.float64),
        "star_tracker_R_rad2": np.dtype(np.float64),
    }
    for name, dtype in expected_event_dtypes.items():
        assert getattr(events, name).dtype == dtype
    assert truth.q_true_NB.shape[1:] == (4,)
    assert truth.gyro_bias_rad_s.shape[1:] == (3,)
    assert truth.omega_true_rad_s.shape[1:] == (3,)
    assert events.gyro_omega_rad_s.shape[1:] == (3,)
    assert events.star_tracker_q_NB.shape[1:] == (4,)
    assert events.star_tracker_R_rad2.shape[1:] == (3, 3)


def test_zero_latency_is_exact(generated) -> None:
    events = generated.dataset.events
    assert np.array_equal(events.arrival_time_s, events.measurement_time_s)
    assert generated.manifest["zero_latency"] is True


def test_same_time_order_is_gyro_then_star_tracker(generated) -> None:
    events = generated.dataset.events
    for trajectory_id in generated.dataset.truth.trajectory_id:
        rows = np.flatnonzero(events.trajectory_id == trajectory_id)
        for time_value in np.unique(events.measurement_time_s[rows]):
            same = rows[events.measurement_time_s[rows] == time_value]
            if same.size == 2:
                assert np.array_equal(
                    events.sensor_code[same],
                    np.asarray([SensorCode.GYRO, SensorCode.STAR_TRACKER], dtype=np.int16),
                )


def test_star_tracker_timestamps_are_gyro_subset(generated) -> None:
    events = generated.dataset.events
    for trajectory_id in generated.dataset.truth.trajectory_id:
        rows = events.trajectory_id == trajectory_id
        gyro_times = set(events.measurement_time_s[rows & (events.sensor_code == SensorCode.GYRO)])
        star_times = set(events.measurement_time_s[rows & (events.sensor_code == SensorCode.STAR_TRACKER)])
        assert star_times
        assert star_times.issubset(gyro_times)


def test_all_nominal_events_are_valid(generated) -> None:
    assert np.all(generated.dataset.events.valid)


def test_same_seed_config_regeneration_has_exact_hashes() -> None:
    config = BasiliskUnitSTConfig(num_trajectories=3, duration_s=0.2, master_seed=420)
    first = generate_basilisk_unit_st(config)
    second = generate_basilisk_unit_st(config)
    assert first.semantic_hashes == second.semantic_hashes


def test_manifest_records_all_locked_identities(generated) -> None:
    manifest = generated.manifest
    assert manifest["generator_id"] == GENERATOR_ID == "basilisk-unit-st-v1"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["convention_id"] == CONVENTION_ID
    assert manifest["seed_policy_version"] == SEED_POLICY_VERSION
    assert manifest["simulator_adapter_version"] == SIMULATOR_ADAPTER_VERSION
    assert manifest["sensor_model_version"] == SENSOR_MODEL_VERSION
    assert manifest["frame_proof_id"] == FRAME_PROOF_ID
    assert set(manifest["source_fingerprints"]) == {
        "bench/estimators/mekf.py",
        "bench/tasks/generator/mekf_events.py",
        "bench/tasks/generator/basilisk_unit_st.py",
        "docs/research/phase1a/P1A_BASILISK_FRAME_CONVENTION_PROOF.md",
    }


def test_gyro_noise_seed_changes_sensor_not_truth() -> None:
    base = BasiliskUnitSTConfig(num_trajectories=3, duration_s=0.2, master_seed=430)
    first = generate_basilisk_unit_st(base)
    second = generate_basilisk_unit_st(replace(base, gyro_noise_seed_namespace="gyro-white-noise-alt"))
    assert first.semantic_hashes.truth_hash == second.semantic_hashes.truth_hash
    assert first.semantic_hashes.sensor_payload_hash != second.semantic_hashes.sensor_payload_hash


def test_star_noise_seed_changes_sensor_not_truth() -> None:
    base = BasiliskUnitSTConfig(num_trajectories=3, duration_s=0.2, master_seed=440)
    first = generate_basilisk_unit_st(base)
    second = generate_basilisk_unit_st(replace(base, star_tracker_noise_seed_namespace="star-tracker-noise-alt"))
    assert first.semantic_hashes.truth_hash == second.semantic_hashes.truth_hash
    assert first.semantic_hashes.sensor_payload_hash != second.semantic_hashes.sensor_payload_hash


def test_truth_seed_changes_basilisk_truth() -> None:
    base = BasiliskUnitSTConfig(num_trajectories=3, duration_s=0.2, master_seed=450)
    first = generate_basilisk_unit_st(base)
    second = generate_basilisk_unit_st(replace(base, truth_attitude_seed_namespace="truth-attitude-alt"))
    assert first.semantic_hashes.truth_hash != second.semantic_hashes.truth_hash


def test_bias_seed_changes_bias_and_gyro_but_not_attitude_rate_truth() -> None:
    base = BasiliskUnitSTConfig(num_trajectories=3, duration_s=0.2, master_seed=460)
    first = generate_basilisk_unit_st(base)
    second = generate_basilisk_unit_st(replace(base, gyro_bias_seed_namespace="gyro-bias-alt"))
    assert np.array_equal(first.dataset.truth.q_true_NB, second.dataset.truth.q_true_NB)
    assert np.array_equal(first.dataset.truth.omega_true_rad_s, second.dataset.truth.omega_true_rad_s)
    assert not np.array_equal(first.dataset.truth.gyro_bias_rad_s, second.dataset.truth.gyro_bias_rad_s)
    assert not np.array_equal(first.dataset.events.gyro_omega_rad_s, second.dataset.events.gyro_omega_rad_s)


def test_star_sign_seed_preserves_physical_measurements() -> None:
    base = BasiliskUnitSTConfig(num_trajectories=3, duration_s=0.4, master_seed=470)
    first = generate_basilisk_unit_st(base)
    second = generate_basilisk_unit_st(replace(base, star_tracker_sign_seed_namespace="star-sign-alt"))
    dots = np.sum(first.dataset.events.star_tracker_q_NB * second.dataset.events.star_tracker_q_NB, axis=1)
    assert np.allclose(np.abs(dots), 1.0, rtol=0.0, atol=2.0e-15)
    assert first.semantic_hashes.sensor_payload_hash != second.semantic_hashes.sensor_payload_hash


def test_trajectory_ids_are_unique_and_splits_disjoint(generated) -> None:
    ids = generated.dataset.truth.trajectory_id
    split = generated.trajectory_split
    assert np.unique(ids).size == ids.size
    groups = [set(map(int, group)) for group in (split.train_ids, split.val_ids, split.test_ids)]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert set(map(int, ids)) == set.union(*groups)


def test_split_seed_changes_only_membership_identity() -> None:
    base = BasiliskUnitSTConfig(num_trajectories=8, duration_s=0.2, master_seed=480)
    first = generate_basilisk_unit_st(base)
    second = generate_basilisk_unit_st(replace(base, split_seed_namespace="trajectory-split-alt"))
    assert first.semantic_hashes.truth_hash == second.semantic_hashes.truth_hash
    assert first.semantic_hashes.sensor_payload_hash == second.semantic_hashes.sensor_payload_hash
    assert first.semantic_hashes.event_order_hash == second.semantic_hashes.event_order_hash
    assert first.manifest["trajectory_split"]["split_membership_hash"] != second.manifest["trajectory_split"]["split_membership_hash"]


def test_serialization_round_trip_and_strict_generator_id(generated, tmp_path) -> None:
    artifact = tmp_path / "artifact"
    saved = save_event_dataset(artifact, generated.dataset, generated.manifest)
    loaded, manifest, hashes = load_event_dataset(
        artifact, expected_generator_id="basilisk-unit-st-v1"
    )
    assert saved == hashes == generated.semantic_hashes
    assert manifest["generator_id"] == "basilisk-unit-st-v1"
    for field in fields(generated.dataset.events):
        assert np.array_equal(getattr(generated.dataset.events, field.name), getattr(loaded.events, field.name))
    for field in fields(generated.dataset.truth):
        assert np.array_equal(getattr(generated.dataset.truth, field.name), getattr(loaded.truth, field.name))


def test_strict_loader_rejects_expected_generator_mismatch(generated, tmp_path) -> None:
    artifact = tmp_path / "artifact"
    save_event_dataset(artifact, generated.dataset, generated.manifest)
    with pytest.raises(ValueError, match="generator_id mismatch"):
        load_event_dataset(artifact, expected_generator_id="synthetic-unit-st-v1")


def test_manifest_identity_tamper_is_hash_rejected(generated, tmp_path) -> None:
    artifact = tmp_path / "artifact"
    save_event_dataset(artifact, generated.dataset, generated.manifest)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["simulator_adapter_version"] = "tampered-v1"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        load_event_dataset(artifact, expected_generator_id="basilisk-unit-st-v1")


def test_zero_noise_direct_replay_matches_basilisk_truth() -> None:
    config = BasiliskUnitSTConfig(
        num_trajectories=3,
        duration_s=0.5,
        gyro_noise_std_rad_s=0.0,
        star_tracker_noise_scale=0.0,
        randomize_star_tracker_sign=True,
        master_seed=490,
    )
    generated = generate_basilisk_unit_st(config)
    truth = generated.dataset.truth
    start, stop = int(truth.truth_offsets[0]), int(truth.truth_offsets[1])
    result = replay_trajectory(
        generated.dataset.events,
        int(truth.trajectory_id[0]),
        _initial_state(generated, exact_bias=True),
        0.0,
        np.diag(np.asarray([1.0e-14] * 3 + [1.0e-16] * 3, dtype=np.float64)),
    )
    assert _attitude_error(truth.q_true_NB[stop - 1], result.final_state.q_NB) <= 2.0e-12
    assert np.linalg.norm(result.final_state.b_g - truth.gyro_bias_rad_s[start]) <= 2.0e-12


def test_representative_noisy_replay_is_finite_unit_and_spd(generated) -> None:
    trajectory_id = int(generated.dataset.truth.trajectory_id[0])
    prior = _initial_state(generated)
    perturbed = MEKFState(
        q_NB=quat_multiply(prior.q_NB, quat_exp([0.02, -0.01, 0.015])),
        b_g=np.asarray([0.002, -0.001, 0.0015]),
        P=np.eye(6, dtype=np.float64) * 1.0e-3,
    )
    result = replay_trajectory(
        generated.dataset.events, trajectory_id, perturbed, 0.0, _process_noise()
    )
    assert np.all(np.isfinite(result.q_NB_history))
    assert np.all(np.isfinite(result.b_g_history))
    assert np.all(np.isfinite(result.P_history))
    assert np.allclose(np.linalg.norm(result.q_NB_history, axis=1), 1.0, rtol=0.0, atol=2.0e-14)
    for covariance in result.P_history:
        assert np.array_equal(covariance, covariance.T)
        np.linalg.cholesky(covariance)
    for covariance in result.star_tracker_S:
        assert np.array_equal(covariance, covariance.T)
        np.linalg.cholesky(covariance)


def test_repeated_replay_is_exactly_deterministic(generated) -> None:
    trajectory_id = int(generated.dataset.truth.trajectory_id[0])
    prior = _initial_state(generated)
    first = replay_trajectory(generated.dataset.events, trajectory_id, prior, 0.0, _process_noise())
    second = replay_trajectory(generated.dataset.events, trajectory_id, prior, 0.0, _process_noise())
    _assert_replay_equal(first, second)


def test_serialization_round_trip_replay_is_exact(generated, tmp_path) -> None:
    artifact = tmp_path / "artifact"
    save_event_dataset(artifact, generated.dataset, generated.manifest)
    loaded, _manifest, _hashes = load_event_dataset(
        artifact, expected_generator_id="basilisk-unit-st-v1"
    )
    trajectory_id = int(generated.dataset.truth.trajectory_id[0])
    prior = _initial_state(generated)
    before = replay_trajectory(generated.dataset.events, trajectory_id, prior, 0.0, _process_noise())
    after = replay_trajectory(loaded.events, trajectory_id, prior, 0.0, _process_noise())
    _assert_replay_equal(before, after)


def test_all_star_signs_negated_have_identical_replay(generated) -> None:
    kwargs = _event_kwargs(generated.dataset.events)
    kwargs["star_tracker_q_NB"] *= -1.0
    negated = MEKFEventTable(**kwargs)
    trajectory_id = int(generated.dataset.truth.trajectory_id[0])
    prior = _initial_state(generated)
    positive = replay_trajectory(generated.dataset.events, trajectory_id, prior, 0.0, _process_noise())
    negative = replay_trajectory(negated, trajectory_id, prior, 0.0, _process_noise())
    _assert_replay_equal(positive, negative)


def test_truth_is_not_replay_input_and_is_not_mutated(generated) -> None:
    parameters = set(inspect.signature(replay_trajectory).parameters)
    assert parameters == {"event_table", "trajectory_id", "initial_state", "initial_time_s", "Q_c"}
    truth_before = {
        field.name: np.array(getattr(generated.dataset.truth, field.name), copy=True)
        for field in fields(generated.dataset.truth)
    }
    replay_trajectory(
        generated.dataset.events,
        int(generated.dataset.truth.trajectory_id[0]),
        _initial_state(generated),
        0.0,
        _process_noise(),
    )
    for name, before in truth_before.items():
        assert np.array_equal(getattr(generated.dataset.truth, name), before)


def test_gate_a_state_arrays_remain_readonly(generated) -> None:
    state = _initial_state(generated)
    for array in (state.q_NB, state.b_g, state.P):
        assert array.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0.0


@pytest.mark.parametrize(
    "updates",
    (
        {"duration_s": -1.0},
        {"gyro_rate_hz": 19, "star_tracker_rate_hz": 5},
        {"representative_mass_kg": 0.0},
        {"representative_spherical_inertia_kg_m2": 0.0},
        {"gyro_noise_std_rad_s": -1.0},
        {"star_tracker_R_rad2": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))},
    ),
)
def test_malformed_config_and_non_spd_covariance_fail_loudly(updates) -> None:
    with pytest.raises(ValueError):
        BasiliskUnitSTConfig(**updates)


@pytest.mark.parametrize("seed", (510, 511, 512, 513, 514))
def test_five_seed_property_sweep(seed: int, tmp_path) -> None:
    config = BasiliskUnitSTConfig(
        num_trajectories=3,
        duration_s=0.2,
        gyro_rate_hz=20,
        star_tracker_rate_hz=5,
        master_seed=seed,
    )
    first = generate_basilisk_unit_st(config)
    second = generate_basilisk_unit_st(config)
    assert first.semantic_hashes == second.semantic_hashes
    artifact = tmp_path / f"artifact-{seed}"
    save_event_dataset(artifact, first.dataset, first.manifest)
    loaded, _manifest, hashes = load_event_dataset(
        artifact, expected_generator_id="basilisk-unit-st-v1"
    )
    assert hashes == first.semantic_hashes
    assert np.max(np.abs(np.linalg.norm(loaded.truth.q_true_NB, axis=1) - 1.0)) <= 2.0e-14
    split = first.trajectory_split
    groups = [set(map(int, group)) for group in (split.train_ids, split.val_ids, split.test_ids)]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    trajectory_id = int(first.dataset.truth.trajectory_id[0])
    prior = _initial_state(first)
    replay = replay_trajectory(first.dataset.events, trajectory_id, prior, 0.0, _process_noise())
    kwargs = _event_kwargs(first.dataset.events)
    kwargs["star_tracker_q_NB"] *= -1.0
    sign_replay = replay_trajectory(MEKFEventTable(**kwargs), trajectory_id, prior, 0.0, _process_noise())
    _assert_replay_equal(replay, sign_replay)
    assert min(float(np.min(np.linalg.eigvalsh(item))) for item in replay.P_history) > 0.0
    assert min(float(np.min(np.linalg.eigvalsh(item))) for item in replay.star_tracker_S) > 0.0
