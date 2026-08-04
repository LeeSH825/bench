from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from bench.estimators.mekf import quat_geodesic_angle
from bench.tasks.generator.mekf_events import SensorCode
from bench.tasks.generator.unit_st_regimes import (
    BASILISK_REGIME_GENERATOR_ID,
    SYNTHETIC_REGIME_GENERATOR_ID,
    OracleContextSidecar,
    RegimeCode,
    UnitSTRegimeConfig,
    WindowCode,
    covariance_increment_scale,
    generate_base_unit_st,
    generate_unit_st_regime,
    load_oracle_context,
    load_unit_st_regime,
    save_unit_st_regime,
)


@pytest.fixture(scope="module")
def base_config() -> UnitSTRegimeConfig:
    return UnitSTRegimeConfig(
        truth_source="synthetic",
        num_trajectories=6,
        duration_s=2.0,
        gyro_rate_hz=10,
        star_tracker_rate_hz=2,
        master_seed=101,
        train_fraction=0.5,
        val_fraction=1.0 / 6.0,
        test_fraction=1.0 / 3.0,
    )


@pytest.fixture(scope="module")
def paired(base_config: UnitSTRegimeConfig):
    base = generate_base_unit_st(base_config)
    stationary = generate_unit_st_regime(base_config, base_generated=base)
    c2 = generate_unit_st_regime(
        UnitSTRegimeConfig(
            **{
                **asdict(base_config),
                "regime_code": int(RegimeCode.C2_GYRO_PROCESS_STEP),
                "event_covariance_multiplier": 4.0,
            }
        ),
        base_generated=base,
    )
    c3 = generate_unit_st_regime(
        UnitSTRegimeConfig(
            **{
                **asdict(base_config),
                "regime_code": int(RegimeCode.C3_STAR_TRACKER_RELIABILITY_STEP),
                "event_covariance_multiplier": 4.0,
            }
        ),
        base_generated=base,
    )
    return stationary, c2, c3


def _payload_event_mask(generated, sensor: SensorCode) -> np.ndarray:
    events = generated.dataset.events
    rows = np.flatnonzero(events.sensor_code == np.int16(sensor))
    context_event = generated.oracle_context.event_window_id == np.int8(WindowCode.EVENT)
    payload = events.payload_index[rows]
    result = np.zeros(
        events.gyro_omega_rad_s.shape[0]
        if sensor == SensorCode.GYRO
        else events.star_tracker_q_NB.shape[0],
        dtype=np.bool_,
    )
    result[payload] = context_event[rows]
    return result


def test_stationary_context_is_all_one(paired) -> None:
    stationary, _c2, _c3 = paired
    assert np.array_equal(stationary.oracle_context.alpha_g, np.ones_like(stationary.oracle_context.alpha_g))
    assert np.array_equal(stationary.oracle_context.alpha_b, np.ones_like(stationary.oracle_context.alpha_b))
    assert np.array_equal(stationary.oracle_context.alpha_R_ST, np.ones_like(stationary.oracle_context.alpha_R_ST))


def test_c2_changes_only_event_gyro_payload(paired) -> None:
    stationary, c2, _c3 = paired
    event = _payload_event_mask(c2, SensorCode.GYRO)
    assert np.array_equal(
        stationary.dataset.events.gyro_omega_rad_s[~event],
        c2.dataset.events.gyro_omega_rad_s[~event],
    )
    assert np.any(stationary.dataset.events.gyro_omega_rad_s[event] != c2.dataset.events.gyro_omega_rad_s[event])
    assert np.array_equal(stationary.dataset.events.star_tracker_q_NB, c2.dataset.events.star_tracker_q_NB)
    assert np.array_equal(stationary.dataset.events.star_tracker_R_rad2, c2.dataset.events.star_tracker_R_rad2)


def test_c3_changes_only_event_star_tracker_payload(paired) -> None:
    stationary, _c2, c3 = paired
    event = _payload_event_mask(c3, SensorCode.STAR_TRACKER)
    before = stationary.dataset.events.star_tracker_q_NB
    after = c3.dataset.events.star_tracker_q_NB
    assert np.array_equal(before[~event], after[~event])
    assert np.any(before[event] != after[event])
    assert np.array_equal(stationary.dataset.events.gyro_omega_rad_s, c3.dataset.events.gyro_omega_rad_s)
    assert np.array_equal(stationary.dataset.events.star_tracker_R_rad2, c3.dataset.events.star_tracker_R_rad2)


@pytest.mark.parametrize("index", [1, 2])
def test_truth_and_order_are_exactly_paired(paired, index: int) -> None:
    stationary = paired[0]
    candidate = paired[index]
    for name in (
        "trajectory_id",
        "truth_offsets",
        "truth_time_s",
        "q_true_NB",
        "gyro_bias_rad_s",
        "omega_true_rad_s",
    ):
        assert np.array_equal(getattr(stationary.dataset.truth, name), getattr(candidate.dataset.truth, name))
    for name in (
        "trajectory_id",
        "sensor_code",
        "measurement_time_s",
        "arrival_time_s",
        "event_order",
        "valid",
        "payload_index",
    ):
        assert np.array_equal(getattr(stationary.dataset.events, name), getattr(candidate.dataset.events, name))


def test_covariance_multiplier_uses_sqrt_standard_deviation() -> None:
    assert covariance_increment_scale(1.0) == 0.0
    assert covariance_increment_scale(2.0) == 1.0
    assert covariance_increment_scale(5.0) == 2.0
    with pytest.raises(ValueError):
        covariance_increment_scale(0.5)


def test_c2_and_c3_sidecar_scales_are_event_only(paired) -> None:
    _stationary, c2, c3 = paired
    event = c2.oracle_context.event_window_id == np.int8(WindowCode.EVENT)
    assert np.array_equal(c2.oracle_context.alpha_g[event], np.full(np.count_nonzero(event), 4.0))
    assert np.array_equal(c2.oracle_context.alpha_g[~event], np.ones(np.count_nonzero(~event)))
    assert np.array_equal(c2.oracle_context.alpha_R_ST, np.ones_like(c2.oracle_context.alpha_R_ST))
    assert np.array_equal(c3.oracle_context.alpha_R_ST[event], np.full(np.count_nonzero(event), 4.0))
    assert np.array_equal(c3.oracle_context.alpha_R_ST[~event], np.ones(np.count_nonzero(~event)))
    assert np.array_equal(c3.oracle_context.alpha_g, np.ones_like(c3.oracle_context.alpha_g))


def test_alpha_b_is_locked_to_one(paired) -> None:
    for generated in paired:
        assert np.array_equal(generated.oracle_context.alpha_b, np.ones_like(generated.oracle_context.alpha_b))


def test_timing_labels_are_absent_from_sensor_manifest(paired) -> None:
    manifest_text = json.dumps(paired[1].sensor_manifest, sort_keys=True)
    for forbidden in (
        "regime_code",
        "event_covariance_multiplier",
        "event_start_fraction",
        "event_end_fraction",
        "alpha_g",
        "alpha_R_ST",
        "event_window_id",
    ):
        assert forbidden not in manifest_text


def test_oracle_arrays_are_read_only(paired) -> None:
    context = paired[1].oracle_context
    for field in (
        context.trajectory_id,
        context.event_order,
        context.alpha_g,
        context.alpha_b,
        context.alpha_R_ST,
        context.event_window_id,
        context.regime_code,
    ):
        assert not field.flags.writeable
        with pytest.raises(ValueError):
            field[0] = field[0]


def test_cursor_is_forward_only_and_rejects_future_access(paired) -> None:
    context = paired[1].oracle_context
    trajectory_id = int(context.trajectory_id[0])
    cursor = context.cursor(trajectory_id)
    assert not hasattr(cursor, "peek")
    with pytest.raises(ValueError):
        cursor.consume(1)
    assert cursor.consume(0) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        cursor.consume(0)


def test_generator_is_deterministic(base_config: UnitSTRegimeConfig) -> None:
    first = generate_unit_st_regime(base_config)
    second = generate_unit_st_regime(base_config)
    assert first.semantic_hashes == second.semantic_hashes
    assert first.oracle_context.semantic_hash == second.oracle_context.semantic_hash


def test_whole_trajectory_split_is_deterministic_and_disjoint(paired) -> None:
    first = paired[0].trajectory_split
    second = paired[2].trajectory_split
    assert np.array_equal(first.train_ids, second.train_ids)
    assert np.array_equal(first.val_ids, second.val_ids)
    assert np.array_equal(first.test_ids, second.test_ids)
    assert not set(first.train_ids) & set(first.val_ids)
    assert not set(first.train_ids) & set(first.test_ids)
    assert not set(first.val_ids) & set(first.test_ids)


def test_zero_latency_and_gyro_then_st_order_are_preserved(paired) -> None:
    events = paired[1].dataset.events
    assert np.array_equal(events.measurement_time_s, events.arrival_time_s)
    for index in range(events.event_count - 1):
        same_trajectory = events.trajectory_id[index] == events.trajectory_id[index + 1]
        same_time = events.measurement_time_s[index] == events.measurement_time_s[index + 1]
        if same_trajectory and same_time:
            assert events.sensor_code[index] == np.int16(SensorCode.GYRO)
            assert events.sensor_code[index + 1] == np.int16(SensorCode.STAR_TRACKER)


def test_randomized_quaternion_sign_changes_only_representation(base_config: UnitSTRegimeConfig) -> None:
    unsigned_cfg = UnitSTRegimeConfig(**{**asdict(base_config), "randomize_star_tracker_sign": False})
    base = generate_base_unit_st(unsigned_cfg)
    unsigned = generate_unit_st_regime(unsigned_cfg, base_generated=base)
    signed_cfg = UnitSTRegimeConfig(**{**asdict(base_config), "randomize_star_tracker_sign": True})
    signed = generate_unit_st_regime(signed_cfg, base_generated=base)
    dots = np.sum(unsigned.dataset.events.star_tracker_q_NB * signed.dataset.events.star_tracker_q_NB, axis=1)
    assert np.allclose(np.abs(dots), 1.0, rtol=0.0, atol=4.0e-16)
    assert np.any(dots < 0.0)


def test_all_star_tracker_quaternions_are_normalized(paired) -> None:
    for generated in paired:
        norms = np.linalg.norm(generated.dataset.events.star_tracker_q_NB, axis=1)
        assert np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-13)


def test_sensor_and_oracle_serialization_are_separate(tmp_path: Path, paired) -> None:
    sensor = tmp_path / "sensor"
    oracle = tmp_path / "oracle"
    save_unit_st_regime(sensor, oracle, paired[1])
    assert {item.name for item in sensor.iterdir()} == {"manifest.json", "truth.npz", "events.npz"}
    assert {item.name for item in oracle.iterdir()} == {"experiment_manifest.json", "oracle_context.npz"}
    sensor_text = (sensor / "manifest.json").read_text(encoding="ascii")
    assert "oracle_context" not in sensor_text
    assert "event_window_id" not in sensor_text


def test_strict_round_trip_preserves_both_hashes(tmp_path: Path, paired) -> None:
    sensor = tmp_path / "sensor"
    oracle = tmp_path / "oracle"
    generated = paired[2]
    save_unit_st_regime(sensor, oracle, generated)
    loaded = load_unit_st_regime(
        sensor,
        oracle,
        expected_generator_id=SYNTHETIC_REGIME_GENERATOR_ID,
    )
    _dataset, _manifest, hashes, context, experiment = loaded
    assert hashes == generated.semantic_hashes
    assert context.semantic_hash == generated.oracle_context.semantic_hash
    assert experiment["raw_sensor_stream_hash"] == hashes.dataset_hash


def test_strict_expected_generator_id_rejects_mismatch(tmp_path: Path, paired) -> None:
    sensor = tmp_path / "sensor"
    oracle = tmp_path / "oracle"
    save_unit_st_regime(sensor, oracle, paired[0])
    with pytest.raises(ValueError, match="generator_id mismatch"):
        load_unit_st_regime(sensor, oracle, expected_generator_id=BASILISK_REGIME_GENERATOR_ID)


def test_oracle_corruption_is_rejected(tmp_path: Path, paired) -> None:
    sensor = tmp_path / "sensor"
    oracle = tmp_path / "oracle"
    save_unit_st_regime(sensor, oracle, paired[1])
    archive_path = oracle / "oracle_context.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["alpha_g"][0] = 7.0
    np.savez(archive_path, **arrays)
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        load_oracle_context(
            oracle,
            expected_raw_sensor_stream_hash=paired[1].semantic_hashes.dataset_hash,
        )


def test_wrong_sensor_pairing_hash_is_rejected(tmp_path: Path, paired) -> None:
    oracle = tmp_path / "oracle"
    from bench.tasks.generator.unit_st_regimes import save_oracle_context

    save_oracle_context(oracle, paired[1].oracle_context, paired[1].experiment_manifest)
    with pytest.raises(ValueError, match="different raw sensor stream"):
        load_oracle_context(oracle, expected_raw_sensor_stream_hash="0" * 64)


def test_regime_raw_hash_changes_but_truth_hash_does_not(paired) -> None:
    stationary, c2, c3 = paired
    assert stationary.semantic_hashes.truth_hash == c2.semantic_hashes.truth_hash == c3.semantic_hashes.truth_hash
    assert len({stationary.semantic_hashes.dataset_hash, c2.semantic_hashes.dataset_hash, c3.semantic_hashes.dataset_hash}) == 3
    assert len({stationary.oracle_context.semantic_hash, c2.oracle_context.semantic_hash, c3.oracle_context.semantic_hash}) == 3


def test_context_constructor_rejects_shape_mismatch(paired) -> None:
    context = paired[0].oracle_context
    kwargs = {field: np.array(getattr(context, field), copy=True) for field in (
        "trajectory_id", "event_order", "alpha_g", "alpha_b", "alpha_R_ST", "event_window_id", "regime_code"
    )}
    kwargs["alpha_g"] = kwargs["alpha_g"][:-1]
    with pytest.raises(ValueError, match="identical length"):
        OracleContextSidecar(**kwargs)


def test_base_qg_psd_matches_sampled_variance(base_config: UnitSTRegimeConfig) -> None:
    expected = base_config.gyro_noise_std_rad_s**2 / base_config.gyro_rate_hz
    assert base_config.base_Q_g_rad2_s == expected


def test_generator_source_has_no_neural_or_visualization_dependency() -> None:
    source = Path("bench/tasks/generator/unit_st_regimes.py").read_text(encoding="utf-8")
    for forbidden in ("torch", "kalman_net", "visualization", "magnetometer", "sun_sensor"):
        assert forbidden not in source.lower()
