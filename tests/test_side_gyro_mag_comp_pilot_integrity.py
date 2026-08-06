"""The six direct integrity checks frozen in PILOT_SPEC.md.

Each test contains one local counterexample and asserts that the corresponding
invariant rejects it.  This file is intentionally not a general mutation
framework.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch

from bench.estimators.mekf import inject_error_state, propagate_state
from bench.side_gyro_mag_comp_pilot.data import (
    assert_same_realization,
    freeze_train_normalization,
    generate_dataset,
    strip_runtime_normalization,
    strip_runtime_trajectory,
    validate_deployable_namespace,
)
from bench.side_gyro_mag_comp_pilot.model import SideEstimator, SplitGainBackbone
from bench.side_gyro_mag_comp_pilot.study import (
    _initial_state,
    deployable_replay,
    n3s_replay_namespace,
    protected_replay_hashes,
    verify_n3s_bridge,
)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(population={"train": 2, "validation": 1, "test": 2})


def test_01_whole_trajectory_split_disjointness_with_negative_fixture(dataset) -> None:
    split = dataset.split
    assert not (set(split.train_ids) & set(split.validation_ids))
    assert not (set(split.train_ids) & set(split.test_ids))
    assert not (set(split.validation_ids) & set(split.test_ids))
    with pytest.raises(ValueError, match="disjoint"):
        dataclasses.replace(
            split,
            validation_ids=(split.train_ids[0],) + split.validation_ids[1:],
        )


def test_02_identical_raw_realization_across_variants_with_negative_fixture(dataset) -> None:
    trajectory = dataset.sensor[dataset.split.test_ids[0]]
    assert_same_realization([trajectory, trajectory])
    other = dataset.sensor[dataset.split.test_ids[1]]
    with pytest.raises(ValueError, match="same trajectory and raw realization"):
        assert_same_realization([trajectory, other])


def test_03_runtime_leakage_firewall_with_negative_fixture(dataset) -> None:
    trajectory = dataset.sensor[dataset.split.test_ids[0]]
    runtime = strip_runtime_trajectory(trajectory)
    validate_deployable_namespace(runtime)
    with pytest.raises(ValueError, match="forbidden deployable key"):
        validate_deployable_namespace({"future_sample": [1.0, 2.0, 3.0]})


def test_04_right_error_gain_and_multiplicative_injection_with_negative_fixture(dataset) -> None:
    trajectory_id = dataset.split.test_ids[0]
    gyro, mag = dataset.sensor[trajectory_id].events[:2]
    estimator = SideEstimator("raw", feature_enabled=False)
    initial = _initial_state()
    estimator.reset_trajectory(initial)
    step = estimator.step_pair(
        gyro.measurement_S, mag.measurement_S, gyro.timestamp_s, dataset.m_model_N_onboard,
    )
    q_c = np.diag(np.r_[np.full(3, 1e-8), np.full(3, 1e-12)]).astype(np.float64)
    prior = propagate_state(initial, gyro.measurement_S, gyro.timestamp_s, q_c).state
    expected_q, expected_b = inject_error_state(prior.q_NB, prior.b_g, step.delta_x)

    def assert_contract(candidate) -> None:
        assert candidate.K.shape == (6, 3)
        assert np.allclose(candidate.state.q_NB, expected_q, rtol=0.0, atol=1e-14)
        assert np.allclose(candidate.state.b_g, expected_b, rtol=0.0, atol=1e-14)

    assert_contract(step)
    violated = dataclasses.replace(step, K=np.zeros((3, 6)), state=initial)
    with pytest.raises(AssertionError):
        assert_contract(violated)


def test_05_n3s_exact_checkpoint_and_feature_association_only_with_negative_fixture(dataset) -> None:
    normalization = freeze_train_normalization(dataset)
    runtime_normalization = strip_runtime_normalization(normalization)
    runtime = {item: strip_runtime_trajectory(dataset.sensor[item]) for item in dataset.split.test_ids}
    r0_ids = tuple(item for item in dataset.split.test_ids if dataset.sensor[item].regime == "R0")
    model = SideEstimator("learned", feature_enabled=True)
    model.install_normalization(
        normalization.gyro_mean, normalization.gyro_std, normalization.mag_mean, normalization.mag_std,
    )
    state = copy.deepcopy(model.state_dict())
    target_id = r0_ids[0]
    n3_model = SideEstimator("learned", feature_enabled=True)
    n3_model.load_state_dict(state, strict=True)
    n3 = deployable_replay(
        runtime[target_id], n3_model, runtime_normalization,
        dataset.m_model_N_onboard, variant="N3",
    )
    n3s, evidence = n3s_replay_namespace(
        runtime, r0_ids, 0, dataset.m_model_N_onboard, target_id, 31001,
        state, "a" * 64, runtime_normalization,
    )
    n3_hashes = protected_replay_hashes(runtime[target_id], n3)
    n3s_hashes = protected_replay_hashes(runtime[target_id], n3s)
    verify_n3s_bridge(
        n3_hashes, n3s_hashes, evidence,
        n3s_recurrent_owner_token=n3s.recurrent_history_owner_token,
        n3s_recurrent_history_sha256=n3s.recurrent_history_provenance_sha256,
    )
    violated = dict(evidence)
    violated["n3s_state_dict_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="state_dict digest changed"):
        verify_n3s_bridge(
            n3_hashes, n3s_hashes, violated,
            n3s_recurrent_owner_token=n3s.recurrent_history_owner_token,
            n3s_recurrent_history_sha256=n3s.recurrent_history_provenance_sha256,
        )


def test_06_feature_off_is_corrected_only_with_negative_fixture() -> None:
    template = SplitGainBackbone()
    with torch.no_grad():
        template.gyro_film.bias[:36].fill_(0.2)
        template.mag_film.bias[:9].fill_(0.1)
    state = copy.deepcopy(template.state_dict())
    prior = np.arange(6, dtype=np.float64) * 0.01
    innovation = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    h = np.zeros((3, 6), dtype=np.float64)
    h[:, :3] = np.eye(3)
    zero = np.zeros(8, dtype=np.float64)
    context = np.ones(8, dtype=np.float64)

    def output(feature_enabled: bool, feature: np.ndarray) -> np.ndarray:
        model = SplitGainBackbone()
        model.load_state_dict(state, strict=True)
        model.reset_trajectory()
        return model.forward_step(
            prior, innovation, h, feature, feature, feature_enabled=feature_enabled,
        ).K.detach().numpy()

    corrected_only = output(False, zero)
    feature_off = output(False, context)
    np.testing.assert_array_equal(feature_off, corrected_only)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(output(True, context), corrected_only)
