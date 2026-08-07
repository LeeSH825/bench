"""The six direct integrity checks frozen in PILOT_SPEC.md.

Each check exercises its production invariant and contains a focused negative
fixture that is rejected.  This remains six tests, not a mutation framework.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch

import bench.side_gyro_mag_comp_pilot.study as study_module
from bench.estimators.mekf import propagate_state, quat_to_dcm, skew
from bench.side_gyro_mag_comp_pilot.data import (
    RuntimeTrajectoryBatch,
    SensorTrajectory,
    assert_same_realization,
    freeze_train_normalization,
    generate_dataset,
    raw_realization_digest,
    runtime_realization_digest,
    strip_runtime_normalization,
    strip_runtime_trajectory,
    validate_deployable_namespace,
)
from bench.side_gyro_mag_comp_pilot.model import (
    GainOutput,
    GyroEncoder,
    SideEstimator,
    SplitGainBackbone,
)
from bench.side_gyro_mag_comp_pilot.study import (
    _initial_state,
    _torch_dcm,
    _torch_quat_exp,
    _torch_quat_multiply,
    _torch_skew,
    deployable_replay,
    n3s_replay_namespace,
    protected_replay_hashes,
    state_dict_digest,
    verify_n3s_bridge,
)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(population={"train": 2, "validation": 1, "test": 2})


def _runtime_with_packets(template: RuntimeTrajectoryBatch, packets: tuple) -> RuntimeTrajectoryBatch:
    return RuntimeTrajectoryBatch(
        trajectory_id=template.trajectory_id,
        realization_sha256=runtime_realization_digest(packets),
        packets=packets,
    )


def _perturb_runtime_tail(runtime: RuntimeTrajectoryBatch, first_changed_sample: int) -> RuntimeTrajectoryBatch:
    packets = tuple(
        dataclasses.replace(
            packet,
            measurement_S=packet.measurement_S + np.array([0.7, -0.4, 0.2]),
        ) if packet.event_order // 2 >= first_changed_sample else packet
        for packet in runtime.packets
    )
    return _runtime_with_packets(runtime, packets)


def _one_step_lookahead_runtime(runtime: RuntimeTrajectoryBatch) -> RuntimeTrajectoryBatch:
    pairs = [runtime.packets[index:index + 2] for index in range(0, len(runtime.packets), 2)]
    packets = []
    for index, current in enumerate(pairs):
        future = pairs[min(index + 1, len(pairs) - 1)]
        for current_packet, future_packet in zip(current, future):
            packets.append(dataclasses.replace(
                current_packet, measurement_S=future_packet.measurement_S,
            ))
    return _runtime_with_packets(runtime, tuple(packets))


def _hamilton_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scalar = left[0] * right[0] - np.dot(left[1:], right[1:])
    vector = left[0] * right[1:] + right[0] * left[1:] + np.cross(left[1:], right[1:])
    value = np.r_[scalar, vector]
    return value / np.linalg.norm(value)


def _quaternion_exp(rotation: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotation))
    if theta < 1e-12:
        value = np.r_[1.0, 0.5 * rotation]
    else:
        value = np.r_[np.cos(0.5 * theta), np.sin(0.5 * theta) * rotation / theta]
    return value / np.linalg.norm(value)


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
    perturbed_events = list(trajectory.events)
    perturbed_events[0] = dataclasses.replace(
        perturbed_events[0], measurement_S=perturbed_events[0].measurement_S + np.array([1e-6, 0.0, 0.0]),
    )
    perturbed_digest = raw_realization_digest(perturbed_events)
    perturbed_events = [
        dataclasses.replace(event, realization_id=perturbed_digest) for event in perturbed_events
    ]
    perturbed = SensorTrajectory(
        trajectory_id=trajectory.trajectory_id,
        regime=trajectory.regime,
        realization_id=perturbed_digest,
        stream_namespace=trajectory.stream_namespace,
        events=tuple(perturbed_events),
    )
    assert perturbed.trajectory_id == trajectory.trajectory_id
    assert perturbed.realization_id != trajectory.realization_id
    with pytest.raises(ValueError, match="same trajectory and raw realization"):
        assert_same_realization([trajectory, perturbed])


def test_03_runtime_leakage_firewall_with_negative_fixture(dataset) -> None:
    trajectory = dataset.sensor[dataset.split.test_ids[0]]
    runtime = strip_runtime_trajectory(trajectory)
    normalization = strip_runtime_normalization(freeze_train_normalization(dataset))
    validate_deployable_namespace(runtime)
    with pytest.raises(ValueError, match="forbidden deployable key"):
        validate_deployable_namespace({"future_sample": [1.0, 2.0, 3.0]})

    template = SideEstimator("learned", feature_enabled=False)
    state = copy.deepcopy(template.state_dict())

    def replay(batch: RuntimeTrajectoryBatch):
        estimator = SideEstimator("learned", feature_enabled=False)
        estimator.load_state_dict(state, strict=True)
        return deployable_replay(
            batch, estimator, normalization, dataset.m_model_N_onboard, variant="N2",
        )

    cutoff = len(runtime.packets) // 4
    tail_perturbed = _perturb_runtime_tail(runtime, cutoff)
    baseline, changed_tail = replay(runtime), replay(tail_perturbed)

    def assert_prefix_identical(left, right) -> None:
        for field in (
            "q_hat_NB", "b_hat_B_rad_s", "corrected_gyro_B", "corrected_mag_B",
            "gyro_feature", "mag_feature",
        ):
            np.testing.assert_array_equal(getattr(left, field)[:cutoff], getattr(right, field)[:cutoff])

    assert_prefix_identical(baseline, changed_tail)
    with pytest.raises(AssertionError):
        assert_prefix_identical(
            replay(_one_step_lookahead_runtime(runtime)),
            replay(_one_step_lookahead_runtime(tail_perturbed)),
        )
    encoder = GyroEncoder()
    encoder.forward_step(np.zeros(3), 1.0, True)
    with pytest.raises(ValueError, match="future/reordered"):
        encoder.forward_step(np.zeros(3), 0.5, True)


def test_04_right_error_gain_and_multiplicative_injection_with_negative_fixture(dataset) -> None:
    torch.manual_seed(4051)
    template = SplitGainBackbone()
    generator = torch.Generator().manual_seed(7713)
    with torch.no_grad():
        for head in (template.g1_head, template.g2_head):
            head.weight.copy_(torch.randn(head.weight.shape, dtype=torch.float64, generator=generator))
            head.bias.copy_(torch.randn(head.bias.shape, dtype=torch.float64, generator=generator))
    state = copy.deepcopy(template.state_dict())
    prior_input = np.array([0.04, -0.02, 0.03, 2e-5, -1e-5, 3e-5])
    innovation = np.array([0.08, -0.05, 0.02])
    h = np.array([
        [0.0, -0.7, 0.3, 0.01, 0.02, -0.03],
        [0.7, 0.0, -0.2, -0.04, 0.01, 0.02],
        [-0.3, 0.2, 0.0, 0.03, -0.02, 0.01],
    ], dtype=np.float64)
    feature = np.linspace(-0.2, 0.3, 8)

    def run_backbone(model: SplitGainBackbone) -> GainOutput:
        model.load_state_dict(state, strict=True)
        model.reset_trajectory()
        return model.forward_step(
            prior_input, innovation, h, feature, -feature, feature_enabled=False,
        )

    output = run_backbone(SplitGainBackbone())
    assert output.K.shape == (6, 3)
    assert not torch.allclose(output.G2, output.G2.T)
    expected_gain = output.G1 @ torch.as_tensor(h).T @ output.G2
    torch.testing.assert_close(output.K, expected_gain, rtol=0.0, atol=1e-14)

    class TransposedG2Backbone(SplitGainBackbone):
        def forward_step(self, *args, **kwargs) -> GainOutput:
            candidate = super().forward_step(*args, **kwargs)
            jacobian = torch.as_tensor(args[2], dtype=torch.float64)
            return GainOutput(
                candidate.G1, candidate.G2,
                candidate.G1 @ jacobian.T @ candidate.G2.T,
            )

    faulty = run_backbone(TransposedG2Backbone())
    with pytest.raises(AssertionError):
        torch.testing.assert_close(faulty.K, expected_gain, rtol=0.0, atol=1e-14)

    trajectory_id = dataset.split.test_ids[0]
    gyro, mag = dataset.sensor[trajectory_id].events[:2]
    runtime_template = SideEstimator("raw", feature_enabled=False)
    runtime_state = copy.deepcopy(runtime_template.state_dict())
    estimator = SideEstimator("raw", feature_enabled=False)
    estimator.load_state_dict(runtime_state, strict=True)
    initial = _initial_state()
    estimator.reset_trajectory(initial)
    step = estimator.step_pair(
        gyro.measurement_S, mag.measurement_S, gyro.timestamp_s, dataset.m_model_N_onboard,
    )
    q_c = np.diag(np.r_[np.full(3, 1e-8), np.full(3, 1e-12)]).astype(np.float64)
    prior = propagate_state(initial, gyro.measurement_S, gyro.timestamp_s, q_c).state
    expected_q = _hamilton_product(prior.q_NB, _quaternion_exp(step.delta_x[:3]))
    np.testing.assert_allclose(step.state.q_NB, expected_q, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(step.state.b_g, prior.b_g + step.delta_x[3:], rtol=0.0, atol=1e-14)
    left_injected = _hamilton_product(_quaternion_exp(step.delta_x[:3]), prior.q_NB)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(left_injected, expected_q, rtol=0.0, atol=1e-14)

    vector = np.array([0.31, -0.22, 0.47], dtype=np.float64)
    quaternion = np.array([0.91, 0.13, -0.25, 0.29], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    np.testing.assert_allclose(_torch_skew(torch.tensor(vector)).numpy(), skew(vector), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(_torch_dcm(torch.tensor(quaternion)).numpy(), quat_to_dcm(quaternion), rtol=0.0, atol=1e-14)

    torch_estimator = SideEstimator("raw", feature_enabled=False)
    torch_estimator.load_state_dict(runtime_state, strict=True)
    q = torch.tensor(initial.q_NB, dtype=torch.float64)
    bias = torch.tensor(initial.b_g, dtype=torch.float64)
    gyro_value = torch.tensor(gyro.measurement_S, dtype=torch.float64)
    q = _torch_quat_multiply(q, _torch_quat_exp((gyro_value - bias) * gyro.timestamp_s))
    m_onboard = torch.tensor(dataset.m_model_N_onboard, dtype=torch.float64)
    prediction = _torch_dcm(q).T @ m_onboard
    torch_h = torch.zeros((3, 6), dtype=torch.float64)
    torch_h[:, :3] = _torch_skew(prediction)
    torch_innovation = torch.tensor(mag.measurement_S, dtype=torch.float64) - prediction
    torch_gain = torch_estimator.backbone.forward_step(
        torch.cat((gyro_value, bias)), torch_innovation, torch_h,
        torch.zeros(8, dtype=torch.float64), torch.zeros(8, dtype=torch.float64),
        feature_enabled=False,
    )
    torch_delta = torch_gain.K @ torch_innovation
    torch_q = _torch_quat_multiply(q, _torch_quat_exp(torch_delta[:3]))
    torch_bias = bias + torch_delta[3:]
    np.testing.assert_allclose(torch_gain.K.detach().numpy(), step.K, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(torch_delta.detach().numpy(), step.delta_x, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(torch_q.detach().numpy(), step.state.q_NB, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(torch_bias.detach().numpy(), step.state.b_g, rtol=0.0, atol=1e-14)


def test_05_n3s_exact_checkpoint_and_feature_association_only_with_negative_fixture(dataset, monkeypatch) -> None:
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
    n3_digest = state_dict_digest(n3_model.state_dict())
    n3s, evidence = n3s_replay_namespace(
        runtime, r0_ids, 0, dataset.m_model_N_onboard, target_id, 31001,
        state, "a" * 64, n3_digest, runtime_normalization,
    )
    n3_hashes = protected_replay_hashes(runtime[target_id], n3)
    n3s_hashes = protected_replay_hashes(runtime[target_id], n3s)
    verify_n3s_bridge(
        n3_hashes, n3s_hashes, evidence,
        n3s_recurrent_owner_token=n3s.recurrent_history_owner_token,
        n3s_recurrent_history_sha256=n3s.recurrent_history_provenance_sha256,
    )

    perturbed_state = copy.deepcopy(state)
    tensor_name = "backbone.g1_head.bias"
    perturbed_state[tensor_name] = perturbed_state[tensor_name].clone()
    perturbed_state[tensor_name].reshape(-1)[0] += 1e-6
    perturbed_model = SideEstimator("learned", feature_enabled=True)
    perturbed_model.load_state_dict(perturbed_state, strict=True)
    deployable_replay(
        runtime[target_id], perturbed_model, runtime_normalization,
        dataset.m_model_N_onboard, variant="N3",
    )
    checkpoint_violation = dict(evidence)
    checkpoint_violation["n3s_state_dict_sha256"] = state_dict_digest(perturbed_model.state_dict())
    with pytest.raises(ValueError, match="state_dict digest changed"):
        verify_n3s_bridge(
            n3_hashes, n3s_hashes, checkpoint_violation,
            n3s_recurrent_owner_token=n3s.recurrent_history_owner_token,
            n3s_recurrent_history_sha256=n3s.recurrent_history_provenance_sha256,
        )

    monkeypatch.setattr(
        study_module, "fixed_derangement",
        lambda trajectory_ids, **_: {item: item for item in trajectory_ids},
    )
    identity_n3s, identity_evidence = n3s_replay_namespace(
        runtime, r0_ids, 0, dataset.m_model_N_onboard, target_id, 31001,
        state, "a" * 64, n3_digest, runtime_normalization,
    )
    with pytest.raises(ValueError, match="association did not change|source and target|fixed point"):
        verify_n3s_bridge(
            n3_hashes, protected_replay_hashes(runtime[target_id], identity_n3s), identity_evidence,
            n3s_recurrent_owner_token=identity_n3s.recurrent_history_owner_token,
            n3s_recurrent_history_sha256=identity_n3s.recurrent_history_provenance_sha256,
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
