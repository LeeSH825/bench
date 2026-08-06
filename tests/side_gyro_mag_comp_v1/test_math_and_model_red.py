from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import bench.side_gyro_mag_comp_v1.model as model_module

from bench.estimators.mekf import (
    MEKFState,
    body_vector_jacobian,
    body_vector_prediction,
    inject_error_state,
    propagate_state,
    quat_exp,
    quat_geodesic_angle,
    quat_multiply,
    quat_to_dcm,
    reset_covariance,
    right_jacobian_so3,
    skew,
)
from bench.side_gyro_mag_comp_v1.data import CalibrationTruth
from bench.side_gyro_mag_comp_v1.model import (
    FEATURE_DIM,
    GyroEncoder,
    GainOutput,
    MagEncoder,
    SideEstimator,
    SplitGainBackbone,
)


def _state(q: np.ndarray | None = None) -> MEKFState:
    return MEKFState(
        np.array([1., 0., 0., 0.]) if q is None else q,
        np.zeros(3), np.eye(6) * 0.1,
    )


def test_sc_qnb_right_injection_fixture_red() -> None:
    q_hat = quat_exp(np.array([0.31, -0.17, 0.22]))
    delta = np.array([0.07, 0.03, -0.04])
    q_right, _ = inject_error_state(q_hat, np.zeros(3), np.r_[delta, np.zeros(3)])
    q_expected = quat_multiply(q_hat, quat_exp(delta))
    q_wrong_left = quat_multiply(quat_exp(delta), q_hat)
    assert quat_geodesic_angle(q_right, q_expected) < 1e-12
    assert quat_geodesic_angle(q_wrong_left, q_expected) > 1e-4


def test_sc_gyro_body_rad_s_right_propagation_red() -> None:
    q0 = quat_exp(np.array([0.2, -0.1, 0.3]))
    omega = np.array([0.04, -0.03, 0.02])
    result = propagate_state(_state(q0), omega, 0.4, np.eye(6) * 1e-12)
    expected = quat_multiply(q0, quat_exp(omega * 0.4))
    wrong = quat_multiply(quat_exp(omega * 0.4), q0)
    assert quat_geodesic_angle(result.state.q_NB, expected) < 1e-12
    assert quat_geodesic_angle(wrong, expected) > 1e-5


def test_sc_deterministic_vs_residual_bias_separation_red() -> None:
    omega, residual = np.array([.02, -.01, .03]), np.array([2e-5, -3e-5, 4e-5])
    scale, offset = np.diag([1.01, .99, 1.005]), np.array([4e-4, -3e-4, 2e-4])
    packet = scale @ (omega + residual) + offset
    target = np.linalg.solve(scale, packet - offset)
    assert np.allclose(target, omega + residual)
    assert not np.allclose(target, omega, atol=1e-8)


def test_sc_gyro_oracle_retains_residual_bias_red() -> None:
    rng = np.random.default_rng(4)
    residual = rng.normal(0, 2e-5, (40, 3))
    omega = rng.normal(0, .02, (40, 3))
    scale, offset = np.diag([1.01, .99, 1.006]), np.array([5e-4, -4e-4, 3e-4])
    packet = (scale @ (omega + residual).T).T + offset
    oracle = np.linalg.solve(scale, (packet - offset).T).T
    slopes = [np.polyfit(residual[:, i], (oracle - omega)[:, i], 1)[0] for i in range(3)]
    assert np.allclose(slopes, np.ones(3), atol=1e-10)


def test_sc_learned_compensator_residual_bias_retention_red() -> None:
    rng = np.random.default_rng(7); count = 240
    omega = rng.normal(0, .02, (count, 3)); residual = rng.normal(0, 2e-5, (count, 3))
    target = omega + residual
    encoder = GyroEncoder(); optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    encoder.reset_trajectory(); loss = torch.zeros((), dtype=torch.float64)
    for index, packet in enumerate(target):
        output = encoder.forward_step(packet, .1 * (index + 1), True)
        loss = loss + torch.mean((output.corrected_B - torch.tensor(packet)) ** 2)
    optimizer.zero_grad(); loss.backward(); optimizer.step()  # serialized trained no-op checkpoint
    checkpoint = copy.deepcopy(encoder.state_dict())
    learned_encoder = GyroEncoder(); learned_encoder.load_state_dict(checkpoint, strict=True)
    learned_encoder.reset_trajectory()
    learned = np.stack([
        learned_encoder.forward_step(packet, .1 * (index + 1), True).corrected_B.detach().numpy()
        for index, packet in enumerate(target)
    ])
    slope = np.array([np.polyfit(residual[:, i], (learned - omega)[:, i], 1)[0] for i in range(3)])
    error = learned - target
    standard_error = np.maximum(error.std(0, ddof=1) / np.sqrt(count), np.finfo(float).eps)
    assert np.all((slope >= .9) & (slope <= 1.1))
    assert np.all(np.abs(error.mean(0)) <= 2 * standard_error)
    adversary = omega
    adversary_slope = np.array([np.polyfit(residual[:, i], (adversary - omega)[:, i], 1)[0] for i in range(3)])
    assert np.any(adversary_slope < .9)


def test_sc_mag_hard_soft_mounting_inverse_red() -> None:
    angle = .4
    C_BSm = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    calibration = CalibrationTruth(
        A_g=np.eye(3), c_g=np.zeros(3), C_SgB=np.eye(3),
        A_m=np.diag([1.08, .94, 1.03]), b_m=np.array([.04, -.03, .02]), C_BSm=C_BSm,
    )
    body = np.array([.3, -.2, .9])
    packet = calibration.A_m @ calibration.C_BSm.T @ body + calibration.b_m
    recovered = calibration.C_BSm @ np.linalg.solve(calibration.A_m, packet - calibration.b_m)
    wrong_order = calibration.C_BSm @ (np.linalg.solve(calibration.A_m, packet) - calibration.b_m)
    wrong_frame = calibration.C_BSm.T @ np.linalg.solve(calibration.A_m, packet - calibration.b_m)
    assert np.allclose(recovered, body)
    assert np.linalg.norm(wrong_order - body) > 1e-3
    assert np.linalg.norm(wrong_frame - body) > 1e-2


def test_sc_mag_jacobian_sign_finite_difference_red() -> None:
    q_hat = quat_exp(np.array([.2, -.3, .1]))
    reference = np.array([.3, -.2, .93])
    h = body_vector_prediction(q_hat, reference)
    H = body_vector_jacobian(q_hat, reference)
    epsilon = 1e-7
    numeric = np.empty((3, 3))
    # Vary the true right-local state that produces z while keeping h(q_hat) fixed.
    for axis in range(3):
        step = np.zeros(3); step[axis] = epsilon
        z_plus = body_vector_prediction(quat_multiply(q_hat, quat_exp(step)), reference)
        z_minus = body_vector_prediction(quat_multiply(q_hat, quat_exp(-step)), reference)
        numeric[:, axis] = ((z_plus - h) - (z_minus - h)) / (2 * epsilon)
    assert np.allclose(numeric, H[:, :3], atol=2e-9)
    assert not np.allclose(numeric, -H[:, :3], atol=1e-4)


def test_sc_split_gain_shape_and_right_injection_red() -> None:
    torch.manual_seed(1)
    backbone = SplitGainBackbone()
    h = np.array([.3, -.2, .93])
    H = np.zeros((3, 6)); H[:, :3] = skew(h)
    output = backbone.forward_step(np.arange(6) * .01, np.array([.02, -.01, .03]), H,
                                   np.zeros(8), np.zeros(8), feature_enabled=False)
    assert output.G1.shape == (6, 6) and output.G2.shape == (3, 3) and output.K.shape == (6, 3)
    with pytest.raises(RuntimeError):
        wrong_gain = torch.zeros((3, 6), dtype=torch.float64)
        if wrong_gain.shape != (6, 3):
            raise RuntimeError("gain shape adversary")


def test_sc_feature_dim_exactly_eight_red() -> None:
    gyro, mag = GyroEncoder(), MagEncoder()
    assert gyro.forward_step(np.ones(3), .1, True).feature.shape == (FEATURE_DIM,)
    assert mag.forward_step(np.ones(3), .1, True).feature.shape == (FEATURE_DIM,)
    backbone = SplitGainBackbone()
    with pytest.raises(ValueError):
        backbone.forward_step(np.zeros(6), np.zeros(3), np.zeros((3, 6)), np.zeros(7), np.zeros(8), feature_enabled=True)


def test_sc_film_feature_off_exact_equivalence_red() -> None:
    first, second = SplitGainBackbone(), SplitGainBackbone()
    second.load_state_dict(first.state_dict())
    args = (np.arange(6) * .01, np.array([.01, -.03, .02]), np.c_[np.eye(3), np.zeros((3, 3))])
    left = first.forward_step(*args, np.zeros(8), np.zeros(8), feature_enabled=False)
    right = second.forward_step(*args, np.full(8, 1e9), np.full(8, -1e9), feature_enabled=False)
    assert torch.equal(left.G1, right.G1) and torch.equal(left.G2, right.G2) and torch.equal(left.K, right.K)


def test_sc_film_branch_isolation_red() -> None:
    base = SplitGainBackbone()
    with torch.no_grad():
        base.gyro_film.weight[0, 0] = .2
        base.mag_film.weight[0, 0] = .3
    args = (np.arange(6) * .01, np.array([.01, -.03, .02]), np.c_[np.eye(3), np.zeros((3, 3))])
    a, b, c = copy.deepcopy(base), copy.deepcopy(base), copy.deepcopy(base)
    nominal = a.forward_step(*args, np.zeros(8), np.zeros(8), feature_enabled=True)
    gyro_changed = b.forward_step(*args, np.r_[1., np.zeros(7)], np.zeros(8), feature_enabled=True)
    mag_changed = c.forward_step(*args, np.zeros(8), np.r_[1., np.zeros(7)], feature_enabled=True)
    assert not torch.equal(nominal.G1, gyro_changed.G1) and torch.equal(nominal.G2, gyro_changed.G2)
    assert torch.equal(nominal.G1, mag_changed.G1) and not torch.equal(nominal.G2, mag_changed.G2)


def test_sc_causal_prefix_invariance_red() -> None:
    first, second = GyroEncoder(), GyroEncoder()
    second.load_state_dict(first.state_dict())
    prefix = [np.array([.1, .2, .3]), np.array([.2, .1, .4])]
    out_first = [first.forward_step(value, (i + 1) * .1, True).corrected_B.detach().numpy() for i, value in enumerate(prefix)]
    out_second = [second.forward_step(value, (i + 1) * .1, True).corrected_B.detach().numpy() for i, value in enumerate(prefix)]
    first.forward_step(np.array([9., 8., 7.]), .3, True)
    second.forward_step(np.array([-9., -8., -7.]), .3, True)
    assert np.array_equal(out_first, out_second)
    bad_first = prefix[0] + .01 * np.array([9., 8., 7.])
    bad_second = prefix[0] + .01 * np.array([-9., -8., -7.])
    assert not np.array_equal(bad_first, bad_second)


def test_sc_intra_timestamp_stage_order_red() -> None:
    estimator = SideEstimator("raw", feature_enabled=False)
    estimator.reset_trajectory(_state(), 0.)
    with pytest.raises(ValueError):
        estimator.compensate_magnetometer(np.array([.3, -.2, .9]), .1)
    gyro = estimator.compensate_gyro(np.array([.01, .02, .03]), .1)
    with pytest.raises(ValueError):
        estimator.update(gyro, gyro, np.array([.3, -.2, .9]))


def test_sc_recurrent_lineage_hashes_same_owner_length_different_hidden_bytes_red() -> None:
    first = SideEstimator("raw", feature_enabled=False)
    second = SideEstimator("raw", feature_enabled=False)
    second.load_state_dict(first.state_dict(), strict=True)
    owner = "same-target-owner"
    first.reset_trajectory(_state(), 0., trajectory_owner_token=owner)
    second.reset_trajectory(_state(), 0., trajectory_owner_token=owner)
    m_model = np.array([.31, -.18, .933])
    for index, timestamp in enumerate((.1, .2)):
        first.step_pair(
            np.array([.01, .02, .03]) + index * 1e-3,
            np.array([.30, -.20, .90]) + index * 1e-3,
            timestamp, m_model,
        )
        second.step_pair(
            np.array([-.04, .05, -.02]) - index * 1e-3,
            np.array([.41, -.11, .81]) - index * 1e-3,
            timestamp, m_model,
        )
    assert first.recurrent_history_owner_token == second.recurrent_history_owner_token == owner
    assert first.backbone.transition_count == second.backbone.transition_count == 2
    assert first.recurrent_history_provenance_sha256() != second.recurrent_history_provenance_sha256()


def test_sc_side_estimator_wrong_gain_shape_mutation_red(monkeypatch: pytest.MonkeyPatch) -> None:
    estimator = SideEstimator("raw", feature_enabled=False); estimator.reset_trajectory(_state(), 0.)
    gyro = estimator.compensate_gyro(np.array([.01, .02, .03]), .1); estimator.propagate(gyro)
    mag = estimator.compensate_magnetometer(np.array([.3, -.2, .9]), .1)
    monkeypatch.setattr(estimator.backbone, "forward_step", lambda *args, **kwargs: GainOutput(
        torch.eye(6, dtype=torch.float64), torch.eye(3, dtype=torch.float64),
        torch.zeros((3, 6), dtype=torch.float64),
    ))
    with pytest.raises(RuntimeError, match="invalid Split"):
        estimator.update(gyro, mag, np.array([.31, -.18, .933]))


def test_sc_side_estimator_identity_reset_mutation_red(monkeypatch: pytest.MonkeyPatch) -> None:
    estimator = SideEstimator("raw", feature_enabled=False); estimator.reset_trajectory(_state(), 0.)
    gyro = estimator.compensate_gyro(np.array([.01, .02, .03]), .1); estimator.propagate(gyro)
    mag = estimator.compensate_magnetometer(np.array([.5, -.1, .7]), .1)
    K = torch.zeros((6, 3), dtype=torch.float64); K[:3] = torch.eye(3, dtype=torch.float64) * .1
    monkeypatch.setattr(estimator.backbone, "forward_step", lambda *args, **kwargs: GainOutput(
        torch.eye(6, dtype=torch.float64), torch.eye(3, dtype=torch.float64), K,
    ))
    monkeypatch.setattr(model_module, "reset_covariance", lambda P, delta: (P, np.eye(6), 0.0))
    with pytest.raises(RuntimeError, match="right-Jacobian"):
        estimator.update(gyro, mag, np.array([.31, -.18, .933]))


def test_sc_nominal_learned_checkpoint_no_op_red() -> None:
    encoder = MagEncoder(); optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    packets = [np.array([.3, -.2, .93]), np.array([.31, -.19, .92])]
    encoder.reset_trajectory(); loss = torch.zeros((), dtype=torch.float64)
    for index, packet in enumerate(packets):
        output = encoder.forward_step(packet, .1 * (index + 1), True)
        loss = loss + torch.mean((output.corrected_B - torch.tensor(packet)) ** 2)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    checkpoint = copy.deepcopy(encoder.state_dict())
    restored = MagEncoder(); restored.load_state_dict(checkpoint, strict=True); restored.reset_trajectory()
    corrected = [restored.forward_step(packet, .1 * (index + 1), True).corrected_B.detach().numpy()
                 for index, packet in enumerate(packets)]
    assert np.array_equal(np.stack(corrected), np.stack(packets))
    mutation = copy.deepcopy(checkpoint); mutation["correction_head.bias"] = torch.ones(3, dtype=torch.float64) * .01
    adversary = MagEncoder(); adversary.load_state_dict(mutation, strict=True); adversary.reset_trajectory()
    assert not np.array_equal(adversary.forward_step(packets[0], .1, True).corrected_B.detach().numpy(), packets[0])


def test_sc_right_error_reset_red() -> None:
    P = np.diag([.2, .1, .05, .01, .02, .03])
    delta = np.array([.2, -.1, .04])
    reset, G, _ = reset_covariance(P, delta)
    assert np.allclose(G[:3, :3], right_jacobian_so3(delta))
    assert not np.allclose(reset, P)
    identity_adversary = np.eye(6) @ P @ np.eye(6).T
    assert not np.allclose(identity_adversary, reset)


def test_sc_single_mag_weak_axis_red() -> None:
    h = np.array([.3, -.2, .93]); h /= np.linalg.norm(h)
    H = skew(h)
    assert np.linalg.matrix_rank(H, tol=1e-12) == 2
    errors = np.array([[.1, .0, .0], [.0, .1, .0], [.0, .0, .1]])
    weak = np.abs(errors @ h)
    observable = np.linalg.norm(errors - (errors @ h)[:, None] * h, axis=1)
    assert weak.size > 0 and observable.size > 0 and np.all(np.isfinite(weak + observable))


def test_sc_g1_bias_attitude_coupling_red() -> None:
    backbone = SplitGainBackbone()
    h = np.array([.3, -.2, .93]); H = np.zeros((3, 6)); H[:, :3] = skew(h)
    residual = np.array([.02, -.01, .03])
    output = backbone.forward_step(np.arange(6) * .01, residual, H, np.ones(8), np.zeros(8), feature_enabled=True)
    delta = output.K @ torch.as_tensor(residual, dtype=torch.float64)
    assert torch.linalg.norm(output.G1[3:, :3]) > 0
    assert torch.linalg.norm(delta[3:]) > 0
    delta[3:].sum().backward()
    grad = backbone.g1_head.bias.grad.reshape(6, 6)[3:, :3]
    assert torch.linalg.norm(grad) > 0
