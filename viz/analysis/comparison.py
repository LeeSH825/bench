from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from viz.analysis import attitude, consistency, gain, units


PHYSICAL_METRIC_LABELS: Mapping[str, str] = {
    "attitude_rpy": "Attitude RPY",
    "attitude_geodesic_error": "Attitude geodesic error",
    "attitude_error_components": "Attitude error components",
    "attitude_uncertainty": "Attitude error and uncertainty",
    "gyro_bias": "Gyro bias estimate",
    "gyro_bias_error": "Gyro bias error",
    "gyro_bias_uncertainty": "Gyro bias error and uncertainty",
    "empirical_attitude_spread": "Empirical attitude spread",
    "empirical_bias_spread": "Empirical bias spread",
    "attitude_correction": "Attitude correction K nu",
    "bias_correction": "Bias correction K nu",
}

STRICT_METRIC_LABELS: Mapping[str, str] = {
    "innovation": "Innovation",
    "innovation_norm": "Innovation norm",
    "gain_norm": "Kalman gain Frobenius norm",
    "gain_element": "Kalman gain matrix element",
    "attitude_gain_block": "Attitude gain block norm",
    "bias_gain_block": "Bias gain block norm",
    "nees": "NEES",
    "nis": "NIS",
    "p_diagonal": "Physical P diagonal",
    "s_diagonal": "Physical S diagonal",
}

ATTITUDE_CONVENTION_FIELDS = (
    "representation",
    "quaternion_order",
    "rotation_direction",
    "frame_from",
    "frame_to",
)


def _spec(meta: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    value = meta.get("comparison_spec")
    return value if isinstance(value, Mapping) else None


def _section(meta: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    spec = _spec(meta)
    if spec is None:
        return {}
    value = spec.get(name)
    return value if isinstance(value, Mapping) else {}


def _result(
    compatible: bool,
    *,
    level: str,
    reasons: Sequence[str] = (),
    warnings: Sequence[str] = (),
    available_metrics: Sequence[str] = (),
) -> Dict[str, Any]:
    return {
        "compatible": bool(compatible),
        "level": str(level),
        "reasons": list(reasons),
        "warnings": list(warnings),
        "available_metrics": list(available_metrics),
    }


def available_physical_metrics(meta: Mapping[str, Any]) -> list[str]:
    spec = _spec(meta)
    if spec is None:
        return []
    metrics: list[str] = []
    if bool(_section(meta, "attitude").get("available")):
        metrics.extend(
            [
                "attitude_rpy",
                "attitude_geodesic_error",
                "attitude_error_components",
                "attitude_uncertainty",
            ]
        )
    if bool(_section(meta, "bias").get("available")):
        metrics.extend(["gyro_bias", "gyro_bias_error", "gyro_bias_uncertainty"])
    empirical = _section(meta, "empirical_uncertainty")
    if bool(empirical.get("available")):
        if bool(_section(meta, "attitude").get("available")):
            metrics.append("empirical_attitude_spread")
        if bool(_section(meta, "bias").get("available")):
            metrics.append("empirical_bias_spread")
    correction = _section(meta, "correction")
    if bool(correction.get("available")):
        if correction.get("attitude_block") is not None:
            metrics.append("attitude_correction")
        if correction.get("bias_block") is not None:
            metrics.append("bias_correction")
    return metrics


def available_internal_metrics(meta: Mapping[str, Any]) -> list[str]:
    caps = meta.get("capabilities", {})
    metrics: list[str] = []
    if bool(caps.get("innovation")) and bool(_section(meta, "innovation").get("available")):
        metrics.extend(["innovation", "innovation_norm"])
    if bool(caps.get("gain")) and bool(_section(meta, "gain").get("available")):
        metrics.extend(["gain_norm", "gain_element"])
        gain_spec = _section(meta, "gain")
        if gain_spec.get("attitude_block") is not None:
            metrics.append("attitude_gain_block")
        if gain_spec.get("bias_block") is not None:
            metrics.append("bias_gain_block")
    if bool(caps.get("covariance")):
        metrics.extend(["nees", "p_diagonal"])
    if bool(caps.get("innovation_cov")) and bool(caps.get("innovation")):
        metrics.extend(["nis", "s_diagonal"])
    return metrics


def _identity_checks(
    base_meta: Mapping[str, Any],
    candidate_meta: Mapping[str, Any],
    *,
    metric: str,
    base_source_trajectory_id: Any = None,
    candidate_source_trajectory_id: Any = None,
    base_time: Optional[np.ndarray] = None,
    candidate_time: Optional[np.ndarray] = None,
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    base_spec = _spec(base_meta)
    candidate_spec = _spec(candidate_meta)
    if base_spec is None:
        reasons.append("base artifact has no comparison_spec")
        return reasons, warnings
    if candidate_spec is None:
        reasons.append("candidate artifact has no comparison_spec")
        return reasons, warnings

    base_split = base_meta.get("data_spec", {}).get("split")
    candidate_split = candidate_meta.get("data_spec", {}).get("split")
    if base_split != candidate_split:
        reasons.append(f"data split mismatch: base={base_split!r}, candidate={candidate_split!r}")

    base_identity = base_spec.get("identity", {})
    candidate_identity = candidate_spec.get("identity", {})
    base_physical = base_identity.get("physical_scenario_id") if isinstance(base_identity, Mapping) else None
    candidate_physical = candidate_identity.get("physical_scenario_id") if isinstance(candidate_identity, Mapping) else None
    if base_physical is not None and candidate_physical is not None:
        if base_physical != candidate_physical:
            reasons.append(
                f"physical scenario mismatch: base={base_physical!r}, candidate={candidate_physical!r}"
            )
    elif base_meta.get("task") != candidate_meta.get("task") or base_meta.get("scenario_id") != candidate_meta.get("scenario_id"):
        reasons.append("task/scenario mismatch and no shared physical_scenario_id is declared")

    fingerprint_name = None
    if metric.startswith("attitude"):
        fingerprint_name = "attitude_truth"
    elif metric.startswith("gyro_bias") or metric == "bias_correction" or metric == "empirical_bias_spread":
        fingerprint_name = "bias_truth"
    if fingerprint_name is not None:
        base_fingerprints = base_identity.get("truth_fingerprints", {}) if isinstance(base_identity, Mapping) else {}
        candidate_fingerprints = (
            candidate_identity.get("truth_fingerprints", {}) if isinstance(candidate_identity, Mapping) else {}
        )
        base_fingerprint = base_fingerprints.get(fingerprint_name) if isinstance(base_fingerprints, Mapping) else None
        candidate_fingerprint = (
            candidate_fingerprints.get(fingerprint_name) if isinstance(candidate_fingerprints, Mapping) else None
        )
        if not isinstance(base_fingerprint, str) or not isinstance(candidate_fingerprint, str):
            reasons.append(f"{fingerprint_name} identity is unavailable")
        elif base_fingerprint != candidate_fingerprint:
            reasons.append(f"{fingerprint_name} mismatch")

    if base_source_trajectory_id is not None or candidate_source_trajectory_id is not None:
        if (
            type(base_source_trajectory_id) is not type(candidate_source_trajectory_id)
            or base_source_trajectory_id != candidate_source_trajectory_id
        ):
            reasons.append(
                "source trajectory mismatch: "
                f"base={base_source_trajectory_id!r}, candidate={candidate_source_trajectory_id!r}"
            )
    if base_time is not None or candidate_time is not None:
        if base_time is None or candidate_time is None or not np.array_equal(base_time, candidate_time):
            reasons.append("time axis mismatch; interpolation is not allowed")

    provenance_values = {
        str(base_meta.get("data_spec", {}).get("source_trajectory_id_source", "")),
        str(candidate_meta.get("data_spec", {}).get("source_trajectory_id_source", "")),
    }
    if any("fallback" in value for value in provenance_values):
        warnings.append(
            "Source ID provenance uses a split-row fallback; comparison is valid only for the same dataset file and row ordering"
        )
    return reasons, warnings


def evaluate_physical_metric_compatibility(
    base_meta: Mapping[str, Any],
    candidate_meta: Mapping[str, Any],
    *,
    metric: str,
    base_source_trajectory_id: Any = None,
    candidate_source_trajectory_id: Any = None,
    base_time: Optional[np.ndarray] = None,
    candidate_time: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    available = available_physical_metrics(candidate_meta)
    reasons, warnings = _identity_checks(
        base_meta,
        candidate_meta,
        metric=metric,
        base_source_trajectory_id=base_source_trajectory_id,
        candidate_source_trajectory_id=candidate_source_trajectory_id,
        base_time=base_time,
        candidate_time=candidate_time,
    )
    if metric not in PHYSICAL_METRIC_LABELS:
        reasons.append(f"unsupported physical metric {metric!r}")
    if metric not in available:
        reasons.append(f"candidate does not provide {PHYSICAL_METRIC_LABELS.get(metric, metric)}")

    if metric.startswith("attitude") or metric == "empirical_attitude_spread":
        base_attitude = _section(base_meta, "attitude")
        candidate_attitude = _section(candidate_meta, "attitude")
        for field in ATTITUDE_CONVENTION_FIELDS:
            if base_attitude.get(field) != candidate_attitude.get(field):
                reasons.append(
                    f"attitude {field} mismatch: base={base_attitude.get(field)!r}, "
                    f"candidate={candidate_attitude.get(field)!r}"
                )
        if metric == "attitude_rpy":
            for field in ("rpy_sequence", "rpy_convention", "rpy_output_order"):
                if base_attitude.get(field) != candidate_attitude.get(field):
                    reasons.append(
                        f"attitude {field} mismatch: base={base_attitude.get(field)!r}, "
                        f"candidate={candidate_attitude.get(field)!r}"
                    )
        if metric == "attitude_uncertainty" and not bool(_section(candidate_meta, "covariance").get("physical")):
            warnings.append("No physical attitude covariance; only the candidate error line can be displayed")

    if metric.startswith("gyro_bias") or metric in {"bias_correction", "empirical_bias_spread"}:
        base_bias = _section(base_meta, "bias")
        candidate_bias = _section(candidate_meta, "bias")
        for field in ("units", "frame"):
            if base_bias.get(field) != candidate_bias.get(field):
                reasons.append(
                    f"bias {field} mismatch: base={base_bias.get(field)!r}, candidate={candidate_bias.get(field)!r}"
                )
        if metric == "gyro_bias_uncertainty" and not bool(_section(candidate_meta, "covariance").get("physical")):
            warnings.append("No physical bias covariance; only the candidate error line can be displayed")

    if metric in {"attitude_correction", "bias_correction"}:
        correction = _section(candidate_meta, "correction")
        if correction.get("source") == "reconstructed_gain_times_innovation":
            warnings.append("Correction is reconstructed as K nu, not captured as the final applied state change")
    return _result(
        not reasons,
        level="physical" if not reasons else "blocked",
        reasons=reasons,
        warnings=warnings,
        available_metrics=available,
    )


def evaluate_internal_metric_compatibility(
    base_meta: Mapping[str, Any],
    candidate_meta: Mapping[str, Any],
    *,
    metric: str,
    base_source_trajectory_id: Any = None,
    candidate_source_trajectory_id: Any = None,
    base_time: Optional[np.ndarray] = None,
    candidate_time: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    available = available_internal_metrics(candidate_meta)
    reasons, warnings = _identity_checks(
        base_meta,
        candidate_meta,
        metric=metric,
        base_source_trajectory_id=base_source_trajectory_id,
        candidate_source_trajectory_id=candidate_source_trajectory_id,
        base_time=base_time,
        candidate_time=candidate_time,
    )
    if metric not in STRICT_METRIC_LABELS:
        reasons.append(f"unsupported strict metric {metric!r}")
    if metric not in available:
        reasons.append(f"candidate does not provide {STRICT_METRIC_LABELS.get(metric, metric)}")

    for key in ("task", "scenario_id", "formulation"):
        if base_meta.get(key) != candidate_meta.get(key):
            reasons.append(f"strict {key} mismatch: base={base_meta.get(key)!r}, candidate={candidate_meta.get(key)!r}")
    if base_meta.get("seed") != candidate_meta.get("seed"):
        reasons.append(
            f"strict seed mismatch: base={base_meta.get('seed')!r}, candidate={candidate_meta.get('seed')!r}"
        )
    if base_meta.get("state_spec") != candidate_meta.get("state_spec"):
        reasons.append("strict state_spec mismatch")

    if metric.startswith("innovation") or metric == "nis" or metric == "s_diagonal":
        base_innov = _section(base_meta, "innovation")
        candidate_innov = _section(candidate_meta, "innovation")
        for field in (
            "measurement_type",
            "residual_definition",
            "units",
            "frame",
            "dimension",
            "measurement_order",
            "valid_mask_key",
        ):
            if base_innov.get(field) != candidate_innov.get(field):
                reasons.append(
                    f"innovation {field} mismatch: base={base_innov.get(field)!r}, "
                    f"candidate={candidate_innov.get(field)!r}"
                )

    if metric.startswith("gain") or metric.endswith("gain_block"):
        base_gain = _section(base_meta, "gain")
        candidate_gain = _section(candidate_meta, "gain")
        for field in (
            "row_state_order",
            "column_measurement_order",
            "row_units",
            "column_units",
            "state_scaling",
            "measurement_scaling",
            "shape",
        ):
            if base_gain.get(field) != candidate_gain.get(field):
                reasons.append(
                    f"gain {field} mismatch: base={base_gain.get(field)!r}, candidate={candidate_gain.get(field)!r}"
                )
    return _result(
        not reasons,
        level="strict" if not reasons else "blocked",
        reasons=reasons,
        warnings=warnings,
        available_metrics=available,
    )


def _array(traj: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in traj:
        raise ValueError(f"trajectory key {key!r} is unavailable")
    return np.asarray(traj[key], dtype=np.float64)


def attitude_rpy_deg(meta: Mapping[str, Any], traj: Mapping[str, np.ndarray], *, estimate: bool) -> np.ndarray:
    spec = _section(meta, "attitude")
    key = str(spec.get("estimate_key" if estimate else "truth_key"))
    q = _array(traj, key)
    return units.rad_to_deg(np.unwrap(attitude.euler321_from_quat(q), axis=0))


def attitude_geodesic_error_deg(meta: Mapping[str, Any], traj: Mapping[str, np.ndarray]) -> np.ndarray:
    spec = _section(meta, "attitude")
    q_true = _array(traj, str(spec.get("truth_key")))
    q_hat = _array(traj, str(spec.get("estimate_key")))
    return units.rad_to_deg(attitude.geodesic_angle_rad(q_true, q_hat))


def attitude_error_components_deg(meta: Mapping[str, Any], traj: Mapping[str, np.ndarray]) -> np.ndarray:
    spec = _section(meta, "attitude")
    q_true = _array(traj, str(spec.get("truth_key")))
    q_hat = _array(traj, str(spec.get("estimate_key")))
    return units.rad_to_deg(attitude.quat_to_rotvec(attitude.relative_quat(q_true, q_hat)))


def _block(section: Mapping[str, Any], key: str) -> slice:
    value = section.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"comparison block {key!r} is unavailable")
    return slice(int(value[0]), int(value[1]))


def physical_attitude_band_deg(meta: Mapping[str, Any], traj: Mapping[str, np.ndarray]) -> Optional[np.ndarray]:
    covariance = _section(meta, "covariance")
    if not bool(covariance.get("available")) or not bool(covariance.get("physical")):
        return None
    block = _block(covariance, "attitude_block")
    p = _array(traj, str(covariance.get("key", "P")))
    return units.covariance_axis_band_deg(p[:, block, block], covariance.get("space"))


def bias_estimate_deg_h(meta: Mapping[str, Any], traj: Mapping[str, np.ndarray]) -> np.ndarray:
    spec = _section(meta, "bias")
    estimate = _array(traj, str(spec.get("estimate_key")))[:, _block(spec, "estimate_block")]
    return units.rad_s_to_deg_h(estimate)


def bias_truth_deg_h(meta: Mapping[str, Any], traj: Mapping[str, np.ndarray]) -> np.ndarray:
    spec = _section(meta, "bias")
    return units.rad_s_to_deg_h(_array(traj, str(spec.get("truth_key"))))


def bias_error_deg_h(meta: Mapping[str, Any], traj: Mapping[str, np.ndarray]) -> np.ndarray:
    return bias_estimate_deg_h(meta, traj) - bias_truth_deg_h(meta, traj)


def physical_bias_band_deg_h(meta: Mapping[str, Any], traj: Mapping[str, np.ndarray]) -> Optional[np.ndarray]:
    covariance = _section(meta, "covariance")
    if not bool(covariance.get("available")) or not bool(covariance.get("physical")):
        return None
    block = _block(covariance, "bias_block")
    p = _array(traj, str(covariance.get("key", "P")))
    sigma = np.sqrt(np.maximum(np.diagonal(p[:, block, block], axis1=-2, axis2=-1), 0.0))
    return units.rad_s_to_deg_h(np.asarray(3.0, dtype=np.float64) * sigma)


def empirical_spread(
    meta: Mapping[str, Any],
    aggregate: Mapping[str, np.ndarray],
    *,
    kind: str,
) -> np.ndarray:
    empirical = _section(meta, "empirical_uncertainty")
    if bool(empirical.get("physical")):
        raise ValueError("empirical ensemble uncertainty cannot be declared physical")
    values = _array(aggregate, str(empirical.get("key", "emp_std")))
    if kind == "attitude":
        block = _block(_section(meta, "attitude"), "state_block")
        spread = values[:, block]
        space = _section(meta, "attitude").get("state_coordinate_space")
        if space == "mrp":
            return units.mrp_delta_to_deg(spread)
        if space == "rotation_vector_rad":
            return units.rad_to_deg(spread)
        raise ValueError(f"unsupported attitude empirical coordinate space {space!r}")
    if kind == "bias":
        block = _block(_section(meta, "bias"), "estimate_block")
        return units.rad_s_to_deg_h(values[:, block])
    raise ValueError(f"unsupported empirical spread kind {kind!r}")


def reconstructed_state_correction(meta: Mapping[str, Any], traj: Mapping[str, np.ndarray]) -> np.ndarray:
    correction = _section(meta, "correction")
    direct_key = correction.get("state_key")
    if isinstance(direct_key, str) and direct_key in traj:
        return _array(traj, direct_key)
    if correction.get("source") != "reconstructed_gain_times_innovation":
        raise ValueError("state correction is unavailable")
    gain_arr = _array(traj, str(correction.get("gain_key", "gain")))
    innovation = _array(traj, str(correction.get("innovation_key", "innov")))
    if gain_arr.shape[0] != innovation.shape[0] or gain_arr.shape[2] != innovation.shape[1]:
        raise ValueError(f"gain/innovation shape mismatch: {gain_arr.shape} vs {innovation.shape}")
    correction_value = np.einsum("tnm,tm->tn", gain_arr, innovation)
    valid_key = correction.get("valid_mask_key")
    if isinstance(valid_key, str) and valid_key in traj:
        valid = np.asarray(traj[valid_key], dtype=bool)
        correction_value = np.where(valid[:, None], correction_value, np.nan)
    return correction_value


def physical_correction(meta: Mapping[str, Any], traj: Mapping[str, np.ndarray], *, kind: str) -> np.ndarray:
    correction = _section(meta, "correction")
    state_value = reconstructed_state_correction(meta, traj)
    if kind == "attitude":
        value = state_value[:, _block(correction, "attitude_block")]
        space = correction.get("attitude_coordinate_space")
        if space == "mrp":
            return units.mrp_delta_to_deg(value)
        if space == "rotation_vector_rad":
            return units.rad_to_deg(value)
        raise ValueError(f"unsupported attitude correction coordinate space {space!r}")
    if kind == "bias":
        return units.rad_s_to_deg_h(state_value[:, _block(correction, "bias_block")])
    raise ValueError(f"unsupported correction kind {kind!r}")


def strict_metric_series(
    meta: Mapping[str, Any],
    traj: Mapping[str, np.ndarray],
    metric: str,
    *,
    row: int = 0,
    col: int = 0,
) -> np.ndarray:
    valid = np.asarray(traj.get("innov_valid", np.ones(len(traj["t"]), dtype=bool)), dtype=bool)
    if metric in {"innovation", "innovation_norm"}:
        values = _array(traj, "innov")
        values = np.where(valid[:, None], values, np.nan)
        return np.linalg.norm(values, axis=1) if metric == "innovation_norm" else values
    if metric.startswith("gain") or metric.endswith("gain_block"):
        values = _array(traj, "gain")
        values = np.where(valid[:, None, None], values, np.nan)
        if metric == "gain_norm":
            return gain.gain_norm(values)
        if metric == "gain_element":
            r = min(max(int(row), 0), values.shape[1] - 1)
            c = min(max(int(col), 0), values.shape[2] - 1)
            return values[:, r, c]
        key = "attitude_block" if metric == "attitude_gain_block" else "bias_block"
        block = _block(_section(meta, "gain"), key)
        return gain.gain_norm(values[:, block, :])
    if metric == "nees":
        return consistency.nees(_array(traj, "x_hat") - _array(traj, "x_true"), _array(traj, "P"))
    if metric == "nis":
        return consistency.nis(_array(traj, "innov"), _array(traj, "S"), valid=valid)
    if metric == "p_diagonal":
        return np.diagonal(_array(traj, "P"), axis1=-2, axis2=-1)
    if metric == "s_diagonal":
        return np.diagonal(_array(traj, "S"), axis1=-2, axis2=-1)
    raise ValueError(f"unsupported strict metric {metric!r}")
