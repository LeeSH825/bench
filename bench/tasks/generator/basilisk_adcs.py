from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .contract import GeneratorOutput, coerce_ntd_float32_output, make_split_cfg, make_task_cfg
from .datasets.common import DatasetMissingError
from ...utils.seeding import numpy_rng_v0, stable_int_seed_v0


def _deep_merge(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _get_nested(mapping: Mapping[str, Any], path: Tuple[str, ...], default: Any = None) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_vec3(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be length 3, got shape {arr.shape}")
    return arr


def _as_diag_inertia(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (3,):
        if np.any(arr <= 0.0):
            raise ValueError(f"inertia diagonal entries must be positive, got {arr}")
        return np.diag(arr)
    if arr.shape == (3, 3):
        if np.any(np.diag(arr) <= 0.0):
            raise ValueError(f"inertia diagonal entries must be positive, got {arr}")
        return arr
    raise ValueError(f"inertia must be length-3 diagonal or 3x3 matrix, got shape {arr.shape}")


def _resolve_noise_r2(task_cfg_dict: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> float:
    resolved = _deep_merge(dict(task_cfg_dict), scenario_cfg)
    noise = resolved.get("noise", {})
    if not isinstance(noise, Mapping):
        return 1.0e-4
    if "sensor_noise_scale_db" in resolved:
        base = float(_get_nested(noise, ("R", "r2"), 1.0e-4))
        return float(base * math.pow(10.0, float(resolved["sensor_noise_scale_db"]) / 10.0))
    if "r2" in noise:
        return float(noise["r2"])
    return float(_get_nested(noise, ("R", "r2"), 1.0e-4))


def _resolve_q2(task_cfg_dict: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> float:
    resolved = _deep_merge(dict(task_cfg_dict), scenario_cfg)
    noise = resolved.get("noise", {})
    if not isinstance(noise, Mapping):
        return 1.0e-8
    if "q2" in noise:
        return float(noise["q2"])
    return float(_get_nested(noise, ("Q", "q2"), 1.0e-8))


def _small_angle_model(*, dt: float, q2: float, r2: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f = np.eye(6, dtype=np.float64)
    f[0:3, 3:6] = 0.25 * float(dt) * np.eye(3, dtype=np.float64)
    h = np.eye(6, dtype=np.float64)
    q = float(q2) * np.eye(6, dtype=np.float64)
    r = float(r2) * np.eye(6, dtype=np.float64)
    return f.astype(np.float32), h.astype(np.float32), q.astype(np.float32), r.astype(np.float32)


def _block_std_vec(cfg: Mapping[str, Any], *, default: float = 0.0) -> np.ndarray:
    base = float(cfg.get("std", default))
    sigma = float(cfg.get("sigma_std", cfg.get("std_sigma", cfg.get("bias_std_sigma", base))))
    omega = float(cfg.get("omega_std", cfg.get("std_omega", cfg.get("bias_std_omega", base))))
    return np.asarray([sigma, sigma, sigma, omega, omega, omega], dtype=np.float64)


def _component_enabled(cfg: Mapping[str, Any]) -> bool:
    if "enabled" in cfg:
        return bool(cfg.get("enabled"))
    return bool(cfg)


def _resolve_measurement_corruption_cfg(resolved_task: Mapping[str, Any]) -> Dict[str, Any]:
    raw_obj = resolved_task.get("measurement_corruption", {})
    if not isinstance(raw_obj, Mapping):
        return {"enabled": False, "profile_id": "gaussian_only"}
    raw = json.loads(json.dumps(dict(raw_obj)))
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, Mapping):
        profiles = {}
    profile_id = str(raw.get("profile_id", raw.get("severity", ""))).strip()
    merged = dict(raw)
    merged.pop("profiles", None)
    if profile_id:
        if profile_id not in profiles:
            raise ValueError(f"measurement_corruption profile_id={profile_id!r} not found in profiles")
        profile_cfg = profiles[profile_id]
        if not isinstance(profile_cfg, Mapping):
            raise ValueError(f"measurement_corruption profile {profile_id!r} must be a mapping")
        merged = _deep_merge(merged, dict(profile_cfg))
        merged["profile_id"] = profile_id
    merged.setdefault("enabled", False)
    merged.setdefault("profile_id", profile_id or "gaussian_only")
    return merged


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = [float(v) for v in np.asarray(vec, dtype=np.float64).reshape(3)]
    return np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def _mean_step_norm(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return 0.0
    if a.ndim == 3:
        return float(np.mean(np.linalg.norm(a, axis=2)))
    if a.ndim == 2:
        return float(np.mean(np.linalg.norm(a, axis=1)))
    return float(np.linalg.norm(a.reshape(-1)))


def _axis_matrix(rng: np.random.Generator, std: float) -> np.ndarray:
    if float(std) <= 0.0:
        return np.eye(3, dtype=np.float64)
    return np.eye(3, dtype=np.float64) + _skew(rng.normal(0.0, float(std), size=3))


def apply_structured_measurement_corruption(
    *,
    x_all: np.ndarray,
    cfg: Mapping[str, Any],
    sensor_std: float,
    suite_name: str,
    task_id: str,
    scenario_id: str,
    seed: int,
    dt: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Bridge-level low-cost sensor corruption for full-state ADCS measurements.

    The Basilisk truth state is not modified. The returned y is a corrupted
    measurement of x=[sigma_BN, omega_BN_B].
    """
    x = np.asarray(x_all, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 6:
        raise ValueError(f"structured corruption expects x shape [N,T,6], got {x.shape}")
    n_total, t_len, y_dim = x.shape
    y = x.copy()
    y_before = y.copy()

    gaussian_seq = np.zeros_like(y)
    bias_seq = np.zeros_like(y)
    drift_seq = np.zeros_like(y)
    outlier_seq = np.zeros_like(y)
    outlier_mask_seq = np.zeros_like(y)
    vibration_seq = np.zeros_like(y)
    scale_effect_seq = np.zeros_like(y)
    misalignment_effect_seq = np.zeros_like(y)
    scale_mats = np.zeros((n_total, y_dim, y_dim), dtype=np.float64)
    misalignment_mats = np.zeros((n_total, y_dim, y_dim), dtype=np.float64)

    cfg_dict = dict(cfg)
    gaussian_cfg = cfg_dict.get("gaussian", {})
    if not isinstance(gaussian_cfg, Mapping):
        gaussian_cfg = {}
    bias_cfg = cfg_dict.get("bias", {})
    if not isinstance(bias_cfg, Mapping):
        bias_cfg = {}
    random_walk_cfg = cfg_dict.get("random_walk", cfg_dict.get("drift_random_walk", {}))
    if not isinstance(random_walk_cfg, Mapping):
        random_walk_cfg = {}
    drift_cfg = cfg_dict.get("drift", {})
    if not isinstance(drift_cfg, Mapping):
        drift_cfg = {}
    scale_cfg = cfg_dict.get("scale", {})
    if not isinstance(scale_cfg, Mapping):
        scale_cfg = {}
    misalignment_cfg = cfg_dict.get("axis_misalignment", cfg_dict.get("misalignment", {}))
    if not isinstance(misalignment_cfg, Mapping):
        misalignment_cfg = {}
    outlier_cfg = cfg_dict.get("outlier", cfg_dict.get("outliers", {}))
    if not isinstance(outlier_cfg, Mapping):
        outlier_cfg = {}
    vibration_cfg = cfg_dict.get("vibration", {})
    if not isinstance(vibration_cfg, Mapping):
        vibration_cfg = {}

    if _component_enabled(gaussian_cfg):
        if "r2" in gaussian_cfg:
            gaussian_std = math.sqrt(float(gaussian_cfg["r2"]))
            gaussian_std_vec = np.full((6,), gaussian_std, dtype=np.float64)
        else:
            gaussian_std_vec = _block_std_vec(gaussian_cfg, default=float(sensor_std))
    else:
        gaussian_std_vec = np.zeros((6,), dtype=np.float64)

    bias_std_vec = _block_std_vec(bias_cfg, default=0.0) if _component_enabled(bias_cfg) else np.zeros((6,), dtype=np.float64)
    rw_std_vec = _block_std_vec(random_walk_cfg, default=0.0) if _component_enabled(random_walk_cfg) else np.zeros((6,), dtype=np.float64)
    drift_slope_std_vec = (
        _block_std_vec(drift_cfg, default=0.0) if _component_enabled(drift_cfg) else np.zeros((6,), dtype=np.float64)
    )
    scale_std = float(scale_cfg.get("std", scale_cfg.get("scale_std", 0.0))) if _component_enabled(scale_cfg) else 0.0
    misalignment_std = (
        float(misalignment_cfg.get("std", misalignment_cfg.get("misalignment_std", 0.0)))
        if _component_enabled(misalignment_cfg)
        else 0.0
    )
    outlier_prob = float(outlier_cfg.get("prob", outlier_cfg.get("p", outlier_cfg.get("outlier_prob", 0.0)))) if _component_enabled(outlier_cfg) else 0.0
    outlier_std_vec = _block_std_vec(outlier_cfg, default=0.0) if _component_enabled(outlier_cfg) else np.zeros((6,), dtype=np.float64)
    vibration_amp_vec = (
        _block_std_vec(vibration_cfg, default=float(vibration_cfg.get("amp", 0.0)))
        if _component_enabled(vibration_cfg)
        else np.zeros((6,), dtype=np.float64)
    )
    freq_range = vibration_cfg.get("freq_hz_range", vibration_cfg.get("frequency_hz_range", [1.0, 5.0]))
    freq_arr = np.asarray(freq_range, dtype=np.float64).reshape(-1)
    if freq_arr.size != 2:
        freq_arr = np.asarray([1.0, 5.0], dtype=np.float64)
    freq_low, freq_high = float(min(freq_arr)), float(max(freq_arr))

    t = np.arange(t_len, dtype=np.float64) * float(dt)

    for i in range(n_total):
        rng = numpy_rng_v0(stable_int_seed_v0("basilisk_adcs_corruption", suite_name, task_id, scenario_id, int(seed), int(i)))

        gaussian = rng.normal(0.0, gaussian_std_vec, size=(t_len, y_dim))
        gaussian_seq[i] = gaussian
        y[i] += gaussian

        bias = rng.normal(0.0, bias_std_vec, size=(y_dim,))
        bias_i = np.broadcast_to(bias.reshape(1, y_dim), (t_len, y_dim))
        bias_seq[i] = bias_i
        y[i] += bias_i

        rw_steps = rng.normal(0.0, rw_std_vec, size=(t_len, y_dim))
        rw = np.cumsum(rw_steps, axis=0)
        rw[0, :] = 0.0
        slope = rng.normal(0.0, drift_slope_std_vec, size=(y_dim,))
        linear = t.reshape(t_len, 1) * slope.reshape(1, y_dim)
        drift = rw + linear
        drift_seq[i] = drift
        y[i] += drift

        before_scale = y[i].copy()
        scale_factors = 1.0 + rng.normal(0.0, scale_std, size=(y_dim,))
        scale_mat = np.diag(scale_factors)
        scale_mats[i] = scale_mat
        y[i] = y[i] @ scale_mat.T
        scale_effect_seq[i] = y[i] - before_scale

        before_misalignment = y[i].copy()
        m_sigma = _axis_matrix(rng, misalignment_std)
        m_omega = _axis_matrix(rng, misalignment_std)
        m = np.eye(y_dim, dtype=np.float64)
        m[0:3, 0:3] = m_sigma
        m[3:6, 3:6] = m_omega
        misalignment_mats[i] = m
        y[i, :, 0:3] = y[i, :, 0:3] @ m_sigma.T
        y[i, :, 3:6] = y[i, :, 3:6] @ m_omega.T
        misalignment_effect_seq[i] = y[i] - before_misalignment

        if outlier_prob > 0.0:
            mask = rng.random(size=(t_len, y_dim)) < outlier_prob
            outlier = rng.normal(0.0, outlier_std_vec, size=(t_len, y_dim)) * mask
        else:
            mask = np.zeros((t_len, y_dim), dtype=bool)
            outlier = np.zeros((t_len, y_dim), dtype=np.float64)
        outlier_mask_seq[i] = mask.astype(np.float64)
        outlier_seq[i] = outlier
        y[i] += outlier

        if np.any(vibration_amp_vec > 0.0):
            amp = rng.normal(0.0, vibration_amp_vec, size=(y_dim,))
            freq = rng.uniform(freq_low, freq_high, size=(y_dim,))
            phase = rng.uniform(0.0, 2.0 * math.pi, size=(y_dim,))
            vib = amp.reshape(1, y_dim) * np.sin(
                2.0 * math.pi * t.reshape(t_len, 1) * freq.reshape(1, y_dim) + phase.reshape(1, y_dim)
            )
        else:
            vib = np.zeros((t_len, y_dim), dtype=np.float64)
        vibration_seq[i] = vib
        y[i] += vib

    total = y - x
    clean_norm = max(_mean_step_norm(x), 1.0e-12)
    meta = {
        "enabled": True,
        "profile_id": str(cfg_dict.get("profile_id", cfg_dict.get("severity", "structured"))),
        "severity": str(cfg_dict.get("severity", cfg_dict.get("profile_id", "structured"))),
        "clean_measurement_reference": "x",
        "applied_to_dims": "all_6_full_state_dims",
        "sigma_dims": [0, 1, 2],
        "omega_dims": [3, 4, 5],
        "per_trajectory_components": ["bias", "random_walk", "linear_drift", "scale", "axis_misalignment", "vibration"],
        "time_varying_components": ["gaussian", "random_walk", "linear_drift", "outlier", "vibration"],
        "gaussian": json.loads(json.dumps(dict(gaussian_cfg))),
        "bias": json.loads(json.dumps(dict(bias_cfg))),
        "random_walk": json.loads(json.dumps(dict(random_walk_cfg))),
        "drift": json.loads(json.dumps(dict(drift_cfg))),
        "scale": json.loads(json.dumps(dict(scale_cfg))),
        "axis_misalignment": json.loads(json.dumps(dict(misalignment_cfg))),
        "outlier": json.loads(json.dumps(dict(outlier_cfg))),
        "vibration": json.loads(json.dumps(dict(vibration_cfg))),
        "stats": {
            "y_clean_norm_mean": _mean_step_norm(x),
            "gaussian_norm_mean": _mean_step_norm(gaussian_seq),
            "bias_norm_mean": _mean_step_norm(bias_seq),
            "drift_norm_mean": _mean_step_norm(drift_seq),
            "scale_effect_norm_mean": _mean_step_norm(scale_effect_seq),
            "misalignment_effect_norm_mean": _mean_step_norm(misalignment_effect_seq),
            "outlier_norm_mean": _mean_step_norm(outlier_seq),
            "outlier_rate_observed": float(np.mean(outlier_mask_seq)),
            "vibration_norm_mean": _mean_step_norm(vibration_seq),
            "total_corruption_norm_mean": _mean_step_norm(total),
            "total_corruption_to_clean_ratio": float(_mean_step_norm(total) / clean_norm),
            "y_raw_delta_from_gaussian_baseline_norm_mean": _mean_step_norm(y - y_before),
        },
    }
    extras = {
        "y_clean_seq": x.astype(np.float32),
        "corruption_total_seq": total.astype(np.float32),
        "corruption_gaussian_seq": gaussian_seq.astype(np.float32),
        "corruption_bias_seq": bias_seq.astype(np.float32),
        "corruption_drift_seq": drift_seq.astype(np.float32),
        "corruption_outlier_seq": outlier_seq.astype(np.float32),
        "corruption_outlier_mask_seq": outlier_mask_seq.astype(np.float32),
        "corruption_vibration_seq": vibration_seq.astype(np.float32),
        "corruption_scale_effect_seq": scale_effect_seq.astype(np.float32),
        "corruption_misalignment_effect_seq": misalignment_effect_seq.astype(np.float32),
        "corruption_scale_matrix_seq": scale_mats.astype(np.float32),
        "corruption_misalignment_matrix_seq": misalignment_mats.astype(np.float32),
    }
    return y.astype(np.float32), extras, meta


def _shadow_mrp(sigma: np.ndarray) -> np.ndarray:
    norm2 = float(np.dot(sigma, sigma))
    if norm2 > 1.0:
        return -sigma / norm2
    return sigma


def _require_avs_basilisk() -> Dict[str, Any]:
    try:
        from Basilisk.architecture import messaging  # type: ignore
        from Basilisk.simulation import spacecraft  # type: ignore
        from Basilisk.utilities import SimulationBaseClass, macros  # type: ignore
    except Exception as exc:
        raise DatasetMissingError(
            dataset="AVS Basilisk",
            env_var="PYTHONPATH",
            message=(
                "AVS Basilisk is required for task_family=basilisk_adcs_v0. "
                "Expected modules: Basilisk.simulation and Basilisk.utilities. "
                "The PyPI package Basilisk==0.1/lowercase basilisk is not the spacecraft simulation framework."
            ),
        ) from exc

    modules: Dict[str, Any] = {
        "messaging": messaging,
        "spacecraft": spacecraft,
        "SimulationBaseClass": SimulationBaseClass,
        "macros": macros,
    }
    try:
        from Basilisk.simulation import extForceTorque  # type: ignore

        modules["extForceTorque"] = extForceTorque
    except Exception:
        modules["extForceTorque"] = None
    return modules


def _simulate_one_trajectory(
    *,
    bsk: Mapping[str, Any],
    sigma0: np.ndarray,
    omega0: np.ndarray,
    inertia: np.ndarray,
    disturbance_torque: np.ndarray,
    dt: float,
    t_len: int,
) -> np.ndarray:
    SimulationBaseClass = bsk["SimulationBaseClass"]
    macros = bsk["macros"]
    spacecraft = bsk["spacecraft"]

    sim = SimulationBaseClass.SimBaseClass()
    process_name = "dynProcess"
    task_name = "dynTask"
    sim_task = sim.CreateNewTask(task_name, macros.sec2nano(float(dt)))
    sim.CreateNewProcess(process_name).addTask(sim_task)

    sc = spacecraft.Spacecraft()
    sc.ModelTag = "basilisk_adcs_spacecraft"
    sc.hub.mHub = 1.0
    sc.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    sc.hub.IHubPntBc_B = np.asarray(inertia, dtype=np.float64)
    sc.hub.sigma_BNInit = [[float(v)] for v in sigma0]
    sc.hub.omega_BN_BInit = [[float(v)] for v in omega0]

    ext_force_torque = None
    if bsk.get("extForceTorque") is not None and float(np.linalg.norm(disturbance_torque)) > 0.0:
        ext_force_torque = bsk["extForceTorque"].ExtForceTorque()
        ext_force_torque.ModelTag = "constant_disturbance_torque"
        ext_force_torque.extTorquePntB_B = [[float(v)] for v in disturbance_torque]
        sc.addDynamicEffector(ext_force_torque)

    sim.AddModelToTask(task_name, sc)

    recorder = sc.scStateOutMsg.recorder(macros.sec2nano(float(dt)))
    sim.AddModelToTask(task_name, recorder)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(float(dt) * max(0, int(t_len) - 1)))
    sim.ExecuteSimulation()

    sigma = np.asarray(recorder.sigma_BN, dtype=np.float64)
    omega = np.asarray(recorder.omega_BN_B, dtype=np.float64)
    if sigma.shape[0] < t_len or omega.shape[0] < t_len:
        raise RuntimeError(
            f"Basilisk recorder returned too few samples: sigma={sigma.shape}, omega={omega.shape}, T={t_len}"
        )
    x = np.concatenate([sigma[:t_len], omega[:t_len]], axis=1)
    x[:, 0:3] = np.asarray([_shadow_mrp(row) for row in x[:, 0:3]], dtype=np.float64)
    return x


def generate_basilisk_adcs_v0(
    *,
    suite_name: str,
    task_cfg_dict: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    seed: int,
    scenario_id: str,
    task_family: str = "basilisk_adcs_v0",
) -> Tuple[GeneratorOutput, np.ndarray, np.ndarray]:
    bsk = _require_avs_basilisk()

    resolved_task = _deep_merge(dict(task_cfg_dict), scenario_cfg)
    task_cfg = make_task_cfg(resolved_task, scenario_cfg=scenario_cfg)
    split_cfg = make_split_cfg(resolved_task)

    x_dim = int(task_cfg.x_dim)
    y_dim = int(task_cfg.y_dim)
    t_len = int(task_cfg.sequence_length_T)
    n_total = int(split_cfg.n_total)
    if x_dim != 6 or y_dim != 6:
        raise ValueError(f"basilisk_adcs_v0 requires x_dim=y_dim=6, got x_dim={x_dim}, y_dim={y_dim}")

    sim_cfg = dict(resolved_task.get("simulation", {}) or {})
    dt = float(sim_cfg.get("dt", 0.1))
    inertia = _as_diag_inertia(sim_cfg.get("inertia", [10.0, 8.0, 6.0]))
    disturbance = _as_vec3(sim_cfg.get("disturbance_torque", [0.0, 0.0, 0.0]), name="disturbance_torque")
    sigma0_std = float(sim_cfg.get("sigma0_std", 0.05))
    omega0_std = float(sim_cfg.get("omega0_std", 0.01))
    sigma0_max_norm = float(sim_cfg.get("sigma0_max_norm", 0.25))

    q2 = _resolve_q2(resolved_task, scenario_cfg)
    r2 = _resolve_noise_r2(resolved_task, scenario_cfg)
    sensor_std = math.sqrt(float(r2))
    corruption_cfg = _resolve_measurement_corruption_cfg(resolved_task)
    corruption_enabled = bool(corruption_cfg.get("enabled", False))

    f_assumed, h_assumed, q_assumed, r_assumed = _small_angle_model(dt=dt, q2=q2, r2=r2)

    data_seed = stable_int_seed_v0("basilisk_adcs_data", suite_name, task_cfg.task_id, scenario_id, int(seed))
    rng = numpy_rng_v0(data_seed)
    x_all = np.zeros((n_total, t_len, x_dim), dtype=np.float64)
    y_all = np.zeros((n_total, t_len, y_dim), dtype=np.float64)

    for i in range(n_total):
        sigma0 = rng.normal(0.0, sigma0_std, size=3)
        sigma_norm = float(np.linalg.norm(sigma0))
        if sigma_norm > sigma0_max_norm:
            sigma0 = sigma0 * (sigma0_max_norm / sigma_norm)
        omega0 = rng.normal(0.0, omega0_std, size=3)
        x_i = _simulate_one_trajectory(
            bsk=bsk,
            sigma0=np.asarray(sigma0, dtype=np.float64),
            omega0=np.asarray(omega0, dtype=np.float64),
            inertia=inertia,
            disturbance_torque=disturbance,
            dt=dt,
            t_len=t_len,
        )
        x_all[i] = x_i

    corruption_extras: Dict[str, np.ndarray] = {}
    corruption_meta: Dict[str, Any] = {
        "enabled": False,
        "profile_id": "gaussian_only",
        "clean_measurement_reference": "x",
        "stats": {},
    }
    if corruption_enabled:
        y_all, corruption_extras, corruption_meta = apply_structured_measurement_corruption(
            x_all=x_all,
            cfg=corruption_cfg,
            sensor_std=sensor_std,
            suite_name=suite_name,
            task_id=task_cfg.task_id,
            scenario_id=scenario_id,
            seed=int(seed),
            dt=float(dt),
        )
    else:
        for i in range(n_total):
            noise = rng.normal(0.0, sensor_std, size=(t_len, y_dim))
            y_all[i] = x_all[i] + noise

    q2_t = np.full((t_len,), float(q2), dtype=np.float32)
    r2_t = np.full((t_len,), float(r2), dtype=np.float32)
    sow_t = q2_t / np.maximum(r2_t, np.float32(1.0e-12))

    meta_common: Dict[str, Any] = {
        "format_version": "0.1",
        "canonical_layout": "NTD",
        "schema_version": 1,
        "task_family": str(task_family),
        "suite_name": suite_name,
        "task_id": task_cfg.task_id,
        "scenario_id": scenario_id,
        "scenario_cfg": json.loads(json.dumps(scenario_cfg)),
        "seed": int(seed),
        "x_dim": int(x_dim),
        "y_dim": int(y_dim),
        "T": int(t_len),
        "control_input_u": False,
        "ground_truth": dict(task_cfg.ground_truth),
        "observation": {
            "type": "full_state_adcs",
            "h_type": "direct_sigma_omega",
            "H": "identity",
        },
        "corruption": corruption_meta,
        "measurement_corruption": corruption_meta,
        "noise": {
            "Q": {"type": "scaled_identity", "q2": float(q2)},
            "R": {"type": "scaled_identity", "r2": float(r2)},
            "sensor_noise_std": float(sensor_std),
        },
        "noise_schedule": {
            "enabled": False,
            "kind": "stationary",
            "q2_t": {"source": "npz:q2_t", "shape": [int(t_len)]},
            "r2_t": {"source": "npz:r2_t", "shape": [int(t_len)]},
            "SoW_t": {"source": "npz:SoW_t", "shape": [int(t_len)]},
            "SoW_hat_t": None,
            "params": {"q2_base": float(q2), "r2_base": float(r2)},
        },
        "ssm": {
            "true": {
                "framework": "AVS Basilisk",
                "system_type": "rigid_body_attitude",
                "dt": float(dt),
                "inertia": np.asarray(inertia, dtype=float).tolist(),
                "disturbance_torque_B_Nm": disturbance.astype(float).tolist(),
                "state": ["sigma_BN_1", "sigma_BN_2", "sigma_BN_3", "omega_BN_B_1", "omega_BN_B_2", "omega_BN_B_3"],
                "measurement": ["sigma_meas_1", "sigma_meas_2", "sigma_meas_3", "omega_meas_1", "omega_meas_2", "omega_meas_3"],
                "initial_condition": {
                    "sigma0_std": float(sigma0_std),
                    "sigma0_max_norm": float(sigma0_max_norm),
                    "omega0_std_rad_s": float(omega0_std),
                },
            },
            "assumed": {
                "system_type": "small_angle_linearized_attitude",
                "valid_for_oracle": False,
                "F": f_assumed.astype(float).tolist(),
                "H": h_assumed.astype(float).tolist(),
                "Q": q_assumed.astype(float).tolist(),
                "R": r_assumed.astype(float).tolist(),
                "note": "Approximate model supplied for adapters needing F/H/Q/R; not an oracle for Basilisk dynamics.",
            },
        },
        "mismatch": {
            "enabled": True,
            "kind": "nonlinear_basilisk_vs_small_angle_linearized_assumed_model",
            "params": {
                "oracle_supported": False,
                "sensor_noise_r2": float(r2),
            },
        },
        "switching": {
            "enabled": False,
            "models": [],
            "t_change": None,
            "retrain_window": 0,
        },
        "units": {
            "sigma_BN": "MRP dimensionless",
            "omega_BN_B": "rad/s",
            "torque": "N*m",
            "dt": "s",
        },
        "attitude_representation": "MRP",
    }

    out = coerce_ntd_float32_output(
        GeneratorOutput(
            x=x_all.astype(np.float32),
            y=y_all.astype(np.float32),
            meta=meta_common,
            extras={
                "q2_t": q2_t,
                "r2_t": r2_t,
                "SoW_t": sow_t.astype(np.float32),
                "sigma_true_seq": x_all[:, :, 0:3].astype(np.float32),
                "omega_true_seq": x_all[:, :, 3:6].astype(np.float32),
                "task_key": f"{task_family}:{task_cfg.task_id}:{scenario_id}",
                **corruption_extras,
            },
        )
    )
    return out, f_assumed, h_assumed
