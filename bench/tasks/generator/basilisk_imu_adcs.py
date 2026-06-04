from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from .basilisk_adcs import (
    _as_diag_inertia,
    _as_vec3,
    _deep_merge,
    _require_avs_basilisk,
    _resolve_q2,
    _shadow_mrp,
    _small_angle_model,
)
from .contract import GeneratorOutput, coerce_ntd_float32_output, make_split_cfg, make_task_cfg
from .datasets.common import DatasetMissingError
from ...utils.seeding import numpy_rng_v0, stable_int_seed_v0


_FIELD_SPECS: Dict[str, List[Tuple[str, str, str]]] = {
    "gyro_only": [
        ("AngVelPlatform", "gyro", "rad/s"),
    ],
    "gyro_accel": [
        ("AngVelPlatform", "gyro", "rad/s"),
        ("AccelPlatform", "accel", "m/s^2"),
    ],
    "gyro_delta_angle": [
        ("AngVelPlatform", "gyro", "rad/s"),
        ("DRFramePlatform", "delta_theta", "rad"),
    ],
    "full_imu": [
        ("AngVelPlatform", "gyro", "rad/s"),
        ("AccelPlatform", "accel", "m/s^2"),
        ("DRFramePlatform", "delta_theta", "rad"),
        ("DVFramePlatform", "delta_v", "m/s"),
    ],
}


def _require_imu_basilisk() -> Dict[str, Any]:
    modules = _require_avs_basilisk()
    try:
        from Basilisk.simulation import imuSensor  # type: ignore
    except Exception as exc:
        raise DatasetMissingError(
            dataset="AVS Basilisk imuSensor",
            env_var="PYTHONPATH",
            message=(
                "AVS Basilisk with Basilisk.simulation.imuSensor is required for "
                "task_family=basilisk_imu_adcs_v0. No synthetic IMU fallback is provided."
            ),
        ) from exc
    modules["imuSensor"] = imuSensor
    return modules


def _json_clone(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def _resolve_imu_cfg(resolved_task: Mapping[str, Any]) -> Dict[str, Any]:
    raw_obj = resolved_task.get("imu", {})
    if not isinstance(raw_obj, Mapping):
        raw_obj = {}
    raw = _json_clone(dict(raw_obj))
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, Mapping):
        profiles = {}
    profile_id = str(raw.get("profile_id", raw.get("severity", "clean_imu"))).strip() or "clean_imu"
    merged = dict(raw)
    merged.pop("profiles", None)
    if profile_id in profiles:
        profile_cfg = profiles[profile_id]
        if not isinstance(profile_cfg, Mapping):
            raise ValueError(f"imu profile {profile_id!r} must be a mapping")
        merged = _deep_merge(merged, dict(profile_cfg))
    elif profile_id != "clean_imu":
        raise ValueError(f"imu profile_id={profile_id!r} not found in imu.profiles")
    merged["profile_id"] = profile_id
    merged.setdefault("severity", profile_id)
    merged.setdefault("measurement_mode", raw.get("measurement_mode", "gyro_delta_angle"))
    merged.setdefault("sensor_frame", "platform")
    merged.setdefault("fake_marker", False)
    return merged


def _list3(value: Any, *, default: float = 0.0) -> List[List[float]]:
    if value is None:
        arr = np.full((3,), float(default), dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            arr = np.full((3,), float(arr[0]), dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"expected scalar or length-3 vector, got shape {arr.shape}")
    return [[float(v)] for v in arr]


def _diag3_from_std(std: float) -> List[List[float]]:
    s = float(std)
    return (np.eye(3, dtype=np.float64) * (s * s)).tolist()


def _std_from_cfg(cfg: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = cfg.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _trajectory_imu_cfg(
    *,
    base_cfg: Mapping[str, Any],
    suite_name: str,
    task_id: str,
    scenario_id: str,
    seed: int,
    trajectory_index: int,
) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    rng = numpy_rng_v0(
        stable_int_seed_v0("basilisk_imu_sensor_cfg", suite_name, task_id, scenario_id, int(seed), int(trajectory_index))
    )

    gyro_bias_std = _std_from_cfg(cfg, "gyro_bias_std", 0.0)
    accel_bias_std = _std_from_cfg(cfg, "accel_bias_std", 0.0)
    if gyro_bias_std > 0.0:
        cfg["gyro_bias"] = rng.normal(0.0, gyro_bias_std, size=3).astype(float).tolist()
    else:
        cfg.setdefault("gyro_bias", [0.0, 0.0, 0.0])
    if accel_bias_std > 0.0:
        cfg["accel_bias"] = rng.normal(0.0, accel_bias_std, size=3).astype(float).tolist()
    else:
        cfg.setdefault("accel_bias", [0.0, 0.0, 0.0])
    cfg["RNGSeed"] = int(stable_int_seed_v0("basilisk_imu_sensor_rng", suite_name, task_id, scenario_id, int(seed), int(trajectory_index)) % (2**31 - 1))
    return cfg


def _configure_imu_sensor(imu: Any, cfg: Mapping[str, Any]) -> None:
    imu.RNGSeed = int(cfg.get("RNGSeed", 1))
    imu.sensorPos_B = _list3(cfg.get("sensor_pos_B", [0.0, 0.0, 0.0]))
    if hasattr(imu, "setBodyToPlatformDCM"):
        dcm_euler = cfg.get("body_to_platform_euler321_rad", [0.0, 0.0, 0.0])
        arr = np.asarray(dcm_euler, dtype=np.float64).reshape(-1)
        if arr.shape != (3,):
            raise ValueError("imu.body_to_platform_euler321_rad must be length 3")
        imu.setBodyToPlatformDCM(float(arr[0]), float(arr[1]), float(arr[2]))

    imu.senRotBias = _list3(cfg.get("gyro_bias", [0.0, 0.0, 0.0]))
    imu.senTransBias = _list3(cfg.get("accel_bias", [0.0, 0.0, 0.0]))

    gyro_noise_std = _std_from_cfg(cfg, "gyro_noise_std", 0.0)
    accel_noise_std = _std_from_cfg(cfg, "accel_noise_std", 0.0)
    if gyro_noise_std > 0.0:
        imu.PMatrixGyro = _diag3_from_std(gyro_noise_std)
        imu.AMatrixGyro = np.eye(3, dtype=np.float64).tolist()
    if accel_noise_std > 0.0:
        imu.PMatrixAccel = _diag3_from_std(accel_noise_std)
        imu.AMatrixAccel = np.eye(3, dtype=np.float64).tolist()

    gyro_walk = _std_from_cfg(cfg, "gyro_walk_bound", 0.0)
    accel_walk = _std_from_cfg(cfg, "accel_walk_bound", 0.0)
    if gyro_walk > 0.0:
        imu.setWalkBoundsGyro(_list3(gyro_walk))
    if accel_walk > 0.0:
        imu.setWalkBoundsAccel(_list3(accel_walk))

    gyro_error_bound = _std_from_cfg(cfg, "gyro_error_bound", 0.0)
    accel_error_bound = _std_from_cfg(cfg, "accel_error_bound", 0.0)
    if gyro_error_bound > 0.0:
        imu.setErrorBoundsGyro(_list3(gyro_error_bound))
    if accel_error_bound > 0.0:
        imu.setErrorBoundsAccel(_list3(accel_error_bound))

    gyro_sat = cfg.get("gyro_saturation", None)
    accel_sat = cfg.get("accel_saturation", None)
    if gyro_sat is not None:
        imu.set_oSatBounds(_list3(gyro_sat))
    if accel_sat is not None:
        imu.set_aSatBounds(_list3(accel_sat))

    accel_lsb = cfg.get("accel_lsb", None)
    gyro_lsb = cfg.get("gyro_lsb", None)
    if accel_lsb is not None or gyro_lsb is not None:
        imu.setLSBs(float(accel_lsb or 0.0), float(gyro_lsb or 0.0))


def _extract_field(recorder: Any, field: str, t_len: int) -> np.ndarray:
    arr = np.asarray(getattr(recorder, field), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < t_len:
        raise RuntimeError(f"Basilisk IMU recorder field {field} returned shape {arr.shape}, expected at least [T,3]")
    return arr[:t_len]


def _select_y(recorder: Any, mode: str, t_len: int) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[Dict[str, Any]]]:
    if mode not in _FIELD_SPECS:
        raise ValueError(f"unsupported IMU measurement_mode={mode!r}; supported={sorted(_FIELD_SPECS)}")
    arrays: Dict[str, np.ndarray] = {
        "gyro": _extract_field(recorder, "AngVelPlatform", t_len),
        "accel": _extract_field(recorder, "AccelPlatform", t_len),
        "delta_theta": _extract_field(recorder, "DRFramePlatform", t_len),
        "delta_v": _extract_field(recorder, "DVFramePlatform", t_len),
    }
    chunks: List[np.ndarray] = []
    mapping: List[Dict[str, Any]] = []
    col0 = 0
    for field, alias, units in _FIELD_SPECS[mode]:
        arr = _extract_field(recorder, field, t_len)
        chunks.append(arr)
        mapping.append({"columns": [col0, col0 + 1, col0 + 2], "field": field, "alias": alias, "units": units})
        col0 += 3
    return np.concatenate(chunks, axis=1), arrays, mapping


def _assumed_imu_model(*, mode: str, dt: float, q2: float, imu_cfg: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    f, _, q, _ = _small_angle_model(dt=dt, q2=q2, r2=1.0e-6)
    y_dim = 3 * len(_FIELD_SPECS[mode])
    h = np.zeros((y_dim, 6), dtype=np.float64)
    variances: List[float] = []
    gyro_std = max(_std_from_cfg(imu_cfg, "gyro_noise_std", 0.0), _std_from_cfg(imu_cfg, "gyro_bias_std", 0.0), 1.0e-6)
    accel_std = max(_std_from_cfg(imu_cfg, "accel_noise_std", 0.0), _std_from_cfg(imu_cfg, "accel_bias_std", 0.0), 1.0e-6)
    col = 0
    for _, alias, _units in _FIELD_SPECS[mode]:
        if alias == "gyro":
            h[col : col + 3, 3:6] = np.eye(3, dtype=np.float64)
            variances.extend([gyro_std * gyro_std] * 3)
        elif alias == "delta_theta":
            h[col : col + 3, 3:6] = float(dt) * np.eye(3, dtype=np.float64)
            variances.extend([(float(dt) * gyro_std) ** 2] * 3)
        else:
            variances.extend([accel_std * accel_std] * 3)
        col += 3
    r = np.diag(np.asarray(variances, dtype=np.float64))
    r2_scalar = float(np.mean(np.diag(r)))
    return f.astype(np.float32), h.astype(np.float32), q.astype(np.float32), r.astype(np.float32), r2_scalar


def _imu_h_metadata(*, mode: str, dt: float, h: np.ndarray, imu_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    h_arr = np.asarray(h, dtype=np.float64)
    platform_euler = np.asarray(imu_cfg.get("body_to_platform_euler321_rad", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
    platform_identity = bool(platform_euler.shape == (3,) and np.allclose(platform_euler, 0.0))
    if mode == "gyro_delta_angle":
        return {
            "measurement_model": "gyro_delta_simple",
            "h_type": "imu_gyro_delta",
            "H": h_arr.astype(float).tolist(),
            "H_rank": int(np.linalg.matrix_rank(h_arr)),
            "direct_observation": False,
            "attitude_directly_observed": False,
            "gyro_model": "AngVelPlatform ~= omega",
            "delta_model": f"DRFramePlatform ~= omega * dt (dt={float(dt):.9g})",
            "platform_frame_identity": platform_identity,
        }
    return {
        "measurement_model": f"imu_sensor_packet:{mode}",
        "h_type": "imu_sensor_packet",
        "H": h_arr.astype(float).tolist(),
        "H_rank": int(np.linalg.matrix_rank(h_arr)),
        "direct_observation": False,
        "attitude_directly_observed": False,
        "gyro_model": "AngVelPlatform ~= omega when gyro field is present",
        "delta_model": "mode does not use the audited gyro_delta_simple packet",
        "platform_frame_identity": platform_identity,
    }


def _resolve_bias_state_cfg(resolved_task: Mapping[str, Any]) -> Dict[str, Any]:
    raw_obj = resolved_task.get("bias_state", resolved_task.get("imu_bias", {}))
    if not isinstance(raw_obj, Mapping):
        raw_obj = {}
    raw = _json_clone(dict(raw_obj))
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, Mapping):
        profiles = {}
    profile_id = str(raw.get("profile_id", raw.get("severity", "clean_bias"))).strip() or "clean_bias"
    merged = dict(raw)
    merged.pop("profiles", None)
    if profile_id in profiles:
        profile_cfg = profiles[profile_id]
        if not isinstance(profile_cfg, Mapping):
            raise ValueError(f"bias_state profile {profile_id!r} must be a mapping")
        merged = _deep_merge(merged, dict(profile_cfg))
    elif profile_id != "clean_bias":
        raise ValueError(f"bias_state profile_id={profile_id!r} not found in bias_state.profiles")
    merged["profile_id"] = profile_id
    merged.setdefault("severity", profile_id)
    merged.setdefault("bias_init_std", 0.0)
    merged.setdefault("bias_rw_std", 0.0)
    merged.setdefault("gyro_noise_std", 0.0)
    merged.setdefault("delta_noise_std", float(merged.get("gyro_noise_std", 0.0)) * float(resolved_task.get("simulation", {}).get("dt", 0.1)))
    return merged


def _assumed_imu_bias_model(
    *,
    dt: float,
    q2: float,
    bias_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    f6, _, q6, _ = _small_angle_model(dt=dt, q2=q2, r2=1.0e-6)
    f = np.eye(9, dtype=np.float64)
    f[0:6, 0:6] = np.asarray(f6, dtype=np.float64)
    h = np.zeros((6, 9), dtype=np.float64)
    h[0:3, 3:6] = np.eye(3, dtype=np.float64)
    h[0:3, 6:9] = np.eye(3, dtype=np.float64)
    h[3:6, 3:6] = float(dt) * np.eye(3, dtype=np.float64)
    h[3:6, 6:9] = float(dt) * np.eye(3, dtype=np.float64)

    q = np.eye(9, dtype=np.float64) * float(q2)
    q[0:6, 0:6] = np.asarray(q6, dtype=np.float64)
    bias_rw_std = float(bias_cfg.get("bias_rw_std", 0.0) or 0.0)
    q_bias = max((bias_rw_std * bias_rw_std) * float(dt), 1.0e-12)
    q[6:9, 6:9] = np.eye(3, dtype=np.float64) * q_bias

    gyro_noise_std = float(bias_cfg.get("gyro_noise_std", 0.0) or 0.0)
    delta_noise_std = float(bias_cfg.get("delta_noise_std", gyro_noise_std * float(dt)) or 0.0)
    variances = [max(gyro_noise_std * gyro_noise_std, 1.0e-12)] * 3
    variances.extend([max(delta_noise_std * delta_noise_std, 1.0e-12)] * 3)
    r = np.diag(np.asarray(variances, dtype=np.float64))
    r2_scalar = float(np.mean(np.diag(r)))
    return f.astype(np.float32), h.astype(np.float32), q.astype(np.float32), r.astype(np.float32), r2_scalar


def _resolve_sparse_ref_cfg(resolved_task: Mapping[str, Any]) -> Dict[str, Any]:
    raw_obj = resolved_task.get("sparse_ref", resolved_task.get("attitude_ref", {}))
    if not isinstance(raw_obj, Mapping):
        raw_obj = {}
    raw = _json_clone(dict(raw_obj))
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, Mapping):
        profiles = {}
    profile_id = str(raw.get("profile_id", raw.get("severity", "mild_bias_ref"))).strip() or "mild_bias_ref"
    if profile_id in {"same_as_bias_state", "match_bias_state"}:
        bias_obj = resolved_task.get("bias_state", resolved_task.get("imu_bias", {}))
        if isinstance(bias_obj, Mapping):
            profile_id = str(bias_obj.get("profile_id", bias_obj.get("severity", "mild_bias_ref"))).strip() or "mild_bias_ref"
    merged = dict(raw)
    merged.pop("profiles", None)
    if profile_id in profiles:
        profile_cfg = profiles[profile_id]
        if not isinstance(profile_cfg, Mapping):
            raise ValueError(f"sparse_ref profile {profile_id!r} must be a mapping")
        merged = _deep_merge(merged, dict(profile_cfg))
    elif profile_id not in {"default", "mild_bias_ref"}:
        raise ValueError(f"sparse_ref profile_id={profile_id!r} not found in sparse_ref.profiles")
    merged["profile_id"] = profile_id
    merged.setdefault("severity", profile_id)
    merged.setdefault("ref_update_period", 10)
    merged.setdefault("ref_noise_std", 0.005)
    merged.setdefault("ref_dropout_prob", 0.0)
    merged.setdefault("ref_mask_mode", "periodic")
    merged.setdefault("missing_ref_value", "zero")
    merged.setdefault("mask_strategy", "zero_ref_rows_and_measurement_mask")
    return merged


def _assumed_imu_sparse_ref_model(
    *,
    dt: float,
    q2: float,
    bias_cfg: Mapping[str, Any],
    ref_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    f, _h_bias, q, _r_bias, _r2_bias = _assumed_imu_bias_model(dt=dt, q2=q2, bias_cfg=bias_cfg)
    h = np.zeros((9, 9), dtype=np.float64)
    h[0:3, 3:6] = np.eye(3, dtype=np.float64)
    h[0:3, 6:9] = np.eye(3, dtype=np.float64)
    h[3:6, 3:6] = float(dt) * np.eye(3, dtype=np.float64)
    h[3:6, 6:9] = float(dt) * np.eye(3, dtype=np.float64)
    h[6:9, 0:3] = np.eye(3, dtype=np.float64)

    gyro_noise_std = float(bias_cfg.get("gyro_noise_std", 0.0) or 0.0)
    delta_noise_std = float(bias_cfg.get("delta_noise_std", gyro_noise_std * float(dt)) or 0.0)
    ref_noise_std = float(ref_cfg.get("ref_noise_std", 0.005) or 0.0)
    variances = [max(gyro_noise_std * gyro_noise_std, 1.0e-12)] * 3
    variances.extend([max(delta_noise_std * delta_noise_std, 1.0e-12)] * 3)
    variances.extend([max(ref_noise_std * ref_noise_std, 1.0e-12)] * 3)
    r = np.diag(np.asarray(variances, dtype=np.float64))
    r2_scalar = float(np.mean(np.diag(r)))
    return f.astype(np.float32), h.astype(np.float32), q.astype(np.float32), r.astype(np.float32), r2_scalar


def _mean_step_norm(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return 0.0
    if a.ndim == 3:
        return float(np.mean(np.linalg.norm(a, axis=2)))
    if a.ndim == 2:
        return float(np.mean(np.linalg.norm(a, axis=1)))
    return float(np.linalg.norm(a.reshape(-1)))


def _simulate_one_imu_trajectory(
    *,
    bsk: Mapping[str, Any],
    sigma0: np.ndarray,
    omega0: np.ndarray,
    inertia: np.ndarray,
    disturbance_torque: np.ndarray,
    dt: float,
    t_len: int,
    imu_cfg: Mapping[str, Any],
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict[str, Any]]]:
    SimulationBaseClass = bsk["SimulationBaseClass"]
    macros = bsk["macros"]
    spacecraft = bsk["spacecraft"]
    imuSensor = bsk["imuSensor"]

    sim = SimulationBaseClass.SimBaseClass()
    process_name = "dynProcess"
    task_name = "dynTask"
    sim_task = sim.CreateNewTask(task_name, macros.sec2nano(float(dt)))
    sim.CreateNewProcess(process_name).addTask(sim_task)

    sc = spacecraft.Spacecraft()
    sc.ModelTag = "basilisk_imu_spacecraft"
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

    imu_clean = imuSensor.ImuSensor()
    imu_clean.ModelTag = "basilisk_imu_clean"
    imu_clean.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    _configure_imu_sensor(imu_clean, {"RNGSeed": 1, **{k: imu_cfg[k] for k in ("sensor_pos_B", "body_to_platform_euler321_rad") if k in imu_cfg}})

    imu_meas = imuSensor.ImuSensor()
    imu_meas.ModelTag = "basilisk_imu_measured"
    imu_meas.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    _configure_imu_sensor(imu_meas, imu_cfg)

    sim.AddModelToTask(task_name, sc)
    sim.AddModelToTask(task_name, imu_clean)
    sim.AddModelToTask(task_name, imu_meas)

    sc_rec = sc.scStateOutMsg.recorder(macros.sec2nano(float(dt)))
    clean_rec = imu_clean.sensorOutMsg.recorder(macros.sec2nano(float(dt)))
    meas_rec = imu_meas.sensorOutMsg.recorder(macros.sec2nano(float(dt)))
    sim.AddModelToTask(task_name, sc_rec)
    sim.AddModelToTask(task_name, clean_rec)
    sim.AddModelToTask(task_name, meas_rec)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(float(dt) * max(0, int(t_len) - 1)))
    sim.ExecuteSimulation()

    sigma = np.asarray(sc_rec.sigma_BN, dtype=np.float64)
    omega = np.asarray(sc_rec.omega_BN_B, dtype=np.float64)
    if sigma.shape[0] < t_len or omega.shape[0] < t_len:
        raise RuntimeError(f"Basilisk recorder returned too few samples: sigma={sigma.shape}, omega={omega.shape}, T={t_len}")
    x = np.concatenate([sigma[:t_len], omega[:t_len]], axis=1)
    x[:, 0:3] = np.asarray([_shadow_mrp(row) for row in x[:, 0:3]], dtype=np.float64)

    y, raw, mapping = _select_y(meas_rec, mode, t_len)
    y_clean, clean_raw, _ = _select_y(clean_rec, mode, t_len)
    return x, y, raw, {"selected_y": y_clean, **clean_raw}, mapping


def generate_basilisk_imu_adcs_v0(
    *,
    suite_name: str,
    task_cfg_dict: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    seed: int,
    scenario_id: str,
    task_family: str = "basilisk_imu_adcs_v0",
) -> Tuple[GeneratorOutput, np.ndarray, np.ndarray]:
    bsk = _require_imu_basilisk()
    resolved_task = _deep_merge(dict(task_cfg_dict), scenario_cfg)
    task_cfg = make_task_cfg(resolved_task, scenario_cfg=scenario_cfg)
    split_cfg = make_split_cfg(resolved_task)

    x_dim = int(task_cfg.x_dim)
    y_dim = int(task_cfg.y_dim)
    t_len = int(task_cfg.sequence_length_T)
    n_total = int(split_cfg.n_total)
    if x_dim != 6:
        raise ValueError(f"basilisk_imu_adcs_v0 requires x_dim=6, got x_dim={x_dim}")

    imu_cfg = _resolve_imu_cfg(resolved_task)
    mode = str(imu_cfg.get("measurement_mode", "gyro_delta_angle"))
    if mode not in _FIELD_SPECS:
        raise ValueError(f"unsupported IMU measurement_mode={mode!r}; supported={sorted(_FIELD_SPECS)}")
    expected_y_dim = 3 * len(_FIELD_SPECS[mode])
    if y_dim != expected_y_dim:
        raise ValueError(f"basilisk_imu_adcs_v0 mode={mode} requires y_dim={expected_y_dim}, got y_dim={y_dim}")

    sim_cfg = dict(resolved_task.get("simulation", {}) or {})
    dt = float(sim_cfg.get("dt", 0.1))
    inertia = _as_diag_inertia(sim_cfg.get("inertia", [10.0, 8.0, 6.0]))
    disturbance = _as_vec3(sim_cfg.get("disturbance_torque", [0.0, 0.0, 0.0]), name="disturbance_torque")
    sigma0_std = float(sim_cfg.get("sigma0_std", 0.05))
    omega0_std = float(sim_cfg.get("omega0_std", 0.01))
    sigma0_max_norm = float(sim_cfg.get("sigma0_max_norm", 0.25))

    q2 = _resolve_q2(resolved_task, scenario_cfg)
    f_assumed, h_assumed, q_assumed, r_assumed, r2_scalar = _assumed_imu_model(
        mode=mode,
        dt=dt,
        q2=q2,
        imu_cfg=imu_cfg,
    )
    h_meta = _imu_h_metadata(mode=mode, dt=dt, h=h_assumed, imu_cfg=imu_cfg)

    data_seed = stable_int_seed_v0("basilisk_imu_adcs_data", suite_name, task_cfg.task_id, scenario_id, int(seed))
    rng = numpy_rng_v0(data_seed)
    x_all = np.zeros((n_total, t_len, x_dim), dtype=np.float64)
    y_all = np.zeros((n_total, t_len, y_dim), dtype=np.float64)
    clean_y_all = np.zeros_like(y_all)
    gyro_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    accel_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    delta_theta_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    delta_v_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    clean_gyro_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    clean_accel_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    clean_delta_theta_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    clean_delta_v_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    per_traj_biases: List[Dict[str, Any]] = []
    mapping: List[Dict[str, Any]] = []

    for i in range(n_total):
        sigma0 = rng.normal(0.0, sigma0_std, size=3)
        sigma_norm = float(np.linalg.norm(sigma0))
        if sigma_norm > sigma0_max_norm:
            sigma0 = sigma0 * (sigma0_max_norm / sigma_norm)
        omega0 = rng.normal(0.0, omega0_std, size=3)
        traj_cfg = _trajectory_imu_cfg(
            base_cfg=imu_cfg,
            suite_name=suite_name,
            task_id=task_cfg.task_id,
            scenario_id=scenario_id,
            seed=int(seed),
            trajectory_index=i,
        )
        x_i, y_i, raw_i, clean_raw_i, mapping = _simulate_one_imu_trajectory(
            bsk=bsk,
            sigma0=np.asarray(sigma0, dtype=np.float64),
            omega0=np.asarray(omega0, dtype=np.float64),
            inertia=inertia,
            disturbance_torque=disturbance,
            dt=dt,
            t_len=t_len,
            imu_cfg=traj_cfg,
            mode=mode,
        )
        x_all[i] = x_i
        y_all[i] = y_i
        clean_y_all[i] = clean_raw_i["selected_y"]
        gyro_all[i] = raw_i["gyro"]
        accel_all[i] = raw_i["accel"]
        delta_theta_all[i] = raw_i["delta_theta"]
        delta_v_all[i] = raw_i["delta_v"]
        clean_gyro_all[i] = clean_raw_i["gyro"]
        clean_accel_all[i] = clean_raw_i["accel"]
        clean_delta_theta_all[i] = clean_raw_i["delta_theta"]
        clean_delta_v_all[i] = clean_raw_i["delta_v"]
        per_traj_biases.append(
            {
                "gyro_bias": traj_cfg.get("gyro_bias", [0.0, 0.0, 0.0]),
                "accel_bias": traj_cfg.get("accel_bias", [0.0, 0.0, 0.0]),
            }
        )

    imu_error = y_all - clean_y_all
    q2_t = np.full((t_len,), float(q2), dtype=np.float32)
    r2_t = np.full((t_len,), float(r2_scalar), dtype=np.float32)
    sow_t = q2_t / np.maximum(r2_t, np.float32(1.0e-12))

    clean_y_norm = max(_mean_step_norm(clean_y_all), 1.0e-12)
    imu_stats = {
        "selected_y_norm_mean": _mean_step_norm(y_all),
        "clean_y_norm_mean": _mean_step_norm(clean_y_all),
        "imu_error_norm_mean": _mean_step_norm(imu_error),
        "imu_error_to_clean_ratio": float(_mean_step_norm(imu_error) / clean_y_norm),
        "gyro_norm_mean": _mean_step_norm(gyro_all),
        "accel_norm_mean": _mean_step_norm(accel_all),
        "delta_theta_norm_mean": _mean_step_norm(delta_theta_all),
        "delta_v_norm_mean": _mean_step_norm(delta_v_all),
        "accel_channel_near_zero": bool(_mean_step_norm(clean_accel_all) < 1.0e-12),
        "delta_v_channel_near_zero": bool(_mean_step_norm(clean_delta_v_all) < 1.0e-12),
    }

    sensor_fields = [spec[0] for spec in _FIELD_SPECS[mode]]
    meta_common: Dict[str, Any] = {
        "format_version": "0.1",
        "canonical_layout": "NTD",
        "schema_version": 1,
        "task_family": str(task_family),
        "suite_name": suite_name,
        "task_id": task_cfg.task_id,
        "scenario_id": scenario_id,
        "scenario_cfg": _json_clone(scenario_cfg),
        "seed": int(seed),
        "fake_marker": False,
        "x_dim": int(x_dim),
        "y_dim": int(y_dim),
        "T": int(t_len),
        "control_input_u": False,
        "ground_truth": dict(task_cfg.ground_truth),
        "observation": {
            "type": "basilisk_imu_sensor",
            "h_type": h_meta["h_type"],
            "measurement_mode": mode,
            "output_fields": sensor_fields,
            "field_mapping": mapping,
            "contains_absolute_attitude": False,
            "observability_note": "IMU gyro/delta-angle do not directly observe absolute MRP attitude.",
        },
        "imu_config": {
            "profile_id": str(imu_cfg.get("profile_id", "clean_imu")),
            "severity": str(imu_cfg.get("severity", imu_cfg.get("profile_id", "clean_imu"))),
            "measurement_mode": mode,
            "sensor_frame": str(imu_cfg.get("sensor_frame", "platform")),
            "platform_frame": "P",
            "sensorPos_B": _json_clone(imu_cfg.get("sensor_pos_B", [0.0, 0.0, 0.0])),
            "body_to_platform_euler321_rad": _json_clone(imu_cfg.get("body_to_platform_euler321_rad", [0.0, 0.0, 0.0])),
            "output_fields": sensor_fields,
            "noise": {
                "gyro_noise_std": float(_std_from_cfg(imu_cfg, "gyro_noise_std", 0.0)),
                "accel_noise_std": float(_std_from_cfg(imu_cfg, "accel_noise_std", 0.0)),
                "gyro_walk_bound": float(_std_from_cfg(imu_cfg, "gyro_walk_bound", 0.0)),
                "accel_walk_bound": float(_std_from_cfg(imu_cfg, "accel_walk_bound", 0.0)),
            },
            "bias": {
                "gyro_bias_std": float(_std_from_cfg(imu_cfg, "gyro_bias_std", 0.0)),
                "accel_bias_std": float(_std_from_cfg(imu_cfg, "accel_bias_std", 0.0)),
                "per_trajectory": True,
                "sampled_biases": per_traj_biases[: min(8, len(per_traj_biases))],
            },
            "saturation": {
                "gyro_saturation": imu_cfg.get("gyro_saturation"),
                "accel_saturation": imu_cfg.get("accel_saturation"),
            },
            "discretization": {
                "gyro_lsb": imu_cfg.get("gyro_lsb"),
                "accel_lsb": imu_cfg.get("accel_lsb"),
            },
            "stats": imu_stats,
        },
        "imu_output_fields": mapping,
        "noise": {
            "Q": {"type": "scaled_identity", "q2": float(q2)},
            "R": {"type": "imu_profile_mean_variance", "r2": float(r2_scalar)},
            "sensor_noise_std": float(math.sqrt(max(r2_scalar, 0.0))),
        },
        "noise_schedule": {
            "enabled": False,
            "kind": "stationary",
            "q2_t": {"source": "npz:q2_t", "shape": [int(t_len)]},
            "r2_t": {"source": "npz:r2_t", "shape": [int(t_len)]},
            "SoW_t": {"source": "npz:SoW_t", "shape": [int(t_len)]},
            "SoW_hat_t": None,
            "params": {"q2_base": float(q2), "r2_base": float(r2_scalar)},
        },
        "ssm": {
            "true": {
                "framework": "AVS Basilisk",
                "system_type": "rigid_body_attitude_with_imu_sensor",
                "dt": float(dt),
                "imu_process_rate_s": float(dt),
                "inertia": np.asarray(inertia, dtype=float).tolist(),
                "disturbance_torque_B_Nm": disturbance.astype(float).tolist(),
                "state": ["sigma_BN_1", "sigma_BN_2", "sigma_BN_3", "omega_BN_B_1", "omega_BN_B_2", "omega_BN_B_3"],
                "measurement": [f"{item['alias']}_{j + 1}" for item in mapping for j in range(3)],
                "initial_condition": {
                    "sigma0_std": float(sigma0_std),
                    "sigma0_max_norm": float(sigma0_max_norm),
                    "omega0_std_rad_s": float(omega0_std),
                },
            },
            "assumed": {
                "system_type": "small_angle_linearized_attitude_with_imu_measurement_projection",
                "valid_for_oracle": False,
                "F": f_assumed.astype(float).tolist(),
                "Q": q_assumed.astype(float).tolist(),
                "R": r_assumed.astype(float).tolist(),
                **h_meta,
                "note": "Approximate model supplied for neural adapters needing F/H/Q/R; not an oracle.",
            },
        },
        "mismatch": {
            "enabled": True,
            "kind": "imu_partial_observation_without_absolute_attitude_reference",
            "params": {
                "oracle_supported": False,
                "measurement_mode": mode,
                "accel_informative": not bool(imu_stats["accel_channel_near_zero"]),
                "delta_v_informative": not bool(imu_stats["delta_v_channel_near_zero"]),
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
            "AngVelPlatform": "rad/s",
            "AccelPlatform": "m/s^2",
            "DRFramePlatform": "rad",
            "DVFramePlatform": "m/s",
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
                "imu_gyro_seq": gyro_all.astype(np.float32),
                "imu_accel_seq": accel_all.astype(np.float32),
                "imu_delta_theta_seq": delta_theta_all.astype(np.float32),
                "imu_delta_v_seq": delta_v_all.astype(np.float32),
                "imu_clean_gyro_seq": clean_gyro_all.astype(np.float32),
                "imu_clean_accel_seq": clean_accel_all.astype(np.float32),
                "imu_clean_delta_theta_seq": clean_delta_theta_all.astype(np.float32),
                "imu_clean_delta_v_seq": clean_delta_v_all.astype(np.float32),
                "imu_clean_y_seq": clean_y_all.astype(np.float32),
                "imu_error_seq": imu_error.astype(np.float32),
                "task_key": f"{task_family}:{task_cfg.task_id}:{scenario_id}",
            },
        )
    )
    return out, f_assumed, h_assumed


def generate_basilisk_imu_bias_adcs_v0(
    *,
    suite_name: str,
    task_cfg_dict: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    seed: int,
    scenario_id: str,
    task_family: str = "basilisk_imu_bias_adcs_v0",
) -> Tuple[GeneratorOutput, np.ndarray, np.ndarray]:
    bsk = _require_imu_basilisk()
    resolved_task = _deep_merge(dict(task_cfg_dict), scenario_cfg)
    task_cfg = make_task_cfg(resolved_task, scenario_cfg=scenario_cfg)
    split_cfg = make_split_cfg(resolved_task)

    x_dim = int(task_cfg.x_dim)
    y_dim = int(task_cfg.y_dim)
    t_len = int(task_cfg.sequence_length_T)
    n_total = int(split_cfg.n_total)
    if x_dim != 9:
        raise ValueError(f"basilisk_imu_bias_adcs_v0 requires x_dim=9, got x_dim={x_dim}")
    if y_dim != 6:
        raise ValueError(f"basilisk_imu_bias_adcs_v0 requires y_dim=6, got y_dim={y_dim}")

    imu_cfg = _resolve_imu_cfg(resolved_task)
    mode = str(imu_cfg.get("measurement_mode", "gyro_delta_angle"))
    if mode != "gyro_delta_angle":
        raise ValueError("basilisk_imu_bias_adcs_v0 currently supports measurement_mode='gyro_delta_angle' only")
    clean_imu_cfg = dict(imu_cfg)
    clean_imu_cfg.update(
        {
            "profile_id": "clean_imu_base",
            "severity": "clean_imu_base",
            "gyro_noise_std": 0.0,
            "accel_noise_std": 0.0,
            "gyro_bias_std": 0.0,
            "accel_bias_std": 0.0,
            "gyro_walk_bound": 0.0,
            "accel_walk_bound": 0.0,
            "gyro_error_bound": 0.0,
            "accel_error_bound": 0.0,
            "gyro_lsb": None,
            "accel_lsb": None,
            "gyro_saturation": None,
            "accel_saturation": None,
        }
    )
    bias_cfg = _resolve_bias_state_cfg(resolved_task)

    sim_cfg = dict(resolved_task.get("simulation", {}) or {})
    dt = float(sim_cfg.get("dt", 0.1))
    inertia = _as_diag_inertia(sim_cfg.get("inertia", [10.0, 8.0, 6.0]))
    disturbance = _as_vec3(sim_cfg.get("disturbance_torque", [0.0, 0.0, 0.0]), name="disturbance_torque")
    sigma0_std = float(sim_cfg.get("sigma0_std", 0.05))
    omega0_std = float(sim_cfg.get("omega0_std", 0.01))
    sigma0_max_norm = float(sim_cfg.get("sigma0_max_norm", 0.25))

    q2 = _resolve_q2(resolved_task, scenario_cfg)
    f_assumed, h_assumed, q_assumed, r_assumed, r2_scalar = _assumed_imu_bias_model(
        dt=dt,
        q2=q2,
        bias_cfg=bias_cfg,
    )
    h_rank = int(np.linalg.matrix_rank(np.asarray(h_assumed, dtype=np.float64)))

    data_seed = stable_int_seed_v0("basilisk_imu_bias_adcs_data", suite_name, task_cfg.task_id, scenario_id, int(seed))
    rng = numpy_rng_v0(data_seed)
    x_all = np.zeros((n_total, t_len, x_dim), dtype=np.float64)
    y_all = np.zeros((n_total, t_len, y_dim), dtype=np.float64)
    clean_y_all = np.zeros((n_total, t_len, y_dim), dtype=np.float64)
    bias_seq_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    bias_component_all = np.zeros((n_total, t_len, y_dim), dtype=np.float64)
    noise_component_all = np.zeros((n_total, t_len, y_dim), dtype=np.float64)
    clean_bias_y_all = np.zeros((n_total, t_len, y_dim), dtype=np.float64)
    clean_gyro_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    clean_delta_theta_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    mapping: List[Dict[str, Any]] = []

    bias_init_std = float(bias_cfg.get("bias_init_std", 0.0) or 0.0)
    bias_rw_std = float(bias_cfg.get("bias_rw_std", 0.0) or 0.0)
    gyro_noise_std = float(bias_cfg.get("gyro_noise_std", 0.0) or 0.0)
    delta_noise_std = float(bias_cfg.get("delta_noise_std", gyro_noise_std * dt) or 0.0)

    for i in range(n_total):
        sigma0 = rng.normal(0.0, sigma0_std, size=3)
        sigma_norm = float(np.linalg.norm(sigma0))
        if sigma_norm > sigma0_max_norm:
            sigma0 = sigma0 * (sigma0_max_norm / sigma_norm)
        omega0 = rng.normal(0.0, omega0_std, size=3)
        traj_cfg = _trajectory_imu_cfg(
            base_cfg=clean_imu_cfg,
            suite_name=suite_name,
            task_id=task_cfg.task_id,
            scenario_id=scenario_id,
            seed=int(seed),
            trajectory_index=i,
        )
        x6_i, _y_clean_i, _raw_i, clean_raw_i, mapping = _simulate_one_imu_trajectory(
            bsk=bsk,
            sigma0=np.asarray(sigma0, dtype=np.float64),
            omega0=np.asarray(omega0, dtype=np.float64),
            inertia=inertia,
            disturbance_torque=disturbance,
            dt=dt,
            t_len=t_len,
            imu_cfg=traj_cfg,
            mode=mode,
        )
        clean_y = np.asarray(clean_raw_i["selected_y"], dtype=np.float64)
        traj_rng = numpy_rng_v0(
            stable_int_seed_v0("basilisk_imu_bias_process", suite_name, task_cfg.task_id, scenario_id, int(seed), i)
        )
        bias_seq = np.zeros((t_len, 3), dtype=np.float64)
        if bias_init_std > 0.0:
            bias_seq[0] = traj_rng.normal(0.0, bias_init_std, size=3)
        if t_len > 1 and bias_rw_std > 0.0:
            steps = traj_rng.normal(0.0, bias_rw_std * math.sqrt(float(dt)), size=(t_len - 1, 3))
            bias_seq[1:] = bias_seq[0] + np.cumsum(steps, axis=0)
        elif t_len > 1:
            bias_seq[1:] = bias_seq[0]

        noise = np.zeros((t_len, 6), dtype=np.float64)
        if gyro_noise_std > 0.0:
            noise[:, 0:3] = traj_rng.normal(0.0, gyro_noise_std, size=(t_len, 3))
        if delta_noise_std > 0.0:
            noise[:, 3:6] = traj_rng.normal(0.0, delta_noise_std, size=(t_len, 3))

        bias_component = np.concatenate([bias_seq, bias_seq * float(dt)], axis=1)
        clean_bias_y = clean_y + bias_component
        y_i = clean_bias_y + noise
        x_all[i] = np.concatenate([x6_i, bias_seq], axis=1)
        y_all[i] = y_i
        clean_y_all[i] = clean_y
        bias_seq_all[i] = bias_seq
        bias_component_all[i] = bias_component
        noise_component_all[i] = noise
        clean_bias_y_all[i] = clean_bias_y
        clean_gyro_all[i] = clean_raw_i["gyro"]
        clean_delta_theta_all[i] = clean_raw_i["delta_theta"]

    imu_error = y_all - clean_y_all
    q2_t = np.full((t_len,), float(q2), dtype=np.float32)
    r2_t = np.full((t_len,), float(r2_scalar), dtype=np.float32)
    sow_t = q2_t / np.maximum(r2_t, np.float32(1.0e-12))

    clean_y_norm = max(_mean_step_norm(clean_y_all), 1.0e-12)
    omega_norm = max(_mean_step_norm(x_all[:, :, 3:6]), 1.0e-12)
    bias_stats = {
        "profile_id": str(bias_cfg.get("profile_id", "clean_bias")),
        "severity": str(bias_cfg.get("severity", bias_cfg.get("profile_id", "clean_bias"))),
        "bias_norm_mean": _mean_step_norm(bias_seq_all),
        "bias_component_norm_mean": _mean_step_norm(bias_component_all),
        "noise_component_norm_mean": _mean_step_norm(noise_component_all),
        "imu_error_norm_mean": _mean_step_norm(imu_error),
        "imu_error_to_clean_ratio": float(_mean_step_norm(imu_error) / clean_y_norm),
        "bias_to_omega_ratio": float(_mean_step_norm(bias_seq_all) / omega_norm),
        "gyro_noise_std": float(gyro_noise_std),
        "delta_noise_std": float(delta_noise_std),
        "bias_init_std": float(bias_init_std),
        "bias_rw_std": float(bias_rw_std),
    }

    sensor_fields = [spec[0] for spec in _FIELD_SPECS[mode]]
    h_model = {
        "measurement_model": "gyro_delta_bias",
        "h_type": "imu_gyro_delta_bias",
        "H": h_assumed.astype(float).tolist(),
        "H_rank": h_rank,
        "direct_observation": False,
        "attitude_directly_observed": False,
        "gyro_model": "AngVelPlatform + gyro_bias ~= omega + b_g",
        "delta_model": f"DRFramePlatform + gyro_bias * dt ~= (omega + b_g) * dt (dt={float(dt):.9g})",
        "platform_frame_identity": bool(
            np.allclose(np.asarray(imu_cfg.get("body_to_platform_euler321_rad", [0.0, 0.0, 0.0]), dtype=np.float64), 0.0)
        ),
    }

    meta_common: Dict[str, Any] = {
        "format_version": "0.1",
        "canonical_layout": "NTD",
        "schema_version": 1,
        "task_family": str(task_family),
        "suite_name": suite_name,
        "task_id": task_cfg.task_id,
        "scenario_id": scenario_id,
        "scenario_cfg": _json_clone(scenario_cfg),
        "seed": int(seed),
        "fake_marker": False,
        "x_dim": int(x_dim),
        "y_dim": int(y_dim),
        "T": int(t_len),
        "control_input_u": False,
        "ground_truth": dict(task_cfg.ground_truth),
        "observation": {
            "type": "basilisk_imu_sensor_with_controlled_gyro_bias",
            "h_type": "imu_gyro_delta_bias",
            "measurement_mode": mode,
            "output_fields": sensor_fields,
            "field_mapping": mapping,
            "contains_absolute_attitude": False,
            "observability_note": "Gyro and delta-angle observe omega+bias, not absolute MRP attitude.",
        },
        "imu_config": {
            "profile_id": str(imu_cfg.get("profile_id", "clean_imu")),
            "measurement_mode": mode,
            "sensor_frame": str(imu_cfg.get("sensor_frame", "platform")),
            "body_to_platform_euler321_rad": _json_clone(imu_cfg.get("body_to_platform_euler321_rad", [0.0, 0.0, 0.0])),
            "output_fields": sensor_fields,
        },
        "bias_state": {
            "profile_id": str(bias_cfg.get("profile_id", "clean_bias")),
            "severity": str(bias_cfg.get("severity", bias_cfg.get("profile_id", "clean_bias"))),
            "dynamics": "b_g[t+1] = b_g[t] + bias_rw_std * sqrt(dt) * eps_t",
            "bias_init_std": float(bias_init_std),
            "bias_rw_std": float(bias_rw_std),
            "gyro_noise_std": float(gyro_noise_std),
            "delta_noise_std": float(delta_noise_std),
            "stats": bias_stats,
        },
        "imu_output_fields": mapping,
        "noise": {
            "Q": {"type": "small_angle_plus_bias_random_walk", "q2": float(q2)},
            "R": {"type": "controlled_gyro_delta_noise", "r2": float(r2_scalar)},
            "sensor_noise_std": float(math.sqrt(max(r2_scalar, 0.0))),
        },
        "noise_schedule": {
            "enabled": False,
            "kind": "stationary",
            "q2_t": {"source": "npz:q2_t", "shape": [int(t_len)]},
            "r2_t": {"source": "npz:r2_t", "shape": [int(t_len)]},
            "SoW_t": {"source": "npz:SoW_t", "shape": [int(t_len)]},
            "SoW_hat_t": None,
            "params": {"q2_base": float(q2), "r2_base": float(r2_scalar)},
        },
        "ssm": {
            "true": {
                "framework": "AVS Basilisk + controlled gyro-bias wrapper",
                "system_type": "rigid_body_attitude_with_imu_sensor_and_bias_state",
                "dt": float(dt),
                "imu_process_rate_s": float(dt),
                "inertia": np.asarray(inertia, dtype=float).tolist(),
                "disturbance_torque_B_Nm": disturbance.astype(float).tolist(),
                "state": [
                    "sigma_BN_1",
                    "sigma_BN_2",
                    "sigma_BN_3",
                    "omega_BN_B_1",
                    "omega_BN_B_2",
                    "omega_BN_B_3",
                    "gyro_bias_1",
                    "gyro_bias_2",
                    "gyro_bias_3",
                ],
                "measurement": [
                    "gyro_meas_1",
                    "gyro_meas_2",
                    "gyro_meas_3",
                    "delta_angle_meas_1",
                    "delta_angle_meas_2",
                    "delta_angle_meas_3",
                ],
                "initial_condition": {
                    "sigma0_std": float(sigma0_std),
                    "sigma0_max_norm": float(sigma0_max_norm),
                    "omega0_std_rad_s": float(omega0_std),
                    "bias_init_std_rad_s": float(bias_init_std),
                },
            },
            "assumed": {
                "system_type": "small_angle_linearized_attitude_bias_random_walk_with_imu_measurement_projection",
                "valid_for_oracle": False,
                "F": f_assumed.astype(float).tolist(),
                "Q": q_assumed.astype(float).tolist(),
                "R": r_assumed.astype(float).tolist(),
                **h_model,
                "note": "Diagnostic model for adapters needing F/H/Q/R; this is not an oracle.",
            },
        },
        "mismatch": {
            "enabled": True,
            "kind": "imu_partial_observation_with_explicit_gyro_bias_state",
            "params": {
                "oracle_supported": False,
                "measurement_mode": mode,
                "H_rank": h_rank,
                "attitude_directly_observed": False,
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
            "gyro_bias": "rad/s",
            "AngVelPlatform": "rad/s",
            "DRFramePlatform": "rad",
            "torque": "N*m",
            "dt": "s",
        },
        "attitude_representation": "MRP",
        "storage": {
            "imu_clean_y_seq": "npz_extras:imu_clean_y_seq",
            "imu_error_seq": "npz_extras:imu_error_seq",
            "gyro_bias_seq": "npz_extras:gyro_bias_seq",
            "bias_component_seq": "npz_extras:bias_component_seq",
            "noise_component_seq": "npz_extras:noise_component_seq",
            "imu_bias_clean_y_seq": "npz_extras:imu_bias_clean_y_seq",
        },
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
                "gyro_bias_seq": bias_seq_all.astype(np.float32),
                "imu_clean_gyro_seq": clean_gyro_all.astype(np.float32),
                "imu_clean_delta_theta_seq": clean_delta_theta_all.astype(np.float32),
                "imu_clean_y_seq": clean_y_all.astype(np.float32),
                "imu_error_seq": imu_error.astype(np.float32),
                "bias_component_seq": bias_component_all.astype(np.float32),
                "noise_component_seq": noise_component_all.astype(np.float32),
                "imu_bias_clean_y_seq": clean_bias_y_all.astype(np.float32),
                "y_raw_seq": y_all.astype(np.float32),
                "task_key": f"{task_family}:{task_cfg.task_id}:{scenario_id}",
            },
        )
    )
    return out, f_assumed, h_assumed


def generate_basilisk_imu_sparse_ref_adcs_v0(
    *,
    suite_name: str,
    task_cfg_dict: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    seed: int,
    scenario_id: str,
    task_family: str = "basilisk_imu_sparse_ref_adcs_v0",
) -> Tuple[GeneratorOutput, np.ndarray, np.ndarray]:
    bsk = _require_imu_basilisk()
    resolved_task = _deep_merge(dict(task_cfg_dict), scenario_cfg)
    task_cfg = make_task_cfg(resolved_task, scenario_cfg=scenario_cfg)
    split_cfg = make_split_cfg(resolved_task)

    x_dim = int(task_cfg.x_dim)
    y_dim = int(task_cfg.y_dim)
    t_len = int(task_cfg.sequence_length_T)
    n_total = int(split_cfg.n_total)
    if x_dim != 9:
        raise ValueError(f"basilisk_imu_sparse_ref_adcs_v0 requires x_dim=9, got x_dim={x_dim}")
    if y_dim != 9:
        raise ValueError(f"basilisk_imu_sparse_ref_adcs_v0 requires y_dim=9, got y_dim={y_dim}")

    imu_cfg = _resolve_imu_cfg(resolved_task)
    mode = str(imu_cfg.get("measurement_mode", "gyro_delta_angle"))
    if mode != "gyro_delta_angle":
        raise ValueError("basilisk_imu_sparse_ref_adcs_v0 currently supports measurement_mode='gyro_delta_angle' only")
    clean_imu_cfg = dict(imu_cfg)
    clean_imu_cfg.update(
        {
            "profile_id": "clean_imu_base",
            "severity": "clean_imu_base",
            "gyro_noise_std": 0.0,
            "accel_noise_std": 0.0,
            "gyro_bias_std": 0.0,
            "accel_bias_std": 0.0,
            "gyro_walk_bound": 0.0,
            "accel_walk_bound": 0.0,
            "gyro_error_bound": 0.0,
            "accel_error_bound": 0.0,
            "gyro_lsb": None,
            "accel_lsb": None,
            "gyro_saturation": None,
            "accel_saturation": None,
        }
    )
    bias_cfg = _resolve_bias_state_cfg(resolved_task)
    ref_cfg = _resolve_sparse_ref_cfg(resolved_task)

    sim_cfg = dict(resolved_task.get("simulation", {}) or {})
    dt = float(sim_cfg.get("dt", 0.1))
    inertia = _as_diag_inertia(sim_cfg.get("inertia", [10.0, 8.0, 6.0]))
    disturbance = _as_vec3(sim_cfg.get("disturbance_torque", [0.0, 0.0, 0.0]), name="disturbance_torque")
    sigma0_std = float(sim_cfg.get("sigma0_std", 0.05))
    omega0_std = float(sim_cfg.get("omega0_std", 0.01))
    sigma0_max_norm = float(sim_cfg.get("sigma0_max_norm", 0.25))

    q2 = _resolve_q2(resolved_task, scenario_cfg)
    f_assumed, h_assumed, q_assumed, r_assumed, r2_scalar = _assumed_imu_sparse_ref_model(
        dt=dt,
        q2=q2,
        bias_cfg=bias_cfg,
        ref_cfg=ref_cfg,
    )
    h_rank_unmasked = int(np.linalg.matrix_rank(np.asarray(h_assumed, dtype=np.float64)))
    h_masked = np.asarray(h_assumed, dtype=np.float64).copy()
    h_masked[6:9, :] = 0.0
    h_rank_masked = int(np.linalg.matrix_rank(h_masked))

    data_seed = stable_int_seed_v0("basilisk_imu_sparse_ref_adcs_data", suite_name, task_cfg.task_id, scenario_id, int(seed))
    rng = numpy_rng_v0(data_seed)
    x_all = np.zeros((n_total, t_len, x_dim), dtype=np.float64)
    y_all = np.zeros((n_total, t_len, y_dim), dtype=np.float64)
    imu_clean_y_all = np.zeros((n_total, t_len, 6), dtype=np.float64)
    measurement_clean_y_all = np.zeros((n_total, t_len, y_dim), dtype=np.float64)
    measurement_error_all = np.zeros((n_total, t_len, y_dim), dtype=np.float64)
    bias_seq_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    bias_component_all = np.zeros((n_total, t_len, 6), dtype=np.float64)
    noise_component_all = np.zeros((n_total, t_len, 6), dtype=np.float64)
    ref_clean_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    ref_mask_all = np.zeros((n_total, t_len, 1), dtype=np.float64)
    ref_error_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    measurement_mask_all = np.zeros((n_total, t_len, y_dim), dtype=np.float64)
    imu_bias_clean_y_all = np.zeros((n_total, t_len, 6), dtype=np.float64)
    clean_gyro_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    clean_delta_theta_all = np.zeros((n_total, t_len, 3), dtype=np.float64)
    mapping: List[Dict[str, Any]] = []

    bias_init_std = float(bias_cfg.get("bias_init_std", 0.0) or 0.0)
    bias_rw_std = float(bias_cfg.get("bias_rw_std", 0.0) or 0.0)
    gyro_noise_std = float(bias_cfg.get("gyro_noise_std", 0.0) or 0.0)
    delta_noise_std = float(bias_cfg.get("delta_noise_std", gyro_noise_std * dt) or 0.0)
    ref_mask_mode = str(ref_cfg.get("ref_mask_mode", "periodic") or "periodic").strip().lower()
    ref_period_raw = ref_cfg.get("ref_update_period", 10)
    if ref_period_raw is None:
        ref_period = max(1, t_len + 1)
    else:
        ref_period = max(1, int(ref_period_raw))
    ref_noise_std = float(ref_cfg.get("ref_noise_std", 0.005) or 0.0)
    ref_dropout_prob = min(max(float(ref_cfg.get("ref_dropout_prob", 0.0) or 0.0), 0.0), 1.0)

    for i in range(n_total):
        sigma0 = rng.normal(0.0, sigma0_std, size=3)
        sigma_norm = float(np.linalg.norm(sigma0))
        if sigma_norm > sigma0_max_norm:
            sigma0 = sigma0 * (sigma0_max_norm / sigma_norm)
        omega0 = rng.normal(0.0, omega0_std, size=3)
        traj_cfg = _trajectory_imu_cfg(
            base_cfg=clean_imu_cfg,
            suite_name=suite_name,
            task_id=task_cfg.task_id,
            scenario_id=scenario_id,
            seed=int(seed),
            trajectory_index=i,
        )
        x6_i, _y_clean_i, _raw_i, clean_raw_i, mapping = _simulate_one_imu_trajectory(
            bsk=bsk,
            sigma0=np.asarray(sigma0, dtype=np.float64),
            omega0=np.asarray(omega0, dtype=np.float64),
            inertia=inertia,
            disturbance_torque=disturbance,
            dt=dt,
            t_len=t_len,
            imu_cfg=traj_cfg,
            mode=mode,
        )
        imu_clean = np.asarray(clean_raw_i["selected_y"], dtype=np.float64)
        traj_rng = numpy_rng_v0(
            stable_int_seed_v0("basilisk_imu_sparse_ref_process", suite_name, task_cfg.task_id, scenario_id, int(seed), i)
        )
        bias_seq = np.zeros((t_len, 3), dtype=np.float64)
        if bias_init_std > 0.0:
            bias_seq[0] = traj_rng.normal(0.0, bias_init_std, size=3)
        if t_len > 1 and bias_rw_std > 0.0:
            steps = traj_rng.normal(0.0, bias_rw_std * math.sqrt(float(dt)), size=(t_len - 1, 3))
            bias_seq[1:] = bias_seq[0] + np.cumsum(steps, axis=0)
        elif t_len > 1:
            bias_seq[1:] = bias_seq[0]

        imu_noise = np.zeros((t_len, 6), dtype=np.float64)
        if gyro_noise_std > 0.0:
            imu_noise[:, 0:3] = traj_rng.normal(0.0, gyro_noise_std, size=(t_len, 3))
        if delta_noise_std > 0.0:
            imu_noise[:, 3:6] = traj_rng.normal(0.0, delta_noise_std, size=(t_len, 3))

        bias_component = np.concatenate([bias_seq, bias_seq * float(dt)], axis=1)
        imu_bias_clean = imu_clean + bias_component
        imu_meas = imu_bias_clean + imu_noise

        ref_mask = np.zeros((t_len, 1), dtype=np.float64)
        if ref_mask_mode in {"all_zero", "none", "disabled", "off"}:
            pass
        elif ref_mask_mode in {"all_one", "dense", "always", "on"}:
            ref_mask[:, 0] = 1.0
        elif ref_mask_mode in {"periodic", "sparse"}:
            ref_mask[::ref_period, 0] = 1.0
        else:
            raise ValueError(f"Unsupported sparse_ref.ref_mask_mode={ref_mask_mode!r}")
        if ref_dropout_prob > 0.0:
            drop = traj_rng.random(t_len) < ref_dropout_prob
            ref_mask[drop, 0] = 0.0
        sigma_ref_clean = x6_i[:, 0:3]
        ref_noise = np.zeros((t_len, 3), dtype=np.float64)
        active = ref_mask[:, 0] > 0.5
        if ref_noise_std > 0.0 and np.any(active):
            ref_noise[active] = traj_rng.normal(0.0, ref_noise_std, size=(int(np.sum(active)), 3))
        ref_clean_masked = sigma_ref_clean * ref_mask
        ref_meas = ref_clean_masked + ref_noise * ref_mask

        measurement_clean = np.concatenate([imu_clean, ref_clean_masked], axis=1)
        y_i = np.concatenate([imu_meas, ref_meas], axis=1)
        measurement_mask = np.ones((t_len, y_dim), dtype=np.float64)
        measurement_mask[:, 6:9] = ref_mask

        x_all[i] = np.concatenate([x6_i, bias_seq], axis=1)
        y_all[i] = y_i
        imu_clean_y_all[i] = imu_clean
        measurement_clean_y_all[i] = measurement_clean
        measurement_error_all[i] = y_i - measurement_clean
        bias_seq_all[i] = bias_seq
        bias_component_all[i] = bias_component
        noise_component_all[i] = imu_noise
        ref_clean_all[i] = sigma_ref_clean
        ref_mask_all[i] = ref_mask
        ref_error_all[i] = ref_meas - ref_clean_masked
        measurement_mask_all[i] = measurement_mask
        imu_bias_clean_y_all[i] = imu_bias_clean
        clean_gyro_all[i] = clean_raw_i["gyro"]
        clean_delta_theta_all[i] = clean_raw_i["delta_theta"]

    q2_t = np.full((t_len,), float(q2), dtype=np.float32)
    r2_t = np.full((t_len,), float(r2_scalar), dtype=np.float32)
    sow_t = q2_t / np.maximum(r2_t, np.float32(1.0e-12))

    imu_error = y_all[:, :, 0:6] - imu_clean_y_all
    clean_imu_norm = max(_mean_step_norm(imu_clean_y_all), 1.0e-12)
    omega_norm = max(_mean_step_norm(x_all[:, :, 3:6]), 1.0e-12)
    ref_mask_rate = float(np.mean(ref_mask_all))
    ref_steps = np.flatnonzero(ref_mask_all[0, :, 0] > 0.5) if n_total > 0 else np.asarray([], dtype=np.int64)
    ref_period_observed = float(np.mean(np.diff(ref_steps))) if ref_steps.size > 1 else float("nan")
    ref_noise_observed = ref_error_all[np.repeat(ref_mask_all > 0.5, 3, axis=2)]
    ref_noise_rmse = float(math.sqrt(float(np.mean(ref_noise_observed * ref_noise_observed)))) if ref_noise_observed.size else 0.0

    sparse_stats = {
        "profile_id": str(ref_cfg.get("profile_id", ref_cfg.get("severity", "mild_bias_ref"))),
        "severity": str(ref_cfg.get("severity", ref_cfg.get("profile_id", "mild_bias_ref"))),
        "ref_update_period": int(ref_period),
        "ref_period_observed": ref_period_observed,
        "ref_mask_rate": ref_mask_rate,
        "ref_mask_mode": ref_mask_mode,
        "ref_dropout_prob": float(ref_dropout_prob),
        "ref_dropout_observed": float(max(0.0, 1.0 - ref_mask_rate * ref_period)),
        "ref_noise_std": float(ref_noise_std),
        "ref_noise_rmse_observed": ref_noise_rmse,
        "missing_ref_value": str(ref_cfg.get("missing_ref_value", "zero")),
        "mask_strategy": str(ref_cfg.get("mask_strategy", "zero_ref_rows_and_measurement_mask")),
    }
    bias_stats = {
        "profile_id": str(bias_cfg.get("profile_id", "clean_ref")),
        "severity": str(bias_cfg.get("severity", bias_cfg.get("profile_id", "clean_ref"))),
        "bias_norm_mean": _mean_step_norm(bias_seq_all),
        "bias_component_norm_mean": _mean_step_norm(bias_component_all),
        "noise_component_norm_mean": _mean_step_norm(noise_component_all),
        "imu_error_norm_mean": _mean_step_norm(imu_error),
        "imu_error_to_clean_ratio": float(_mean_step_norm(imu_error) / clean_imu_norm),
        "bias_to_omega_ratio": float(_mean_step_norm(bias_seq_all) / omega_norm),
        "gyro_noise_std": float(gyro_noise_std),
        "delta_noise_std": float(delta_noise_std),
        "bias_init_std": float(bias_init_std),
        "bias_rw_std": float(bias_rw_std),
    }

    h_model = {
        "measurement_model": "imu_bias_sparse_attitude_ref",
        "h_type": "imu_bias_sparse_attitude_ref",
        "H": h_assumed.astype(float).tolist(),
        "H_unmasked": h_assumed.astype(float).tolist(),
        "H_rank": h_rank_unmasked,
        "H_rank_unmasked": h_rank_unmasked,
        "H_rank_masked_ref_unavailable": h_rank_masked,
        "direct_observation": False,
        "attitude_directly_observed": "sparse_only",
        "gyro_model": "AngVelPlatform + gyro_bias ~= omega + b_g",
        "delta_model": f"DRFramePlatform + gyro_bias * dt ~= (omega + b_g) * dt (dt={float(dt):.9g})",
        "ref_model": "sparse_sigma_ref ~= sigma_BN when ref_mask=1",
        "ref_mask_semantics": "measurement_mask_seq[:, :, 6:9] is zero when sparse reference is unavailable",
        "missing_ref_value": str(ref_cfg.get("missing_ref_value", "zero")),
        "platform_frame_identity": bool(
            np.allclose(np.asarray(imu_cfg.get("body_to_platform_euler321_rad", [0.0, 0.0, 0.0]), dtype=np.float64), 0.0)
        ),
    }

    meta_common: Dict[str, Any] = {
        "format_version": "0.1",
        "canonical_layout": "NTD",
        "schema_version": 1,
        "task_family": str(task_family),
        "suite_name": suite_name,
        "task_id": task_cfg.task_id,
        "scenario_id": scenario_id,
        "scenario_cfg": _json_clone(scenario_cfg),
        "seed": int(seed),
        "fake_marker": False,
        "x_dim": int(x_dim),
        "y_dim": int(y_dim),
        "T": int(t_len),
        "control_input_u": False,
        "ground_truth": dict(task_cfg.ground_truth),
        "observation": {
            "type": "basilisk_imu_sensor_with_controlled_gyro_bias_and_sparse_attitude_reference",
            "h_type": "imu_bias_sparse_attitude_ref",
            "measurement_mode": "gyro_delta_angle_plus_sparse_sigma_ref",
            "output_fields": ["AngVelPlatform", "DRFramePlatform", "sparse_sigma_ref"],
            "field_mapping": [
                {"columns": [0, 1, 2], "field": "AngVelPlatform", "alias": "gyro", "units": "rad/s"},
                {"columns": [3, 4, 5], "field": "DRFramePlatform", "alias": "delta_theta", "units": "rad"},
                {"columns": [6, 7, 8], "field": "sparse_sigma_ref", "alias": "sigma_ref", "units": "MRP dimensionless"},
            ],
            "contains_absolute_attitude": "sparse",
            "ref_mask_extra": "ref_mask_seq",
            "measurement_mask_extra": "measurement_mask_seq",
            "observability_note": "Sparse sigma reference anchors attitude only when ref_mask_seq is one.",
        },
        "imu_config": {
            "profile_id": str(imu_cfg.get("profile_id", "clean_imu")),
            "measurement_mode": mode,
            "sensor_frame": str(imu_cfg.get("sensor_frame", "platform")),
            "body_to_platform_euler321_rad": _json_clone(imu_cfg.get("body_to_platform_euler321_rad", [0.0, 0.0, 0.0])),
            "output_fields": [spec[0] for spec in _FIELD_SPECS[mode]],
        },
        "bias_state": {
            "profile_id": str(bias_cfg.get("profile_id", "clean_ref")),
            "severity": str(bias_cfg.get("severity", bias_cfg.get("profile_id", "clean_ref"))),
            "dynamics": "b_g[t+1] = b_g[t] + bias_rw_std * sqrt(dt) * eps_t",
            "bias_init_std": float(bias_init_std),
            "bias_rw_std": float(bias_rw_std),
            "gyro_noise_std": float(gyro_noise_std),
            "delta_noise_std": float(delta_noise_std),
            "stats": bias_stats,
        },
        "sparse_ref": {
            "profile_id": str(ref_cfg.get("profile_id", ref_cfg.get("severity", "mild_bias_ref"))),
            "severity": str(ref_cfg.get("severity", ref_cfg.get("profile_id", "mild_bias_ref"))),
            "ref_update_period": int(ref_period),
            "ref_noise_std": float(ref_noise_std),
            "ref_dropout_prob": float(ref_dropout_prob),
            "ref_mask_mode": ref_mask_mode,
            "ref_mask_semantics": "ref_mask_seq[n,t,0] == 1 means y[n,t,6:9] is sigma_BN + noise",
            "missing_ref_value": str(ref_cfg.get("missing_ref_value", "zero")),
            "mask_strategy": str(ref_cfg.get("mask_strategy", "zero_ref_rows_and_measurement_mask")),
            "stats": sparse_stats,
        },
        "imu_output_fields": mapping,
        "noise": {
            "Q": {"type": "small_angle_plus_bias_random_walk", "q2": float(q2)},
            "R": {"type": "controlled_gyro_delta_ref_noise", "r2": float(r2_scalar)},
            "sensor_noise_std": float(math.sqrt(max(r2_scalar, 0.0))),
        },
        "noise_schedule": {
            "enabled": False,
            "kind": "stationary",
            "q2_t": {"source": "npz:q2_t", "shape": [int(t_len)]},
            "r2_t": {"source": "npz:r2_t", "shape": [int(t_len)]},
            "SoW_t": {"source": "npz:SoW_t", "shape": [int(t_len)]},
            "SoW_hat_t": None,
            "params": {"q2_base": float(q2), "r2_base": float(r2_scalar)},
        },
        "ssm": {
            "true": {
                "framework": "AVS Basilisk + controlled gyro-bias and sparse attitude-reference wrapper",
                "system_type": "rigid_body_attitude_with_imu_sensor_bias_state_and_sparse_attitude_reference",
                "dt": float(dt),
                "imu_process_rate_s": float(dt),
                "inertia": np.asarray(inertia, dtype=float).tolist(),
                "disturbance_torque_B_Nm": disturbance.astype(float).tolist(),
                "state": [
                    "sigma_BN_1",
                    "sigma_BN_2",
                    "sigma_BN_3",
                    "omega_BN_B_1",
                    "omega_BN_B_2",
                    "omega_BN_B_3",
                    "gyro_bias_1",
                    "gyro_bias_2",
                    "gyro_bias_3",
                ],
                "measurement": [
                    "gyro_meas_1",
                    "gyro_meas_2",
                    "gyro_meas_3",
                    "delta_angle_meas_1",
                    "delta_angle_meas_2",
                    "delta_angle_meas_3",
                    "sparse_sigma_ref_1",
                    "sparse_sigma_ref_2",
                    "sparse_sigma_ref_3",
                ],
                "initial_condition": {
                    "sigma0_std": float(sigma0_std),
                    "sigma0_max_norm": float(sigma0_max_norm),
                    "omega0_std_rad_s": float(omega0_std),
                    "bias_init_std_rad_s": float(bias_init_std),
                },
            },
            "assumed": {
                "system_type": "small_angle_linearized_attitude_bias_random_walk_with_imu_and_sparse_ref_projection",
                "valid_for_oracle": False,
                "F": f_assumed.astype(float).tolist(),
                "Q": q_assumed.astype(float).tolist(),
                "R": r_assumed.astype(float).tolist(),
                **h_model,
                "note": "Diagnostic model for adapters needing F/H/Q/R; this is not an oracle. Current adapters receive y with zero-filled missing ref rows plus measurement_mask_seq extras.",
            },
        },
        "mismatch": {
            "enabled": True,
            "kind": "imu_bias_state_with_sparse_absolute_attitude_reference",
            "params": {
                "oracle_supported": False,
                "measurement_mode": "gyro_delta_angle_plus_sparse_sigma_ref",
                "H_rank_unmasked": h_rank_unmasked,
                "H_rank_masked_ref_unavailable": h_rank_masked,
                "attitude_directly_observed": "sparse_only",
                "ref_mask_rate": ref_mask_rate,
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
            "gyro_bias": "rad/s",
            "AngVelPlatform": "rad/s",
            "DRFramePlatform": "rad",
            "sparse_sigma_ref": "MRP dimensionless",
            "torque": "N*m",
            "dt": "s",
        },
        "attitude_representation": "MRP",
        "storage": {
            "imu_clean_y_seq": "npz_extras:imu_clean_y_seq",
            "measurement_clean_y_seq": "npz_extras:measurement_clean_y_seq",
            "measurement_error_seq": "npz_extras:measurement_error_seq",
            "measurement_mask_seq": "npz_extras:measurement_mask_seq",
            "ref_mask_seq": "npz_extras:ref_mask_seq",
            "ref_clean_seq": "npz_extras:ref_clean_seq",
            "ref_error_seq": "npz_extras:ref_error_seq",
            "gyro_bias_seq": "npz_extras:gyro_bias_seq",
            "bias_component_seq": "npz_extras:bias_component_seq",
            "noise_component_seq": "npz_extras:noise_component_seq",
        },
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
                "gyro_bias_seq": bias_seq_all.astype(np.float32),
                "imu_clean_gyro_seq": clean_gyro_all.astype(np.float32),
                "imu_clean_delta_theta_seq": clean_delta_theta_all.astype(np.float32),
                "imu_clean_y_seq": imu_clean_y_all.astype(np.float32),
                "imu_error_seq": imu_error.astype(np.float32),
                "bias_component_seq": bias_component_all.astype(np.float32),
                "noise_component_seq": noise_component_all.astype(np.float32),
                "imu_bias_clean_y_seq": imu_bias_clean_y_all.astype(np.float32),
                "ref_clean_seq": ref_clean_all.astype(np.float32),
                "ref_mask_seq": ref_mask_all.astype(np.float32),
                "ref_error_seq": ref_error_all.astype(np.float32),
                "measurement_clean_y_seq": measurement_clean_y_all.astype(np.float32),
                "measurement_error_seq": measurement_error_all.astype(np.float32),
                "measurement_mask_seq": measurement_mask_all.astype(np.float32),
                "y_raw_seq": y_all.astype(np.float32),
                "task_key": f"{task_family}:{task_cfg.task_id}:{scenario_id}",
            },
        )
    )
    return out, f_assumed, h_assumed
