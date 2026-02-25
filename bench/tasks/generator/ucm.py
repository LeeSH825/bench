from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ..data_format import CANONICAL_LAYOUT_V0
from ..generator.contract import GeneratorOutput, coerce_ntd_float32_output, make_split_cfg, make_task_cfg
from ...utils.seeding import numpy_rng_v0, stable_int_seed_v0


def _deep_merge(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def _matrix_to_json(mat: np.ndarray) -> list[list[float]]:
    arr = np.asarray(mat, dtype=np.float64)
    return [[float(v) for v in row] for row in arr]


def _rotation_matrix(theta_deg: float) -> np.ndarray:
    theta = math.radians(float(theta_deg))
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def _resolve_q2_r2(noise_cfg: Mapping[str, Any]) -> Tuple[float, float]:
    q2: Optional[float] = None
    r2: Optional[float] = None

    if "q2" in noise_cfg:
        q2 = float(noise_cfg["q2"])
    if "r2" in noise_cfg:
        r2 = float(noise_cfg["r2"])

    q_map = noise_cfg.get("Q")
    if q2 is None and isinstance(q_map, Mapping) and "q2" in q_map:
        q2 = float(q_map["q2"])
    r_map = noise_cfg.get("R")
    if r2 is None and isinstance(r_map, Mapping) and "r2" in r_map:
        r2 = float(r_map["r2"])

    pre_shift = noise_cfg.get("pre_shift")
    if isinstance(pre_shift, Mapping):
        pre_q = pre_shift.get("Q")
        pre_r = pre_shift.get("R")
        if q2 is None and isinstance(pre_q, Mapping) and "q2" in pre_q:
            q2 = float(pre_q["q2"])
        if r2 is None and isinstance(pre_r, Mapping) and "r2" in pre_r:
            r2 = float(pre_r["r2"])

    if q2 is None or r2 is None:
        raise ValueError("ucm_v0 requires noise q2/r2 (noise.Q.q2 and noise.R.r2, or noise.q2/r2)")
    if q2 <= 0.0 or r2 <= 0.0:
        raise ValueError(f"q2/r2 must be positive, got q2={q2}, r2={r2}")
    return float(q2), float(r2)


def _resolve_theta_deg(task_cfg_dict: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> float:
    theta = None
    ssm = task_cfg_dict.get("ssm")
    if isinstance(ssm, Mapping):
        params = ssm.get("params")
        if isinstance(params, Mapping) and "theta_deg" in params:
            theta = float(params["theta_deg"])
    if theta is None and "theta_deg" in task_cfg_dict:
        theta = float(task_cfg_dict["theta_deg"])
    ucm_cfg = task_cfg_dict.get("ucm")
    if theta is None and isinstance(ucm_cfg, Mapping) and "theta_deg" in ucm_cfg:
        theta = float(ucm_cfg["theta_deg"])

    sc_ssm = scenario_cfg.get("ssm")
    if isinstance(sc_ssm, Mapping):
        sc_params = sc_ssm.get("params")
        if isinstance(sc_params, Mapping) and "theta_deg" in sc_params:
            theta = float(sc_params["theta_deg"])
    if "theta_deg" in scenario_cfg:
        theta = float(scenario_cfg["theta_deg"])
    sc_ucm = scenario_cfg.get("ucm")
    if isinstance(sc_ucm, Mapping) and "theta_deg" in sc_ucm:
        theta = float(sc_ucm["theta_deg"])

    if theta is None:
        theta = 10.0
    return float(theta)


def _resolve_obs_mode(task_cfg_dict: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> str:
    obs_mode_raw: Optional[Any] = None
    obs = task_cfg_dict.get("observation")
    if isinstance(obs, Mapping):
        obs_mode_raw = obs.get("mode", obs.get("obs_mode", obs.get("h_type")))
    ssm = task_cfg_dict.get("ssm")
    if isinstance(ssm, Mapping):
        params = ssm.get("params")
        if isinstance(params, Mapping):
            obs_mode_raw = params.get("obs_mode", obs_mode_raw)
    sc_obs = scenario_cfg.get("observation")
    if isinstance(sc_obs, Mapping):
        obs_mode_raw = sc_obs.get("mode", sc_obs.get("obs_mode", obs_mode_raw))
    sc_ssm = scenario_cfg.get("ssm")
    if isinstance(sc_ssm, Mapping):
        sc_params = sc_ssm.get("params")
        if isinstance(sc_params, Mapping):
            obs_mode_raw = sc_params.get("obs_mode", obs_mode_raw)
    obs_mode_raw = scenario_cfg.get("obs_mode", obs_mode_raw)

    mode = str(obs_mode_raw if obs_mode_raw is not None else "linear").strip().lower()
    if mode in {"linear", "identity"}:
        return "linear"
    if mode in {"nonlinear", "polar"}:
        return "nonlinear"
    raise ValueError("ucm_v0 observation mode must be one of: linear, nonlinear")


def _sample_distribution(
    dist_cfg: Mapping[str, Any],
    *,
    n: int,
    rng: np.random.Generator,
    name: str,
) -> np.ndarray:
    kind = str(dist_cfg.get("kind", "uniform")).strip().lower()

    if kind in {"fixed", "constant"}:
        value = float(dist_cfg.get("value", dist_cfg.get("v", 1.0e-3)))
        if value <= 0.0:
            raise ValueError(f"{name} fixed value must be > 0, got {value}")
        return np.full((int(n),), value, dtype=np.float64)

    if kind in {"uniform"}:
        low = float(dist_cfg.get("low", dist_cfg.get("min", 1.0e-4)))
        high = float(dist_cfg.get("high", dist_cfg.get("max", low)))
        if low <= 0.0 or high <= 0.0 or high < low:
            raise ValueError(f"{name} uniform bounds must satisfy 0<low<=high, got low={low}, high={high}")
        return np.asarray(rng.uniform(low, high, size=int(n)), dtype=np.float64)

    if kind in {"loguniform", "log_uniform"}:
        low = float(dist_cfg.get("low", dist_cfg.get("min", 1.0e-4)))
        high = float(dist_cfg.get("high", dist_cfg.get("max", low)))
        if low <= 0.0 or high <= 0.0 or high < low:
            raise ValueError(f"{name} loguniform bounds must satisfy 0<low<=high, got low={low}, high={high}")
        z = rng.uniform(np.log(low), np.log(high), size=int(n))
        return np.asarray(np.exp(z), dtype=np.float64)

    if kind in {"list", "discrete"}:
        values = np.asarray(list(dist_cfg.get("values", [])), dtype=np.float64)
        if values.ndim != 1 or int(values.size) <= 0:
            raise ValueError(f"{name} list distribution requires non-empty 1D 'values'")
        if np.any(values <= 0.0):
            raise ValueError(f"{name} list values must be > 0")
        probs_raw = dist_cfg.get("probs")
        if probs_raw is None:
            idx = rng.integers(0, int(values.size), size=int(n))
            return values[idx]
        probs = np.asarray(list(probs_raw), dtype=np.float64)
        if probs.shape != values.shape:
            raise ValueError(f"{name} probs shape must match values shape, got {probs.shape} vs {values.shape}")
        probs = probs / np.maximum(np.sum(probs), 1.0e-12)
        idx = rng.choice(int(values.size), size=int(n), p=probs)
        return values[idx]

    raise ValueError(f"unsupported {name}.kind: {kind}")


def _task_set_block(task_cfg_dict: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    raw = task_cfg_dict.get("task_set", {})
    task_set = dict(raw) if isinstance(raw, Mapping) else {}
    sc_raw = scenario_cfg.get("task_set")
    if isinstance(sc_raw, Mapping):
        task_set = _deep_merge(task_set, sc_raw)
    return task_set


def _task_key_mode(task_set_cfg: Mapping[str, Any]) -> str:
    task_key_cfg = task_set_cfg.get("task_key", {})
    if isinstance(task_key_cfg, Mapping):
        mode_raw = task_key_cfg.get("mode", "(q2,r2)")
    else:
        mode_raw = "(q2,r2)"
    mode = str(mode_raw).strip().lower()
    if mode in {"(q2,r2)", "q2_r2", "tuple"}:
        return "(q2,r2)"
    if mode in {"v_db", "sow_db"}:
        return "SoW_dB"
    if mode in {"sow_linear", "ratio"}:
        return "SoW_linear"
    raise ValueError("task_set.task_key.mode must be one of: (q2,r2), V_dB/SoW_dB, SoW_linear")


def _task_key_array(mode: str, q2_seq: np.ndarray, r2_seq: np.ndarray) -> np.ndarray:
    q2 = np.asarray(q2_seq, dtype=np.float64)
    r2 = np.asarray(r2_seq, dtype=np.float64)
    if mode == "(q2,r2)":
        return np.stack([q2, r2], axis=1).astype(np.float32, copy=False)
    sow = q2 / np.maximum(r2, 1.0e-12)
    if mode == "SoW_dB":
        return np.asarray(10.0 * np.log10(np.maximum(sow, 1.0e-12)), dtype=np.float32)
    if mode == "SoW_linear":
        return np.asarray(sow, dtype=np.float32)
    raise ValueError(f"unsupported task key mode: {mode}")


def _rollout_ucm(
    *,
    rng: np.random.Generator,
    n_seq: int,
    t_len: int,
    theta_deg: float,
    q2_seq: np.ndarray,
    r2_seq: np.ndarray,
    obs_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    r_theta = _rotation_matrix(float(theta_deg))
    q_std = np.sqrt(np.asarray(q2_seq, dtype=np.float64))
    r_std = np.sqrt(np.asarray(r2_seq, dtype=np.float64))
    if q_std.shape != (int(n_seq),):
        raise ValueError(f"q2_seq must have shape [{n_seq}], got {q_std.shape}")
    if r_std.shape != (int(n_seq),):
        raise ValueError(f"r2_seq must have shape [{n_seq}], got {r_std.shape}")

    x = np.zeros((int(n_seq), int(t_len), 2), dtype=np.float64)
    y = np.zeros((int(n_seq), int(t_len), 2), dtype=np.float64)

    # Keep x0 convention aligned with existing linear generators: standard normal on R^2.
    x_t = np.asarray(rng.standard_normal((int(n_seq), 2)), dtype=np.float64)
    for t in range(int(t_len)):
        if obs_mode == "linear":
            y_det = x_t
        elif obs_mode == "nonlinear":
            radius = np.linalg.norm(x_t, axis=1)
            angle = np.arctan2(x_t[:, 1], x_t[:, 0])
            y_det = np.stack([radius, angle], axis=1)
        else:
            raise ValueError(f"unsupported obs_mode: {obs_mode}")

        v_t = np.asarray(rng.standard_normal((int(n_seq), 2)), dtype=np.float64) * r_std[:, None]
        y_t = y_det + v_t

        x[:, t, :] = x_t
        y[:, t, :] = y_t

        w_t = np.asarray(rng.standard_normal((int(n_seq), 2)), dtype=np.float64) * q_std[:, None]
        x_t = (x_t @ r_theta.T) + w_t

    return x, y


def generate_ucm_v0(
    *,
    suite_name: str,
    task_cfg_dict: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    seed: int,
    scenario_id: str,
    task_family: str = "ucm_v0",
) -> Tuple[GeneratorOutput, np.ndarray, np.ndarray]:
    task_cfg = make_task_cfg(task_cfg_dict, scenario_cfg=scenario_cfg)
    split_cfg = make_split_cfg(task_cfg_dict)

    x_dim = int(task_cfg.x_dim)
    y_dim = int(task_cfg.y_dim)
    t_len = int(task_cfg.sequence_length_T)
    n_total = int(split_cfg.n_total)
    if x_dim != 2 or y_dim != 2:
        raise ValueError(f"ucm_v0 expects x_dim=y_dim=2, got x_dim={x_dim}, y_dim={y_dim}")

    obs_mode = _resolve_obs_mode(task_cfg_dict, scenario_cfg)
    theta_deg = _resolve_theta_deg(task_cfg_dict, scenario_cfg)
    q2_base, r2_base = _resolve_q2_r2(dict(task_cfg.noise))

    task_set_cfg = _task_set_block(task_cfg_dict, scenario_cfg)
    task_set_enabled = bool(task_set_cfg.get("enabled", False))
    sample_per_sequence = bool(task_set_cfg.get("sample_per_sequence", True))
    q2_seq = np.full((n_total,), float(q2_base), dtype=np.float64)
    r2_seq = np.full((n_total,), float(r2_base), dtype=np.float64)

    if task_set_enabled:
        ts_seed = stable_int_seed_v0("task_set", suite_name, task_cfg.task_id, scenario_id, int(seed))
        rng_ts = numpy_rng_v0(ts_seed)
        q2_dist = task_set_cfg.get("q2_dist", {"kind": "fixed", "value": q2_base})
        r2_dist = task_set_cfg.get("r2_dist", {"kind": "fixed", "value": r2_base})
        if not isinstance(q2_dist, Mapping) or not isinstance(r2_dist, Mapping):
            raise ValueError("task_set q2_dist/r2_dist must be mappings")
        n_draw = n_total if sample_per_sequence else 1
        q2_draw = _sample_distribution(q2_dist, n=n_draw, rng=rng_ts, name="task_set.q2_dist")
        r2_draw = _sample_distribution(r2_dist, n=n_draw, rng=rng_ts, name="task_set.r2_dist")
        if sample_per_sequence:
            q2_seq = np.asarray(q2_draw, dtype=np.float64)
            r2_seq = np.asarray(r2_draw, dtype=np.float64)
        else:
            q2_seq[:] = float(q2_draw[0])
            r2_seq[:] = float(r2_draw[0])

    data_seed = stable_int_seed_v0("data", suite_name, task_cfg.task_id, scenario_id, int(seed))
    rng_data = numpy_rng_v0(data_seed)
    x_all, y_all = _rollout_ucm(
        rng=rng_data,
        n_seq=n_total,
        t_len=t_len,
        theta_deg=theta_deg,
        q2_seq=q2_seq,
        r2_seq=r2_seq,
        obs_mode=obs_mode,
    )

    f_true = _rotation_matrix(theta_deg)
    f_assumed = f_true.copy()
    h_proxy = np.eye(2, dtype=np.float64)
    h_true_meta = _matrix_to_json(h_proxy) if obs_mode == "linear" else None
    h_assumed_meta = _matrix_to_json(h_proxy) if obs_mode == "linear" else None
    q_mat = float(q2_base) * np.eye(2, dtype=np.float64)
    r_mat = float(r2_base) * np.eye(2, dtype=np.float64)

    extras: Dict[str, Any] = {}
    if task_set_enabled:
        extras["q2_seq"] = np.asarray(q2_seq, dtype=np.float32)
        extras["r2_seq"] = np.asarray(r2_seq, dtype=np.float32)
        task_key_mode = _task_key_mode(task_set_cfg)
        task_key_cfg = task_set_cfg.get("task_key", {})
        store_task_key = True
        if isinstance(task_key_cfg, Mapping):
            store_task_key = bool(task_key_cfg.get("store_in_extras", True))
        if store_task_key:
            extras["task_key"] = _task_key_array(task_key_mode, q2_seq=q2_seq, r2_seq=r2_seq)

    obs_params: Dict[str, Any] = {
        "obs_mode": str(obs_mode),
        "theta_deg": float(theta_deg),
        "theta_rad": float(math.radians(theta_deg)),
        "x0_kind": "standard_normal",
    }
    if obs_mode == "nonlinear":
        obs_params["measurement_units"] = {"radius": "linear", "angle": "radians"}
        obs_params["observation_function"] = "[||x_t||, atan2(x2, x1)]"

    ssm_true: Dict[str, Any] = {
        "type": "ucm",
        "F": _matrix_to_json(f_true),
        "H": h_true_meta,
        "Q": _matrix_to_json(q_mat),
        "R": _matrix_to_json(r_mat),
        "q2": float(q2_base),
        "r2": float(r2_base),
        "params": dict(obs_params),
    }
    ssm_assumed = {
        "type": "ucm",
        "F": _matrix_to_json(f_assumed),
        "H": h_assumed_meta,
        "Q": _matrix_to_json(q_mat),
        "R": _matrix_to_json(r_mat),
        "q2": float(q2_base),
        "r2": float(r2_base),
        "params": dict(obs_params),
    }

    meta_common: Dict[str, Any] = {
        "format_version": "0.1",
        "canonical_layout": CANONICAL_LAYOUT_V0,
        "schema_version": 1,
        "task_family": str(task_family),
        "suite_name": str(suite_name),
        "task_id": str(task_cfg.task_id),
        "scenario_id": str(scenario_id),
        "scenario_cfg": dict(scenario_cfg),
        "seed": int(seed),
        "x_dim": int(x_dim),
        "y_dim": int(y_dim),
        "T": int(t_len),
        "control_input_u": bool(task_cfg.control_input_u),
        "ground_truth": dict(task_cfg.ground_truth),
        "observation": _deep_merge(dict(task_cfg.observation), {"mode": str(obs_mode)}),
        "noise": {
            "Q": {"type": "scaled_identity", "q2": float(q2_base)},
            "R": {"type": "scaled_identity", "r2": float(r2_base)},
            "obs_distribution": {"name": "gaussian"},
            "per_sequence": bool(task_set_enabled),
        },
        "ssm": {
            "true": ssm_true,
            "assumed": ssm_assumed,
        },
        "mismatch": {
            "enabled": False,
            "kind": "none",
            "params": {},
        },
        "noise_schedule": {
            "enabled": False,
            "kind": "stationary",
            "params": {"q2_base": float(q2_base), "r2_base": float(r2_base)},
            "q2_t": "meta.noise.Q.q2",
            "r2_t": "meta.noise.R.r2",
            "SoW_t": "meta.noise_schedule.derived_from_q2_over_r2",
            "SoW_hat_t": None,
        },
        "switching": {
            "enabled": False,
            "models": [],
            "t_change": None,
            "retrain_window": 0,
        },
    }
    if task_set_enabled:
        task_key_cfg = task_set_cfg.get("task_key", {})
        task_key_mode = _task_key_mode(task_set_cfg)
        store_task_key = True
        if isinstance(task_key_cfg, Mapping):
            store_task_key = bool(task_key_cfg.get("store_in_extras", True))
        meta_common["task_set"] = {
            "enabled": True,
            "sample_per_sequence": bool(sample_per_sequence),
            "q2_dist": dict(task_set_cfg.get("q2_dist", {})) if isinstance(task_set_cfg.get("q2_dist"), Mapping) else {},
            "r2_dist": dict(task_set_cfg.get("r2_dist", {})) if isinstance(task_set_cfg.get("r2_dist"), Mapping) else {},
            "task_key": {
                "mode": str(task_key_mode),
                "stored_in": "npz_extras:task_key" if store_task_key else "disabled",
            },
            "storage": {
                "q2_seq": "npz_extras:q2_seq",
                "r2_seq": "npz_extras:r2_seq",
            },
        }

    out = coerce_ntd_float32_output(
        GeneratorOutput(
            x=x_all.astype(np.float32, copy=False),
            y=y_all.astype(np.float32, copy=False),
            meta=meta_common,
            extras=extras,
        )
    )
    return out, f_assumed.astype(np.float32), h_proxy.astype(np.float32)
