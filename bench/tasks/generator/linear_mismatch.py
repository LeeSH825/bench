from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ..data_format import CANONICAL_LAYOUT_V0
from ..generator.contract import GeneratorOutput, coerce_ntd_float32_output, make_split_cfg, make_task_cfg
from ..generators.linear import build_linear_system_matrices_v0, generate_linear_gaussian_sequences_v0
from ...utils.seeding import numpy_rng_v0, stable_int_seed_v0


_MISMATCH_MODE_CANONICAL = {
    "f_rotation": "F_rotation",
    "h_rotation": "H_rotation",
}


def _deep_merge(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def _coerce_matrix(value: Any, *, shape: Tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def _matrix_to_json(mat: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(mat, dtype=np.float64)]


def _normalize_mode(raw: Any) -> str:
    key = str(raw if raw is not None else "F_rotation").strip().lower()
    if key not in _MISMATCH_MODE_CANONICAL:
        raise ValueError("mismatch.mode must be one of: F_rotation, H_rotation")
    return _MISMATCH_MODE_CANONICAL[key]


def _embedded_rotation(dim: int, rotation_dims: int, alpha_deg: float) -> np.ndarray:
    d = int(dim)
    k = int(rotation_dims)
    if k < 2:
        raise ValueError(f"rotation_dims must be >=2, got {k}")
    if k > d:
        raise ValueError(f"rotation_dims={k} exceeds target dimension={d}")

    alpha = math.radians(float(alpha_deg))
    c, s = math.cos(alpha), math.sin(alpha)
    r2 = np.asarray([[c, -s], [s, c]], dtype=np.float64)

    rk = np.eye(k, dtype=np.float64)
    rk[:2, :2] = r2
    r_full = np.eye(d, dtype=np.float64)
    r_full[:k, :k] = rk
    return r_full


def _resolve_base_cfg(task_cfg_dict: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    raw = task_cfg_dict.get("base", {})
    if isinstance(raw, str):
        base = {"id": str(raw)}
    elif isinstance(raw, Mapping):
        base = dict(raw)
    else:
        base = {}
    sc_base = scenario_cfg.get("base")
    if isinstance(sc_base, Mapping):
        base = _deep_merge(base, sc_base)
    return base


def _resolve_base_matrices(
    *,
    task_cfg_dict: Mapping[str, Any],
    scenario_cfg: Mapping[str, Any],
    x_dim: int,
    y_dim: int,
    suite_name: str,
    task_id: str,
    scenario_id: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    base_cfg = _resolve_base_cfg(task_cfg_dict, scenario_cfg)
    obs = task_cfg_dict.get("observation", {}) or {}
    obs_map = dict(obs) if isinstance(obs, Mapping) else {}
    base_id = str(base_cfg.get("id", base_cfg.get("canonical_id", "linear_gaussian_v0")))
    default_h_spec = "canonical_inverse" if base_id in {"linear_canonical", "linear_canonical_v0"} else "identity"
    h_spec = str(base_cfg.get("H_spec", obs_map.get("H", default_h_spec)))

    f0_raw = base_cfg.get("F0", task_cfg_dict.get("F0", None))
    h0_raw = base_cfg.get("H0", task_cfg_dict.get("H0", None))
    f_default: Optional[np.ndarray] = None
    h_default: Optional[np.ndarray] = None
    if f0_raw is None or h0_raw is None:
        sys_seed = stable_int_seed_v0("system", suite_name, task_id, scenario_id, int(seed))
        rng_sys = numpy_rng_v0(sys_seed)
        f_default, h_default = build_linear_system_matrices_v0(
            rng=rng_sys,
            x_dim=int(x_dim),
            y_dim=int(y_dim),
            H_spec=h_spec,
        )

    f0 = _coerce_matrix(f0_raw, shape=(int(x_dim), int(x_dim)), name="F0") if f0_raw is not None else np.asarray(f_default, dtype=np.float64)
    h0 = _coerce_matrix(h0_raw, shape=(int(y_dim), int(x_dim)), name="H0") if h0_raw is not None else np.asarray(h_default, dtype=np.float64)

    desc = {"base_id": base_id, "H_spec": h_spec}
    return f0, h0, desc


def _resolve_mismatch_cfg(task_cfg_dict: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    raw = task_cfg_dict.get("mismatch", {})
    mm = dict(raw) if isinstance(raw, Mapping) else {}
    sc_mm = scenario_cfg.get("mismatch")
    if isinstance(sc_mm, Mapping):
        mm = _deep_merge(mm, sc_mm)

    mode = _normalize_mode(mm.get("mode", mm.get("kind", "F_rotation")))
    alpha_true_deg = float(mm.get("alpha_true_deg", mm.get("alpha_deg", 10.0)))
    alpha_assumed_deg = float(mm.get("alpha_assumed_deg", 0.0))
    rotation_dims = int(mm.get("rotation_dims", 2))

    return {
        "mode": mode,
        "alpha_true_deg": float(alpha_true_deg),
        "alpha_assumed_deg": float(alpha_assumed_deg),
        "rotation_dims": int(rotation_dims),
    }


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
        raise ValueError("linear_mismatch requires noise q2/r2 (noise.Q.q2 and noise.R.r2, or noise.q2/r2)")
    if q2 <= 0.0 or r2 <= 0.0:
        raise ValueError(f"q2/r2 must be positive, got q2={q2}, r2={r2}")
    return float(q2), float(r2)


def _true_assumed_matrices(
    *,
    mode: str,
    rotation_dims: int,
    alpha_true_deg: float,
    alpha_assumed_deg: float,
    f0: np.ndarray,
    h0: np.ndarray,
    x_dim: int,
    y_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mode == "F_rotation":
        if int(rotation_dims) > int(x_dim):
            raise ValueError(f"F_rotation requires x_dim >= rotation_dims. got x_dim={x_dim}, rotation_dims={rotation_dims}")
        r_true = _embedded_rotation(int(x_dim), int(rotation_dims), float(alpha_true_deg))
        r_assumed = _embedded_rotation(int(x_dim), int(rotation_dims), float(alpha_assumed_deg))
        f_true = r_true @ f0
        f_assumed = r_assumed @ f0
        h_true = h0.copy()
        h_assumed = h0.copy()
        return f_true, h_true, f_assumed, h_assumed

    if mode == "H_rotation":
        if int(rotation_dims) > int(y_dim):
            raise ValueError(f"H_rotation requires y_dim >= rotation_dims. got y_dim={y_dim}, rotation_dims={rotation_dims}")
        r_true = _embedded_rotation(int(y_dim), int(rotation_dims), float(alpha_true_deg))
        r_assumed = _embedded_rotation(int(y_dim), int(rotation_dims), float(alpha_assumed_deg))
        h_true = r_true @ h0
        h_assumed = r_assumed @ h0
        f_true = f0.copy()
        f_assumed = f0.copy()
        return f_true, h_true, f_assumed, h_assumed

    raise ValueError(f"unsupported mismatch mode: {mode}")


def generate_linear_mismatch_v0(
    *,
    suite_name: str,
    task_cfg_dict: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    seed: int,
    scenario_id: str,
    task_family: str = "linear_mismatch_v0",
) -> Tuple[GeneratorOutput, np.ndarray, np.ndarray]:
    task_cfg = make_task_cfg(task_cfg_dict, scenario_cfg=scenario_cfg)
    split_cfg = make_split_cfg(task_cfg_dict)

    x_dim = int(task_cfg.x_dim)
    y_dim = int(task_cfg.y_dim)
    t_len = int(task_cfg.sequence_length_T)
    n_total = int(split_cfg.n_total)

    f0, h0, base_desc = _resolve_base_matrices(
        task_cfg_dict=task_cfg_dict,
        scenario_cfg=scenario_cfg,
        x_dim=x_dim,
        y_dim=y_dim,
        suite_name=suite_name,
        task_id=task_cfg.task_id,
        scenario_id=scenario_id,
        seed=int(seed),
    )
    if f0.shape != (x_dim, x_dim):
        raise ValueError(f"F0 must be [{x_dim},{x_dim}], got {f0.shape}")
    if h0.shape != (y_dim, x_dim):
        raise ValueError(f"H0 must be [{y_dim},{x_dim}], got {h0.shape}")

    mm = _resolve_mismatch_cfg(task_cfg_dict, scenario_cfg)
    mode = str(mm["mode"])
    alpha_true_deg = float(mm["alpha_true_deg"])
    alpha_assumed_deg = float(mm["alpha_assumed_deg"])
    rotation_dims = int(mm["rotation_dims"])
    f_true, h_true, f_assumed, h_assumed = _true_assumed_matrices(
        mode=mode,
        rotation_dims=rotation_dims,
        alpha_true_deg=alpha_true_deg,
        alpha_assumed_deg=alpha_assumed_deg,
        f0=f0,
        h0=h0,
        x_dim=x_dim,
        y_dim=y_dim,
    )

    q2, r2 = _resolve_q2_r2(dict(task_cfg.noise))
    q_mat = float(q2) * np.eye(x_dim, dtype=np.float64)
    r_mat = float(r2) * np.eye(y_dim, dtype=np.float64)

    data_seed = stable_int_seed_v0("data", suite_name, task_cfg.task_id, scenario_id, int(seed))
    rng_data = numpy_rng_v0(data_seed)
    x_all, y_all = generate_linear_gaussian_sequences_v0(
        rng=rng_data,
        n_seq=n_total,
        T=t_len,
        F=f_true,
        H=h_true,
        q2=float(q2),
        r2=float(r2),
        t0_shift=None,
        r_scale_post=1.0,
        obs_dist_name="gaussian",
        obs_dist_params={"name": "gaussian"},
        q2_t=None,
        r2_t=None,
    )

    mm_params: Dict[str, Any] = {
        "alpha_true_deg": float(alpha_true_deg),
        "alpha_assumed_deg": float(alpha_assumed_deg),
        "rotation_dims": int(rotation_dims),
        "base_id": str(base_desc.get("base_id", "linear_gaussian_v0")),
        "mode": str(mode),
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
        "observation": dict(task_cfg.observation),
        "noise": {
            "Q": {"type": "scaled_identity", "q2": float(q2)},
            "R": {"type": "scaled_identity", "r2": float(r2)},
            "obs_distribution": {"name": "gaussian"},
        },
        "ssm": {
            "true": {
                "type": "linear_gaussian",
                "F": _matrix_to_json(f_true),
                "H": _matrix_to_json(h_true),
                "Q": _matrix_to_json(q_mat),
                "R": _matrix_to_json(r_mat),
                "q2": float(q2),
                "r2": float(r2),
            },
            "assumed": {
                "type": "linear_gaussian",
                "F": _matrix_to_json(f_assumed),
                "H": _matrix_to_json(h_assumed),
                "Q": _matrix_to_json(q_mat),
                "R": _matrix_to_json(r_mat),
                "q2": float(q2),
                "r2": float(r2),
            },
        },
        "mismatch": {
            "enabled": True,
            "kind": str(mode),
            "params": mm_params,
        },
        "noise_schedule": {
            "enabled": False,
            "kind": "stationary",
            "params": {"q2_base": float(q2), "r2_base": float(r2)},
            "q2_t": "meta.noise.Q.q2",
            "r2_t": "meta.noise.R.r2",
            "SoW_t": "meta.noise_schedule.derived_from_q2_over_r2",
            "SoW_hat_t": None,
        },
    }

    out = coerce_ntd_float32_output(
        GeneratorOutput(
            x=x_all.astype(np.float32, copy=False),
            y=y_all.astype(np.float32, copy=False),
            meta=meta_common,
            extras={"task_key": f"{task_family}:{task_cfg.task_id}:{scenario_id}"},
        )
    )
    return out, f_assumed.astype(np.float32), h_assumed.astype(np.float32)
