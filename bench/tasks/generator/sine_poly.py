from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..data_format import CANONICAL_LAYOUT_V0
from ..generator.contract import GeneratorOutput, coerce_ntd_float32_output, make_split_cfg, make_task_cfg
from ...utils.seeding import numpy_rng_v0, stable_int_seed_v0


_SP_PARAM_NAMES: Tuple[str, ...] = ("alpha", "beta", "phi", "delta", "a", "b", "c")


def _deep_merge(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def _matrix_to_json(mat: np.ndarray) -> List[List[float]]:
    arr = np.asarray(mat, dtype=np.float64)
    return [[float(v) for v in row] for row in arr]


def _param_vector(raw: Any, *, dim: int, name: str, default: float) -> np.ndarray:
    if raw is None:
        return np.full((int(dim),), float(default), dtype=np.float64)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((int(dim),), float(arr.item()), dtype=np.float64)
    if arr.ndim == 1 and int(arr.shape[0]) == int(dim):
        return arr.astype(np.float64, copy=False)
    raise ValueError(f"{name} must be scalar or length-{dim} vector, got shape={arr.shape}")


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
        raise ValueError("sine_poly_v0 requires noise q2/r2 (noise.Q.q2 and noise.R.r2, or noise.q2/r2)")
    if q2 <= 0.0 or r2 <= 0.0:
        raise ValueError(f"q2/r2 must be positive, got q2={q2}, r2={r2}")
    return float(q2), float(r2)


def _params_block(task_cfg_dict: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    raw_params = task_cfg_dict.get("params")
    if isinstance(raw_params, Mapping):
        out = _deep_merge(out, raw_params)
    ssm = task_cfg_dict.get("ssm")
    if isinstance(ssm, Mapping):
        ssm_params = ssm.get("params")
        if isinstance(ssm_params, Mapping):
            out = _deep_merge(out, ssm_params)
    sp = task_cfg_dict.get("sine_poly")
    if isinstance(sp, Mapping):
        out = _deep_merge(out, sp)
    for k in _SP_PARAM_NAMES:
        if k in task_cfg_dict:
            out[k] = task_cfg_dict[k]

    sc_params = scenario_cfg.get("params")
    if isinstance(sc_params, Mapping):
        out = _deep_merge(out, sc_params)
    sc_ssm = scenario_cfg.get("ssm")
    if isinstance(sc_ssm, Mapping):
        sc_ssm_params = sc_ssm.get("params")
        if isinstance(sc_ssm_params, Mapping):
            out = _deep_merge(out, sc_ssm_params)
    sc_sp = scenario_cfg.get("sine_poly")
    if isinstance(sc_sp, Mapping):
        out = _deep_merge(out, sc_sp)
    for k in _SP_PARAM_NAMES:
        if k in scenario_cfg:
            out[k] = scenario_cfg[k]

    return out


def _true_params(param_cfg: Mapping[str, Any], *, dim: int) -> Dict[str, np.ndarray]:
    return {
        "alpha": _param_vector(param_cfg.get("alpha"), dim=dim, name="alpha", default=1.0),
        "beta": _param_vector(param_cfg.get("beta"), dim=dim, name="beta", default=1.0),
        "phi": _param_vector(param_cfg.get("phi"), dim=dim, name="phi", default=0.0),
        "delta": _param_vector(param_cfg.get("delta"), dim=dim, name="delta", default=0.0),
        "a": _param_vector(param_cfg.get("a"), dim=dim, name="a", default=1.0),
        "b": _param_vector(param_cfg.get("b"), dim=dim, name="b", default=1.0),
        "c": _param_vector(param_cfg.get("c"), dim=dim, name="c", default=0.0),
    }


def _assumed_params_from_mismatch(
    *,
    true_params: Mapping[str, np.ndarray],
    mismatch_cfg: Mapping[str, Any],
    dim: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    assumed = {k: np.asarray(v, dtype=np.float64).copy() for k, v in true_params.items()}
    if not bool(mismatch_cfg.get("enabled", False)):
        return assumed, {"enabled": False, "kind": "none", "params": {}}

    kind = str(mismatch_cfg.get("kind", "param_perturb"))
    params = mismatch_cfg.get("params", {})
    if not isinstance(params, Mapping):
        params = {}
    if kind != "param_perturb":
        raise ValueError("sine_poly mismatch.kind currently supports only 'param_perturb'")

    explicit = params.get("assumed", {})
    explicit_map = dict(explicit) if isinstance(explicit, Mapping) else {}
    which_raw = params.get("which", list(_SP_PARAM_NAMES))
    if isinstance(which_raw, (list, tuple)):
        which = [str(k) for k in which_raw if str(k) in _SP_PARAM_NAMES]
    else:
        which = [str(which_raw)] if str(which_raw) in _SP_PARAM_NAMES else []
    if not which:
        which = list(_SP_PARAM_NAMES)

    perturb_raw = params.get("perturb", {})
    perturb = dict(perturb_raw) if isinstance(perturb_raw, Mapping) else {}
    mode = str(perturb.get("mode", "add")).strip().lower()
    if mode not in {"add", "mul"}:
        raise ValueError("mismatch.params.perturb.mode must be 'add' or 'mul'")
    scale_raw = perturb.get("scale", 0.0 if mode == "add" else 1.0)
    if isinstance(scale_raw, Mapping):
        scale_map = {str(k): scale_raw[k] for k in scale_raw.keys()}
    else:
        scale_map = {}

    for name in _SP_PARAM_NAMES:
        if name in explicit_map:
            assumed[name] = _param_vector(explicit_map[name], dim=dim, name=f"assumed.{name}", default=0.0)
            continue
        if name not in which:
            continue
        true_v = np.asarray(true_params[name], dtype=np.float64)
        neutral = 0.0 if mode == "add" else 1.0
        if isinstance(scale_raw, Mapping):
            if name in scale_map:
                sc = _param_vector(scale_map[name], dim=dim, name=f"perturb.scale.{name}", default=neutral)
            else:
                sc = np.full((dim,), float(neutral), dtype=np.float64)
        else:
            sc = _param_vector(scale_raw, dim=dim, name="perturb.scale", default=neutral)
        if mode == "add":
            assumed[name] = true_v + sc
        else:
            assumed[name] = true_v * sc

    mm_meta = {
        "enabled": True,
        "kind": "param_perturb",
        "params": {
            "which": list(which),
            "perturb": {
                "mode": str(mode),
                "scale": scale_raw,
            },
            "assumed": {k: np.asarray(assumed[k], dtype=np.float64).tolist() for k in _SP_PARAM_NAMES},
        },
    }
    return assumed, mm_meta


def _resolve_mismatch_cfg(task_cfg_dict: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    raw = task_cfg_dict.get("mismatch", {})
    mm = dict(raw) if isinstance(raw, Mapping) else {}
    sc_mm = scenario_cfg.get("mismatch")
    if isinstance(sc_mm, Mapping):
        mm = _deep_merge(mm, sc_mm)
    return mm


def _f_step(x_t: np.ndarray, params: Mapping[str, np.ndarray]) -> np.ndarray:
    alpha = np.asarray(params["alpha"], dtype=np.float64)
    beta = np.asarray(params["beta"], dtype=np.float64)
    phi = np.asarray(params["phi"], dtype=np.float64)
    delta = np.asarray(params["delta"], dtype=np.float64)
    return alpha * np.sin(beta * x_t + phi) + delta


def _h_step(x_t: np.ndarray, params: Mapping[str, np.ndarray]) -> np.ndarray:
    a = np.asarray(params["a"], dtype=np.float64)
    b = np.asarray(params["b"], dtype=np.float64)
    c = np.asarray(params["c"], dtype=np.float64)
    return a * np.square(b * x_t + c)


def _rollout_sine_poly(
    *,
    rng: np.random.Generator,
    n_seq: int,
    t_len: int,
    x_dim: int,
    y_dim: int,
    q2: float,
    r2: float,
    true_params: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if int(y_dim) != int(x_dim):
        raise ValueError("sine_poly_v0 currently requires y_dim == x_dim (component-wise observation)")

    x = np.zeros((int(n_seq), int(t_len), int(x_dim)), dtype=np.float64)
    y = np.zeros((int(n_seq), int(t_len), int(y_dim)), dtype=np.float64)

    x_t = np.asarray(rng.standard_normal((int(n_seq), int(x_dim))), dtype=np.float64)
    q_std = np.sqrt(float(q2))
    r_std = np.sqrt(float(r2))

    for t in range(int(t_len)):
        y_det = _h_step(x_t, true_params)
        v_t = r_std * np.asarray(rng.standard_normal((int(n_seq), int(y_dim))), dtype=np.float64)
        y_t = y_det + v_t
        x[:, t, :] = x_t
        y[:, t, :] = y_t

        x_det_next = _f_step(x_t, true_params)
        w_t = q_std * np.asarray(rng.standard_normal((int(n_seq), int(x_dim))), dtype=np.float64)
        x_t = x_det_next + w_t

    return x, y


def _params_to_jsonable(params: Mapping[str, np.ndarray]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for k in _SP_PARAM_NAMES:
        out[k] = [float(v) for v in np.asarray(params[k], dtype=np.float64).tolist()]
    return out


def generate_sine_poly_v0(
    *,
    suite_name: str,
    task_cfg_dict: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    seed: int,
    scenario_id: str,
    task_family: str = "sine_poly_v0",
) -> Tuple[GeneratorOutput, np.ndarray, np.ndarray]:
    task_cfg = make_task_cfg(task_cfg_dict, scenario_cfg=scenario_cfg)
    split_cfg = make_split_cfg(task_cfg_dict)

    x_dim = int(task_cfg.x_dim)
    y_dim = int(task_cfg.y_dim)
    t_len = int(task_cfg.sequence_length_T)
    n_total = int(split_cfg.n_total)
    if x_dim <= 0 or y_dim <= 0:
        raise ValueError(f"sine_poly_v0 requires positive dims, got x_dim={x_dim}, y_dim={y_dim}")
    if y_dim != x_dim:
        raise ValueError(f"sine_poly_v0 currently requires y_dim==x_dim, got x_dim={x_dim}, y_dim={y_dim}")

    q2, r2 = _resolve_q2_r2(dict(task_cfg.noise))
    param_cfg = _params_block(task_cfg_dict, scenario_cfg)
    true_params = _true_params(param_cfg, dim=x_dim)
    mismatch_cfg = _resolve_mismatch_cfg(task_cfg_dict, scenario_cfg)
    assumed_params, mismatch_meta = _assumed_params_from_mismatch(
        true_params=true_params,
        mismatch_cfg=mismatch_cfg,
        dim=x_dim,
    )

    data_seed = stable_int_seed_v0("data", suite_name, task_cfg.task_id, scenario_id, int(seed))
    rng_data = numpy_rng_v0(data_seed)
    x_all, y_all = _rollout_sine_poly(
        rng=rng_data,
        n_seq=n_total,
        t_len=t_len,
        x_dim=x_dim,
        y_dim=y_dim,
        q2=float(q2),
        r2=float(r2),
        true_params=true_params,
    )

    q_mat = float(q2) * np.eye(x_dim, dtype=np.float64)
    r_mat = float(r2) * np.eye(y_dim, dtype=np.float64)
    f_proxy = np.eye(x_dim, dtype=np.float64)
    h_proxy = np.eye(y_dim, x_dim, dtype=np.float64)

    true_params_json = _params_to_jsonable(true_params)
    assumed_params_json = _params_to_jsonable(assumed_params)
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
        "observation": _deep_merge(dict(task_cfg.observation), {"mode": "nonlinear_componentwise"}),
        "noise": {
            "Q": {"type": "scaled_identity", "q2": float(q2)},
            "R": {"type": "scaled_identity", "r2": float(r2)},
            "obs_distribution": {"name": "gaussian"},
        },
        "ssm": {
            "true": {
                "type": "sine_poly",
                "dims": {"x_dim": int(x_dim), "y_dim": int(y_dim)},
                "params": _deep_merge(true_params_json, {"x0_kind": "standard_normal"}),
                "Q": _matrix_to_json(q_mat),
                "R": _matrix_to_json(r_mat),
                "q2": float(q2),
                "r2": float(r2),
                "f": "alpha*sin(beta*x+phi)+delta",
                "h": "a*(b*x+c)^2",
            },
            "assumed": {
                "type": "sine_poly",
                "dims": {"x_dim": int(x_dim), "y_dim": int(y_dim)},
                "params": _deep_merge(assumed_params_json, {"x0_kind": "standard_normal"}),
                "Q": _matrix_to_json(q_mat),
                "R": _matrix_to_json(r_mat),
                "q2": float(q2),
                "r2": float(r2),
                "f": "alpha*sin(beta*x+phi)+delta",
                "h": "a*(b*x+c)^2",
            },
        },
        "mismatch": mismatch_meta,
        "noise_schedule": {
            "enabled": False,
            "kind": "stationary",
            "params": {"q2_base": float(q2), "r2_base": float(r2)},
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

    out = coerce_ntd_float32_output(
        GeneratorOutput(
            x=x_all.astype(np.float32, copy=False),
            y=y_all.astype(np.float32, copy=False),
            meta=meta_common,
            extras={},
        )
    )
    return out, f_proxy.astype(np.float32), h_proxy.astype(np.float32)
