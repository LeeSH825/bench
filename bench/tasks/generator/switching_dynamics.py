from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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


def _coerce_matrix(value: Any, *, shape: Tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def _diag_observation(y_dim: int, x_dim: int) -> np.ndarray:
    h = np.zeros((int(y_dim), int(x_dim)), dtype=np.float64)
    d = min(int(y_dim), int(x_dim))
    for i in range(d):
        h[i, i] = 1.0
    return h


def _resolve_global_q2_r2(noise_cfg: Mapping[str, Any]) -> Tuple[float, float]:
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

    if q2 is None:
        q2 = 1.0e-3
    if r2 is None:
        r2 = 1.0e-3
    if q2 <= 0.0 or r2 <= 0.0:
        raise ValueError(f"q2/r2 must be positive, got q2={q2}, r2={r2}")
    return float(q2), float(r2)


def _resolve_model_cfg(
    *,
    name: str,
    task_cfg_dict: Mapping[str, Any],
    scenario_cfg: Mapping[str, Any],
    x_dim: int,
    y_dim: int,
    q2_default: float,
    r2_default: float,
) -> Dict[str, Any]:
    key = f"model_{str(name).upper()}"
    key_lo = f"model_{str(name).lower()}"
    raw = task_cfg_dict.get(key)
    if raw is None:
        raw = task_cfg_dict.get(key_lo, {})
    model_cfg = dict(raw) if isinstance(raw, Mapping) else {}

    sc_raw = scenario_cfg.get(key)
    if sc_raw is None:
        sc_raw = scenario_cfg.get(key_lo)
    if isinstance(sc_raw, Mapping):
        model_cfg = _deep_merge(model_cfg, sc_raw)

    f_raw = model_cfg.get("F")
    h_raw = model_cfg.get("H")

    if f_raw is None:
        if str(name).upper() == "A":
            f = np.eye(int(x_dim), dtype=np.float64)
        else:
            f = -np.eye(int(x_dim), dtype=np.float64)
    else:
        f = _coerce_matrix(f_raw, shape=(int(x_dim), int(x_dim)), name=f"model_{name}.F")

    if h_raw is None:
        h = _diag_observation(int(y_dim), int(x_dim))
    else:
        h = _coerce_matrix(h_raw, shape=(int(y_dim), int(x_dim)), name=f"model_{name}.H")

    q2: Optional[float] = None
    r2: Optional[float] = None
    if "q2" in model_cfg:
        q2 = float(model_cfg["q2"])
    if "r2" in model_cfg:
        r2 = float(model_cfg["r2"])
    q_map = model_cfg.get("Q")
    if q2 is None and isinstance(q_map, Mapping) and "q2" in q_map:
        q2 = float(q_map["q2"])
    r_map = model_cfg.get("R")
    if r2 is None and isinstance(r_map, Mapping) and "r2" in r_map:
        r2 = float(r_map["r2"])
    if q2 is None:
        q2 = float(q2_default)
    if r2 is None:
        r2 = float(r2_default)
    if q2 <= 0.0 or r2 <= 0.0:
        raise ValueError(f"model_{name} q2/r2 must be positive, got q2={q2}, r2={r2}")

    return {
        "F": f,
        "H": h,
        "q2": float(q2),
        "r2": float(r2),
    }


def _resolve_switching_cfg(
    *,
    task_cfg_dict: Mapping[str, Any],
    scenario_cfg: Mapping[str, Any],
    t_len: int,
) -> Dict[str, Any]:
    raw = task_cfg_dict.get("switching", {})
    cfg = dict(raw) if isinstance(raw, Mapping) else {}
    sc_raw = scenario_cfg.get("switching")
    if isinstance(sc_raw, Mapping):
        cfg = _deep_merge(cfg, sc_raw)

    if "t_change" in cfg:
        t_val = int(cfg["t_change"])
        lo = t_val
        hi = t_val
        sample_per_sequence = bool(cfg.get("sample_per_sequence", False))
    else:
        range_raw = cfg.get("t_change_range", cfg.get("t_change_window", [int(t_len // 2), int(t_len // 2)]))
        if not isinstance(range_raw, Sequence) or len(range_raw) != 2:
            raise ValueError("switching t_change_range/t_change_window must be [t0,t1]")
        lo = int(range_raw[0])
        hi = int(range_raw[1])
        if hi < lo:
            lo, hi = hi, lo
        sample_per_sequence = bool(cfg.get("sample_per_sequence", lo != hi))

    lo = max(0, min(int(lo), int(t_len) - 1))
    hi = max(0, min(int(hi), int(t_len) - 1))
    if hi < lo:
        lo, hi = hi, lo
    retrain_window = int(cfg.get("retrain_window", 0))
    return {
        "t_change_lo": int(lo),
        "t_change_hi": int(hi),
        "sample_per_sequence": bool(sample_per_sequence),
        "retrain_window": int(retrain_window),
    }


def _resolve_assumed_policy(task_cfg_dict: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> str:
    raw = scenario_cfg.get("assumed_policy", task_cfg_dict.get("assumed_policy", "A_only"))
    if isinstance(raw, Mapping):
        raw = raw.get("policy", "A_only")
    policy = str(raw).strip().lower()
    if policy in {"a_only", "a", "model_a_only"}:
        return "A_only"
    if policy in {"oracle_piecewise", "oracle"}:
        return "oracle_piecewise"
    raise ValueError("assumed_policy must be one of: A_only, oracle_piecewise")


def _sample_t_change_seq(
    *,
    n_total: int,
    lo: int,
    hi: int,
    sample_per_sequence: bool,
    suite_name: str,
    task_id: str,
    scenario_id: str,
    seed: int,
) -> np.ndarray:
    if not bool(sample_per_sequence) or int(lo) == int(hi):
        return np.full((int(n_total),), int(lo), dtype=np.int64)
    ts_seed = stable_int_seed_v0("switching_t_change", suite_name, task_id, scenario_id, int(seed))
    rng = numpy_rng_v0(ts_seed)
    return np.asarray(rng.integers(int(lo), int(hi) + 1, size=int(n_total)), dtype=np.int64)


def _rollout_linear_switching(
    *,
    rng: np.random.Generator,
    n_seq: int,
    t_len: int,
    x_dim: int,
    y_dim: int,
    model_A: Mapping[str, Any],
    model_B: Mapping[str, Any],
    t_change_seq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    f_a = np.asarray(model_A["F"], dtype=np.float64)
    h_a = np.asarray(model_A["H"], dtype=np.float64)
    q2_a = float(model_A["q2"])
    r2_a = float(model_A["r2"])
    f_b = np.asarray(model_B["F"], dtype=np.float64)
    h_b = np.asarray(model_B["H"], dtype=np.float64)
    q2_b = float(model_B["q2"])
    r2_b = float(model_B["r2"])

    x = np.zeros((int(n_seq), int(t_len), int(x_dim)), dtype=np.float64)
    y = np.zeros((int(n_seq), int(t_len), int(y_dim)), dtype=np.float64)

    x_t = np.asarray(rng.standard_normal((int(n_seq), int(x_dim))), dtype=np.float64)
    t_change_seq_i64 = np.asarray(t_change_seq, dtype=np.int64)
    if t_change_seq_i64.shape != (int(n_seq),):
        raise ValueError(f"t_change_seq must have shape [{n_seq}], got {t_change_seq_i64.shape}")

    for t in range(int(t_len)):
        is_post = (int(t) >= t_change_seq_i64)
        y_det_a = x_t @ h_a.T
        y_det_b = x_t @ h_b.T
        y_det = np.where(is_post[:, None], y_det_b, y_det_a)

        r_std = np.where(is_post, np.sqrt(r2_b), np.sqrt(r2_a))
        v_t = np.asarray(rng.standard_normal((int(n_seq), int(y_dim))), dtype=np.float64) * r_std[:, None]
        y_t = y_det + v_t

        x[:, t, :] = x_t
        y[:, t, :] = y_t

        if t < int(t_len) - 1:
            x_det_a = x_t @ f_a.T
            x_det_b = x_t @ f_b.T
            x_det = np.where(is_post[:, None], x_det_b, x_det_a)
            q_std = np.where(is_post, np.sqrt(q2_b), np.sqrt(q2_a))
            w_t = np.asarray(rng.standard_normal((int(n_seq), int(x_dim))), dtype=np.float64) * q_std[:, None]
            x_t = x_det + w_t

    return x, y


def generate_switching_dynamics_v0(
    *,
    suite_name: str,
    task_cfg_dict: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    seed: int,
    scenario_id: str,
    task_family: str = "switching_dynamics_v0",
) -> Tuple[GeneratorOutput, np.ndarray, np.ndarray]:
    task_cfg = make_task_cfg(task_cfg_dict, scenario_cfg=scenario_cfg)
    split_cfg = make_split_cfg(task_cfg_dict)

    x_dim = int(task_cfg.x_dim)
    y_dim = int(task_cfg.y_dim)
    t_len = int(task_cfg.sequence_length_T)
    n_total = int(split_cfg.n_total)
    if x_dim <= 0 or y_dim <= 0:
        raise ValueError(f"switching_dynamics_v0 requires positive dims, got x_dim={x_dim}, y_dim={y_dim}")

    q2_default, r2_default = _resolve_global_q2_r2(dict(task_cfg.noise))
    model_A = _resolve_model_cfg(
        name="A",
        task_cfg_dict=task_cfg_dict,
        scenario_cfg=scenario_cfg,
        x_dim=x_dim,
        y_dim=y_dim,
        q2_default=q2_default,
        r2_default=r2_default,
    )
    model_B = _resolve_model_cfg(
        name="B",
        task_cfg_dict=task_cfg_dict,
        scenario_cfg=scenario_cfg,
        x_dim=x_dim,
        y_dim=y_dim,
        q2_default=q2_default,
        r2_default=r2_default,
    )
    sw_cfg = _resolve_switching_cfg(task_cfg_dict=task_cfg_dict, scenario_cfg=scenario_cfg, t_len=t_len)
    assumed_policy = _resolve_assumed_policy(task_cfg_dict, scenario_cfg)

    t_change_seq = _sample_t_change_seq(
        n_total=n_total,
        lo=int(sw_cfg["t_change_lo"]),
        hi=int(sw_cfg["t_change_hi"]),
        sample_per_sequence=bool(sw_cfg["sample_per_sequence"]),
        suite_name=suite_name,
        task_id=task_cfg.task_id,
        scenario_id=scenario_id,
        seed=int(seed),
    )

    data_seed = stable_int_seed_v0("data", suite_name, task_cfg.task_id, scenario_id, int(seed))
    rng_data = numpy_rng_v0(data_seed)
    x_all, y_all = _rollout_linear_switching(
        rng=rng_data,
        n_seq=n_total,
        t_len=t_len,
        x_dim=x_dim,
        y_dim=y_dim,
        model_A=model_A,
        model_B=model_B,
        t_change_seq=t_change_seq,
    )

    f_a = np.asarray(model_A["F"], dtype=np.float64)
    h_a = np.asarray(model_A["H"], dtype=np.float64)
    q_a = float(model_A["q2"]) * np.eye(x_dim, dtype=np.float64)
    r_a = float(model_A["r2"]) * np.eye(y_dim, dtype=np.float64)
    f_b = np.asarray(model_B["F"], dtype=np.float64)
    h_b = np.asarray(model_B["H"], dtype=np.float64)
    q_b = float(model_B["q2"]) * np.eye(x_dim, dtype=np.float64)
    r_b = float(model_B["r2"]) * np.eye(y_dim, dtype=np.float64)

    ssm_true = {
        "type": "linear_switching",
        "models": {
            "A": {
                "F": _matrix_to_json(f_a),
                "H": _matrix_to_json(h_a),
                "Q": _matrix_to_json(q_a),
                "R": _matrix_to_json(r_a),
                "q2": float(model_A["q2"]),
                "r2": float(model_A["r2"]),
            },
            "B": {
                "F": _matrix_to_json(f_b),
                "H": _matrix_to_json(h_b),
                "Q": _matrix_to_json(q_b),
                "R": _matrix_to_json(r_b),
                "q2": float(model_B["q2"]),
                "r2": float(model_B["r2"]),
            },
        },
    }
    if assumed_policy == "oracle_piecewise":
        ssm_assumed = {
            "type": "linear_switching_piecewise",
            "policy": "oracle_piecewise",
            "models": {
                "A": ssm_true["models"]["A"],
                "B": ssm_true["models"]["B"],
            },
            "t_change_source": "npz_extras:t_change_seq",
            # Keep A-only fallback fields for baseline compatibility.
            "F": _matrix_to_json(f_a),
            "H": _matrix_to_json(h_a),
            "Q": _matrix_to_json(q_a),
            "R": _matrix_to_json(r_a),
            "q2": float(model_A["q2"]),
            "r2": float(model_A["r2"]),
            "fallback_policy": "A_only",
        }
    else:
        ssm_assumed = {
            "type": "linear_gaussian",
            "policy": "A_only",
            "model": "A",
            "F": _matrix_to_json(f_a),
            "H": _matrix_to_json(h_a),
            "Q": _matrix_to_json(q_a),
            "R": _matrix_to_json(r_a),
            "q2": float(model_A["q2"]),
            "r2": float(model_A["r2"]),
        }

    t_lo = int(sw_cfg["t_change_lo"])
    t_hi = int(sw_cfg["t_change_hi"])
    sample_per_sequence = bool(sw_cfg["sample_per_sequence"])
    retrain_window = int(sw_cfg["retrain_window"])
    switching_block = {
        "enabled": True,
        "models": ["A", "B"],
        "t_change": [int(t_lo), int(t_hi)],
        "retrain_window": int(retrain_window),
        "sample_per_sequence": bool(sample_per_sequence),
        "storage": {"t_change_seq": "npz_extras:t_change_seq"},
    }
    mismatch_block = {
        "enabled": True,
        "kind": "switching_dynamics",
        "params": {
            "assumed_policy": str(assumed_policy),
            "t_change_range": [int(t_lo), int(t_hi)],
            "sample_per_sequence": bool(sample_per_sequence),
            "retrain_window": int(retrain_window),
        },
    }
    noise_schedule_block = {
        "enabled": False,
        "kind": "stationary",
        "params": {
            "q2_A": float(model_A["q2"]),
            "r2_A": float(model_A["r2"]),
            "q2_B": float(model_B["q2"]),
            "r2_B": float(model_B["r2"]),
        },
        "q2_t": "meta.ssm.true.models.{A,B}.q2",
        "r2_t": "meta.ssm.true.models.{A,B}.r2",
        "SoW_t": "meta.noise_schedule.derived_from_q2_over_r2",
        "SoW_hat_t": None,
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
            "Q": {"type": "scaled_identity", "q2": float(model_A["q2"])},
            "R": {"type": "scaled_identity", "r2": float(model_A["r2"])},
            "obs_distribution": {"name": "gaussian"},
        },
        "ssm": {
            "true": ssm_true,
            "assumed": ssm_assumed,
        },
        "mismatch": mismatch_block,
        "noise_schedule": noise_schedule_block,
        "switching": switching_block,
    }
    extras: Dict[str, Any] = {
        "t_change_seq": np.asarray(t_change_seq, dtype=np.float32),
    }

    out = coerce_ntd_float32_output(
        GeneratorOutput(
            x=x_all.astype(np.float32, copy=False),
            y=y_all.astype(np.float32, copy=False),
            meta=meta_common,
            extras=extras,
        )
    )

    # Return assumed A-model matrices for baseline compatibility.
    return out, f_a.astype(np.float32), h_a.astype(np.float32)
