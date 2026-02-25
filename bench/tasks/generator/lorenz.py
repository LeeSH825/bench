from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ..data_format import CANONICAL_LAYOUT_V0
from ..generator.contract import GeneratorOutput, coerce_ntd_float32_output, make_split_cfg, make_task_cfg
from ..generator.datasets.common import INTERNAL_SPLIT_PAYLOADS_KEY
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
        raise ValueError("lorenz_v0 requires q2/r2 from noise.Q/noise.R or noise.q2/noise.r2")
    if q2 <= 0.0 or r2 <= 0.0:
        raise ValueError(f"lorenz_v0 q2/r2 must be positive, got q2={q2}, r2={r2}")
    return float(q2), float(r2)


def _lorenz_cfg_block(task_cfg_dict: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    ssm = task_cfg_dict.get("ssm")
    if isinstance(ssm, Mapping):
        p = ssm.get("params")
        if isinstance(p, Mapping):
            out = _deep_merge(out, p)
    lz = task_cfg_dict.get("lorenz")
    if isinstance(lz, Mapping):
        out = _deep_merge(out, lz)

    for k in ("sigma", "rho", "beta", "J_true", "J_assumed", "delta_tau_true", "delta_tau_assumed"):
        if k in task_cfg_dict:
            out[k] = task_cfg_dict[k]

    sc_ssm = scenario_cfg.get("ssm")
    if isinstance(sc_ssm, Mapping):
        sc_p = sc_ssm.get("params")
        if isinstance(sc_p, Mapping):
            out = _deep_merge(out, sc_p)
    sc_lz = scenario_cfg.get("lorenz")
    if isinstance(sc_lz, Mapping):
        out = _deep_merge(out, sc_lz)
    for k in ("sigma", "rho", "beta", "J_true", "J_assumed", "delta_tau_true", "delta_tau_assumed"):
        if k in scenario_cfg:
            out[k] = scenario_cfg[k]

    return out


def _resolve_obs_mode(task_cfg_dict: Mapping[str, Any], scenario_cfg: Mapping[str, Any], lz_cfg: Mapping[str, Any]) -> str:
    mode_raw: Any = lz_cfg.get("obs_mode")
    obs = task_cfg_dict.get("observation")
    if isinstance(obs, Mapping):
        mode_raw = obs.get("mode", obs.get("h_type", mode_raw))
    sc_obs = scenario_cfg.get("observation")
    if isinstance(sc_obs, Mapping):
        mode_raw = sc_obs.get("mode", sc_obs.get("h_type", mode_raw))
    if "obs_mode" in task_cfg_dict:
        mode_raw = task_cfg_dict["obs_mode"]
    if "obs_mode" in scenario_cfg:
        mode_raw = scenario_cfg["obs_mode"]

    mode = str(mode_raw if mode_raw is not None else "identity").strip().lower()
    if mode in {"identity", "full", "linear"}:
        return "identity"
    if mode in {"partial", "select_first_dims", "select_first_two_dims"}:
        return "partial"
    raise ValueError("lorenz_v0 obs_mode must be one of: identity/full/linear, partial")


def _resolve_mismatch_cfg(task_cfg_dict: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    raw = task_cfg_dict.get("mismatch", {})
    mm = dict(raw) if isinstance(raw, Mapping) else {}
    sc_mm = scenario_cfg.get("mismatch")
    if isinstance(sc_mm, Mapping):
        mm = _deep_merge(mm, sc_mm)

    params = mm.get("params", {})
    params_map = dict(params) if isinstance(params, Mapping) else {}

    obs_rotation_deg = float(
        params_map.get(
            "obs_rotation_deg",
            mm.get(
                "obs_rotation_deg",
                0.0,
            ),
        )
    )
    obs_rotation_assumed_deg = float(
        params_map.get(
            "obs_rotation_assumed_deg",
            mm.get("obs_rotation_assumed_deg", 0.0),
        )
    )

    sampling_decimation = int(
        params_map.get(
            "sampling_decimation",
            params_map.get("decimation", mm.get("sampling_decimation", 1)),
        )
    )
    if sampling_decimation <= 0:
        raise ValueError(f"sampling_decimation must be >=1, got {sampling_decimation}")

    time_axis_shift_steps = int(
        params_map.get(
            "time_axis_shift_steps",
            params_map.get("shift_steps", mm.get("time_axis_shift_steps", 0)),
        )
    )
    time_axis_resample_stride = int(
        params_map.get(
            "time_axis_resample_stride",
            params_map.get("resample_stride", mm.get("time_axis_resample_stride", 1)),
        )
    )
    if time_axis_resample_stride <= 0:
        raise ValueError(f"time_axis_resample_stride must be >=1, got {time_axis_resample_stride}")

    return {
        "enabled": bool(mm.get("enabled", False)),
        "obs_rotation_deg": float(obs_rotation_deg),
        "obs_rotation_assumed_deg": float(obs_rotation_assumed_deg),
        "sampling_decimation": int(sampling_decimation),
        "time_axis_shift_steps": int(time_axis_shift_steps),
        "time_axis_resample_stride": int(time_axis_resample_stride),
    }


def _resolve_split_plan(
    *,
    task_cfg_dict: Mapping[str, Any],
    n_train: int,
    n_val: int,
    n_test: int,
    default_t: int,
) -> Dict[str, Dict[str, int]]:
    raw = task_cfg_dict.get("splits", {})
    splits = dict(raw) if isinstance(raw, Mapping) else {}

    def one(split: str, default_n: int) -> Dict[str, int]:
        s_raw = splits.get(split, {})
        s_cfg = dict(s_raw) if isinstance(s_raw, Mapping) else {}
        return {
            "N": int(s_cfg.get("N", default_n)),
            "T": int(s_cfg.get("T", default_t)),
        }

    plan = {
        "train": one("train", int(n_train)),
        "val": one("val", int(n_val)),
        "test": one("test", int(n_test)),
    }
    for split_name, cfg in plan.items():
        if cfg["N"] < 0:
            raise ValueError(f"lorenz_v0 split '{split_name}' N must be >=0, got {cfg['N']}")
        if cfg["T"] <= 0:
            raise ValueError(f"lorenz_v0 split '{split_name}' T must be >0, got {cfg['T']}")
    return plan


def _rotation_y(y_dim: int, deg: float) -> np.ndarray:
    r = np.eye(int(y_dim), dtype=np.float64)
    if abs(float(deg)) <= 0.0:
        return r
    if int(y_dim) < 2:
        raise ValueError(f"obs rotation requires y_dim>=2, got y_dim={y_dim}")
    th = math.radians(float(deg))
    c, s = math.cos(th), math.sin(th)
    r[:2, :2] = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return r


def _observation_matrices(
    *,
    x_dim: int,
    y_dim: int,
    obs_mode: str,
    obs_rotation_deg: float,
    obs_rotation_assumed_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if int(x_dim) != 3:
        raise ValueError(f"lorenz_v0 currently expects x_dim=3, got x_dim={x_dim}")
    if int(y_dim) <= 0 or int(y_dim) > int(x_dim):
        raise ValueError(f"lorenz_v0 expects 1<=y_dim<=x_dim, got y_dim={y_dim}, x_dim={x_dim}")

    h_base = np.zeros((int(y_dim), int(x_dim)), dtype=np.float64)
    if obs_mode == "identity":
        for i in range(int(y_dim)):
            h_base[i, i] = 1.0
    elif obs_mode == "partial":
        for i in range(int(y_dim)):
            h_base[i, i] = 1.0
    else:
        raise ValueError(f"unsupported obs_mode: {obs_mode}")

    h_true = _rotation_y(int(y_dim), float(obs_rotation_deg)) @ h_base
    h_assumed = _rotation_y(int(y_dim), float(obs_rotation_assumed_deg)) @ h_base
    return h_true, h_assumed


def _lorenz_jacobian(x: np.ndarray, *, sigma: float, rho: float, beta: float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 2 or int(x_arr.shape[1]) != 3:
        raise ValueError(f"lorenz state must be [N,3], got {x_arr.shape}")

    n = int(x_arr.shape[0])
    a = np.zeros((n, 3, 3), dtype=np.float64)
    x1 = x_arr[:, 0]
    x2 = x_arr[:, 1]
    x3 = x_arr[:, 2]

    a[:, 0, 0] = -float(sigma)
    a[:, 0, 1] = float(sigma)
    a[:, 1, 0] = float(rho) - x3
    a[:, 1, 1] = -1.0
    a[:, 1, 2] = -x1
    a[:, 2, 0] = x2
    a[:, 2, 1] = x1
    a[:, 2, 2] = -float(beta)
    return a


def _lorenz_taylor_step(
    x: np.ndarray,
    *,
    sigma: float,
    rho: float,
    beta: float,
    delta_tau: float,
    J: int,
) -> np.ndarray:
    x_now = np.asarray(x, dtype=np.float64)
    n = int(x_now.shape[0])
    a = _lorenz_jacobian(x_now, sigma=sigma, rho=rho, beta=beta)
    adt = a * float(delta_tau)
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), (n, 3, 3)).copy()
    f = eye.copy()
    term = eye.copy()
    for j in range(1, int(J) + 1):
        term = np.matmul(term, adt) / float(j)
        f = f + term
    return np.einsum("nij,nj->ni", f, x_now)


def _observation_source_idx(T: int, *, shift_steps: int, resample_stride: int) -> np.ndarray:
    base = np.arange(int(T), dtype=np.int64) * int(resample_stride) + int(shift_steps)
    clipped = np.clip(base, 0, int(T) - 1)
    return np.asarray(clipped, dtype=np.int64)


def _rollout_lorenz_split(
    *,
    rng: np.random.Generator,
    n_seq: int,
    T: int,
    sigma: float,
    rho: float,
    beta: float,
    J_true: int,
    delta_tau_true: float,
    q2: float,
    r2: float,
    h_true: np.ndarray,
    sampling_decimation: int,
    time_axis_shift_steps: int,
    time_axis_resample_stride: int,
    state_clip: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(n_seq)
    t_len = int(T)
    y_dim = int(h_true.shape[0])

    x = np.zeros((n, t_len, 3), dtype=np.float64)
    if n > 0:
        x[:, 0, :] = np.asarray(rng.standard_normal((n, 3)), dtype=np.float64)

    q_std = math.sqrt(float(q2))
    r_std = math.sqrt(float(r2))

    for t in range(max(0, t_len - 1)):
        x_next = np.asarray(x[:, t, :], dtype=np.float64)
        for _ in range(int(sampling_decimation)):
            x_next = _lorenz_taylor_step(
                x_next,
                sigma=float(sigma),
                rho=float(rho),
                beta=float(beta),
                delta_tau=float(delta_tau_true),
                J=int(J_true),
            )
            if n > 0:
                x_next = x_next + q_std * np.asarray(rng.standard_normal((n, 3)), dtype=np.float64)
            if float(state_clip) > 0.0:
                x_next = np.clip(x_next, -float(state_clip), float(state_clip))
        x[:, t + 1, :] = x_next

    obs_idx = _observation_source_idx(
        int(t_len),
        shift_steps=int(time_axis_shift_steps),
        resample_stride=int(time_axis_resample_stride),
    )
    x_obs = x[:, obs_idx, :]
    y_det = np.matmul(x_obs, h_true.T)
    if n > 0:
        y = y_det + r_std * np.asarray(rng.standard_normal((n, t_len, y_dim)), dtype=np.float64)
    else:
        y = np.zeros((0, t_len, y_dim), dtype=np.float64)
    return x.astype(np.float32), y.astype(np.float32), obs_idx.astype(np.float32)


def generate_lorenz_v0(
    *,
    suite_name: str,
    task_cfg_dict: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    seed: int,
    scenario_id: str,
    task_family: str = "lorenz_v0",
) -> Tuple[GeneratorOutput, Optional[np.ndarray], Optional[np.ndarray]]:
    task_cfg = make_task_cfg(task_cfg_dict, scenario_cfg=scenario_cfg)
    split_cfg = make_split_cfg(task_cfg_dict)

    x_dim = int(task_cfg.x_dim)
    y_dim = int(task_cfg.y_dim)
    if int(x_dim) != 3:
        raise ValueError(f"lorenz_v0 currently supports x_dim=3 only, got x_dim={x_dim}")
    if int(y_dim) <= 0 or int(y_dim) > int(x_dim):
        raise ValueError(f"lorenz_v0 requires 1<=y_dim<=x_dim, got y_dim={y_dim}, x_dim={x_dim}")

    q2, r2 = _resolve_q2_r2(dict(task_cfg.noise))
    lz_cfg = _lorenz_cfg_block(task_cfg_dict, scenario_cfg)
    mm_cfg = _resolve_mismatch_cfg(task_cfg_dict, scenario_cfg)

    sigma = float(lz_cfg.get("sigma", 10.0))
    rho = float(lz_cfg.get("rho", 28.0))
    beta = float(lz_cfg.get("beta", 8.0 / 3.0))
    J_true = int(lz_cfg.get("J_true", lz_cfg.get("J", 5)))
    J_assumed = int(lz_cfg.get("J_assumed", J_true))
    delta_tau_true = float(lz_cfg.get("delta_tau_true", lz_cfg.get("delta_tau", 0.02)))
    delta_tau_assumed = float(lz_cfg.get("delta_tau_assumed", delta_tau_true))
    if J_true <= 0 or J_assumed <= 0:
        raise ValueError(f"lorenz_v0 J_true/J_assumed must be >=1, got J_true={J_true}, J_assumed={J_assumed}")
    if delta_tau_true <= 0.0 or delta_tau_assumed <= 0.0:
        raise ValueError(
            f"lorenz_v0 delta_tau_true/delta_tau_assumed must be >0, "
            f"got {delta_tau_true}, {delta_tau_assumed}"
        )

    obs_mode = _resolve_obs_mode(task_cfg_dict, scenario_cfg, lz_cfg)
    h_true, h_assumed = _observation_matrices(
        x_dim=x_dim,
        y_dim=y_dim,
        obs_mode=obs_mode,
        obs_rotation_deg=float(mm_cfg["obs_rotation_deg"]),
        obs_rotation_assumed_deg=float(mm_cfg["obs_rotation_assumed_deg"]),
    )

    sampling_decimation = int(mm_cfg["sampling_decimation"])
    time_axis_shift_steps = int(mm_cfg["time_axis_shift_steps"])
    time_axis_resample_stride = int(mm_cfg["time_axis_resample_stride"])
    state_clip = float(lz_cfg.get("state_clip", 100.0))

    split_plan = _resolve_split_plan(
        task_cfg_dict=task_cfg_dict,
        n_train=int(split_cfg.n_train),
        n_val=int(split_cfg.n_val),
        n_test=int(split_cfg.n_test),
        default_t=int(task_cfg.sequence_length_T),
    )

    split_payloads: Dict[str, Dict[str, Any]] = {}
    split_meta: Dict[str, Any] = {}

    for split_name in ("train", "val", "test"):
        n_split = int(split_plan[split_name]["N"])
        t_split = int(split_plan[split_name]["T"])
        split_seed = stable_int_seed_v0("data", suite_name, task_cfg.task_id, scenario_id, int(seed), split_name)
        rng = numpy_rng_v0(split_seed)
        x_split, y_split, obs_idx_t = _rollout_lorenz_split(
            rng=rng,
            n_seq=n_split,
            T=t_split,
            sigma=sigma,
            rho=rho,
            beta=beta,
            J_true=J_true,
            delta_tau_true=delta_tau_true,
            q2=q2,
            r2=r2,
            h_true=h_true,
            sampling_decimation=sampling_decimation,
            time_axis_shift_steps=time_axis_shift_steps,
            time_axis_resample_stride=time_axis_resample_stride,
            state_clip=state_clip,
        )
        split_payloads[split_name] = {
            "x": x_split.astype(np.float32, copy=False),
            "y": y_split.astype(np.float32, copy=False),
            "extras": {
                "obs_src_t": obs_idx_t.astype(np.float32, copy=False),
            },
        }
        split_meta[split_name] = {
            "N": int(n_split),
            "L": int(n_split),
            "T": int(t_split),
            "sampling_decimation": int(sampling_decimation),
            "time_axis_shift_steps": int(time_axis_shift_steps),
            "time_axis_resample_stride": int(time_axis_resample_stride),
        }

    mismatch_kinds = []
    if int(J_assumed) != int(J_true) or abs(float(delta_tau_assumed) - float(delta_tau_true)) > 0.0:
        mismatch_kinds.append("lorenz_J")
    if abs(float(mm_cfg["obs_rotation_deg"]) - float(mm_cfg["obs_rotation_assumed_deg"])) > 0.0:
        mismatch_kinds.append("obs_rotation")
    if int(sampling_decimation) != 1:
        mismatch_kinds.append("sampling_decimation")
    if int(time_axis_shift_steps) != 0 or int(time_axis_resample_stride) != 1:
        mismatch_kinds.append("time_axis_shift")

    mismatch_enabled = bool(mismatch_kinds) or bool(mm_cfg.get("enabled", False))
    if not mismatch_kinds:
        mismatch_kind_value: Any = "none"
    elif len(mismatch_kinds) == 1:
        mismatch_kind_value = mismatch_kinds[0]
    else:
        mismatch_kind_value = list(mismatch_kinds)

    q_mat = float(q2) * np.eye(int(x_dim), dtype=np.float64)
    r_mat = float(r2) * np.eye(int(y_dim), dtype=np.float64)
    ssm_true = {
        "type": "lorenz",
        "params": {
            "sigma": float(sigma),
            "rho": float(rho),
            "beta": float(beta),
            "J": int(J_true),
            "delta_tau": float(delta_tau_true),
            "obs_mode": str(obs_mode),
            "x0_kind": "normal_0_1",
            "sampling_decimation": int(sampling_decimation),
            "time_axis_shift_steps": int(time_axis_shift_steps),
            "time_axis_resample_stride": int(time_axis_resample_stride),
        },
        "H": _matrix_to_json(h_true),
        "Q": _matrix_to_json(q_mat),
        "R": _matrix_to_json(r_mat),
        "q2": float(q2),
        "r2": float(r2),
    }
    ssm_assumed = {
        "type": "lorenz",
        "params": {
            "sigma": float(sigma),
            "rho": float(rho),
            "beta": float(beta),
            "J": int(J_assumed),
            "delta_tau": float(delta_tau_assumed),
            "obs_mode": str(obs_mode),
            "x0_kind": "normal_0_1",
            "sampling_decimation": 1,
            "time_axis_shift_steps": 0,
            "time_axis_resample_stride": 1,
        },
        "H": _matrix_to_json(h_assumed),
        "Q": _matrix_to_json(q_mat),
        "R": _matrix_to_json(r_mat),
        "q2": float(q2),
        "r2": float(r2),
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
        "T": int(split_plan["train"]["T"]),
        "control_input_u": bool(task_cfg.control_input_u),
        "ground_truth": dict(task_cfg.ground_truth),
        "observation": dict(task_cfg.observation),
        "noise": {
            "Q": {"type": "scaled_identity", "q2": float(q2)},
            "R": {"type": "scaled_identity", "r2": float(r2)},
            "obs_distribution": {"name": "gaussian"},
        },
        "ssm": {
            "true": ssm_true,
            "assumed": ssm_assumed,
        },
        "mismatch": {
            "enabled": bool(mismatch_enabled),
            "kind": mismatch_kind_value,
            "params": {
                "J_true": int(J_true),
                "J_assumed": int(J_assumed),
                "delta_tau_true": float(delta_tau_true),
                "delta_tau_assumed": float(delta_tau_assumed),
                "obs_rotation_deg": float(mm_cfg["obs_rotation_deg"]),
                "obs_rotation_assumed_deg": float(mm_cfg["obs_rotation_assumed_deg"]),
                "sampling_decimation": int(sampling_decimation),
                "time_axis_shift_steps": int(time_axis_shift_steps),
                "time_axis_resample_stride": int(time_axis_resample_stride),
            },
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
        "switching": {
            "enabled": False,
            "models": [],
            "t_change": None,
            "retrain_window": 0,
        },
        "splits": split_meta,
    }

    proxy_split = "train"
    if int(split_payloads["train"]["x"].shape[0]) == 0:
        for split_name in ("val", "test"):
            if int(split_payloads[split_name]["x"].shape[0]) > 0:
                proxy_split = split_name
                break
    x_proxy = np.asarray(split_payloads[proxy_split]["x"], dtype=np.float32)
    y_proxy = np.asarray(split_payloads[proxy_split]["y"], dtype=np.float32)

    out = GeneratorOutput(
        x=x_proxy,
        y=y_proxy,
        meta=meta_common,
        extras={INTERNAL_SPLIT_PAYLOADS_KEY: split_payloads},
    )
    return coerce_ntd_float32_output(out), None, None
