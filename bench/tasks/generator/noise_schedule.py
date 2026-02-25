from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


_SOW_EPS = 1e-12
_SPLIT_EQ20_T0_ALLOWED = {0, 10, 20, 30}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _stable_seed(seed: int, *, tag: str, kind: str, params: Mapping[str, Any]) -> int:
    payload = {
        "seed": int(seed),
        "tag": str(tag),
        "kind": str(kind),
        "params": _jsonable(params),
    }
    txt = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    dig = hashlib.sha1(txt.encode("utf-8")).digest()
    return int.from_bytes(dig[:4], byteorder="little", signed=False)


def _local_rng(seed: int, *, tag: str, kind: str, params: Mapping[str, Any]) -> np.random.Generator:
    return np.random.default_rng(_stable_seed(seed, tag=tag, kind=kind, params=params))


def _as_float_array(values: Sequence[Any], *, T: int, name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.shape != (int(T),):
        raise ValueError(f"{name} must have length T={T}, got shape={arr.shape}")
    return arr


def _sample_jump_multiplier(dist_cfg: Mapping[str, Any], rng: np.random.Generator) -> float:
    kind = str(dist_cfg.get("kind", "discrete")).lower()
    if kind == "discrete":
        values = list(dist_cfg.get("values", []))
        if not values:
            raise ValueError("discrete jump_mult_dist requires non-empty 'values'")
        probs = dist_cfg.get("probs")
        if probs is None:
            idx = int(rng.integers(0, len(values)))
            return float(values[idx])
        probs_arr = np.asarray(list(probs), dtype=np.float64)
        probs_arr = probs_arr / np.maximum(probs_arr.sum(), 1e-12)
        idx = int(rng.choice(len(values), p=probs_arr))
        return float(values[idx])
    if kind == "lognormal":
        mean = float(dist_cfg.get("mean", 0.0))
        sigma = float(dist_cfg.get("sigma", 0.25))
        return float(rng.lognormal(mean=mean, sigma=sigma))
    raise ValueError(f"unsupported jump_mult_dist.kind={kind}")


def _build_step_change(params: Mapping[str, Any], T: int) -> Tuple[np.ndarray, np.ndarray]:
    t0 = int(params.get("t0", 0))
    q2_pre = float(params.get("q2_pre", params.get("base_q2", 1.0)))
    r2_pre = float(params.get("r2_pre", params.get("base_r2", 1.0)))

    q2_post = params.get("q2_post")
    if q2_post is None:
        q2_post = q2_pre * float(params.get("q2_scale_post", 1.0))
    r2_post = params.get("r2_post")
    if r2_post is None:
        r2_post = r2_pre * float(params.get("r2_scale_post", 1.0))

    q2_t = np.full((int(T),), float(q2_pre), dtype=np.float64)
    r2_t = np.full((int(T),), float(r2_pre), dtype=np.float64)
    if 0 <= t0 < int(T):
        q2_t[t0:] = float(q2_post)
        r2_t[t0:] = float(r2_post)
    return q2_t, r2_t


def _build_per_step_jump(
    params: Mapping[str, Any],
    T: int,
    *,
    seed: int,
    rng: Optional[np.random.Generator],
) -> Tuple[np.ndarray, np.ndarray]:
    base_q2 = float(params.get("base_q2", params.get("q2_pre", 1.0)))
    base_r2 = float(params.get("base_r2", params.get("r2_pre", 1.0)))

    if "q2_mult_t" in params or "r2_mult_t" in params:
        q_mult = _as_float_array(params.get("q2_mult_t", [1.0] * int(T)), T=T, name="q2_mult_t")
        r_mult = _as_float_array(params.get("r2_mult_t", [1.0] * int(T)), T=T, name="r2_mult_t")
        return base_q2 * q_mult, base_r2 * r_mult

    q2_t = np.full((int(T),), base_q2, dtype=np.float64)
    r2_t = np.full((int(T),), base_r2, dtype=np.float64)

    events = params.get("jump_events")
    if isinstance(events, list) and events:
        for ev in events:
            if not isinstance(ev, Mapping):
                continue
            t = int(ev.get("t", -1))
            if not (0 <= t < int(T)):
                continue
            q2_t[t] = base_q2 * float(ev.get("q2_mult", 1.0))
            r2_t[t] = base_r2 * float(ev.get("r2_mult", 1.0))
        return q2_t, r2_t

    p_jump = float(params.get("p_jump", 0.1))
    dist_cfg = params.get("jump_mult_dist", {"kind": "discrete", "values": [0.5, 1.0, 2.0]})
    if not isinstance(dist_cfg, Mapping):
        raise ValueError("jump_mult_dist must be a mapping")

    run_rng = rng if rng is not None else _local_rng(seed, tag="per_step_jump", kind="per_step_jump", params=params)
    for t in range(int(T)):
        if float(run_rng.uniform()) < p_jump:
            q2_t[t] = base_q2 * _sample_jump_multiplier(dist_cfg, run_rng)
            r2_t[t] = base_r2 * _sample_jump_multiplier(dist_cfg, run_rng)
    return q2_t, r2_t


def _build_split_eq20(params: Mapping[str, Any], T: int) -> Tuple[np.ndarray, np.ndarray]:
    phase = str(params.get("phase", params.get("split", "train"))).lower()
    t0 = int(params.get("t0", 0))
    if t0 not in _SPLIT_EQ20_T0_ALLOWED:
        raise ValueError(f"split_eq20 requires t0 in {sorted(_SPLIT_EQ20_T0_ALLOWED)}, got {t0}")
    q2_base = float(params.get("q2_base", params.get("base_q2", 1.0e-3)))

    tt = np.arange(int(T), dtype=np.int64)
    if phase == "train":
        # Eq.(20) train: sigma_v,t^2[dB] = (floor(t/2) + t0) mod 50
        r2_db_t = np.mod(np.floor_divide(tt, 2) + int(t0), 50).astype(np.float64)
    elif phase == "test":
        # Eq.(20) test: sigma_v,t^2[dB] = (floor(t/10) + 30) mod 50
        r2_db_t = np.mod(np.floor_divide(tt, 10) + 30, 50).astype(np.float64)
    else:
        raise ValueError("split_eq20 phase must be 'train' or 'test'")

    r2_t = np.power(10.0, r2_db_t / 10.0, dtype=np.float64)
    q2_t = np.full((int(T),), float(q2_base), dtype=np.float64)
    return q2_t, r2_t


def build_noise_schedule(
    cfg: Mapping[str, Any],
    T: int,
    *,
    seed: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    enabled = bool(cfg.get("enabled", False))
    kind = str(cfg.get("kind", "step_change")).lower()
    params_raw = cfg.get("params", {})
    params = dict(params_raw) if isinstance(params_raw, Mapping) else {}

    if kind == "step_change":
        q2_t, r2_t = _build_step_change(params, int(T))
    elif kind == "per_step_jump":
        q2_t, r2_t = _build_per_step_jump(params, int(T), seed=int(seed), rng=rng)
    elif kind == "split_eq20":
        q2_t, r2_t = _build_split_eq20(params, int(T))
    else:
        raise ValueError(f"unsupported noise schedule kind: {kind}")

    if np.any(q2_t < 0) or np.any(r2_t < 0):
        raise ValueError("q2_t/r2_t must be non-negative")

    sow_raw = np.asarray(q2_t, dtype=np.float64) / np.maximum(np.asarray(r2_t, dtype=np.float64), _SOW_EPS)
    clamp_mask = sow_raw <= _SOW_EPS
    if np.any(clamp_mask):
        sow_raw = np.where(clamp_mask, _SOW_EPS, sow_raw)
    sow_db = 10.0 * np.log10(sow_raw)

    arrays: Dict[str, np.ndarray] = {
        "q2_t": np.asarray(q2_t, dtype=np.float32),
        "r2_t": np.asarray(r2_t, dtype=np.float32),
        "SoW_t": np.asarray(sow_raw, dtype=np.float32),
        "SoW_dB_t": np.asarray(sow_db, dtype=np.float32),
    }

    sow_hat_cfg_raw = cfg.get("sow_hat", {})
    sow_hat_cfg = dict(sow_hat_cfg_raw) if isinstance(sow_hat_cfg_raw, Mapping) else {}
    sow_hat_enabled = bool(sow_hat_cfg.get("enabled", False))
    sow_hat_mode = str(sow_hat_cfg.get("mode", "add_db")).lower()
    if sow_hat_enabled:
        sow_rng = _local_rng(int(seed), tag="sow_hat", kind=kind, params={"params": params, "sow_hat": sow_hat_cfg})
        if sow_hat_mode == "add_db":
            sigma_db = float(sow_hat_cfg.get("sigma_db", 1.0))
            noise_db = sow_rng.normal(loc=0.0, scale=sigma_db, size=int(T))
            sow_hat_db = np.asarray(sow_db, dtype=np.float64) + noise_db
            sow_hat = np.power(10.0, sow_hat_db / 10.0, dtype=np.float64)
        elif sow_hat_mode == "mul_linear":
            sigma_mul = float(sow_hat_cfg.get("sigma_mul", 0.1))
            factor = 1.0 + sow_rng.normal(loc=0.0, scale=sigma_mul, size=int(T))
            sow_hat = np.asarray(sow_raw, dtype=np.float64) * factor
        else:
            raise ValueError(f"unsupported sow_hat.mode={sow_hat_mode}")

        sow_hat = np.where(sow_hat <= _SOW_EPS, _SOW_EPS, sow_hat)
        arrays["SoW_hat_t"] = np.asarray(sow_hat, dtype=np.float32)

    meta_desc: Dict[str, Any] = {
        "enabled": bool(enabled),
        "kind": str(kind),
        "params": _jsonable(params),
        "storage": {
            "q2_t": "npz_extras:q2_t",
            "r2_t": "npz_extras:r2_t",
            "SoW_t": "npz_extras:SoW_t",
            "SoW_dB_t": "npz_extras:SoW_dB_t",
            "SoW_hat_t": "npz_extras:SoW_hat_t" if "SoW_hat_t" in arrays else "disabled",
        },
        "sow": {
            "definition": "SoW_t = q2_t / r2_t",
            "eps": float(_SOW_EPS),
            "num_clamped": int(np.count_nonzero(clamp_mask)),
            "clamped": bool(np.any(clamp_mask)),
        },
        "sow_hat": {
            "enabled": bool(sow_hat_enabled),
            "mode": str(sow_hat_mode),
            "sigma_db": float(sow_hat_cfg.get("sigma_db", 0.0)) if sow_hat_mode == "add_db" else None,
            "sigma_mul": float(sow_hat_cfg.get("sigma_mul", 0.0)) if sow_hat_mode == "mul_linear" else None,
        },
    }
    if kind == "split_eq20":
        meta_desc["formula"] = {
            "train": "r2_db_t = (floor(t/2) + t0) mod 50",
            "test": "r2_db_t = (floor(t/10) + 30) mod 50",
        }

    return arrays, meta_desc
