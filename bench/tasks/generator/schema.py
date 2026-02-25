from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .contract import (
    GENERATOR_SCHEMA_TAG_V1,
    GENERATOR_SCHEMA_VERSION_V1,
    SplitCfg,
    TaskCfg,
)


REQUIRED_META_V1_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("schema_version",),
    ("task_family",),
    ("dims",),
    ("splits",),
    ("ssm", "true"),
    ("ssm", "assumed"),
    ("mismatch", "enabled"),
    ("mismatch", "kind"),
    ("mismatch", "params"),
    ("noise_schedule", "enabled"),
    ("noise_schedule", "kind"),
    ("noise_schedule", "q2_t"),
    ("noise_schedule", "r2_t"),
    ("noise_schedule", "SoW_t"),
    ("switching", "enabled"),
    ("switching", "models"),
    ("switching", "t_change"),
    ("switching", "retrain_window"),
)


def _copy_jsonish(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        if isinstance(obj, dict):
            return {k: _copy_jsonish(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_copy_jsonish(v) for v in obj]
        if isinstance(obj, tuple):
            return [_copy_jsonish(v) for v in obj]
        return obj


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _get_nested(obj: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if isinstance(cur, Mapping) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _has_nested(obj: Mapping[str, Any], path: Sequence[str]) -> bool:
    sentinel = object()
    return _get_nested(obj, path, default=sentinel) is not sentinel


def _npz_locator(extras: Mapping[str, Any], key: str, fallback: str) -> Any:
    v = extras.get(key)
    if isinstance(v, np.ndarray):
        return {"source": f"npz:{key}", "shape": list(v.shape)}
    return fallback


def _infer_q2_r2_t0(meta: Mapping[str, Any], task_cfg: Optional[TaskCfg]) -> Tuple[Optional[float], Optional[float], Optional[int], bool]:
    noise = _get_nested(meta, ("noise",), default=None)
    if not isinstance(noise, Mapping) and task_cfg is not None:
        noise = task_cfg.noise
    if not isinstance(noise, Mapping):
        return None, None, None, False

    has_shift = ("pre_shift" in noise) and ("shift" in noise)
    if has_shift:
        q2 = _to_float(_get_nested(noise, ("pre_shift", "Q", "q2")))
        r2 = _to_float(_get_nested(noise, ("pre_shift", "R", "r2")))
        t0 = _to_int(_get_nested(noise, ("shift", "t0")))
        return q2, r2, t0, True

    q2 = _to_float(_get_nested(noise, ("Q", "q2")))
    r2 = _to_float(_get_nested(noise, ("R", "r2")))
    t0 = _to_int(_get_nested(noise, ("shift", "t0")))
    return q2, r2, t0, bool(t0 is not None)


def enforce_meta_v1(
    meta: Mapping[str, Any],
    *,
    task_cfg: Optional[TaskCfg],
    split_cfg: Optional[SplitCfg],
    x: np.ndarray,
    y: np.ndarray,
    extras: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    upgraded: Dict[str, Any] = _copy_jsonish(dict(meta or {}))
    extras_map: Dict[str, Any] = dict(extras or {})

    x_dim = int(x.shape[2])
    y_dim = int(y.shape[2])
    T = int(x.shape[1])

    upgraded.setdefault("schema_version", GENERATOR_SCHEMA_VERSION_V1)
    upgraded.setdefault("schema_tag", GENERATOR_SCHEMA_TAG_V1)
    if task_cfg is not None:
        upgraded.setdefault("task_family", str(task_cfg.task_family))
    else:
        upgraded.setdefault("task_family", str(upgraded.get("task_family", "unknown")))

    upgraded["dims"] = {
        "x_dim": int(x_dim),
        "y_dim": int(y_dim),
        "T": int(T),
    }

    # Legacy compatibility fields stay append-only.
    upgraded.setdefault("x_dim", int(x_dim))
    upgraded.setdefault("y_dim", int(y_dim))
    upgraded.setdefault("T", int(T))

    if task_cfg is not None:
        upgraded.setdefault("system_type", str(task_cfg.system_type))
        upgraded.setdefault("control_input_u", bool(task_cfg.control_input_u))
        upgraded.setdefault("ground_truth", _copy_jsonish(task_cfg.ground_truth))

    ssm = upgraded.get("ssm")
    if not isinstance(ssm, dict):
        ssm = {}
    ssm_true = ssm.get("true")
    if not isinstance(ssm_true, dict):
        ssm_true = {}
    ssm_true.setdefault("system_type", str(upgraded.get("system_type", "unknown")))
    if task_cfg is not None:
        ssm_true.setdefault("observation", _copy_jsonish(task_cfg.observation))
    if isinstance(extras_map.get("F"), np.ndarray):
        ssm_true.setdefault("F_shape", list(extras_map["F"].shape))
    if isinstance(extras_map.get("H"), np.ndarray):
        ssm_true.setdefault("H_shape", list(extras_map["H"].shape))
    ssm_assumed = ssm.get("assumed")
    if not isinstance(ssm_assumed, dict):
        ssm_assumed = _copy_jsonish(ssm_true)
    ssm["true"] = ssm_true
    ssm["assumed"] = ssm_assumed
    upgraded["ssm"] = ssm

    mismatch = upgraded.get("mismatch")
    if not isinstance(mismatch, dict):
        mismatch = {}
    mismatch_cfg = {}
    if task_cfg is not None:
        raw_mm = task_cfg.raw.get("mismatch")
        if isinstance(raw_mm, dict):
            mismatch_cfg = dict(raw_mm)
    mismatch.setdefault("enabled", bool(mismatch_cfg.get("enabled", False)))
    mismatch.setdefault("kind", str(mismatch_cfg.get("kind", "none")))
    mm_params = mismatch.get("params")
    if not isinstance(mm_params, dict):
        mm_params = dict(mismatch_cfg.get("params", {}) or {})
    mismatch["params"] = mm_params
    upgraded["mismatch"] = mismatch

    q2, r2, t0, has_shift = _infer_q2_r2_t0(upgraded, task_cfg)
    noise_schedule = upgraded.get("noise_schedule")
    if not isinstance(noise_schedule, dict):
        noise_schedule = {}
    noise_schedule.setdefault(
        "enabled",
        bool(has_shift or isinstance(extras_map.get("q2_t"), np.ndarray) or isinstance(extras_map.get("r2_t"), np.ndarray)),
    )
    noise_schedule.setdefault("kind", "piecewise_constant" if has_shift else "stationary")
    noise_schedule.setdefault(
        "q2_t",
        _npz_locator(extras_map, "q2_t", "meta.noise.pre_shift.Q.q2|meta.noise.Q.q2"),
    )
    noise_schedule.setdefault(
        "r2_t",
        _npz_locator(extras_map, "r2_t", "meta.noise.pre_shift.R.r2|meta.noise.R.r2"),
    )
    noise_schedule.setdefault(
        "SoW_t",
        _npz_locator(extras_map, "SoW_t", "meta.noise_schedule.derived_from_q2_over_r2"),
    )
    if "SoW_hat_t" in extras_map:
        noise_schedule.setdefault("SoW_hat_t", _npz_locator(extras_map, "SoW_hat_t", None))
    else:
        noise_schedule.setdefault("SoW_hat_t", None)
    params = noise_schedule.get("params")
    if not isinstance(params, dict):
        params = {}
    if q2 is not None:
        params.setdefault("q2_base", float(q2))
    if r2 is not None:
        params.setdefault("r2_base", float(r2))
    if t0 is not None:
        params.setdefault("t_change", int(t0))
    noise_schedule["params"] = params
    upgraded["noise_schedule"] = noise_schedule

    switching = upgraded.get("switching")
    if not isinstance(switching, dict):
        switching = {}
    switching.setdefault("enabled", bool(t0 is not None))
    models = switching.get("models")
    if not isinstance(models, list):
        models = []
    switching["models"] = models
    if t0 is None:
        switching.setdefault("t_change", None)
    else:
        switching.setdefault("t_change", int(t0))
    retrain_default = 0
    if task_cfg is not None:
        raw_sw = task_cfg.raw.get("switching", {})
        if isinstance(raw_sw, dict):
            retrain_default = int(raw_sw.get("retrain_window", 0))
    switching.setdefault("retrain_window", int(retrain_default))
    upgraded["switching"] = switching

    splits = upgraded.get("splits")
    if not isinstance(splits, dict):
        splits = {}
    if split_cfg is not None:
        splits.setdefault("train", {"N": int(split_cfg.n_train)})
        splits.setdefault("val", {"N": int(split_cfg.n_val)})
        splits.setdefault("test", {"N": int(split_cfg.n_test)})
    elif task_cfg is not None:
        sizes = task_cfg.raw.get("dataset_sizes", {}) if isinstance(task_cfg.raw, dict) else {}
        if isinstance(sizes, dict):
            splits.setdefault("train", {"N": int(sizes.get("N_train", 0))})
            splits.setdefault("val", {"N": int(sizes.get("N_val", 0))})
            splits.setdefault("test", {"N": int(sizes.get("N_test", 0))})
    split_name = upgraded.get("split")
    if split_name is not None:
        splits.setdefault("active_split", str(split_name))
    upgraded["splits"] = splits

    return upgraded


def missing_required_meta_paths(meta: Mapping[str, Any], required_paths: Iterable[Sequence[str]] = REQUIRED_META_V1_PATHS) -> List[str]:
    missing: List[str] = []
    for path in required_paths:
        if not _has_nested(meta, path):
            missing.append(".".join(path))
    return missing
