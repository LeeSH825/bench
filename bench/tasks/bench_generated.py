from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .generators.linear import (
    build_linear_system_matrices_v0,
    generate_linear_gaussian_sequences_v0,
)
from .data_format import (
    CANONICAL_LAYOUT_V0,
    DatasetArtifactsV0,
    DatasetSplitV0,
    load_npz_split_v0,
    save_npz_split_v0,
)
from .generator.contract import (
    GeneratorOutput,
    coerce_ntd_float32_output,
    make_split_cfg,
    make_task_cfg,
    resolve_task_family,
)
from .generator.linear_mismatch import generate_linear_mismatch_v0
from .generator.lorenz import generate_lorenz_v0
from .generator.noise_schedule import build_noise_schedule
from .generator.sine_poly import generate_sine_poly_v0
from .generator.switching_dynamics import generate_switching_dynamics_v0
from .generator.ucm import generate_ucm_v0
from .generator.datasets.common import INTERNAL_SPLIT_PAYLOADS_KEY
from .generator.datasets.nclt import generate_nclt_v0
from .generator.datasets.uzh_fpv import generate_uzh_fpv_v0
from .generator.schema import enforce_meta_v1
from .generator.validate import validate_artifacts
from ..utils.seeding import stable_int_seed_v0, numpy_rng_v0
from ..utils.io import ensure_dir
from ..utils.sweep import expand_sweep_grid
from .generator.basilisk_adcs import generate_basilisk_adcs_v0
from .generator.basilisk_imu_adcs import (
    generate_basilisk_imu_adcs_v0,
    generate_basilisk_imu_bias_adcs_v0,
    generate_basilisk_imu_sparse_ref_adcs_v0,
)


def _repo_root_from_here() -> Path:
    # .../bench/bench/tasks/bench_generated.py -> parents[2] == repo root (.../bench)
    return Path(__file__).resolve().parents[2]


def default_cache_root() -> Path:
    env = os.environ.get("BENCH_DATA_CACHE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root_from_here() / "bench_data_cache"


def canonicalize_scenario_id(task_id: str, scenario_cfg: Dict[str, Any]) -> str:
    payload = {"task_id": task_id, "scenario": scenario_cfg}
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:12]


def _dotted_set(d: Dict[str, Any], dotted_key: str, val: Any) -> None:
    cur = d
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = val


def _deep_merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _is_inv_r2_db_sweep_key(key: str) -> bool:
    kk = str(key).strip()
    return kk in {"inv_r2_db", "noise.inv_r2_db"}


def _resolve_r2_sweep_target_key(task_cfg: Dict[str, Any]) -> Optional[str]:
    noise = task_cfg.get("noise", {})
    if not isinstance(noise, Mapping):
        return None

    pre = noise.get("pre_shift", {})
    if isinstance(pre, Mapping):
        pre_r = pre.get("R", {})
        if isinstance(pre_r, Mapping) and ("r2" in pre_r):
            return "noise.pre_shift.R.r2"

    r_map = noise.get("R", {})
    if isinstance(r_map, Mapping) and ("r2" in r_map):
        return "noise.R.r2"
    # Fallback: allow alias sweep even when task omits explicit noise.R.r2.
    return "noise.R.r2"


def _inv_r2_db_to_r2(value: Any) -> float:
    db = float(value)
    return float(np.power(10.0, -db / 10.0))


def _scenario_cfg_basis_for_id(task_cfg: Dict[str, Any], scenario_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonical scenario basis for dataset-cache scenario_id:
      deep_copy(task.noise) + scenario overrides.
    """
    basis: Dict[str, Any] = copy.deepcopy(task_cfg.get("noise", {}) or {})
    for k, v in (scenario_cfg or {}).items():
        if str(k) == "noise" and isinstance(v, dict):
            basis = _deep_merge_dicts(basis, dict(v))
        elif k in basis and isinstance(basis[k], dict) and isinstance(v, dict):
            basis[k] = _deep_merge_dicts(basis[k], v)
        else:
            basis[k] = v
    return basis


def expand_scenarios_from_sweep(task_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Supports flat sweep keys like:
      sweep:
        shift.post_shift.R_scale: [3.0, 10.0, 30.0]
        noise.R.r2:
          start: 0.0
          stop: 30.0
          step: 1.0
        inv_r2_db:
          start: -10
          stop: 40
          step: 5
    where inv_r2_db means r2 = 10^(-inv_r2_db/10).
    Returns list of nested dict scenario_cfg.
    """
    scenarios: List[Dict[str, Any]] = []
    grid = expand_sweep_grid(task_cfg.get("sweep"), sort_keys=False)
    r2_target_key = _resolve_r2_sweep_target_key(task_cfg)

    for flat in grid:
        nested: Dict[str, Any] = {}
        for k, v in flat.items():
            kk = str(k)
            if _is_inv_r2_db_sweep_key(kk):
                if r2_target_key is None:
                    raise ValueError(
                        "inv_r2_db sweep requires task noise config with either "
                        "noise.pre_shift.R.r2 or noise.R.r2"
                    )
                _dotted_set(nested, r2_target_key, _inv_r2_db_to_r2(v))
                continue
            _dotted_set(nested, kk, v)
        scenarios.append(nested)
    return scenarios


def merge_scenario_overrides(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return base

    def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    return deep_merge(base, overrides)


def _task_cache_dir(cache_root: Path, suite_name: str, task_id: str, scenario_id: str, seed: int) -> Path:
    return cache_root / suite_name / task_id / f"scenario_{scenario_id}" / f"seed_{seed}"


_LINEAR_TASK_FAMILIES = {
    "linear",
    "linear_gaussian",
    "linear_gaussian_v0",
    "linear_canonical_v0",
    "default",
}
_LINEAR_MISMATCH_TASK_FAMILIES = {
    "linear_mismatch",
    "linear_mismatch_v0",
}
_UCM_TASK_FAMILIES = {
    "ucm",
    "ucm_v0",
    "uniform_circular_motion",
    "uniform_circular_motion_v0",
}
_SINE_POLY_TASK_FAMILIES = {
    "sine_poly",
    "sine_poly_v0",
    "synthetic_nonlinear",
    "synthetic_nonlinear_v0",
}
_SWITCHING_TASK_FAMILIES = {
    "switching",
    "switching_v0",
    "switching_dynamics",
    "switching_dynamics_v0",
}
_LORENZ_TASK_FAMILIES = {
    "lorenz",
    "lorenz_v0",
    "lorenz_dt",
    "lorenz_dt_v0",
}
_NCLT_TASK_FAMILIES = {
    "nclt",
    "nclt_v0",
    "nclt_segway",
    "nclt_segway_v0",
}
_UZH_FPV_TASK_FAMILIES = {
    "uzh_fpv",
    "uzh_fpv_v0",
    "uzh",
    "uzh_v0",
    "uzh_fpv_ca",
    "uzh_fpv_ca_v0",
}
_BASILISK_ADCS_TASK_FAMILIES = {
    "basilisk_adcs",
    "basilisk_adcs_v0",
    "bsk_adcs",
    "bsk_adcs_v0",
}
_BASILISK_IMU_ADCS_TASK_FAMILIES = {
    "basilisk_imu_adcs",
    "basilisk_imu_adcs_v0",
    "bsk_imu_adcs",
    "bsk_imu_adcs_v0",
}
_BASILISK_IMU_BIAS_ADCS_TASK_FAMILIES = {
    "basilisk_imu_bias_adcs",
    "basilisk_imu_bias_adcs_v0",
    "bsk_imu_bias_adcs",
    "bsk_imu_bias_adcs_v0",
}
_BASILISK_IMU_SPARSE_REF_ADCS_TASK_FAMILIES = {
    "basilisk_imu_sparse_ref_adcs",
    "basilisk_imu_sparse_ref_adcs_v0",
    "bsk_imu_sparse_ref_adcs",
    "bsk_imu_sparse_ref_adcs_v0",
}


def _normalized_task_family(task_cfg: Dict[str, Any]) -> str:
    family = resolve_task_family(task_cfg).lower()
    if family in _LINEAR_TASK_FAMILIES:
        return "linear_gaussian_v0"
    if family in _LINEAR_MISMATCH_TASK_FAMILIES:
        return "linear_mismatch_v0"
    if family in _UCM_TASK_FAMILIES:
        return "ucm_v0"
    if family in _SINE_POLY_TASK_FAMILIES:
        return "sine_poly_v0"
    if family in _SWITCHING_TASK_FAMILIES:
        return "switching_dynamics_v0"
    if family in _LORENZ_TASK_FAMILIES:
        return "lorenz_v0"
    if family in _NCLT_TASK_FAMILIES:
        return "nclt_v0"
    if family in _UZH_FPV_TASK_FAMILIES:
        return "uzh_fpv_v0"
    if family in _BASILISK_ADCS_TASK_FAMILIES:
        return "basilisk_adcs_v0"
    if family in _BASILISK_IMU_ADCS_TASK_FAMILIES:
        return "basilisk_imu_adcs_v0"
    if family in _BASILISK_IMU_BIAS_ADCS_TASK_FAMILIES:
        return "basilisk_imu_bias_adcs_v0"
    if family in _BASILISK_IMU_SPARSE_REF_ADCS_TASK_FAMILIES:
        return "basilisk_imu_sparse_ref_adcs_v0"
    return family


def _legacy_noise_schedule_cfg(
    *,
    T: int,
    q2: float,
    r2: float,
    t0_shift: Optional[int],
    r_scale_post: float,
) -> Dict[str, Any]:
    if t0_shift is not None:
        return {
            "enabled": True,
            "kind": "step_change",
            "params": {
                "t0": int(t0_shift),
                "q2_pre": float(q2),
                "r2_pre": float(r2),
                "q2_post": float(q2),
                "r2_post": float(r2) * float(r_scale_post),
            },
        }
    return {
        "enabled": False,
        "kind": "step_change",
        "params": {
            "t0": int(T),
            "q2_pre": float(q2),
            "r2_pre": float(r2),
            "q2_post": float(q2),
            "r2_post": float(r2),
        },
    }


def _resolved_noise_schedule_cfg(
    *,
    task_cfg_dict: Dict[str, Any],
    q2: float,
    r2: float,
    t0_shift: Optional[int],
    r_scale_post: float,
    T: int,
) -> Dict[str, Any]:
    ns_raw = task_cfg_dict.get("noise_schedule", {})
    if not isinstance(ns_raw, Mapping) or not bool(ns_raw.get("enabled", False)):
        return _legacy_noise_schedule_cfg(
            T=T,
            q2=q2,
            r2=r2,
            t0_shift=t0_shift,
            r_scale_post=r_scale_post,
        )

    cfg = json.loads(json.dumps(dict(ns_raw)))
    params = cfg.get("params", {})
    if not isinstance(params, dict):
        params = {}
    params.setdefault("q2_pre", float(q2))
    params.setdefault("r2_pre", float(r2))
    params.setdefault("base_q2", float(q2))
    params.setdefault("base_r2", float(r2))
    if t0_shift is not None:
        params.setdefault("t0", int(t0_shift))
        params.setdefault("q2_post", float(q2))
        params.setdefault("r2_post", float(r2) * float(r_scale_post))
    cfg["params"] = params
    return cfg


def _noise_extras_only(arrays: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in arrays.items():
        if isinstance(v, np.ndarray):
            out[str(k)] = v.astype(np.float32, copy=False)
    return out


def _merge_noise_schedule_meta(
    noise_schedule_meta: Mapping[str, Any],
    *,
    enabled: bool,
    kind: str,
) -> Dict[str, Any]:
    out = json.loads(json.dumps(dict(noise_schedule_meta)))
    out["enabled"] = bool(enabled)
    out["kind"] = str(kind)
    return out


def _with_noise_storage_markers(meta: Dict[str, Any], arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    storage = dict(meta.get("storage", {})) if isinstance(meta.get("storage"), dict) else {}
    for key in arrays.keys():
        storage.setdefault(str(key), f"npz_extras:{key}")
    meta["storage"] = storage
    return meta


def _noise_schedule_enabled_kind(cfg: Mapping[str, Any]) -> Tuple[bool, str]:
    return bool(cfg.get("enabled", False)), str(cfg.get("kind", "step_change"))


def _task_noise_seed(
    *,
    suite_name: str,
    task_id: str,
    scenario_id: str,
    seed: int,
) -> int:
    return stable_int_seed_v0("noise_schedule", suite_name, task_id, scenario_id, seed)


def _schedule_arrays_and_meta(
    *,
    schedule_cfg: Mapping[str, Any],
    T: int,
    noise_seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    arrays, meta_desc = build_noise_schedule(schedule_cfg, int(T), seed=int(noise_seed), rng=None)
    meta_desc = _with_noise_storage_markers(dict(meta_desc), arrays)
    return arrays, meta_desc


def _sow_t_from_arrays(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if "SoW_t" in arrays:
        return np.asarray(arrays["SoW_t"], dtype=np.float32)
    q2_t = np.asarray(arrays["q2_t"], dtype=np.float64)
    r2_t = np.asarray(arrays["r2_t"], dtype=np.float64)
    return np.asarray(q2_t / np.maximum(r2_t, 1e-12), dtype=np.float32)


def _schedule_extras_with_task_key(
    *,
    arrays: Mapping[str, np.ndarray],
    task_family: str,
    task_id: str,
    scenario_id: str,
) -> Dict[str, Any]:
    extras = _noise_extras_only(arrays)
    if "SoW_t" not in extras:
        extras["SoW_t"] = _sow_t_from_arrays(arrays)
    extras["task_key"] = f"{task_family}:{task_id}:{scenario_id}"
    return extras


def _float_schedule(arrays: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    return np.asarray(arrays[key], dtype=np.float64)


def _noise_schedule_q2r2(arrays: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    return _float_schedule(arrays, "q2_t"), _float_schedule(arrays, "r2_t")


def _noise_schedule_arrays_have_required(arrays: Mapping[str, np.ndarray]) -> None:
    for k in ("q2_t", "r2_t", "SoW_t"):
        if k not in arrays:
            raise ValueError(f"noise_schedule output missing required array: {k}")


def _noise_schedule_cfg_for_generation(
    *,
    task_cfg_dict: Dict[str, Any],
    q2: float,
    r2: float,
    t0_shift: Optional[int],
    r_scale_post: float,
    T: int,
) -> Dict[str, Any]:
    return _resolved_noise_schedule_cfg(
        task_cfg_dict=task_cfg_dict,
        q2=q2,
        r2=r2,
        t0_shift=t0_shift,
        r_scale_post=r_scale_post,
        T=T,
    )


def _noise_schedule_output(
    *,
    task_cfg_dict: Dict[str, Any],
    suite_name: str,
    task_id: str,
    scenario_id: str,
    seed: int,
    q2: float,
    r2: float,
    t0_shift: Optional[int],
    r_scale_post: float,
    T: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], bool, str]:
    schedule_cfg = _noise_schedule_cfg_for_generation(
        task_cfg_dict=task_cfg_dict,
        q2=q2,
        r2=r2,
        t0_shift=t0_shift,
        r_scale_post=r_scale_post,
        T=T,
    )
    noise_seed = _task_noise_seed(
        suite_name=suite_name,
        task_id=task_id,
        scenario_id=scenario_id,
        seed=seed,
    )
    arrays, meta_desc = _schedule_arrays_and_meta(
        schedule_cfg=schedule_cfg,
        T=T,
        noise_seed=noise_seed,
    )
    _noise_schedule_arrays_have_required(arrays)
    enabled, kind = _noise_schedule_enabled_kind(schedule_cfg)
    meta_desc = _merge_noise_schedule_meta(meta_desc, enabled=enabled, kind=kind)
    return arrays, meta_desc, enabled, kind


def _noise_schedule_generation_args(arrays: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    q2_t, r2_t = _noise_schedule_q2r2(arrays)
    return q2_t, r2_t


def _noise_schedule_meta_block(meta_desc: Mapping[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(dict(meta_desc)))


def _build_noise_series(
    *,
    task_cfg_dict: Dict[str, Any],
    suite_name: str,
    task_id: str,
    scenario_id: str,
    seed: int,
    q2: float,
    r2: float,
    t0_shift: Optional[int],
    r_scale_post: float,
    T: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], bool, str]:
    return _noise_schedule_output(
        task_cfg_dict=task_cfg_dict,
        suite_name=suite_name,
        task_id=task_id,
        scenario_id=scenario_id,
        seed=seed,
        q2=q2,
        r2=r2,
        t0_shift=t0_shift,
        r_scale_post=r_scale_post,
        T=T,
    )


def _dispatch_generate_v1(
    *,
    suite_name: str,
    task_cfg_dict: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    seed: int,
    scenario_id: str,
) -> Tuple[GeneratorOutput, Optional[np.ndarray], Optional[np.ndarray]]:
    task_family = _normalized_task_family(task_cfg_dict)
    if task_family == "linear_mismatch_v0":
        return generate_linear_mismatch_v0(
            suite_name=suite_name,
            task_cfg_dict=task_cfg_dict,
            scenario_cfg=scenario_cfg,
            seed=int(seed),
            scenario_id=scenario_id,
            task_family="linear_mismatch_v0",
        )
    if task_family == "ucm_v0":
        return generate_ucm_v0(
            suite_name=suite_name,
            task_cfg_dict=task_cfg_dict,
            scenario_cfg=scenario_cfg,
            seed=int(seed),
            scenario_id=scenario_id,
            task_family="ucm_v0",
        )
    if task_family == "sine_poly_v0":
        return generate_sine_poly_v0(
            suite_name=suite_name,
            task_cfg_dict=task_cfg_dict,
            scenario_cfg=scenario_cfg,
            seed=int(seed),
            scenario_id=scenario_id,
            task_family="sine_poly_v0",
        )
    if task_family == "switching_dynamics_v0":
        return generate_switching_dynamics_v0(
            suite_name=suite_name,
            task_cfg_dict=task_cfg_dict,
            scenario_cfg=scenario_cfg,
            seed=int(seed),
            scenario_id=scenario_id,
            task_family="switching_dynamics_v0",
        )
    if task_family == "lorenz_v0":
        return generate_lorenz_v0(
            suite_name=suite_name,
            task_cfg_dict=task_cfg_dict,
            scenario_cfg=scenario_cfg,
            seed=int(seed),
            scenario_id=scenario_id,
            task_family="lorenz_v0",
        )
    if task_family == "nclt_v0":
        return generate_nclt_v0(
            suite_name=suite_name,
            task_cfg_dict=task_cfg_dict,
            scenario_cfg=scenario_cfg,
            seed=int(seed),
            scenario_id=scenario_id,
            task_family="nclt_v0",
        )
    if task_family == "uzh_fpv_v0":
        return generate_uzh_fpv_v0(
            suite_name=suite_name,
            task_cfg_dict=task_cfg_dict,
            scenario_cfg=scenario_cfg,
            seed=int(seed),
            scenario_id=scenario_id,
            task_family="uzh_fpv_v0",
        )
    if task_family == "basilisk_adcs_v0":
        return generate_basilisk_adcs_v0(
            suite_name=suite_name,
            task_cfg_dict=task_cfg_dict,
            scenario_cfg=scenario_cfg,
            seed=int(seed),
            scenario_id=scenario_id,
            task_family="basilisk_adcs_v0",
        )
    if task_family == "basilisk_imu_adcs_v0":
        return generate_basilisk_imu_adcs_v0(
            suite_name=suite_name,
            task_cfg_dict=task_cfg_dict,
            scenario_cfg=scenario_cfg,
            seed=int(seed),
            scenario_id=scenario_id,
            task_family="basilisk_imu_adcs_v0",
        )
    if task_family == "basilisk_imu_bias_adcs_v0":
        return generate_basilisk_imu_bias_adcs_v0(
            suite_name=suite_name,
            task_cfg_dict=task_cfg_dict,
            scenario_cfg=scenario_cfg,
            seed=int(seed),
            scenario_id=scenario_id,
            task_family="basilisk_imu_bias_adcs_v0",
        )
    if task_family == "basilisk_imu_sparse_ref_adcs_v0":
        return generate_basilisk_imu_sparse_ref_adcs_v0(
            suite_name=suite_name,
            task_cfg_dict=task_cfg_dict,
            scenario_cfg=scenario_cfg,
            seed=int(seed),
            scenario_id=scenario_id,
            task_family="basilisk_imu_sparse_ref_adcs_v0",
        )

    if task_family != "linear_gaussian_v0":
        supported = sorted(
            _LINEAR_TASK_FAMILIES
            | _LINEAR_MISMATCH_TASK_FAMILIES
            | _UCM_TASK_FAMILIES
            | _SINE_POLY_TASK_FAMILIES
            | _SWITCHING_TASK_FAMILIES
            | _LORENZ_TASK_FAMILIES
            | _NCLT_TASK_FAMILIES
            | _UZH_FPV_TASK_FAMILIES
            | _BASILISK_ADCS_TASK_FAMILIES
            | _BASILISK_IMU_ADCS_TASK_FAMILIES
            | _BASILISK_IMU_BIAS_ADCS_TASK_FAMILIES
            | _BASILISK_IMU_SPARSE_REF_ADCS_TASK_FAMILIES
        )
        raise NotImplementedError(
            f"task_family '{task_family}' is not registered. "
            f"Supported families: {supported}"
        )

    resolved_task_cfg = merge_scenario_overrides(task_cfg_dict, scenario_cfg)
    task_cfg = make_task_cfg(resolved_task_cfg, scenario_cfg=scenario_cfg)
    split_cfg = make_split_cfg(resolved_task_cfg)

    x_dim = int(task_cfg.x_dim)
    y_dim = int(task_cfg.y_dim)
    T = int(task_cfg.sequence_length_T)
    n_total = int(split_cfg.n_total)

    obs = dict(task_cfg.observation)
    H_spec = obs.get("H", "identity")

    noise = dict(task_cfg.noise)
    is_shift = ("pre_shift" in noise) and ("shift" in noise)
    if is_shift:
        q2 = float(noise["pre_shift"]["Q"]["q2"])
        r2 = float(noise["pre_shift"]["R"]["r2"])
        t0 = int(noise["shift"]["t0"])
        post = noise["shift"].get("post_shift", {})
        r_scale = float(
            scenario_cfg.get("shift", {})
            .get("post_shift", {})
            .get("R_scale", post.get("R_scale", 1.0))
        )
        obs_dist = post.get("obs_distribution", {"name": "gaussian"})
        dist_name = str(obs_dist.get("name", "gaussian"))
        dist_params = dict(obs_dist)
    else:
        q2 = float(noise["Q"]["q2"])
        r2 = float(noise["R"]["r2"])
        t0 = None
        r_scale = 1.0
        dist_name = "gaussian"
        dist_params = {"name": "gaussian"}

    sys_seed = stable_int_seed_v0("system", suite_name, task_cfg.task_id, scenario_id, seed)
    rng_sys = numpy_rng_v0(sys_seed)
    F, H = build_linear_system_matrices_v0(rng=rng_sys, x_dim=x_dim, y_dim=y_dim, H_spec=str(H_spec))

    noise_arrays, noise_meta_desc, ns_enabled, ns_kind = _build_noise_series(
        task_cfg_dict=task_cfg_dict,
        suite_name=suite_name,
        task_id=task_cfg.task_id,
        scenario_id=scenario_id,
        seed=int(seed),
        q2=q2,
        r2=r2,
        t0_shift=t0,
        r_scale_post=r_scale,
        T=T,
    )
    q2_t_arr, r2_t_arr = _noise_schedule_generation_args(noise_arrays)

    data_seed = stable_int_seed_v0("data", suite_name, task_cfg.task_id, scenario_id, seed)
    rng_data = numpy_rng_v0(data_seed)
    x_all, y_all = generate_linear_gaussian_sequences_v0(
        rng=rng_data,
        n_seq=n_total,
        T=T,
        F=F,
        H=H,
        q2=q2,
        r2=r2,
        t0_shift=t0,
        r_scale_post=r_scale,
        obs_dist_name=dist_name,
        obs_dist_params=dist_params,
        q2_t=q2_t_arr,
        r2_t=r2_t_arr,
    )

    noise_meta = json.loads(json.dumps(noise))
    if is_shift:
        noise_meta.setdefault("shift", {})
        noise_meta["shift"]["t0"] = int(t0)
        noise_meta.setdefault("shift", {}).setdefault("post_shift", {})
        noise_meta["shift"]["post_shift"]["R_scale"] = float(r_scale)
        noise_meta["shift"]["post_shift"]["obs_distribution"] = dict(dist_params)

    extras: Dict[str, Any] = _schedule_extras_with_task_key(
        arrays=noise_arrays,
        task_family=task_family,
        task_id=task_cfg.task_id,
        scenario_id=scenario_id,
    )

    meta_common: Dict[str, Any] = {
        "format_version": "0.1",
        "canonical_layout": CANONICAL_LAYOUT_V0,
        "schema_version": 1,
        "task_family": str(task_family),
        "suite_name": suite_name,
        "task_id": task_cfg.task_id,
        "scenario_id": scenario_id,
        "scenario_cfg": dict(scenario_cfg),
        "seed": int(seed),
        "x_dim": int(x_dim),
        "y_dim": int(y_dim),
        "T": int(T),
        "control_input_u": bool(task_cfg.control_input_u),
        "ground_truth": dict(task_cfg.ground_truth),
        "observation": dict(obs),
        "noise": noise_meta,
        "noise_schedule": _noise_schedule_meta_block(
            _merge_noise_schedule_meta(noise_meta_desc, enabled=ns_enabled, kind=ns_kind)
        ),
    }
    out = coerce_ntd_float32_output(
        GeneratorOutput(
            x=x_all.astype(np.float32),
            y=y_all.astype(np.float32),
            meta=meta_common,
            extras=extras,
        )
    )
    return out, F.astype(np.float32), H.astype(np.float32)


def _is_per_sequence_extra_key(key: str) -> bool:
    k = str(key).strip().lower()
    if k in {"task_key", "q2_seq", "r2_seq", "sow_seq", "sow_db_seq", "v_db_seq"}:
        return True
    return k.endswith("_seq")


def _npz_extras_only(
    extras: Mapping[str, Any],
    *,
    idx: Optional[np.ndarray] = None,
    n_total: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in extras.items():
        if isinstance(v, np.ndarray):
            key = str(k)
            arr = v.astype(np.float32, copy=False)
            if (
                idx is not None
                and n_total is not None
                and arr.ndim >= 1
                and int(arr.shape[0]) == int(n_total)
                and _is_per_sequence_extra_key(key)
            ):
                arr = arr[np.asarray(idx, dtype=np.int64)]
            out[key] = arr
    return out


def _split_payloads_from_extras(extras: Mapping[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
    raw = extras.get(INTERNAL_SPLIT_PAYLOADS_KEY)
    if not isinstance(raw, Mapping):
        return None
    out: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        payload = raw.get(split)
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"internal split payloads must include mapping for split '{split}' under key "
                f"{INTERNAL_SPLIT_PAYLOADS_KEY}"
            )
        x = payload.get("x")
        y = payload.get("y")
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError(f"split payload for '{split}' must include numpy arrays x and y")
        out[split] = {
            "x": x,
            "y": y,
            "extras": dict(payload.get("extras", {}) or {}) if isinstance(payload.get("extras", {}), Mapping) else {},
            "meta": dict(payload.get("meta", {}) or {}) if isinstance(payload.get("meta", {}), Mapping) else {},
            "F": payload.get("F"),
            "H": payload.get("H"),
        }
    return out


def _maybe_generate_and_cache_one(
    *,
    cache_root: Path,
    suite_name: str,
    task_cfg: Dict[str, Any],
    task_id: str,
    scenario_cfg: Dict[str, Any],
    seed: int,
) -> DatasetArtifactsV0:
    """
    Create (train/val/test).npz if missing; otherwise load.
    Deterministic for (suite_name, task_id, scenario_cfg, seed).
    """
    scenario_cfg_basis = _scenario_cfg_basis_for_id(task_cfg, scenario_cfg)
    scenario_id = canonicalize_scenario_id(task_id, scenario_cfg_basis)
    out_dir = _task_cache_dir(cache_root, suite_name, task_id, scenario_id, seed)
    ensure_dir(out_dir)

    train_path = out_dir / "train.npz"
    val_path = out_dir / "val.npz"
    test_path = out_dir / "test.npz"
    split_cfg_obj = make_split_cfg(task_cfg)
    task_cfg_obj = make_task_cfg(task_cfg, scenario_cfg=scenario_cfg)

    if train_path.exists() and val_path.exists() and test_path.exists():
        loaded_train = load_npz_split_v0(train_path)
        meta = enforce_meta_v1(
            loaded_train.meta,
            task_cfg=task_cfg_obj,
            split_cfg=split_cfg_obj,
            x=loaded_train.x,
            y=loaded_train.y,
            extras=loaded_train.extras,
        )
        return DatasetArtifactsV0(
            format_version="0.1",
            canonical_layout=CANONICAL_LAYOUT_V0,
            suite_name=suite_name,
            task_id=task_id,
            scenario_id=scenario_id,
            seed=seed,
            cache_dir=out_dir,
            train=DatasetSplitV0(path=train_path, split="train"),
            val=DatasetSplitV0(path=val_path, split="val"),
            test=DatasetSplitV0(path=test_path, split="test"),
            meta_common={k: v for k, v in meta.items() if k != "split"},
        )

    generated, F, H = _dispatch_generate_v1(
        suite_name=suite_name,
        task_cfg_dict=task_cfg,
        scenario_cfg=scenario_cfg,
        seed=int(seed),
        scenario_id=scenario_id,
    )

    split_payloads = _split_payloads_from_extras(generated.extras)
    x_all = generated.x
    y_all = generated.y

    n_train = int(split_cfg_obj.n_train)
    n_val = int(split_cfg_obj.n_val)
    n_test = int(split_cfg_obj.n_test)
    n_total = int(split_cfg_obj.n_total)
    if split_payloads is None:
        if int(x_all.shape[0]) != n_total or int(y_all.shape[0]) != n_total:
            raise ValueError(
                f"generated n_total mismatch: split_cfg={n_total}, x={x_all.shape}, y={y_all.shape}"
            )

        # deterministic split permutation
        split_seed = stable_int_seed_v0("split", suite_name, task_id, scenario_id, seed)
        rng_split = numpy_rng_v0(split_seed)
        perm = rng_split.permutation(n_total)

        idx_train = perm[:n_train]
        idx_val = perm[n_train : n_train + n_val]
        idx_test = perm[n_train + n_val :]
    else:
        idx_train = idx_val = idx_test = None

    # Save per split (npz). Store F/H when provided by generator.
    def save_split(
        path: Path,
        split: str,
        *,
        idx: Optional[np.ndarray] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if payload is None:
            if idx is None:
                raise ValueError("save_split requires either idx or payload")
            x_split = x_all[idx].astype(np.float32, copy=False)
            y_split = y_all[idx].astype(np.float32, copy=False)
            extras_np = _npz_extras_only(generated.extras, idx=idx, n_total=n_total)
            split_f = F
            split_h = H
            meta_overrides: Dict[str, Any] = {}
        else:
            x_split = np.asarray(payload.get("x"), dtype=np.float32)
            y_split = np.asarray(payload.get("y"), dtype=np.float32)
            if x_split.ndim != 3 or y_split.ndim != 3:
                raise ValueError(f"split payload '{split}' must provide rank-3 x/y arrays")
            extras_np = _npz_extras_only(generated.extras)
            payload_extras = payload.get("extras", {})
            if isinstance(payload_extras, Mapping):
                extras_np.update(_npz_extras_only(payload_extras))
            split_f = payload.get("F", F)
            split_h = payload.get("H", H)
            meta_overrides = dict(payload.get("meta", {}) or {}) if isinstance(payload.get("meta", {}), Mapping) else {}

        meta = dict(generated.meta)
        meta.update(meta_overrides)
        meta["split"] = split
        meta = enforce_meta_v1(
            meta,
            task_cfg=task_cfg_obj,
            split_cfg=split_cfg_obj,
            x=x_split,
            y=y_split,
            extras=extras_np,
        )
        validate_artifacts(x_split, y_split, meta, strict=True)
        save_npz_split_v0(
            path=path,
            x=x_split,
            y=y_split,
            u=None,
            F=None if split_f is None else np.asarray(split_f, dtype=np.float32),
            H=None if split_h is None else np.asarray(split_h, dtype=np.float32),
            meta=meta,
            extras=extras_np,
        )
        return meta

    if split_payloads is None:
        meta_train = save_split(train_path, "train", idx=idx_train)
        save_split(val_path, "val", idx=idx_val)
        save_split(test_path, "test", idx=idx_test)
    else:
        meta_train = save_split(train_path, "train", payload=split_payloads["train"])
        save_split(val_path, "val", payload=split_payloads["val"])
        save_split(test_path, "test", payload=split_payloads["test"])
    meta_common = {k: v for k, v in meta_train.items() if k != "split"}

    return DatasetArtifactsV0(
        format_version="0.1",
        canonical_layout=CANONICAL_LAYOUT_V0,
        suite_name=suite_name,
        task_id=task_id,
        scenario_id=scenario_id,
        seed=seed,
        cache_dir=out_dir,
        train=DatasetSplitV0(path=train_path, split="train"),
        val=DatasetSplitV0(path=val_path, split="val"),
        test=DatasetSplitV0(path=test_path, split="test"),
        meta_common=meta_common,
    )


@dataclass(frozen=True)
class LoaderCfgV0:
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    drop_last: bool = False


class BenchNpzDatasetV0(Dataset):
    """
    Loads a single split .npz with canonical layout:
      x: [N,T,x_dim], y: [N,T,y_dim] (float32)
    """
    def __init__(self, npz_path: Path):
        loaded = load_npz_split_v0(npz_path)
        self._x = loaded.x
        self._y = loaded.y
        self._u = loaded.u
        self.meta = loaded.meta
        self.path = npz_path

    def __len__(self) -> int:
        return int(self._y.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(self._x[idx])
        y = torch.from_numpy(self._y[idx])
        if self._u is None:
            return {"x": x, "y": y}
        u = torch.from_numpy(self._u[idx])
        return {"x": x, "y": y, "u": u}


def _collate_with_meta_v0(samples: List[Dict[str, torch.Tensor]], meta: Dict[str, Any]) -> Dict[str, Any]:
    x = torch.stack([s["x"] for s in samples], dim=0)  # [B,T,x_dim]
    y = torch.stack([s["y"] for s in samples], dim=0)  # [B,T,y_dim]
    u = None
    if "u" in samples[0]:
        u = torch.stack([s["u"] for s in samples], dim=0)
    return {"x": x, "y": y, "u": u, "meta": meta}


def make_dataloaders_v0(
    artifacts: DatasetArtifactsV0,
    loader_cfg: Optional[LoaderCfgV0] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    cfg = loader_cfg or LoaderCfgV0()

    ds_train = BenchNpzDatasetV0(artifacts.train.path)
    ds_val = BenchNpzDatasetV0(artifacts.val.path)
    ds_test = BenchNpzDatasetV0(artifacts.test.path)

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        collate_fn=lambda s: _collate_with_meta_v0(s, ds_train.meta),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        collate_fn=lambda s: _collate_with_meta_v0(s, ds_val.meta),
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        collate_fn=lambda s: _collate_with_meta_v0(s, ds_test.meta),
    )
    return train_loader, val_loader, test_loader


def log_first_batch_v0(loader: DataLoader) -> Dict[str, Any]:
    batch = next(iter(loader))
    x = batch["x"]
    y = batch["y"]
    u = batch.get("u", None)
    info = {
        "x.shape": tuple(x.shape),
        "y.shape": tuple(y.shape),
        "x.dtype": str(x.dtype),
        "y.dtype": str(y.dtype),
        "x.device": str(x.device),
        "y.device": str(y.device),
        "u": None if u is None else {"shape": tuple(u.shape), "dtype": str(u.dtype), "device": str(u.device)},
    }
    return info


def prepare_bench_generated_v0(
    *,
    suite_name: str,
    task_cfg: Dict[str, Any],
    seed: int,
    cache_root: Optional[Path] = None,
    scenario_overrides: Optional[Dict[str, Any]] = None,
) -> List[DatasetArtifactsV0]:
    """
    Generate/cache datasets for (task_cfg + sweep grid) for one seed.
    Returns list of artifacts (one per scenario).
    """
    cache_root = cache_root or default_cache_root()
    task_id = str(task_cfg["task_id"])

    scenarios = expand_scenarios_from_sweep(task_cfg)
    out: List[DatasetArtifactsV0] = []
    for base_s in scenarios:
        scenario_cfg = merge_scenario_overrides(base_s, scenario_overrides)
        art = _maybe_generate_and_cache_one(
            cache_root=cache_root,
            suite_name=suite_name,
            task_cfg=task_cfg,
            task_id=task_id,
            scenario_cfg=scenario_cfg,
            seed=seed,
        )
        out.append(art)
    return out


# P1A-CP4 typed-event integration.  This path is deliberately separate from the
# legacy float32 x/y cache and its DatasetArtifactsV0 contract.
P1A_MEKF_TASK_FAMILY = "mekf_unit_st_v1"
P1A_SYNTHETIC_GENERATOR_ID = "synthetic-unit-st-v1"
P1A_BASILISK_GENERATOR_ID = "basilisk-unit-st-v1"
P1A_TYPED_CACHE_VERSION = "p1a-cp4-typed-event-cache-v1"


@dataclass(frozen=True)
class MEKFTypedDatasetArtifacts:
    """Strict three-file UNIT-ST sidecar plus its verified identity."""

    suite_name: str
    task_id: str
    scenario_id: str
    seed: int
    producer_id: str
    cache_state: str
    cache_dir: Path
    dataset_dir: Path
    dataset: Any
    manifest: Dict[str, Any]
    semantic_hashes: Any
    trajectory_split: Any
    dataset_config_hash: str


def _p1a_cp4_canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("CP4 configuration must be finite canonical-JSON data") from error


def _p1a_cp4_path_component(value: Any, name: str) -> str:
    import re

    component = str(value)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", component) is None:
        raise ValueError(f"{name} must be a safe nonempty path component")
    return component


def _p1a_cp4_resolve_dataset_config(
    task_cfg: Mapping[str, Any],
    *,
    seed: int,
    scenario_overrides: Optional[Mapping[str, Any]],
) -> tuple[str, Any, Dict[str, Any], str, str]:
    from dataclasses import asdict

    if str(task_cfg.get("task_family", "")) != P1A_MEKF_TASK_FAMILY:
        raise ValueError(f"task_family must equal {P1A_MEKF_TASK_FAMILY!r}")
    raw = task_cfg.get("typed_event_dataset")
    if not isinstance(raw, Mapping):
        raise ValueError("mekf_unit_st_v1 requires a typed_event_dataset mapping")
    resolved = copy.deepcopy(dict(raw))
    if scenario_overrides:
        override = scenario_overrides.get("typed_event_dataset", scenario_overrides)
        if not isinstance(override, Mapping):
            raise ValueError("typed event scenario override must be a mapping")
        resolved = _deep_merge_dicts(resolved, copy.deepcopy(dict(override)))

    allowed = {"producer_id", "generator_config", "cache_namespace"}
    unexpected = set(resolved) - allowed
    if unexpected:
        raise ValueError(f"unexpected typed_event_dataset keys: {sorted(unexpected)}")
    producer_id = str(resolved.get("producer_id", ""))
    generator_config = resolved.get("generator_config", {})
    if not isinstance(generator_config, Mapping):
        raise ValueError("generator_config must be a mapping")
    generator_kwargs = copy.deepcopy(dict(generator_config))
    if "master_seed" in generator_kwargs and int(generator_kwargs["master_seed"]) != int(seed):
        raise ValueError("generator_config.master_seed must equal the suite seed")
    generator_kwargs["master_seed"] = int(seed)

    if producer_id == P1A_SYNTHETIC_GENERATOR_ID:
        from bench.tasks.generator.unit_st_synthetic import UnitSTSyntheticConfig

        config = UnitSTSyntheticConfig(**generator_kwargs)
    elif producer_id == P1A_BASILISK_GENERATOR_ID:
        from bench.tasks.generator.basilisk_unit_st import BasiliskUnitSTConfig

        config = BasiliskUnitSTConfig(**generator_kwargs)
    else:
        raise ValueError(
            "producer_id must be synthetic-unit-st-v1 or basilisk-unit-st-v1"
        )

    canonical_config = json.loads(_p1a_cp4_canonical_json_bytes(asdict(config)).decode("ascii"))
    cache_namespace = _p1a_cp4_path_component(
        resolved.get("cache_namespace", P1A_TYPED_CACHE_VERSION), "cache_namespace"
    )
    identity = {
        "cache_contract": P1A_TYPED_CACHE_VERSION,
        "generator_config": canonical_config,
        "producer_id": producer_id,
        "task_family": P1A_MEKF_TASK_FAMILY,
    }
    config_hash = hashlib.sha256(_p1a_cp4_canonical_json_bytes(identity)).hexdigest()
    scenario_id = config_hash[:12]
    return producer_id, config, canonical_config, cache_namespace, scenario_id


def _p1a_cp4_expected_runtime_and_sources(producer_id: str) -> tuple[Dict[str, str], Dict[str, str]]:
    import platform
    import scipy

    from bench.estimators import mekf as mekf_core
    from bench.tasks.generator import mekf_events as event_schema

    def digest(module_or_path: Any) -> str:
        path = (
            Path(module_or_path.__file__).resolve()
            if hasattr(module_or_path, "__file__")
            else Path(module_or_path).resolve()
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if producer_id == P1A_SYNTHETIC_GENERATOR_ID:
        from bench.tasks.generator import unit_st_synthetic as generator

        runtime = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        }
        sources = {
            "bench/estimators/mekf.py": digest(mekf_core),
            "bench/tasks/generator/mekf_events.py": digest(event_schema),
            "bench/tasks/generator/unit_st_synthetic.py": digest(generator),
        }
        return runtime, sources

    from bench.tasks.generator import basilisk_unit_st as generator

    proof = _repo_root_from_here() / "docs/research/phase1a/P1A_BASILISK_FRAME_CONVENTION_PROOF.md"
    sources = {
        "bench/estimators/mekf.py": digest(mekf_core),
        "bench/tasks/generator/mekf_events.py": digest(event_schema),
        "bench/tasks/generator/basilisk_unit_st.py": digest(generator),
        "docs/research/phase1a/P1A_BASILISK_FRAME_CONVENTION_PROOF.md": digest(proof),
    }
    return dict(generator._runtime_identity()), sources


def _p1a_cp4_expected_seed_identity(producer_id: str, config: Any) -> tuple[list[int], Dict[str, Any]]:
    if producer_id == P1A_SYNTHETIC_GENERATOR_ID:
        from bench.tasks.generator import unit_st_synthetic as generator

        namespaces = {
            "truth": config.truth_seed_namespace,
            "gyro_noise": config.gyro_noise_seed_namespace,
            "star_tracker_noise": config.star_tracker_noise_seed_namespace,
            "star_tracker_sign": config.star_tracker_sign_seed_namespace,
        }
    else:
        from bench.tasks.generator import basilisk_unit_st as generator

        namespaces = {
            "truth_attitude": config.truth_attitude_seed_namespace,
            "truth_rate": config.truth_rate_seed_namespace,
            "gyro_bias": config.gyro_bias_seed_namespace,
            "gyro_noise": config.gyro_noise_seed_namespace,
            "star_tracker_noise": config.star_tracker_noise_seed_namespace,
            "star_tracker_sign": config.star_tracker_sign_seed_namespace,
        }
    trajectory_ids = [
        int(generator._trajectory_id(config, index)) for index in range(config.num_trajectories)
    ]
    stream_roots = {
        name: int(generator._stream_seed(config, namespace))
        for name, namespace in namespaces.items()
    }
    stream_roots["trajectory_split"] = int(
        generator._stream_seed(config, config.split_seed_namespace)
    )
    per_trajectory = {
        str(trajectory_id): {
            name: int(generator._stream_seed(config, namespace, trajectory_id))
            for name, namespace in namespaces.items()
        }
        for trajectory_id in trajectory_ids
    }
    return trajectory_ids, {"stream_roots": stream_roots, "per_trajectory": per_trajectory}


def _p1a_cp4_validate_loaded_dataset(
    *,
    dataset: Any,
    manifest: Mapping[str, Any],
    producer_id: str,
    config: Any,
    canonical_config: Mapping[str, Any],
) -> Any:
    from bench.tasks.generator.mekf_events import (
        CONVENTION_ID,
        SCHEMA_VERSION,
        SEED_POLICY_VERSION,
        TrajectorySplit,
        split_trajectory_ids,
    )

    required = {
        "schema_version": SCHEMA_VERSION,
        "generator_id": producer_id,
        "seed_policy_version": SEED_POLICY_VERSION,
        "convention_id": CONVENTION_ID,
        "zero_latency": True,
        "same_timestamp_order": ["gyro", "star_tracker"],
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"cached manifest {key} mismatch")
    if manifest.get("generator_config") != dict(canonical_config):
        raise ValueError("cached manifest generator_config mismatch")
    if manifest.get("master_seed") != int(canonical_config["master_seed"]):
        raise ValueError("cached manifest master_seed mismatch")
    expected_runtime, expected_sources = _p1a_cp4_expected_runtime_and_sources(producer_id)
    if manifest.get("software_versions") != expected_runtime:
        raise ValueError("cached manifest runtime identity mismatch")
    if manifest.get("source_fingerprints") != expected_sources:
        raise ValueError("cached manifest source fingerprint mismatch")

    expected_ids, expected_derived_seeds = _p1a_cp4_expected_seed_identity(
        producer_id, config
    )
    if manifest.get("trajectory_ids") != expected_ids:
        raise ValueError("cached manifest deterministic trajectory_ids mismatch")
    if manifest.get("derived_seeds") != expected_derived_seeds:
        raise ValueError("cached manifest derived seed identity mismatch")
    trajectory_ids = np.asarray(manifest.get("trajectory_ids"), dtype=np.int64)
    if not np.array_equal(trajectory_ids, dataset.truth.trajectory_id):
        raise ValueError("cached manifest trajectory_ids mismatch")
    split_record = manifest.get("trajectory_split")
    if not isinstance(split_record, Mapping):
        raise ValueError("cached manifest trajectory_split is missing")
    expected_split = split_trajectory_ids(
        trajectory_ids,
        split_seed=int(split_record.get("split_seed")),
        train_fraction=float(canonical_config["train_fraction"]),
        val_fraction=float(canonical_config["val_fraction"]),
        test_fraction=float(canonical_config["test_fraction"]),
    )
    for name in ("train_ids", "val_ids", "test_ids"):
        recorded = np.asarray(split_record.get(name), dtype=np.int64)
        if not np.array_equal(recorded, getattr(expected_split, name)):
            raise ValueError(f"cached manifest {name} mismatch")
    return TrajectorySplit(
        train_ids=expected_split.train_ids,
        val_ids=expected_split.val_ids,
        test_ids=expected_split.test_ids,
        split_seed=expected_split.split_seed,
    )


def prepare_mekf_unit_st_v1(
    *,
    suite_name: str,
    task_cfg: Mapping[str, Any],
    seed: int,
    cache_root: Optional[Path] = None,
    scenario_overrides: Optional[Mapping[str, Any]] = None,
) -> MEKFTypedDatasetArtifacts:
    """Generate or strictly verify a typed UNIT-ST dataset without dense coercion."""

    import tempfile

    from bench.tasks.generator.mekf_events import load_event_dataset, save_event_dataset

    root = Path(cache_root) if cache_root is not None else default_cache_root()
    producer_id, config, canonical_config, namespace, scenario_id = (
        _p1a_cp4_resolve_dataset_config(
            task_cfg, seed=int(seed), scenario_overrides=scenario_overrides
        )
    )
    task_id = _p1a_cp4_path_component(task_cfg.get("task_id"), "task_id")
    suite_component = _p1a_cp4_path_component(suite_name, "suite_name")
    config_identity = {
        "cache_contract": P1A_TYPED_CACHE_VERSION,
        "generator_config": canonical_config,
        "producer_id": producer_id,
        "task_family": P1A_MEKF_TASK_FAMILY,
    }
    config_hash = hashlib.sha256(_p1a_cp4_canonical_json_bytes(config_identity)).hexdigest()
    cache_dir = (
        root
        / namespace
        / suite_component
        / task_id
        / producer_id
        / config_hash
        / f"seed_{int(seed)}"
    )
    dataset_dir = cache_dir / "dataset"

    if dataset_dir.exists():
        dataset, manifest, hashes = load_event_dataset(
            dataset_dir, expected_generator_id=producer_id
        )
        split = _p1a_cp4_validate_loaded_dataset(
            dataset=dataset,
            manifest=manifest,
            producer_id=producer_id,
            config=config,
            canonical_config=canonical_config,
        )
        cache_state = "verified_cache_hit"
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".p1a_cp4_partial_", dir=cache_dir) as temporary:
            temporary_dataset = Path(temporary) / "dataset"
            if producer_id == P1A_SYNTHETIC_GENERATOR_ID:
                from bench.tasks.generator.unit_st_synthetic import generate_unit_st

                generated = generate_unit_st(config)
            else:
                from bench.tasks.generator.basilisk_unit_st import generate_basilisk_unit_st

                generated = generate_basilisk_unit_st(config)
            save_event_dataset(temporary_dataset, generated.dataset, generated.manifest)
            dataset, manifest, hashes = load_event_dataset(
                temporary_dataset, expected_generator_id=producer_id
            )
            split = _p1a_cp4_validate_loaded_dataset(
                dataset=dataset,
                manifest=manifest,
                producer_id=producer_id,
                config=config,
                canonical_config=canonical_config,
            )
            if dataset_dir.exists():
                raise FileExistsError("typed event cache target appeared during generation")
            os.replace(temporary_dataset, dataset_dir)
        cache_state = "fresh_generation"

    return MEKFTypedDatasetArtifacts(
        suite_name=str(suite_name),
        task_id=task_id,
        scenario_id=scenario_id,
        seed=int(seed),
        producer_id=producer_id,
        cache_state=cache_state,
        cache_dir=cache_dir,
        dataset_dir=dataset_dir,
        dataset=dataset,
        manifest=dict(manifest),
        semantic_hashes=hashes,
        trajectory_split=split,
        dataset_config_hash=config_hash,
    )
