from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from viz.analysis.attitude import continuous_quat_sign, mrp_to_quat
from viz.contract import (
    ARTIFACT_VERSION,
    VALID_DATA_SPLITS,
    VALID_SPLIT_SOURCES,
    capabilities_for,
    deterministic_traj_index,
    formulation_for_task,
    meas_spec_for,
    sanity_benchmark_only,
    source_key_map,
    state_spec_for,
    validate_meta,
    validate_traj_arrays,
    validate_trajectory_capabilities,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _git_text(args: Sequence[str], cwd: Path) -> Optional[str]:
    try:
        cp = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if cp.returncode != 0:
        return None
    return (cp.stdout or "").strip()


def _git_state(repo_root: Path) -> tuple[Optional[str], bool]:
    commit = _git_text(["rev-parse", "HEAD"], repo_root)
    status = _git_text(["status", "--porcelain"], repo_root)
    return commit, bool(status)


def _as_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return np.asarray(value)


def _f32(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _f16(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float16)


def _bool_1d(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 2 and arr.shape[-1] == 1:
        arr = arr[:, 0]
    return np.asarray(arr > 0.5, dtype=bool)


def _extra_for_traj(extras: Mapping[str, Any], source_key: str, traj_idx: int) -> Optional[np.ndarray]:
    if source_key not in extras:
        return None
    arr = np.asarray(extras[source_key])
    if arr.ndim < 2 or arr.shape[0] <= int(traj_idx):
        return None
    return np.asarray(arr[int(traj_idx)])


def _state_has_kind(state_spec: Mapping[str, Any], kind: str) -> bool:
    for item in state_spec.get("layout", []):
        if isinstance(item, Mapping) and item.get("kind") == kind:
            return True
    return False


def _savez_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_json_atomic(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _aggregate_arrays(
    *,
    t: np.ndarray,
    x_true: np.ndarray,
    x_hat: np.ndarray,
    diagnostics: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {"t": _f32(t)}
    err = np.asarray(x_hat, dtype=np.float32) - np.asarray(x_true, dtype=np.float32)
    if err.ndim == 3:
        arrays["err_mean"] = np.mean(err, axis=0, dtype=np.float32).astype(np.float32)
        if err.shape[0] >= 2:
            arrays["emp_std"] = np.std(err, axis=0, ddof=1, dtype=np.float32).astype(np.float32)
    p_arr = _as_numpy(diagnostics.get("P"))
    if p_arr is not None and p_arr.ndim == 4:
        diag = np.diagonal(np.asarray(p_arr, dtype=np.float32), axis1=2, axis2=3)
        diag = np.maximum(diag, np.float32(0.0))
        arrays["pred_sigma_mean"] = np.mean(np.sqrt(diag).astype(np.float32), axis=0).astype(np.float32)
    return arrays


def _traj_arrays(
    *,
    traj_idx: int,
    t: np.ndarray,
    x_true: np.ndarray,
    x_hat: np.ndarray,
    state_spec: Mapping[str, Any],
    capabilities: Mapping[str, bool],
    extras: Mapping[str, Any],
    source_map: Mapping[str, str],
    diagnostics: Mapping[str, Any],
    diagnostic_semantics: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {
        "t": _f32(t),
        "x_true": _f32(x_true[traj_idx]),
        "x_hat": _f32(x_hat[traj_idx]),
    }
    if _state_has_kind(state_spec, "attitude") and arrays["x_true"].shape[-1] >= 3:
        arrays["q_true"] = continuous_quat_sign(mrp_to_quat(arrays["x_true"][:, 0:3])).astype(np.float32)
        arrays["q_hat"] = continuous_quat_sign(mrp_to_quat(arrays["x_hat"][:, 0:3])).astype(np.float32)

    for key, dtype_fn in (("P", _f16), ("S", _f16), ("gain", _f16), ("gain_g1", _f16), ("gain_g2", _f16)):
        value = _as_numpy(diagnostics.get(key))
        if value is not None and value.ndim >= 3 and value.shape[0] > int(traj_idx):
            arrays[key] = dtype_fn(value[int(traj_idx)])
    innov = _as_numpy(diagnostics.get("innov"))
    if innov is not None and innov.ndim >= 3 and innov.shape[0] > int(traj_idx):
        arrays["innov"] = _f32(innov[int(traj_idx)])
        diagnostic_mask = _as_numpy(diagnostics.get("innov_valid"))
        if diagnostic_mask is not None:
            if diagnostic_mask.ndim != 2 or diagnostic_mask.shape[0] <= int(traj_idx):
                raise ValueError(
                    "diagnostics.innov_valid must have shape [N,T] matching innovation history"
                )
            valid_mask = _bool_1d(diagnostic_mask[int(traj_idx)])
        else:
            valid_mask = np.ones((arrays["innov"].shape[0],), dtype=bool)
        if "ref_mask" in source_map:
            ref_mask = _extra_for_traj(extras, source_map["ref_mask"], traj_idx)
            if ref_mask is not None:
                valid_mask = np.logical_and(valid_mask, _bool_1d(ref_mask))
        arrays["innov_valid"] = valid_mask.astype(bool, copy=False)

    for artifact_key in ("bias_component", "noise_component", "imu_error", "b_true"):
        source_key = source_map.get(artifact_key)
        if source_key is None:
            continue
        arr = _extra_for_traj(extras, source_key, traj_idx)
        if arr is not None:
            arrays[artifact_key] = _f32(arr)

    for artifact_key in ("eclipse_flag", "event_flag", "ref_mask"):
        source_key = source_map.get(artifact_key)
        if source_key is None:
            continue
        arr = _extra_for_traj(extras, source_key, traj_idx)
        if arr is not None:
            arrays[artifact_key] = _bool_1d(arr)

    validate_traj_arrays(arrays)
    validate_trajectory_capabilities(
        {"capabilities": capabilities, "diagnostic_semantics": diagnostic_semantics},
        arrays,
    )
    return arrays


def _source_trajectory_ids(values: Any, n_seq: int) -> np.ndarray:
    if values is None:
        return np.arange(int(n_seq), dtype=np.int64)
    arr = np.asarray(values)
    if arr.ndim != 1 or int(arr.shape[0]) != int(n_seq):
        raise ValueError(f"trajectory_ids must have shape [{n_seq}], got {arr.shape}")
    return arr


def _source_id_json(value: Any) -> Any:
    converted = _jsonable(value)
    if converted is None or isinstance(converted, (dict, list, bool)):
        raise ValueError(f"source trajectory ID must be a non-null scalar, got {converted!r}")
    return converted


def write_viz_artifacts(
    *,
    run_dir: str | Path,
    repo_root: str | Path,
    suite_name: str,
    task_id: str,
    task_family: str,
    scenario_id: str,
    model_id: str,
    seed: int,
    track_id: str,
    init_id: str,
    run_status: str,
    time_s: np.ndarray,
    time_meta: Mapping[str, Any],
    x_true: np.ndarray,
    y_obs: np.ndarray,
    x_hat: np.ndarray,
    split_extras: Optional[Mapping[str, Any]],
    diagnostics: Optional[Mapping[str, Any]],
    adapter_meta: Optional[Mapping[str, Any]],
    config_hash: Optional[str] = None,
    k_traj: int = 8,
    data_split: str = "unknown",
    split_source: Optional[str] = None,
    trajectory_ids: Any = None,
    trajectory_id_source: Optional[str] = None,
    scenario_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    extras = split_extras if isinstance(split_extras, Mapping) else {}
    diag = diagnostics if isinstance(diagnostics, Mapping) else {}
    adapter_meta_dict = dict(adapter_meta or {})
    raw_diagnostic_semantics = adapter_meta_dict.get("diagnostic_semantics", {})
    diagnostic_semantics = (
        dict(raw_diagnostic_semantics)
        if isinstance(raw_diagnostic_semantics, Mapping)
        else {}
    )
    x_true_arr = _f32(x_true)
    y_arr = _f32(y_obs)
    x_hat_arr = _f32(x_hat)
    t_arr = _f32(time_s)
    if x_true_arr.shape != x_hat_arr.shape:
        raise ValueError(f"x_true/x_hat shape mismatch: {x_true_arr.shape} != {x_hat_arr.shape}")
    if x_true_arr.ndim != 3:
        raise ValueError(f"x_true must have shape [N,T,D], got {x_true_arr.shape}")
    if t_arr.ndim != 1 or t_arr.shape[0] != x_true_arr.shape[1]:
        raise ValueError(f"time_s must have shape [T] with T={x_true_arr.shape[1]}, got {t_arr.shape}")

    n_seq, n_step, x_dim = (int(v) for v in x_true_arr.shape)
    y_dim = int(y_arr.shape[-1]) if y_arr.ndim == 3 else 0
    split_name = str(data_split)
    if split_name not in VALID_DATA_SPLITS:
        raise ValueError(f"unsupported data_split={split_name!r}")
    split_source_name = str(split_source or ("legacy_unknown" if split_name == "unknown" else "explicit"))
    if split_source_name not in VALID_SPLIT_SOURCES:
        raise ValueError(f"unsupported split_source={split_source_name!r}")
    source_ids = _source_trajectory_ids(trajectory_ids, n_seq)
    source_id_source = str(
        trajectory_id_source
        or ("split_row_index_fallback" if trajectory_ids is None else "provided_trajectory_id")
    )
    traj_index = deterministic_traj_index(n_seq, k=k_traj)
    selected_source_ids = [_source_id_json(source_ids[int(row_index)]) for row_index in traj_index]
    selected_source_keys = [(type(value).__name__, str(value)) for value in selected_source_ids]
    if len(set(selected_source_keys)) != len(selected_source_keys):
        raise ValueError(f"selected source trajectory IDs must be unique, got {selected_source_ids!r}")
    source_map = source_key_map(extras)
    formulation = formulation_for_task(task_family, task_id=task_id)
    commit, dirty = _git_state(repo)
    caps = capabilities_for(model_id=str(model_id), diagnostics=diag, source_map=source_map)
    state_spec = state_spec_for(formulation, x_dim)

    series_dir = root / "series"
    aggregate = _aggregate_arrays(t=t_arr, x_true=x_true_arr, x_hat=x_hat_arr, diagnostics=diag)
    _savez_atomic(series_dir / "aggregate.npz", aggregate)

    manifest: list[Dict[str, Any]] = []
    expected_files: set[Path] = set()
    for stored_index, row_index in enumerate(traj_index):
        arrays = _traj_arrays(
            traj_idx=int(row_index),
            t=t_arr,
            x_true=x_true_arr,
            x_hat=x_hat_arr,
            state_spec=state_spec,
            capabilities=caps,
            extras=extras,
            source_map=source_map,
            diagnostics=diag,
            diagnostic_semantics=diagnostic_semantics,
        )
        relative_file = f"series/traj_{stored_index:04d}.npz"
        traj_path = root / relative_file
        _savez_atomic(traj_path, arrays)
        expected_files.add(traj_path.resolve())
        manifest.append(
            {
                "stored_index": int(stored_index),
                "source_trajectory_id": selected_source_ids[stored_index],
                "source_trajectory_id_source": source_id_source,
                "selection_row_index": int(row_index),
                "file": relative_file,
                "length_T": int(arrays["t"].shape[0]),
                "time_start": float(arrays["t"][0]),
                "time_end": float(arrays["t"][-1]),
                "has_event": bool(np.any(arrays["event_flag"])) if "event_flag" in arrays else None,
                "has_eclipse": bool(np.any(arrays["eclipse_flag"])) if "eclipse_flag" in arrays else None,
                "run_status": str(run_status),
            }
        )
    for stale_path in series_dir.glob("traj_*.npz"):
        if stale_path.resolve() not in expected_files:
            stale_path.unlink()

    scenario = dict(scenario_meta or {})
    scenario.setdefault("display_name", None)
    scenario.setdefault("parameters", {})
    meta: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "commit": commit,
        "worktree_dirty": bool(dirty),
        "config_hash": config_hash,
        "suite": str(suite_name),
        "task": str(task_id),
        "task_family": str(task_family),
        "scenario_id": str(scenario_id),
        "scenario": scenario,
        "model_id": str(model_id),
        "seed": int(seed),
        "track_id": str(track_id),
        "init_id": str(init_id),
        "formulation": formulation,
        "sanity_benchmark_only": sanity_benchmark_only(formulation),
        "state_spec": state_spec,
        "meas_spec": meas_spec_for(task_family, y_dim),
        "capabilities": caps,
        "diagnostic_semantics": diagnostic_semantics,
        "source_key_map": source_map,
        "traj_index": [int(v) for v in traj_index],
        "data_spec": {
            "split": split_name,
            "split_source": split_source_name,
            "num_trajectories": n_seq,
            "num_stored_trajectories": len(manifest),
            "trajectory_selection": "deterministic_uniform",
            "source_trajectory_id_source": source_id_source,
            "is_live": False,
        },
        "trajectories": manifest,
        "dt": time_meta.get("dt_s"),
        "time": dict(time_meta),
        "T": n_step,
        "N_test": n_seq,
        "run_status": str(run_status),
        "d1_error_state_implemented": False,
        "validation_scope": "v0_contract_structure_only_d1_not_implemented",
        "adapter_meta": adapter_meta_dict,
    }
    validate_meta(meta)
    _write_json_atomic(root / "meta.json", meta)
    return meta
