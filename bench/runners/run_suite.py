from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("PyYAML(yaml) is required for run_suite. Install pyyaml.") from e

try:
    import torch
except Exception as e:
    raise RuntimeError("PyTorch(torch) is required for run_suite.") from e

try:
    from torch.utils.data import DataLoader, Dataset
except Exception as e:
    raise RuntimeError("PyTorch DataLoader is required for run_suite.") from e

from bench.metrics.core import (
    mse_per_step,
    compute_shift_recovery_k,
)
from bench.metrics.adcs_event import compute_adcs_event_metrics
from bench.visualization.pred_artifact import (
    PRED_ARTIFACT_FILENAME,
    PRED_META_FILENAME,
    save_pred_artifact,
)
from bench.utils.diagnostics import (
    array_stats,
    format_array_stats,
    has_nonfinite,
    residual_array,
    residual_stats,
    short_window,
    summarize_mapping_arrays,
    validate_exact_layout,
    write_diagnostic_dump,
)
from bench.utils.logging import (
    clear_logging_context,
    configure_logging,
    get_logger,
    is_debug_enabled,
    set_logging_context,
)
from bench.utils.sweep import expand_sweep_grid


logger = get_logger(__name__)

# Control-plane observability. `active_observer()` returns a NullObserver unless a
# control-plane worker installed one, so every call below is a no-op under the
# existing CLI and changes no runner behaviour. Phase boundaries are emitted here,
# at the runner level, so that even an adapter with no internal instrumentation
# still reports setup/train/test transitions to the dashboard.
try:
    from bench.control.events.observer import active_observer  # type: ignore
except Exception:  # pragma: no cover - control plane is optional

    def active_observer():  # type: ignore[misc]
        class _Null:
            def status(self, *args: Any, **kwargs: Any) -> None: ...
            def metric(self, *args: Any, **kwargs: Any) -> None: ...
            def log(self, *args: Any, **kwargs: Any) -> None: ...
            def artifact(self, *args: Any, **kwargs: Any) -> None: ...

        return _Null()


# Optional: use existing bench utils if available
def _try_import_utils():
    io_mod = None
    seed_mod = None
    env_mod = None
    try:
        from bench.utils import io as io_mod  # type: ignore
    except Exception:
        io_mod = None
    try:
        from bench.utils import seeding as seed_mod  # type: ignore
    except Exception:
        seed_mod = None
    try:
        from bench.utils import env_detect as env_mod  # type: ignore
    except Exception:
        env_mod = None
    return io_mod, seed_mod, env_mod


def _bench_root() -> Path:
    # .../bench/bench/runners/run_suite.py -> parents[2] == .../bench
    return Path(__file__).resolve().parents[2]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _runner_logging_cfg(
    runner_cfg: Dict[str, Any],
    *,
    log_level: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    log_file: Optional[str] = None,
    debug_every: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = dict(runner_cfg.get("logging", {}) or {})
    return {
        "level": str(log_level or cfg.get("level", "INFO")).strip().upper(),
        "log_to_file": bool(cfg.get("log_to_file", False) if log_to_file is None else log_to_file),
        "log_file": (str(log_file) if log_file else (str(cfg.get("log_file")) if cfg.get("log_file") else None)),
        "debug_every": int(cfg.get("debug_every", 0) if debug_every is None else debug_every),
    }


def _fix_yaml_block_scalar_indentation(text: str) -> str:
    """
    suite yaml에서 block scalar(> 또는 |) 다음 줄 들여쓰기가 깨진 경우를
    "파싱용으로만" 완화한다. (파일은 수정하지 않음)
    """
    lines = text.splitlines()
    out: List[str] = []
    in_block = False
    key_indent = 0
    content_indent = 0

    def leading_spaces(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    i = 0
    while i < len(lines):
        line = lines[i]
        if not in_block:
            stripped = line.lstrip(" ")
            if (": >" in stripped or ": |" in stripped) and not stripped.endswith(("<",)):
                if stripped.endswith(">") or stripped.endswith("|"):
                    key_indent = leading_spaces(line)
                    content_indent = key_indent + 2
                    in_block = True
                    out.append(line)
                    i += 1
                    continue
            out.append(line)
            i += 1
            continue

        if line.strip() == "":
            out.append(line)
            i += 1
            continue

        indent = leading_spaces(line)
        if indent <= key_indent and ":" in line:
            in_block = False
            continue  # re-process in normal mode

        if indent < content_indent:
            out.append((" " * content_indent) + line.lstrip(" "))
        else:
            out.append(line)
        i += 1

    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def load_suite_yaml(path: Path) -> Dict[str, Any]:
    raw = path.read_text()
    try:
        return yaml.safe_load(raw)
    except Exception:
        fixed = _fix_yaml_block_scalar_indentation(raw)
        return yaml.safe_load(fixed)


def _expand_sweep(sweep: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return expand_sweep_grid(sweep, sort_keys=True)


def _resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        return torch.device("cpu")
    return torch.device(device_str)


def _tensorize(x: np.ndarray, device: torch.device) -> torch.Tensor:
    # canonical layout: [B,T,D]
    t = torch.from_numpy(x).float()
    return t.to(device)


def _split_batch_stats(split_name: str, x: np.ndarray, y: np.ndarray, batch_size: int) -> List[str]:
    bs = max(1, int(batch_size))
    x_b = x[:bs]
    y_b = y[:bs]
    return [
        f"{split_name} first batch",
        format_array_stats("x", x_b),
        format_array_stats("y", y_b),
    ]


def _extract_shift_debug(meta: Optional[Dict[str, Any]], task: Dict[str, Any]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    noise_meta = meta.get("noise", {}) if isinstance(meta, dict) else {}
    shift_meta = noise_meta.get("shift", {}) if isinstance(noise_meta, dict) else {}
    pre_shift = noise_meta.get("pre_shift", {}) if isinstance(noise_meta, dict) else {}
    post_shift = shift_meta.get("post_shift", {}) if isinstance(shift_meta, dict) else {}
    noise_schedule = meta.get("noise_schedule", {}) if isinstance(meta, dict) else {}

    t0 = _extract_t0(task, meta)
    if t0 is not None:
        info["t0"] = int(t0)
    pre_r2 = None
    if isinstance(pre_shift, dict):
        pre_r = pre_shift.get("R", {})
        if isinstance(pre_r, dict) and "r2" in pre_r:
            try:
                pre_r2 = float(pre_r["r2"])
            except Exception:
                pre_r2 = None
    if pre_r2 is not None:
        info["pre_shift_r2"] = pre_r2
    if isinstance(post_shift, dict):
        for key in ("R_scale", "R_scale(applied)", "R_scale_applied"):
            if key in post_shift:
                try:
                    info["post_shift_R_scale"] = float(post_shift[key])
                    break
                except Exception:
                    break
    if isinstance(noise_schedule, dict) and noise_schedule:
        info["noise_schedule_keys"] = sorted(str(k) for k in noise_schedule.keys())
        kind = noise_schedule.get("kind")
        if kind is not None:
            info["noise_schedule_kind"] = str(kind)
    return info


def _log_split_debug(
    split_name: str,
    split: Dict[str, Any],
    *,
    batch_size: int,
) -> None:
    for line in _split_batch_stats(split_name, split["x"], split["y"], batch_size):
        logger.debug(line)


def _capture_first_batch(
    *,
    split: Dict[str, Any],
    eval_bs: int,
    x_hat_full: Optional[torch.Tensor],
    mse_t_mean: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    bs = max(1, int(eval_bs))
    x = short_window(split["x"][:bs], batch_limit=bs)
    y = short_window(split["y"][:bs], batch_limit=bs)
    payload: Dict[str, Any] = {"x": x, "y": y}
    if x_hat_full is not None:
        x_hat = short_window(x_hat_full[:bs], batch_limit=bs)
        payload["x_hat"] = x_hat
        resid = residual_array(x_hat, x)
        if resid is not None:
            payload["residual"] = resid
            if resid.ndim == 3:
                payload["residual_norm_t"] = np.linalg.norm(resid, axis=(0, 2))
    if mse_t_mean is not None:
        payload["mse_t"] = np.array(mse_t_mean[: min(len(mse_t_mean), 64)], copy=True)
    return payload


def _write_run_diagnostics(
    *,
    run_dir: Path,
    suite_name: str,
    task_id: str,
    scenario_id: str,
    model_id: str,
    track_id: str,
    init_id: str,
    seed: int,
    split_test: Dict[str, Any],
    x_hat_full: Optional[torch.Tensor],
    mse_val: Optional[float],
    mse_db_val: Optional[float],
    mse_t_mean: Optional[np.ndarray],
    thresholds_hit: Sequence[str],
    shift_info: Dict[str, Any],
    adapter_meta: Dict[str, Any],
    adapter_runtime: Optional[Dict[str, Any]],
    reason: str,
) -> Dict[str, str]:
    stats_obj: Dict[str, Any] = {
        "reason": str(reason),
        "suite_name": str(suite_name),
        "task_id": str(task_id),
        "scenario_id": str(scenario_id),
        "model_id": str(model_id),
        "track_id": str(track_id),
        "init_id": str(init_id),
        "seed": int(seed),
        "thresholds_hit": list(str(x) for x in thresholds_hit),
        "shift": shift_info,
        "metrics": {"mse": mse_val, "mse_db": mse_db_val},
        "x_stats": array_stats(split_test["x"]),
        "y_stats": array_stats(split_test["y"]),
        "x_hat_stats": (array_stats(x_hat_full) if x_hat_full is not None else None),
        "residual_stats": (residual_stats(x_hat_full, split_test["x"]) if x_hat_full is not None else None),
        "adapter_meta": adapter_meta,
        "adapter_runtime_stats": summarize_mapping_arrays("adapter_runtime", adapter_runtime),
    }
    arrays = _capture_first_batch(
        split=split_test,
        eval_bs=int(min(max(1, split_test["x"].shape[0]), 8)),
        x_hat_full=x_hat_full,
        mse_t_mean=mse_t_mean,
    )
    extra_arrays: Dict[str, Any] = {}
    if adapter_runtime:
        for key, value in adapter_runtime.items():
            try:
                arr = np.asarray(value)
            except Exception:
                continue
            if arr.ndim >= 1:
                extra_arrays[f"adapter_{key}"] = arr
    return write_diagnostic_dump(run_dir=run_dir, stats=stats_obj, arrays=arrays, extra_arrays=extra_arrays)


def _extract_t0(task: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> Optional[int]:
    """
    meta_json에 t0가 없을 수 있으므로:
    1) meta에서 최대한 찾아보고
    2) 없으면 suite task.noise.shift.t0로 fallback
    """
    if meta:
        candidates = [
            ("noise", "shift", "t0"),
            ("shift", "t0"),
            ("t0_shift",),
            ("noise", "t0_shift"),
        ]
        for c in candidates:
            cur: Any = meta
            ok = True
            for k in c:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok:
                try:
                    return int(cur)
                except Exception:
                    pass

    try:
        t0 = task.get("noise", {}).get("shift", {}).get("t0", None)
        if t0 is not None:
            return int(t0)
    except Exception:
        pass
    return None


def _adcs_event_metrics_if_available(
    *,
    x_true: np.ndarray,
    x_pred: np.ndarray,
    split_extras: Optional[Mapping[str, np.ndarray]],
) -> Optional[Dict[str, Any]]:
    extras = split_extras if isinstance(split_extras, Mapping) else {}
    event_flag = extras.get("event_flag_seq")
    if event_flag is None:
        return None
    metrics = compute_adcs_event_metrics(
        x_true=x_true,
        x_pred=x_pred,
        event_flag_seq=event_flag,
    )
    json_metrics = {
        key: (float(value) if np.isfinite(float(value)) else None)
        for key, value in metrics.items()
    }
    event_mask = np.asarray(event_flag) > 0.5
    return {
        **json_metrics,
        "event_sample_count": int(np.sum(event_mask)),
        "event_flag_source": "test.npz:event_flag_seq",
        "attitude_error_definition": "exact_shortest_quaternion_angle_from_mrp",
        "angular_velocity_units": "match_state_omega_units",
    }


def _nested_value(mapping: Any, path: Sequence[str]) -> Any:
    cur = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _prediction_artifact_policy(
    *,
    suite: Dict[str, Any],
    task: Dict[str, Any],
    runner_cfg: Dict[str, Any],
) -> Tuple[bool, bool]:
    configured: Optional[bool] = None
    for container in (
        task.get("artifacts"),
        suite.get("artifacts"),
        runner_cfg.get("artifacts"),
    ):
        if isinstance(container, Mapping) and "save_predictions" in container:
            configured = bool(container["save_predictions"])
            break

    visualization_enabled = False
    for container in (task.get("visualization"), suite.get("visualization")):
        if isinstance(container, Mapping) and bool(container.get("enabled", False)):
            visualization_enabled = True
            break

    # D20: prediction artifacts default on. Visualization requires the artifact
    # even if a conflicting save_predictions=false is present.
    save_predictions = True if configured is None else configured
    return bool(save_predictions or visualization_enabled), bool(visualization_enabled)


def _viz_artifact_policy(
    *,
    suite: Dict[str, Any],
    task: Dict[str, Any],
    runner_cfg: Dict[str, Any],
    cli_emit: Optional[bool],
) -> bool:
    if bool(cli_emit):
        return True
    for container in (task.get("viz"), suite.get("viz"), runner_cfg.get("viz")):
        if isinstance(container, Mapping) and bool(container.get("emit", False)):
            return True
    return False


def _prediction_time_s(
    *,
    task: Dict[str, Any],
    split_test: Dict[str, Any],
    meta: Optional[Dict[str, Any]],
    n_step: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    direct = split_test.get("time_s")
    if direct is None:
        direct = (split_test.get("extras") or {}).get("time_s")
    if direct is not None:
        return np.asarray(direct), {"time_source": "dataset:time_s", "time_unit": "s"}

    dt_candidates = (
        ("meta.ssm.true.dt", _nested_value(meta, ("ssm", "true", "dt"))),
        ("meta.simulation.dt", _nested_value(meta, ("simulation", "dt"))),
        ("meta.dt", _nested_value(meta, ("dt",))),
        ("task.simulation.dt", _nested_value(task, ("simulation", "dt"))),
        ("task.dt", _nested_value(task, ("dt",))),
    )
    for source, raw in dt_candidates:
        try:
            dt = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(dt) and dt > 0.0:
            return (
                np.arange(int(n_step), dtype=np.float64) * dt,
                {"time_source": source, "time_unit": "s", "dt_s": dt},
            )

    hz_candidates = (
        ("meta.sampling_hz", _nested_value(meta, ("sampling_hz",))),
        ("meta.protocol.sampling_hz", _nested_value(meta, ("protocol", "sampling_hz"))),
        ("task.sampling_hz", _nested_value(task, ("sampling_hz",))),
    )
    for source, raw in hz_candidates:
        try:
            hz = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(hz) and hz > 0.0:
            dt = 1.0 / hz
            return (
                np.arange(int(n_step), dtype=np.float64) * dt,
                {"time_source": source, "time_unit": "s", "dt_s": dt},
            )

    return (
        np.arange(int(n_step), dtype=np.float64),
        {
            "time_source": "sample_index_fallback",
            "time_unit": "sample_index",
            "time_warning": "No task timestamp or sampling interval was available; time_s stores sample indices.",
        },
    )


def _prediction_trajectory_id(split_test: Dict[str, Any], n_seq: int) -> np.ndarray:
    direct = split_test.get("trajectory_id")
    if direct is None:
        direct = (split_test.get("extras") or {}).get("trajectory_id")
    if direct is None:
        return np.arange(int(n_seq), dtype=np.int64)
    return np.asarray(direct)


def _prediction_state_meta(
    *,
    task: Dict[str, Any],
    meta: Optional[Dict[str, Any]],
    x_dim: int,
) -> Dict[str, Any]:
    if int(x_dim) != 9:
        return {}

    task_family = str(
        task.get("task_family")
        or (meta or {}).get("task_family")
        or ""
    ).lower()
    state_names = _nested_value(meta, ("ssm", "true", "state"))
    state_text = " ".join(str(v).lower() for v in state_names) if isinstance(state_names, list) else ""
    has_bias_state = (
        "bias_adcs" in task_family
        or "gyro_bias" in state_text
        or isinstance((meta or {}).get("bias_state"), Mapping)
    )
    if not has_bias_state:
        return {}

    return {
        "state_schema": {
            "attitude": {
                "type": "mrp",
                "name": "sigma_BN",
                "indices": [0, 1, 2],
            },
            "angular_rate": {
                "type": "rad_s",
                "name": "omega_BN_B",
                "indices": [3, 4, 5],
            },
            "gyro_bias": {
                "type": "rad_s",
                "name": "gyro_bias",
                "indices": [6, 7, 8],
                "optional": True,
            },
        },
        "attitude_convention": "MRP sigma_BN",
        "time_unit": "s",
    }


def _cache_root(default_bench_root: Path) -> Path:
    env = os.environ.get("BENCH_DATA_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return (default_bench_root / "bench_data_cache").resolve()


def _npz_path(cache_root: Path, suite_name: str, task_id: str, scenario_id: str, seed: int, split: str) -> Path:
    return cache_root / suite_name / task_id / f"scenario_{scenario_id}" / f"seed_{seed}" / f"{split}.npz"


def _write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def _write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_yaml(p: Path, obj: Any) -> None:
    _ensure_dir(p.parent)
    p.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _append_summary_row(summary_csv: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    _ensure_dir(summary_csv.parent)
    file_exists = summary_csv.exists()
    if file_exists:
        try:
            with summary_csv.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_fields = list(reader.fieldnames or [])
                if existing_fields != fieldnames:
                    existing_rows = list(reader)
                    with summary_csv.open("w", newline="", encoding="utf-8") as wf:
                        ww = csv.DictWriter(wf, fieldnames=fieldnames)
                        ww.writeheader()
                        for r in existing_rows:
                            ww.writerow({k: r.get(k, "") for k in fieldnames})
        except Exception:
            # If summary is malformed, recreate from this row forward.
            file_exists = False
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        safe_row = {k: row.get(k, "") for k in fieldnames}
        w.writerow(safe_row)


def _load_adapter(model_id: str):
    """
    Prefer bench.models.registry if exists.
    Fallback to direct imports.
    """
    try:
        from bench.models.registry import get_model_adapter_class  # type: ignore
        cls = get_model_adapter_class(model_id)
        return cls
    except Exception:
        if model_id == "kalmannet_tsp":
            from bench.models.kalmannet_tsp import KalmanNetTSPAdapter  # type: ignore
            return KalmanNetTSPAdapter
        if model_id == "adaptive_knet":
            from bench.models.adaptive_knet import AdaptiveKNetAdapter  # type: ignore
            return AdaptiveKNetAdapter
        raise


def _track_cfg(runner_cfg: Dict[str, Any], track_id: str) -> Dict[str, Any]:
    for t in runner_cfg.get("tracks", []) or []:
        if t.get("track_id") == track_id:
            return t
    raise KeyError(f"track_id={track_id} not found in suite.runner.tracks")


def _enabled(obj: Dict[str, Any], default: bool) -> bool:
    if "enabled" not in obj:
        return default
    return bool(obj["enabled"])


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer, np.int32, np.int64)):
        return int(v)
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    return v


def _dotted_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: Dict[str, Any] = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]  # type: ignore[assignment]
    cur[parts[-1]] = _to_jsonable(value)


def _is_inv_r2_db_sweep_key(key: str) -> bool:
    kk = str(key).strip()
    return kk in {"inv_r2_db", "noise.inv_r2_db"}


def _resolve_r2_basis_key(noise_cfg: Dict[str, Any]) -> Optional[str]:
    pre = noise_cfg.get("pre_shift", {})
    if isinstance(pre, dict):
        pre_r = pre.get("R", {})
        if isinstance(pre_r, dict) and ("r2" in pre_r):
            return "pre_shift.R.r2"

    r_map = noise_cfg.get("R", {})
    if isinstance(r_map, dict) and ("r2" in r_map):
        return "R.r2"
    # Fallback: allow alias sweep even when task omits explicit noise.R.r2.
    return "R.r2"


def _inv_r2_db_to_r2(value: Any) -> float:
    db = float(value)
    return float(np.power(10.0, -db / 10.0))


def _build_scenario_cfg_basis(task: Dict[str, Any], scenario_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonical scenario basis:
      deep_copy(task.noise) + sweep overrides.
    This basis is used for scenario_id hashing and config snapshots.
    """
    scenario_cfg = copy.deepcopy(task.get("noise", {}) or {})
    r2_basis_key = _resolve_r2_basis_key(scenario_cfg)
    for k, v in (scenario_settings or {}).items():
        kk = str(k)
        if _is_inv_r2_db_sweep_key(kk):
            if r2_basis_key is None:
                raise ValueError(
                    "config_error: inv_r2_db sweep requires task noise config with "
                    "either noise.pre_shift.R.r2 or noise.R.r2"
                )
            _dotted_set(scenario_cfg, r2_basis_key, _inv_r2_db_to_r2(v))
            continue
        if kk.startswith("noise."):
            kk = kk[len("noise."):]
        _dotted_set(scenario_cfg, kk, v)

    def _normalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _normalize(_to_jsonable(v)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_normalize(_to_jsonable(x)) for x in obj]
        return _to_jsonable(obj)

    return _normalize(scenario_cfg)


def _canonicalize_scenario_id(task_id: str, scenario_cfg_basis: Dict[str, Any]) -> str:
    """
    Step4(bench_generated)와 동일한 규칙을 최우선으로 사용.
    """
    try:
        from bench.tasks.bench_generated import canonicalize_scenario_id as _bg  # type: ignore
        return str(_bg(task_id, scenario_cfg_basis))
    except Exception:
        payload = {"task_id": task_id, "scenario": scenario_cfg_basis}
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _scenario_id_for_settings(task: Dict[str, Any], scenario_settings: Dict[str, Any]) -> str:
    task_id = str(task.get("task_id", ""))
    return _canonicalize_scenario_id(task_id, _build_scenario_cfg_basis(task, scenario_settings))


def _meta_get(meta: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = meta
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def _match_scenario_settings_from_meta(meta: Dict[str, Any], scenario_settings: Dict[str, Any]) -> bool:
    """
    cache 스캔 fallback을 위한 meta_json 매칭.
    """
    for k, v in (scenario_settings or {}).items():
        if _is_inv_r2_db_sweep_key(str(k)):
            r2_candidates = [
                _meta_get(meta, ("noise", "pre_shift", "R", "r2")),
                _meta_get(meta, ("noise", "R", "r2")),
                _meta_get(meta, ("pre_shift", "R", "r2")),
                _meta_get(meta, ("R", "r2")),
            ]
            found_r2 = None
            for rr in r2_candidates:
                if rr is not None:
                    found_r2 = rr
                    break
            if found_r2 is None:
                return False
            try:
                vv = float(v)
                inv_found = float(-10.0 * math.log10(max(float(found_r2), 1.0e-30)))
            except Exception:
                return False
            if abs(float(vv) - float(inv_found)) > 1e-6:
                return False
            continue

        parts = tuple(k.split("."))
        candidates = [
            ("noise",) + parts,
            parts,
        ]
        found = None
        for cand in candidates:
            found = _meta_get(meta, cand)
            if found is not None:
                break

        if found is None and k.endswith("R_scale"):
            parts2 = list(parts)
            parts2[-1] = "R_scale(applied)"
            found = _meta_get(meta, ("noise",) + tuple(parts2))
            if found is None:
                parts2[-1] = "R_scale_applied"
                found = _meta_get(meta, ("noise",) + tuple(parts2))

        if found is None:
            return False

        vv = _to_jsonable(v)
        ff = _to_jsonable(found)
        if isinstance(vv, (int, float)) and isinstance(ff, (int, float)):
            if abs(float(vv) - float(ff)) > 1e-9:
                return False
        else:
            if vv != ff:
                return False

    return True


def _resolve_scenario_id_from_cache(
    cache_root: Path,
    suite_name: str,
    task_id: str,
    seed: int,
    scenario_settings: Dict[str, Any],
) -> Optional[str]:
    """
    computed scenario_id로 파일이 없을 때, cache를 스캔해서 meta_json과 scenario_settings가 일치하는
    scenario_id를 찾아준다(안전장치).
    """
    task_root = cache_root / suite_name / task_id
    if not task_root.exists():
        return None

    matches: List[str] = []
    for scen_dir in sorted(task_root.glob("scenario_*")):
        seed_dir = scen_dir / f"seed_{seed}"
        test_npz = seed_dir / "test.npz"
        if not test_npz.exists():
            continue
        try:
            d = np.load(test_npz, allow_pickle=True)
            if "meta_json" not in d:
                continue
            meta = json.loads(d["meta_json"].item())
            if isinstance(meta, dict) and _match_scenario_settings_from_meta(meta, scenario_settings):
                sid = scen_dir.name.replace("scenario_", "", 1)
                matches.append(sid)
        except Exception:
            continue

    if len(matches) == 1:
        return matches[0]
    return None


def _validate_batch_x_hat(x_hat: Any, B: int, T: int, D: int) -> torch.Tensor:
    if isinstance(x_hat, np.ndarray):
        x_hat = torch.from_numpy(x_hat)
    if not isinstance(x_hat, torch.Tensor):
        raise TypeError(f"adapter.predict must return Tensor/tuple/dict. Got {type(x_hat)}")
    x_hat = x_hat.detach().cpu().float()
    try:
        validate_exact_layout(
            x_hat,
            expected=(int(B), int(T), int(D)),
            axis_names=("B", "T", "D"),
            label="x_hat",
        )
    except ValueError as exc:
        logger.error("Adapter output contract violation: %s", exc)
        raise
    return x_hat.contiguous()


def _classify_failure(exc: Exception) -> str:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "eval_nonfinite" in msg:
        return "runtime_error"
    if isinstance(exc, ImportError) or "import" in msg or "module" in msg:
        return "import_failure"
    if "policy_violation" in msg:
        return "policy_violation"
    if "shape_mismatch" in msg or "shape" in msg:
        return "shape_mismatch"
    if "budget_overflow" in msg:
        return "budget_overflow"
    if "train_nan" in msg or "non-finite" in msg or "nan" in msg:
        return "train_nan"
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg:
        return "oom"
    if isinstance(exc, (OSError, IOError, FileNotFoundError, PermissionError)):
        return "io_error"
    return "runtime_error"


def _classify_phase_from_traceback(traceback_text: str) -> str:
    tb = traceback_text.lower()
    if "_try_call_train" in tb or "adapter.train" in tb:
        return "train"
    if "_try_call_adapt" in tb or "adapter.adapt" in tb:
        return "adapt"
    if "_try_call_eval" in tb or "adapter.eval" in tb or "_predict_batches" in tb:
        return "eval"
    if "_try_call_setup" in tb or "adapter.setup" in tb:
        return "setup"
    if "_load_split_npz" in tb or "missing_data" in tb:
        return "data_loading"
    return "runner"


def _normalize_jsonish(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _normalize_jsonish(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize_jsonish(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return _to_jsonable(obj)


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _dataset_hash(paths: Sequence[Path]) -> str:
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(str(p.resolve()).encode("utf-8"))
        if p.exists():
            h.update(_sha1_file(p).encode("utf-8"))
        else:
            h.update(b"missing")
    return h.hexdigest()


def _git_versions_digest(bench_root: Path, run_dir: Path) -> str:
    git_versions_path = run_dir / "git_versions.txt"
    if git_versions_path.exists():
        return _sha1_text(git_versions_path.read_text(encoding="utf-8"))
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(bench_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if cp.returncode == 0:
            return _sha1_text((cp.stdout or "").strip())
    except Exception:
        pass
    return _sha1_text("unknown_git")


def _env_digest(device: torch.device, precision: str, deterministic: bool) -> str:
    payload = {
        "python": sys.version.replace(os.linesep, " "),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "precision": str(precision),
        "deterministic": bool(deterministic),
    }
    return _sha1_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _resolve_model_cache_dir(bench_root: Path, runner_cfg: Dict[str, Any], model_cfg: Dict[str, Any]) -> Optional[Path]:
    raw = model_cfg.get("model_cache_dir", runner_cfg.get("model_cache_dir"))
    if raw in (None, "", False):
        return None
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = (bench_root / p)
    return p.resolve()


def _cache_entry_paths(model_cache_dir: Path, model_id: str, cache_key: str) -> Dict[str, Path]:
    entry_dir = model_cache_dir / str(model_id) / str(cache_key)
    return {
        "entry_dir": entry_dir,
        "ckpt_path": entry_dir / "model.pt",
        "train_state_path": entry_dir / "train_state.json",
        "meta_path": entry_dir / "cache_meta.json",
    }


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _update_ledger_file(ledger_path: Path, patch: Dict[str, Any]) -> Dict[str, Any]:
    obj = _read_json_if_exists(ledger_path)
    for k, v in patch.items():
        obj[k] = v
    _write_json(ledger_path, obj)
    return obj


def _normalize_train_update_accounting(
    *,
    ledger_obj: Dict[str, Any],
    adapter: Any,
    train_max_updates: int,
    train_skipped: bool,
) -> Dict[str, Any]:
    """
    Normalize training update counters before policy checks.

    Convention: train_updates_used is the backward-compatible alias for
    train_outer_updates_used. Single-level training adapters may only fill the
    alias; keep that valid while preserving budget checks.
    """
    raw_train_updates = int(ledger_obj.get("train_updates_used", getattr(adapter, "train_updates_used", 0)) or 0)
    raw_outer_updates = int(
        ledger_obj.get(
            "train_outer_updates_used",
            getattr(adapter, "train_outer_updates_used", raw_train_updates),
        )
        or 0
    )
    train_inner_updates_used = int(
        ledger_obj.get("train_inner_updates_used", getattr(adapter, "train_inner_updates_used", 0)) or 0
    )
    train_skipped_flag = bool(ledger_obj.get("train_skipped", train_skipped))

    normalized_from_alias = False
    train_outer_updates_used = int(raw_outer_updates)
    if train_outer_updates_used <= 0 and raw_train_updates > 0:
        train_outer_updates_used = int(raw_train_updates)
        normalized_from_alias = True

    ledger_obj["train_outer_updates_used"] = int(train_outer_updates_used)
    ledger_obj["train_inner_updates_used"] = int(train_inner_updates_used)
    ledger_obj["train_updates_used"] = int(train_outer_updates_used)
    ledger_obj["train_max_updates"] = int(train_max_updates)
    ledger_obj["train_skipped"] = bool(train_skipped_flag)
    if normalized_from_alias:
        ledger_obj["train_update_accounting_normalized_from_alias"] = True
    return ledger_obj


def _read_adapter_meta(adapter: Any) -> Dict[str, Any]:
    if not hasattr(adapter, "get_adapter_meta"):
        return {}
    try:
        maybe = adapter.get_adapter_meta()  # type: ignore
        if isinstance(maybe, dict):
            return maybe
    except Exception:
        return {}
    return {}


def _sanitize_extra_metric_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_extra_metric_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_extra_metric_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return _sanitize_extra_metric_value(tensor.item())
        return _sanitize_extra_metric_value(tensor.tolist())
    if isinstance(value, np.ndarray):
        return _sanitize_extra_metric_value(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    raise TypeError(f"unsupported extra metric value type: {type(value).__name__}")


def _read_adapter_extra_metrics(adapter: Any) -> Dict[str, Any]:
    if not hasattr(adapter, "get_extra_metrics"):
        return {}
    try:
        maybe = adapter.get_extra_metrics()  # type: ignore
        if not isinstance(maybe, Mapping):
            return {}
        sanitized = _sanitize_extra_metric_value(maybe)
        return dict(sanitized) if isinstance(sanitized, dict) else {}
    except Exception as exc:
        logger.warning("Ignoring invalid adapter extra metrics: %s: %s", type(exc).__name__, exc)
        return {}


def _finalize_adapter_evaluation_diagnostics(
    *,
    adapter: Any,
    run_dir: Path,
    split_extras: Optional[Mapping[str, np.ndarray]],
    x_true: np.ndarray,
    x_pred: np.ndarray,
) -> Dict[str, Any]:
    if not hasattr(adapter, "finalize_evaluation_diagnostics"):
        return {}
    maybe = adapter.finalize_evaluation_diagnostics(  # type: ignore[attr-defined]
        run_dir=run_dir,
        split_extras=split_extras,
        x_true=x_true,
        x_pred=x_pred,
    )
    if maybe is None:
        return {}
    if not isinstance(maybe, Mapping):
        raise TypeError(
            "finalize_evaluation_diagnostics must return a mapping or None"
        )
    return dict(maybe)


def _compute_train_cache_key(
    *,
    model_id: str,
    adapter_version: str,
    task_id: str,
    scenario_id: str,
    seed: int,
    train_budget: Dict[str, Any],
    model_cfg: Dict[str, Any],
    data_hash: str,
    git_digest: str,
    env_digest: str,
) -> str:
    payload = {
        "model_id": str(model_id),
        "adapter_version": str(adapter_version),
        "task_id": str(task_id),
        "scenario_id": str(scenario_id),
        "seed": int(seed),
        "train_budget": _normalize_jsonish(train_budget),
        "model_cfg": _normalize_jsonish(model_cfg),
        "data_hash": str(data_hash),
        "git_digest": str(git_digest),
        "env_digest": str(env_digest),
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]


_ALLOWED_PLANS: Tuple[Tuple[str, str], ...] = (
    ("pretrained", "frozen"),
    ("trained", "frozen"),
    ("trained", "budgeted"),
    ("untrained", "frozen"),
)


def _parse_plan_token(token: str) -> Tuple[str, str]:
    t = token.strip()
    if not t:
        raise ValueError("Empty plan token in --plans.")
    if ":" in t:
        init_id, track_id = t.split(":", 1)
    elif "," in t:
        init_id, track_id = t.split(",", 1)
    else:
        raise ValueError(f"Invalid plan token '{token}'. Use '<init_id>:<track_id>'.")
    return str(init_id).strip().lower(), str(track_id).strip().lower()


def _resolve_plans(args: argparse.Namespace, runner_cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    plan_specs: List[Tuple[str, str]] = []
    if args.plans:
        for raw in args.plans:
            for part in str(raw).split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    raise ValueError(
                        f"Invalid --plans item '{part}'. Use '<init_id>:<track_id>' entries."
                    )
                plan_specs.append(_parse_plan_token(part))
    else:
        plan_specs = [(str(args.init_id).lower(), str(args.track).lower())]

    # Deduplicate while preserving order.
    deduped: List[Tuple[str, str]] = []
    seen = set()
    for p in plan_specs:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)

    track_ids = {str(t.get("track_id")) for t in (runner_cfg.get("tracks", []) or [])}
    for init_id, track_id in deduped:
        if (init_id, track_id) not in _ALLOWED_PLANS:
            raise ValueError(
                f"Unsupported plan ({init_id},{track_id}). Allowed: {_ALLOWED_PLANS}"
            )
        if track_id not in track_ids:
            raise ValueError(f"track_id={track_id} not found in suite.runner.tracks")
        tcfg = _track_cfg(runner_cfg, track_id)
        if track_id == "budgeted" and not bool(tcfg.get("adaptation_enabled", False)):
            raise ValueError("track_id=budgeted requires adaptation_enabled=true in suite.runner.tracks")

    return deduped


def _extract_x_hat_from_pred(pred: Any) -> Any:
    x_hat = pred
    if isinstance(pred, tuple) and len(pred) >= 1:
        x_hat = pred[0]
    if isinstance(pred, dict) and "x_hat" in pred:
        x_hat = pred["x_hat"]
    return x_hat


def _validate_full_x_hat(x_hat: Any, N: int, T: int, D: int) -> torch.Tensor:
    if isinstance(x_hat, list):
        tensors = []
        for item in x_hat:
            if isinstance(item, np.ndarray):
                item = torch.from_numpy(item)
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"x_hat list item must be Tensor/ndarray, got {type(item)}")
            tensors.append(item.detach().cpu())
        if not tensors:
            raise ValueError("x_hat list is empty.")
        x_hat = torch.cat(tensors, dim=0)
    elif isinstance(x_hat, np.ndarray):
        x_hat = torch.from_numpy(x_hat)

    if not isinstance(x_hat, torch.Tensor):
        raise TypeError(f"x_hat must be Tensor/ndarray/list, got {type(x_hat)}")

    x_hat = x_hat.detach().cpu().float()
    try:
        validate_exact_layout(
            x_hat,
            expected=(int(N), int(T), int(D)),
            axis_names=("B", "T", "D"),
            label="x_hat",
        )
    except ValueError as exc:
        logger.error("Adapter eval output contract violation: %s", exc)
        raise
    return x_hat.contiguous()


def _load_split_npz(npz_path: Path) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as z:
        x = z["x"].astype(np.float32, copy=False)
        y = z["y"].astype(np.float32, copy=False)
        u = z["u"].astype(np.float32, copy=False) if "u" in z.files else None
        F = z["F"] if "F" in z.files else None
        H = z["H"] if "H" in z.files else None
        time_s = np.array(z["time_s"], copy=True) if "time_s" in z.files else None
        trajectory_id = (
            np.array(z["trajectory_id"], copy=True)
            if "trajectory_id" in z.files
            else None
        )
        extras: Dict[str, np.ndarray] = {}
        core_keys = {"x", "y", "u", "F", "H", "time_s", "trajectory_id", "meta_json"}
        n = int(x.shape[0])
        for key in z.files:
            if key in core_keys:
                continue
            arr = z[key]
            if getattr(arr, "ndim", 0) >= 1 and int(arr.shape[0]) == n and np.issubdtype(arr.dtype, np.number):
                extras[str(key)] = arr.astype(np.float32, copy=False)
        meta = None
        if "meta_json" in z.files:
            try:
                meta = json.loads(z["meta_json"].item())
            except Exception:
                meta = None
    return {
        "x": x,
        "y": y,
        "u": u,
        "F": F,
        "H": H,
        "time_s": time_s,
        "trajectory_id": trajectory_id,
        "meta": meta,
        "extras": extras,
    }


class _SeqDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: Optional[np.ndarray] = None,
        extras: Optional[Dict[str, np.ndarray]] = None,
    ):
        self._x = x
        self._y = y
        self._u = u
        self._extras = dict(extras or {})

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(self._x[idx])
        y = torch.from_numpy(self._y[idx])
        sample: Dict[str, torch.Tensor] = {"x": x, "y": y}
        if self._u is not None:
            sample["u"] = torch.from_numpy(self._u[idx])
        for key, arr in self._extras.items():
            sample[key] = torch.as_tensor(arr[idx])
        return sample


def _make_loader(
    *,
    x: np.ndarray,
    y: np.ndarray,
    u: Optional[np.ndarray],
    extras: Optional[Dict[str, np.ndarray]] = None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    ds = _SeqDataset(x=x, y=y, u=u, extras=extras)
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        generator=g if shuffle else None,
    )


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def _predict_batches(
    *,
    adapter: Any,
    y_full: np.ndarray,
    x_dim: int,
    T: int,
    eval_bs: int,
    device: torch.device,
    context: Dict[str, Any],
) -> Tuple[torch.Tensor, List[float]]:
    N = int(y_full.shape[0])
    num_batches = (N + eval_bs - 1) // eval_bs
    out_batches: List[torch.Tensor] = []
    batch_times_ms: List[float] = []
    debug_every = int(context.get("debug_every", 0) or 0)

    with torch.no_grad():
        for bi in range(num_batches):
            s = bi * eval_bs
            e = min(N, (bi + 1) * eval_bs)
            y_b = _tensorize(y_full[s:e], device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = adapter.predict(y_b, context=context, return_cov=False)  # type: ignore
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            batch_times_ms.append((t1 - t0) * 1000.0)

            x_hat = _extract_x_hat_from_pred(pred)
            x_hat = _validate_batch_x_hat(x_hat, B=(e - s), T=T, D=x_dim)
            if debug_every > 0 and ((bi % debug_every) == 0 or bi == (num_batches - 1)):
                logger.debug(
                    "predict batch=%s/%s %s",
                    bi + 1,
                    num_batches,
                    format_array_stats("x_hat", x_hat),
                )
            out_batches.append(x_hat.detach().cpu())

    if not out_batches:
        raise RuntimeError("runtime_error: no predictions produced.")
    return torch.cat(out_batches, dim=0).contiguous(), batch_times_ms


def _try_call_setup(adapter: Any, model_cfg: Dict[str, Any], system_info: Dict[str, Any], run_ctx: Dict[str, Any]) -> None:
    observer = active_observer()
    observer.status("PHASE_START", phase="setup", message=f"adapter setup: {type(adapter).__name__}")
    try:
        try:
            adapter.setup(model_cfg, system_info, run_ctx)  # type: ignore
            return
        except TypeError:
            pass
        try:
            adapter.setup(model_cfg, system_info)  # type: ignore
            return
        except TypeError:
            adapter.setup(model_cfg)  # type: ignore
    finally:
        observer.status("PHASE_END", phase="setup")


def _try_call_train(
    adapter: Any,
    train_loader: Any,
    val_loader: Any,
    budget: Dict[str, Any],
    ckpt_dir: Path,
) -> Any:
    observer = active_observer()
    observer.status(
        "PHASE_START",
        phase="train",
        message=f"adapter train: {type(adapter).__name__} budget={budget.get('train_max_updates')}",
    )
    try:
        try:
            return adapter.train(train_loader, val_loader, budget=budget, ckpt_dir=ckpt_dir)  # type: ignore
        except TypeError:
            return adapter.train(train_loader, val_loader)  # type: ignore
    finally:
        observer.status("PHASE_END", phase="train")


def _try_call_eval(
    adapter: Any,
    test_loader: Any,
    ckpt_path: Optional[Path],
    track_cfg: Dict[str, Any],
) -> Any:
    observer = active_observer()
    observer.status("PHASE_START", phase="test", message=f"adapter eval: {type(adapter).__name__}")
    try:
        if ckpt_path is not None:
            try:
                return adapter.eval(test_loader, ckpt_path=ckpt_path, track_cfg=track_cfg)  # type: ignore
            except TypeError:
                pass
            try:
                return adapter.eval(test_loader, str(ckpt_path), track_cfg)  # type: ignore
            except TypeError:
                pass
        try:
            return adapter.eval(test_loader, ckpt_path=None, track_cfg=track_cfg)  # type: ignore
        except TypeError:
            return adapter.eval(test_loader)  # type: ignore
    finally:
        observer.status("PHASE_END", phase="test")


def _adapter_supports_viz_diagnostics(adapter: Any) -> bool:
    supports = getattr(adapter, "supports_viz_diagnostics", None)
    if callable(supports):
        return bool(supports())
    return callable(getattr(adapter, "set_viz_diagnostics_enabled", None))


def _collect_viz_diagnostics(
    *,
    adapter: Any,
    test_loader: Any,
    ckpt_path: Optional[Path],
    track_cfg: Dict[str, Any],
    x_hat_np: np.ndarray,
    N: int,
    T: int,
    D: int,
) -> Dict[str, Any]:
    if not _adapter_supports_viz_diagnostics(adapter):
        return {}
    set_enabled = getattr(adapter, "set_viz_diagnostics_enabled", None)
    if not callable(set_enabled):
        raise RuntimeError("adapter reports visualization diagnostics support but has no set_viz_diagnostics_enabled hook")
    try:
        set_enabled(True)
        diag_res = _try_call_eval(adapter, test_loader, ckpt_path, track_cfg)
        if not isinstance(diag_res, Mapping) or "x_hat" not in diag_res:
            raise RuntimeError("viz_diagnostics_eval_missing_x_hat")
        diag_x_hat = _validate_full_x_hat(diag_res["x_hat"], N=N, T=T, D=D)
        diag_x_np = diag_x_hat.detach().cpu().numpy()
        if not np.array_equal(diag_x_np, x_hat_np):
            max_abs = float(np.max(np.abs(diag_x_np - x_hat_np)))
            raise RuntimeError(
                "viz_diagnostics_changed_predictions: "
                f"max_abs_diff={max_abs}"
            )
        maybe_diag = diag_res.get("diagnostics")
        if isinstance(maybe_diag, Mapping):
            return dict(maybe_diag)
        return {}
    finally:
        set_enabled(False)


def _try_call_adapt(
    adapter: Any,
    stream_or_loader: Any,
    budget: Dict[str, Any],
    t0: Optional[int],
    allowed_after_t0_only: bool,
    context: Dict[str, Any],
) -> Any:
    try:
        return adapter.adapt(  # type: ignore
            stream_or_loader,
            budget=budget,
            t0=t0,
            allowed_after_t0_only=allowed_after_t0_only,
            context=context,
        )
    except TypeError:
        pass
    try:
        return adapter.adapt(stream_or_loader, budget=budget, context=context)  # type: ignore
    except TypeError:
        pass
    try:
        return adapter.adapt(stream_or_loader, budget)  # type: ignore
    except TypeError:
        return adapter.adapt(stream_or_loader)  # type: ignore


def _normalize_adapt_updates_per_step(raw: Any) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = list(enumerate(raw))
    else:
        return out

    for k, v in items:
        try:
            t_idx = int(k)
            count = int(v)
        except Exception:
            continue
        out[t_idx] = count
    return out


def _write_run_manifest(
    *,
    bench_root: Path,
    suite_name: str,
    suite_yaml: Path,
    run_dirs: Sequence[str],
) -> Path:
    manifests_dir = bench_root / "runs" / str(suite_name) / "_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    manifest_path = manifests_dir / f"{ts}_{uuid.uuid4().hex[:8]}.json"
    payload = {
        "suite_name": str(suite_name),
        "suite_yaml": str(suite_yaml),
        "created_at_unix": float(time.time()),
        "run_count": int(len(run_dirs)),
        "run_dirs": [str(Path(p).expanduser().resolve()) for p in run_dirs],
    }
    _write_json(manifest_path, payload)
    return manifest_path


# P1A-CP4 typed-event replay is an exact-pair branch.  None of these helpers
# accepts or creates the legacy dense float32 observation sequence.
P1A_MEKF_TASK_FAMILY = "mekf_unit_st_v1"
P1A_MEKF_MODEL_ID = "mekf_event_replay_v1"
P1A_MEKF_ARTIFACT_VERSION = "p1a-cp4-mekf-replay-artifact-v1"
P1A_MEKF_METRIC_CONTRACT = "p1a-canonical-mekf-metrics-v1"


def _is_p1a_mekf_event_replay_pair(task: Mapping[str, Any], model: Mapping[str, Any]) -> bool:
    return (
        str(task.get("task_family", "")) == P1A_MEKF_TASK_FAMILY
        and str(model.get("model_id", "")) == P1A_MEKF_MODEL_ID
    )


def _p1a_mekf_canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("MEKF runner metadata must be finite canonical-JSON data") from error


def _p1a_mekf_filter_configuration(task: Mapping[str, Any]) -> tuple[Any, float, np.ndarray, Dict[str, Any]]:
    from bench.estimators.mekf import MEKFState

    raw = task.get("mekf_replay")
    if not isinstance(raw, Mapping):
        raise ValueError("mekf_unit_st_v1 requires a mekf_replay mapping")
    unexpected = set(raw) - {
        "initial_time_s",
        "initial_state",
        "process_noise",
        "evaluation_split",
        "metric_confidence_level",
    }
    if unexpected:
        raise ValueError(f"unexpected mekf_replay keys: {sorted(unexpected)}")
    initial = raw.get("initial_state")
    process = raw.get("process_noise")
    if not isinstance(initial, Mapping) or not isinstance(process, Mapping):
        raise ValueError("initial_state and process_noise must be mappings")
    if set(initial) != {"q_NB", "b_g_rad_s", "P_diagonal"}:
        raise ValueError("initial_state must contain exactly q_NB, b_g_rad_s, P_diagonal")
    if set(process) != {"Q_c_diagonal"}:
        raise ValueError("process_noise must contain exactly Q_c_diagonal")

    q = np.asarray(initial["q_NB"], dtype=np.float64)
    bias = np.asarray(initial["b_g_rad_s"], dtype=np.float64)
    p_diagonal = np.asarray(initial["P_diagonal"], dtype=np.float64)
    q_diagonal = np.asarray(process["Q_c_diagonal"], dtype=np.float64)
    if q.shape != (4,) or bias.shape != (3,):
        raise ValueError("initial q_NB/b_g_rad_s must have shapes [4]/[3]")
    if p_diagonal.shape != (6,) or not np.all(np.isfinite(p_diagonal)) or np.any(p_diagonal <= 0.0):
        raise ValueError("P_diagonal must contain six finite positive values")
    if q_diagonal.shape != (6,) or not np.all(np.isfinite(q_diagonal)) or np.any(q_diagonal < 0.0):
        raise ValueError("Q_c_diagonal must contain six finite nonnegative values")
    initial_time_s = float(raw.get("initial_time_s", 0.0))
    if not np.isfinite(initial_time_s) or initial_time_s < 0.0:
        raise ValueError("initial_time_s must be finite and nonnegative")
    evaluation_split = str(raw.get("evaluation_split", "test"))
    if evaluation_split not in {"train", "val", "test"}:
        raise ValueError("evaluation_split must be train, val, or test")
    confidence = float(raw.get("metric_confidence_level", 0.95))
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("metric_confidence_level must be strictly between zero and one")

    P = np.diag(p_diagonal).astype(np.float64, copy=False)
    Q_c = np.diag(q_diagonal).astype(np.float64, copy=False)
    state = MEKFState(q_NB=q, b_g=bias, P=P)
    resolved = {
        "evaluation_split": evaluation_split,
        "initial_state": {
            "P": P.tolist(),
            "b_g_rad_s": bias.tolist(),
            "q_NB": q.tolist(),
        },
        "initial_time_s": initial_time_s,
        "metric_confidence_level": confidence,
        "process_noise": {
            "Q_c": Q_c.tolist(),
            "units": ["rad^2/s", "rad^2/s", "rad^2/s", "rad^2/s^3", "rad^2/s^3", "rad^2/s^3"],
        },
        "state_convention": "active scalar-first Hamilton q_NB; right-local error",
        "bias_units": "rad/s",
        "time_units": "s",
    }
    return state, initial_time_s, Q_c, resolved


def _p1a_exact_truth_join(truth: Any, artifact: Any) -> tuple[np.ndarray, np.ndarray]:
    """Join only after estimation by exact trajectory_id and float64 timestamp."""

    trajectory_rows = np.flatnonzero(truth.trajectory_id == np.int64(artifact.trajectory_id))
    if trajectory_rows.size != 1:
        raise ValueError("truth join requires exactly one matching trajectory_id")
    trajectory_index = int(trajectory_rows[0])
    start = int(truth.truth_offsets[trajectory_index])
    stop = int(truth.truth_offsets[trajectory_index + 1])
    truth_time = truth.truth_time_s[start:stop]
    locations = np.searchsorted(truth_time, artifact.timestamp_s)
    if np.any(locations >= truth_time.size):
        raise ValueError("exact truth join timestamp is outside the trajectory")
    if not np.array_equal(truth_time[locations], artifact.timestamp_s):
        raise ValueError("exact truth join timestamp mismatch; interpolation is forbidden")
    return truth.q_true_NB[start:stop][locations], truth.gyro_bias_rad_s[start:stop][locations]


def _p1a_consistency_dict(summary: Any) -> Dict[str, Any]:
    return {
        "count": int(summary.count),
        "dof_per_sample": int(summary.dof_per_sample),
        "sum": float(summary.sum),
        "mean": float(summary.mean),
        "normalized_mean": float(summary.normalized_mean),
        "confidence_level": float(summary.confidence_level),
        "chi_square_sum_lower": float(summary.chi_square_sum_lower),
        "chi_square_sum_upper": float(summary.chi_square_sum_upper),
    }


def _p1a_spd_dict(diagnostics: Any) -> Dict[str, Any]:
    return {
        "count": int(diagnostics.minimum_eigenvalue.size),
        "dimension": int(diagnostics.dimension),
        "all_cholesky_succeeded": bool(np.all(diagnostics.cholesky_succeeded)),
        "minimum_eigenvalue": float(np.min(diagnostics.minimum_eigenvalue)),
        "maximum_relative_asymmetry": float(np.max(diagnostics.relative_asymmetry)),
    }


def _p1a_mekf_metrics(
    *,
    artifacts: Sequence[Any],
    truth: Any,
    confidence_level: float,
) -> Dict[str, Any]:
    from bench.metrics.mekf import (
        attitude_geodesic_error_deg,
        attitude_geodesic_error_rad,
        bias_error_summary,
        consistency_summary,
        right_local_nees,
        spd_diagnostics,
        star_tracker_nis,
    )

    q_hat = np.concatenate([artifact.q_hat_NB for artifact in artifacts], axis=0)
    b_hat = np.concatenate([artifact.b_hat_rad_s for artifact in artifacts], axis=0)
    P = np.concatenate([artifact.P for artifact in artifacts], axis=0)
    time_s = np.concatenate([artifact.timestamp_s for artifact in artifacts])
    trajectory_id = np.concatenate(
        [
            np.full(artifact.timestamp_s.shape, artifact.trajectory_id, dtype=np.int64)
            for artifact in artifacts
        ]
    )
    joined = [_p1a_exact_truth_join(truth, artifact) for artifact in artifacts]
    q_true = np.concatenate([item[0] for item in joined], axis=0)
    b_true = np.concatenate([item[1] for item in joined], axis=0)

    attitude_rad = attitude_geodesic_error_rad(q_hat, q_true)
    attitude_deg = attitude_geodesic_error_deg(q_hat, q_true)
    bias = bias_error_summary(b_hat, b_true)
    nees = right_local_nees(
        q_hat,
        b_hat,
        P,
        q_true,
        b_true,
        estimate_time_s=time_s,
        covariance_time_s=time_s,
        truth_time_s=time_s,
        estimate_trajectory_id=trajectory_id,
        covariance_trajectory_id=trajectory_id,
        truth_trajectory_id=trajectory_id,
    )

    residual = np.concatenate([artifact.st_residual for artifact in artifacts], axis=0)
    S = np.concatenate([artifact.st_S for artifact in artifacts], axis=0)
    st_time = np.concatenate([artifact.st_timestamp_s for artifact in artifacts])
    st_trajectory_id = np.concatenate(
        [
            np.full(artifact.st_timestamp_s.shape, artifact.trajectory_id, dtype=np.int64)
            for artifact in artifacts
        ]
    )
    nis = star_tracker_nis(
        residual,
        S,
        residual_time_s=st_time,
        covariance_time_s=st_time,
        residual_trajectory_id=st_trajectory_id,
        covariance_trajectory_id=st_trajectory_id,
    )
    nis_summary = consistency_summary(
        nis, dof_per_sample=3, confidence_level=confidence_level
    )
    nees_summary = consistency_summary(
        nees, dof_per_sample=6, confidence_level=confidence_level
    )
    p_diagnostics = spd_diagnostics(P, name="P")
    s_diagnostics = spd_diagnostics(S, name="S")

    return {
        "metric_contract": P1A_MEKF_METRIC_CONTRACT,
        "attitude": {
            "count": int(attitude_rad.size),
            "rmse_rad": float(np.sqrt(np.mean(attitude_rad * attitude_rad))),
            "rmse_deg": float(np.sqrt(np.mean(attitude_deg * attitude_deg))),
            "p95_rad": float(np.percentile(attitude_rad, 95.0)),
            "p95_deg": float(np.percentile(attitude_deg, 95.0)),
            "max_rad": float(np.max(attitude_rad)),
            "max_deg": float(np.max(attitude_deg)),
        },
        "bias": {
            "per_axis_rmse_rad_s": bias.per_axis_rmse_rad_s.tolist(),
            "vector_rmse_rad_s": float(bias.vector_rmse_rad_s),
        },
        "nis": _p1a_consistency_dict(nis_summary),
        "nees": _p1a_consistency_dict(nees_summary),
        "spd": {
            "P": _p1a_spd_dict(p_diagnostics),
            "S": _p1a_spd_dict(s_diagnostics),
        },
    }


def _p1a_write_mekf_replay_artifacts(
    *,
    run_dir: Path,
    artifacts: Sequence[Any],
    dataset_artifacts: Any,
    estimator_config: Mapping[str, Any],
    evaluation_split: str,
) -> Dict[str, Any]:
    artifact_parent = run_dir / "artifacts"
    artifact_parent.mkdir(parents=True, exist_ok=True)
    final_dir = artifact_parent / "mekf_replay"
    pending_dir = artifact_parent / f".mekf_replay.partial.{uuid.uuid4().hex}"
    backup_dir = artifact_parent / f".mekf_replay.previous.{uuid.uuid4().hex}"
    pending_dir.mkdir(parents=False, exist_ok=False)
    trajectory_records: List[Dict[str, Any]] = []
    expected_fields = {
        "event_index",
        "event_order",
        "timestamp_s",
        "sensor_code",
        "q_hat_NB",
        "b_hat_rad_s",
        "P",
        "st_event_index",
        "st_event_order",
        "st_timestamp_s",
        "st_residual",
        "st_S",
    }
    try:
        for artifact in artifacts:
            filename = f"trajectory_{artifact.trajectory_id}.npz"
            arrays = {name: getattr(artifact, name) for name in sorted(expected_fields)}
            np.savez(pending_dir / filename, **arrays)
            with np.load(pending_dir / filename, allow_pickle=False) as archive:
                if set(archive.files) != expected_fields:
                    raise ValueError("runner trajectory artifact fields mismatch")
                for name, expected in arrays.items():
                    observed = archive[name]
                    if observed.dtype.hasobject or not np.array_equal(observed, expected):
                        raise ValueError(f"runner artifact round trip mismatch for {name}")
            trajectory_records.append(
                {
                    "filename": filename,
                    "trajectory_id": int(artifact.trajectory_id),
                    "processed_event_count": int(artifact.processed_event_count),
                    "gyro_event_count": int(artifact.gyro_event_count),
                    "star_tracker_update_count": int(artifact.star_tracker_update_count),
                }
            )

        estimator_config_hash = hashlib.sha256(
            _p1a_mekf_canonical_json_bytes(dict(estimator_config))
        ).hexdigest()
        first = artifacts[0]
        manifest = {
            "artifact_contract_version": P1A_MEKF_ARTIFACT_VERSION,
            "task_family": P1A_MEKF_TASK_FAMILY,
            "model_id": P1A_MEKF_MODEL_ID,
            "adapter_id": first.provenance.adapter_id,
            "adapter_version": first.provenance.adapter_version,
            "dataset_identity": first.provenance.dataset_identity.as_dict(),
            "dataset_config_hash": dataset_artifacts.dataset_config_hash,
            "producer_id": dataset_artifacts.producer_id,
            "cache_state": dataset_artifacts.cache_state,
            "cache_directory": str(dataset_artifacts.dataset_dir),
            "evaluation_split": evaluation_split,
            "trajectory_ids": [int(artifact.trajectory_id) for artifact in artifacts],
            "processed_event_count": int(sum(item.processed_event_count for item in artifacts)),
            "gyro_event_count": int(sum(item.gyro_event_count for item in artifacts)),
            "star_tracker_update_count": int(
                sum(item.star_tracker_update_count for item in artifacts)
            ),
            "trajectory_files": trajectory_records,
            "estimator_config": dict(estimator_config),
            "estimator_config_hash": estimator_config_hash,
            "metric_contract": P1A_MEKF_METRIC_CONTRACT,
            "truth_in_trajectory_npz": False,
        }
        (pending_dir / "manifest.json").write_bytes(
            _p1a_mekf_canonical_json_bytes(manifest)
        )

        if final_dir.exists():
            os.replace(final_dir, backup_dir)
        try:
            os.replace(pending_dir, final_dir)
        except Exception:
            if backup_dir.exists() and not final_dir.exists():
                os.replace(backup_dir, final_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        return manifest
    finally:
        if pending_dir.exists():
            shutil.rmtree(pending_dir)


def _run_p1a_mekf_event_replay(
    *,
    suite: Mapping[str, Any],
    task: Mapping[str, Any],
    model: Mapping[str, Any],
    scenario_settings: Mapping[str, Any],
    seed: int,
    track_id: str,
    init_id: str,
    run_dir: Path,
    cache_root: Path,
    scenario_id: str,
) -> Dict[str, Any]:
    suite_name = str(suite["suite"]["name"])
    task_id = str(task["task_id"])
    model_id = str(model["model_id"])
    try:
        if not _is_p1a_mekf_event_replay_pair(task, model):
            raise ValueError(
                "mekf_unit_st_v1 and mekf_event_replay_v1 must be selected as an exact pair"
            )
        if track_id != "frozen" or init_id != "untrained":
            raise ValueError("mekf_event_replay_v1 disables training and adaptation; use untrained:frozen")
        initial_state, initial_time_s, Q_c, estimator_config = (
            _p1a_mekf_filter_configuration(task)
        )
        _write_yaml(
            run_dir / "config_snapshot.yaml",
            {
                "suite": suite.get("suite", {}),
                "task": dict(task),
                "model": dict(model),
                "scenario_settings": dict(scenario_settings),
                "scenario_id": scenario_id,
                "seed": int(seed),
                "track_id": track_id,
                "init_id": init_id,
                "typed_event_cache_root": str(cache_root),
                "resolved_estimator_config": estimator_config,
            },
        )
        _write_json(
            run_dir / "run_plan.json",
            {
                "plan_id": f"{init_id}__{track_id}",
                "task_family": P1A_MEKF_TASK_FAMILY,
                "model_id": P1A_MEKF_MODEL_ID,
                "training_enabled": False,
                "adaptation_enabled": False,
                "truth_available_to_estimator": False,
            },
        )
        _write_json(
            run_dir / "budget_ledger.json",
            {
                "train_updates_used": 0,
                "adapt_updates_used": 0,
                "train_skipped": True,
                "track_id": track_id,
                "init_id": init_id,
            },
        )

        from bench.estimators.mekf import MEKFState
        from bench.models.mekf import DatasetIdentity
        from bench.models.registry import get_typed_event_bridge_class
        from bench.tasks.bench_generated import prepare_mekf_unit_st_v1

        prepared = prepare_mekf_unit_st_v1(
            suite_name=suite_name,
            task_cfg=task,
            seed=int(seed),
            cache_root=cache_root,
            scenario_overrides=scenario_settings,
        )
        identity = DatasetIdentity.from_verified(prepared.manifest, prepared.semantic_hashes)
        bridge_class = get_typed_event_bridge_class(model_id)
        bridge = bridge_class(expected_dataset_identity=identity)
        evaluation_split = str(estimator_config["evaluation_split"])
        trajectory_ids = getattr(prepared.trajectory_split, f"{evaluation_split}_ids")
        if trajectory_ids.size == 0:
            raise ValueError("selected whole-trajectory split is empty")

        replay_start = time.perf_counter()
        replay_artifacts = []
        for trajectory_id in trajectory_ids:
            per_trajectory_state = MEKFState(
                q_NB=initial_state.q_NB,
                b_g=initial_state.b_g,
                P=initial_state.P,
            )
            replay_artifacts.append(
                bridge.replay_events(
                    prepared.dataset.events,
                    int(trajectory_id),
                    per_trajectory_state,
                    initial_time_s,
                    Q_c,
                    identity,
                )
            )
        replay_elapsed_s = time.perf_counter() - replay_start

        # Truth first becomes reachable here, after every estimator replay completed.
        canonical_metrics = _p1a_mekf_metrics(
            artifacts=replay_artifacts,
            truth=prepared.dataset.truth,
            confidence_level=float(estimator_config["metric_confidence_level"]),
        )
        artifact_manifest = _p1a_write_mekf_replay_artifacts(
            run_dir=run_dir,
            artifacts=replay_artifacts,
            dataset_artifacts=prepared,
            estimator_config=estimator_config,
            evaluation_split=evaluation_split,
        )
        attitude = canonical_metrics["attitude"]
        mse_value = float(attitude["rmse_rad"]) ** 2
        metrics_obj = {
            "status": "ok",
            "suite": suite_name,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "seed": int(seed),
            "model_id": model_id,
            "track_id": track_id,
            "init_id": init_id,
            "cache_state": prepared.cache_state,
            "dataset_identity": identity.as_dict(),
            "canonical_mekf": canonical_metrics,
            "artifact_manifest": "artifacts/mekf_replay/manifest.json",
        }
        _write_json(run_dir / "metrics.json", metrics_obj)
        stale_failure = run_dir / "failure.json"
        if stale_failure.exists():
            stale_failure.unlink()
        processed_count = int(artifact_manifest["processed_event_count"])
        logger.info(
            "P1A-CP4 typed replay complete producer=%s cache_state=%s dataset_hash=%s trajectories=%s",
            prepared.producer_id,
            prepared.cache_state,
            identity.dataset_hash,
            len(replay_artifacts),
        )
        clear_logging_context()
        return {
            "status": "ok",
            "run_dir": str(run_dir),
            "suite": suite_name,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "seed": int(seed),
            "model_id": model_id,
            "track_id": track_id,
            "init_id": init_id,
            "mse": mse_value,
            "rmse": float(attitude["rmse_rad"]),
            "mse_db": float(10.0 * np.log10(mse_value)),
            "timing_ms_per_step": float(1000.0 * replay_elapsed_s / processed_count),
            "recovery_k": None,
            "cache_state": prepared.cache_state,
            "dataset_hash": identity.dataset_hash,
        }
    except Exception as error:
        failure = {
            "status": "failed",
            "failure_type": _classify_failure(error),
            "phase": "p1a_cp4_typed_event_replay",
            "failure_stage": "p1a_cp4_typed_event_replay",
            "message": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "context": {
                "suite_name": suite_name,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "seed": int(seed),
                "model_id": model_id,
                "track_id": track_id,
                "init_id": init_id,
            },
        }
        _write_json(run_dir / "failure.json", failure)
        clear_logging_context()
        return {
            "status": "failed",
            "run_dir": str(run_dir),
            "suite": suite_name,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "seed": int(seed),
            "model_id": model_id,
            "track_id": track_id,
            "init_id": init_id,
            "failure_type": failure["failure_type"],
            "error": failure["message"],
        }


def run_one(
    suite: Dict[str, Any],
    task: Dict[str, Any],
    model: Dict[str, Any],
    scenario_settings: Dict[str, Any],
    seed: int,
    track_id: str,
    device_str: str,
    precision: str,
    init_id: str = "untrained",
    plan_isolation: bool = False,
    log_level: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    log_file: Optional[str] = None,
    debug_every: Optional[int] = None,
    emit_viz_artifacts: Optional[bool] = None,
) -> Dict[str, Any]:
    bench_root = _bench_root()
    cache_root = _cache_root(bench_root)
    runner_cfg = suite.get("runner", {}) or {}
    log_cfg = _runner_logging_cfg(
        runner_cfg,
        log_level=log_level,
        log_to_file=log_to_file,
        log_file=log_file,
        debug_every=debug_every,
    )

    suite_name = suite["suite"]["name"]
    task_id = task["task_id"]
    model_id = model["model_id"]

    scenario_cfg_basis = _build_scenario_cfg_basis(task, scenario_settings)
    scenario_id = _canonicalize_scenario_id(task_id, scenario_cfg_basis)
    cache_scenario_id = str(scenario_id)

    train_path = _npz_path(cache_root, suite_name, task_id, cache_scenario_id, int(seed), "train")
    val_path = _npz_path(cache_root, suite_name, task_id, cache_scenario_id, int(seed), "val")
    test_path = _npz_path(cache_root, suite_name, task_id, cache_scenario_id, int(seed), "test")
    resolved_from_cache = False
    if not test_path.exists():
        alt = _resolve_scenario_id_from_cache(cache_root, suite_name, task_id, int(seed), scenario_settings)
        if alt is not None:
            cache_scenario_id = str(alt)
            train_path = _npz_path(cache_root, suite_name, task_id, cache_scenario_id, int(seed), "train")
            val_path = _npz_path(cache_root, suite_name, task_id, cache_scenario_id, int(seed), "val")
            test_path = _npz_path(cache_root, suite_name, task_id, cache_scenario_id, int(seed), "test")
            resolved_from_cache = True

    out_tmpl = suite.get("reporting", {}).get(
        "output_dir_template",
        "runs/{suite.name}/{task_id}/{model_id}/{track_id}/seed_{seed}/scenario_{scenario_id}",
    )
    run_dir_rel = (
        out_tmpl
        .replace("{suite.name}", str(suite_name))
        .replace("{task_id}", str(task_id))
        .replace("{model_id}", str(model_id))
        .replace("{track_id}", str(track_id))
        .replace("{seed}", str(seed))
        .replace("{scenario_id}", str(scenario_id))
    )
    if plan_isolation:
        run_dir_rel = str(Path(run_dir_rel) / f"init_{init_id}")
    run_dir = (bench_root / run_dir_rel).resolve()
    _ensure_dir(run_dir)
    _ensure_dir(run_dir / "checkpoints")
    _ensure_dir(run_dir / "artifacts")
    configure_logging(
        log_cfg["level"],
        run_dir=run_dir,
        log_to_file=bool(log_cfg["log_to_file"]),
        log_file=(Path(str(log_cfg["log_file"])) if log_cfg["log_file"] else None),
    )
    set_logging_context(
        task_id=str(task_id),
        scenario_id=str(scenario_id),
        model_id=str(model_id),
        track_id=str(track_id),
        init_id=str(init_id),
        seed=int(seed),
    )

    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    io_mod, seed_mod, _ = _try_import_utils()

    def log_out(msg: str) -> None:
        with stdout_log.open("a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")

    def log_err(msg: str) -> None:
        with stderr_log.open("a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")

    deterministic = bool(runner_cfg.get("deterministic", True))
    logger.info(
        "Starting run device=%s precision=%s deterministic=%s resolved_from_cache=%s",
        device_str,
        precision,
        deterministic,
        resolved_from_cache,
    )
    if seed_mod and hasattr(seed_mod, "set_seed"):
        seed_mod.set_seed(int(seed), deterministic=deterministic)  # type: ignore
    else:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

    device = _resolve_device(device_str)
    if device.type == "cpu" and device_str.startswith("cuda"):
        log_out(f"[WARN] requested device={device_str} but cuda not available -> using cpu")
        logger.warning("Requested device=%s but CUDA is unavailable; falling back to cpu", device_str)
    if deterministic and device.type == "cuda":
        if not str(os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")).strip():
            # Required by CUDA deterministic mode for cuBLAS-backed ops (e.g., batched matmul/bmm).
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            log_out("[INFO] set CUBLAS_WORKSPACE_CONFIG=:4096:8 (deterministic CUDA)")
            logger.info("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic CUDA")

    if (
        str(task.get("task_family", "")) == P1A_MEKF_TASK_FAMILY
        or str(model.get("model_id", "")) == P1A_MEKF_MODEL_ID
    ):
        return _run_p1a_mekf_event_replay(
            suite=suite,
            task=task,
            model=model,
            scenario_settings=scenario_settings,
            seed=int(seed),
            track_id=str(track_id),
            init_id=str(init_id),
            run_dir=run_dir,
            cache_root=cache_root,
            scenario_id=str(scenario_id),
        )

    # Early missing_data
    if not test_path.exists():
        data_mode = str(runner_cfg.get("data_mode", "bench_generated")).strip().lower()
        if data_mode == "replay_generated":
            try:
                from bench.tasks.bench_generated import prepare_bench_generated_v0  # lazy import

                replay_arts = prepare_bench_generated_v0(
                    suite_name=suite_name,
                    task_cfg=task,
                    seed=int(seed),
                    cache_root=cache_root,
                    scenario_overrides=scenario_settings,
                )
                if not replay_arts:
                    raise RuntimeError(
                        "replay_generated bridge produced no dataset artifacts"
                    )
                selected_art = None
                for art in replay_arts:
                    if str(art.scenario_id) == str(cache_scenario_id):
                        selected_art = art
                        break
                if selected_art is None:
                    selected_art = replay_arts[0]
                train_path = Path(selected_art.train.path)
                val_path = Path(selected_art.val.path)
                test_path = Path(selected_art.test.path)
                cache_scenario_id = str(selected_art.scenario_id)
                resolved_from_cache = True
                logger.info(
                    "Generated replay_generated cache suite=%s task=%s scenario=%s",
                    suite_name,
                    task_id,
                    cache_scenario_id,
                )
            except Exception as exc:
                logger.error("replay_generated cache generation failed: %s", exc)
                raise

        if test_path.exists():
            logger.info("replay_generated data cache available at %s", test_path)
        else:
            context = {
                "suite_name": suite_name,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "cache_scenario_id": cache_scenario_id,
                "seed": int(seed),
                "model_id": model_id,
                "track_id": track_id,
                "init_id": str(init_id),
                "cache_root": str(cache_root),
                "scenario_settings": {k: _to_jsonable(v) for k, v in (scenario_settings or {}).items()},
                "scenario_cfg_basis": scenario_cfg_basis,
                "scenario_id_method": "bench_generated.canonicalize_scenario_id(task_id, scenario_cfg_basis) (+ fallback scan)",
                "resolved_from_cache": bool(resolved_from_cache),
            }
            failure = {
                "status": "missing_data",
                "failure_type": "io_error",
                "phase": "data_loading",
                "failure_stage": "data_loading",
                "message": f"missing_data: expected test split at {test_path}",
                "context": context,
                "missing_path": str(test_path),
                "hint": "Run bench.tasks.smoke_data/bench_generated for this suite/task/scenario/seed.",
            }
            _write_json(run_dir / "failure.json", failure)
            log_err(json.dumps(failure, indent=2, ensure_ascii=False))
            logger.error("Missing test split: %s", test_path)
            clear_logging_context()
            return {
                "status": "missing_data",
                "run_dir": str(run_dir),
                "suite": suite_name,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "seed": seed,
                "model_id": model_id,
                "track_id": track_id,
                "init_id": str(init_id),
                "failure_type": "io_error",
                "error": "missing_data",
            }

    track_cfg = _track_cfg(runner_cfg, track_id)
    budget_cfg = dict(runner_cfg.get("budget", {}) or {})
    # Budget policy: train/update caps are managed at suite-level runner config.
    train_max_updates = int(budget_cfg.get("train_max_updates", 0))
    eval_bs = int(model.get("eval_batch_size", budget_cfg.get("eval_batch_size", 32)))
    train_bs = int(model.get("batch_size", budget_cfg.get("train_batch_size", eval_bs)))
    adaptation_enabled = bool(track_cfg.get("adaptation_enabled", False))
    adaptation_budget_cfg = dict(track_cfg.get("adaptation_budget", {}) or {})
    adapt_max_updates = int(adaptation_budget_cfg.get("max_updates", 200))
    adapt_max_updates_per_step = int(adaptation_budget_cfg.get("max_updates_per_step", 1))
    allowed_after_t0_only = bool(adaptation_budget_cfg.get("allowed_after_t0_only", False))
    save_predictions, visualization_enabled = _prediction_artifact_policy(
        suite=suite,
        task=task,
        runner_cfg=runner_cfg,
    )
    viz_emit = _viz_artifact_policy(
        suite=suite,
        task=task,
        runner_cfg=runner_cfg,
        cli_emit=emit_viz_artifacts,
    )
    for stale_name in (PRED_ARTIFACT_FILENAME, PRED_META_FILENAME):
        stale_path = run_dir / "artifacts" / stale_name
        if stale_path.exists():
            stale_path.unlink()

    run_plan = {
        "plan_id": f"{init_id}__{track_id}",
        "init_id": str(init_id),
        "track_id": str(track_id),
        "official_benchmark_eligible": bool(str(init_id).lower() != "untrained"),
        "suite_name": str(suite_name),
        "task_id": str(task_id),
        "scenario_id": str(scenario_id),
        "seed": int(seed),
        "model_id": str(model_id),
        "deterministic": bool(deterministic),
        "device_requested": str(device_str),
        "device_resolved": str(device),
        "precision": str(precision),
        "logging": {
            "level": str(log_cfg["level"]),
            "log_to_file": bool(log_cfg["log_to_file"]),
            "log_file": log_cfg["log_file"],
            "debug_every": int(log_cfg["debug_every"]),
        },
        "budgets": {
            "train_max_updates": int(train_max_updates),
            "adapt_max_updates": int(adapt_max_updates) if adaptation_enabled else 0,
            "adapt_max_updates_per_step": int(adapt_max_updates_per_step) if adaptation_enabled else 0,
            "t0_gate_enabled": bool(allowed_after_t0_only),
            "adaptation_budget": _normalize_jsonish(track_cfg.get("adaptation_budget", {})),
        },
        "artifacts": {
            "save_predictions": bool(save_predictions),
            "save_predictions_default": True,
            "visualization_enabled": bool(visualization_enabled),
        },
    }
    _write_json(run_dir / "run_plan.json", run_plan)

    # Initialize ledger early; adapter may update this file.
    ledger_path = run_dir / "budget_ledger.json"
    _write_json(
        ledger_path,
        {
            "train_updates_used": 0,
            "train_outer_updates_used": 0,
            "train_inner_updates_used": 0,
            "adapt_updates_used": 0,
            "train_max_updates": int(train_max_updates),
            "train_skipped": False,
            "cache_enabled": False,
            "cache_hit": False,
            "cache_key": None,
            "track_id": str(track_id),
            "init_id": str(init_id),
        },
    )

    cfg_snapshot = {
        "suite": suite.get("suite", {}),
        "task": task,
        "model": model,
        "scenario_settings": scenario_settings,
        "scenario_cfg_basis": scenario_cfg_basis,
        "scenario_id": scenario_id,
        "cache_scenario_id": cache_scenario_id,
        "scenario_id_resolved_from_cache": bool(resolved_from_cache),
        "seed": int(seed),
        "track_id": track_id,
        "track_cfg": track_cfg,
        "init_id": str(init_id),
        "runner_overrides": {
            "device": device_str,
            "precision": precision,
            "logging": {
                "level": str(log_cfg["level"]),
                "log_to_file": bool(log_cfg["log_to_file"]),
                "log_file": log_cfg["log_file"],
                "debug_every": int(log_cfg["debug_every"]),
            },
        },
        "data": {
            "cache_root": str(cache_root),
            "train_npz": str(train_path),
            "val_npz": str(val_path),
            "test_npz": str(test_path),
            "canonical_layout": "NTD",
        },
    }
    _write_yaml(run_dir / "config_snapshot.yaml", cfg_snapshot)

    try:
        if io_mod and hasattr(io_mod, "write_env_snapshot"):
            io_mod.write_env_snapshot(run_dir)  # type: ignore
        if io_mod and hasattr(io_mod, "write_git_snapshot"):
            io_mod.write_git_snapshot(run_dir)  # type: ignore
    except Exception as e:
        log_err(f"[WARN] write_env_snapshot/write_git_snapshot failed: {e}")

    env_txt_lines = [
        f"python: {sys.version.replace(os.linesep, ' ')}",
        f"platform: {sys.platform}",
        f"torch: {getattr(torch, '__version__', 'unknown')}",
        f"cuda_available: {torch.cuda.is_available()}",
        f"device_used: {device.type}",
    ]
    _write_text(run_dir / "env.txt", "\n".join(env_txt_lines) + "\n")

    split_test: Optional[Dict[str, Any]] = None
    x_hat_full: Optional[torch.Tensor] = None
    mse_t_mean: Optional[np.ndarray] = None
    mse_val: Optional[float] = None
    mse_db_val: Optional[float] = None
    shift_info: Dict[str, Any] = {}
    adapter_meta: Dict[str, Any] = {}
    adapter_runtime: Optional[Dict[str, Any]] = None
    prediction_artifact_info: Dict[str, Any] = {
        "enabled": bool(save_predictions),
        "status": "pending" if save_predictions else "disabled",
    }
    run_warnings: List[Dict[str, Any]] = []

    try:
        if adaptation_enabled and adapt_max_updates > 200:
            raise RuntimeError(
                f"budget_overflow: adaptation_budget.max_updates must be <= 200, got {adapt_max_updates}"
            )
        if adaptation_enabled and adapt_max_updates_per_step > 1:
            raise RuntimeError(
                "budget_overflow: adaptation_budget.max_updates_per_step must be <= 1 "
                f"(got {adapt_max_updates_per_step})"
            )

        split_train = _load_split_npz(train_path) if train_path.exists() else None
        split_val = _load_split_npz(val_path) if val_path.exists() else None
        split_test = _load_split_npz(test_path)

        if str(init_id).lower() == "trained" and (split_train is None or split_val is None):
            raise FileNotFoundError(
                f"missing_data: expected train/val splits for trained plan. train={train_path} val={val_path}"
            )

        x_gt = split_test["x"]  # [N,T,Dx]
        y_test = split_test["y"]  # [N,T,Dy]
        meta = split_test.get("meta")
        F = split_test.get("F")
        H = split_test.get("H")
        t0_shift = _extract_t0(task, meta)
        shift_info = _extract_shift_debug(meta, task)

        N, T, Dx = x_gt.shape
        _, _, Dy = y_test.shape
        logger.info(
            "Loaded test split shapes x=%s y=%s dims=(N=%s,T=%s,Dx=%s,Dy=%s)",
            tuple(x_gt.shape),
            tuple(y_test.shape),
            N,
            T,
            Dx,
            Dy,
        )
        if shift_info:
            logger.info("Shift/noise metadata: %s", json.dumps(shift_info, sort_keys=True))

        run_plan["shift"] = {"t0": int(t0_shift) if t0_shift is not None else None}
        run_plan["adaptation"] = {
            "enabled": bool(adaptation_enabled),
            "allowed_after_t0_only": bool(allowed_after_t0_only),
            "max_updates": int(adapt_max_updates) if adaptation_enabled else 0,
            "max_updates_per_step": int(adapt_max_updates_per_step) if adaptation_enabled else 0,
        }
        _write_json(run_dir / "run_plan.json", run_plan)

        if split_train is None:
            split_train = split_test
        if split_val is None:
            split_val = split_test

        _log_split_debug("train", split_train, batch_size=train_bs)
        _log_split_debug("val", split_val, batch_size=eval_bs)
        _log_split_debug("test", split_test, batch_size=eval_bs)

        train_loader = _make_loader(
            x=split_train["x"],
            y=split_train["y"],
            u=split_train.get("u"),
            extras=split_train.get("extras"),
            batch_size=train_bs,
            shuffle=True,
            seed=int(seed),
        )
        val_loader = _make_loader(
            x=split_val["x"],
            y=split_val["y"],
            u=split_val.get("u"),
            extras=split_val.get("extras"),
            batch_size=eval_bs,
            shuffle=False,
            seed=int(seed),
        )
        test_loader = _make_loader(
            x=split_test["x"],
            y=split_test["y"],
            u=split_test.get("u"),
            extras=split_test.get("extras"),
            batch_size=eval_bs,
            shuffle=False,
            seed=int(seed),
        )

        AdapterCls = _load_adapter(model_id)
        adapter = AdapterCls()

        system_info = {
            "x_dim": int(Dx),
            "y_dim": int(Dy),
            "T": int(T),
            "F": F,
            "H": H,
            "meta": meta,
            "task_id": task_id,
            "suite_name": suite_name,
            "scenario_settings": scenario_settings,
            "debug_every": int(log_cfg["debug_every"]),
            "log_level": str(log_cfg["level"]),
        }
        run_ctx = {
            "run_dir": str(run_dir),
            "seed": int(seed),
            "deterministic": bool(deterministic),
            "scenario_id": str(scenario_id),
            "task_id": str(task_id),
            "suite_name": str(suite_name),
            "model_id": str(model_id),
            "track_id": str(track_id),
            "init_id": str(init_id),
            "device": str(device),
            "debug_every": int(log_cfg["debug_every"]),
            "log_level": str(log_cfg["level"]),
            "log_to_file": bool(log_cfg["log_to_file"]),
        }
        _try_call_setup(adapter, model, system_info, run_ctx)

        adapter_meta_seed = _read_adapter_meta(adapter)
        adapter_version_for_cache = str(adapter_meta_seed.get("adapter_version", "unknown"))
        logger.info(
            "Adapter ready class=%s layout=%s version=%s",
            getattr(adapter, "last_class", None),
            getattr(adapter, "last_layout", None),
            adapter_version_for_cache,
        )
        model_cache_dir = _resolve_model_cache_dir(bench_root, runner_cfg, model)
        cache_enabled = bool(model_cache_dir is not None and str(init_id).lower() == "trained")
        train_skipped = False
        cache_hit = False
        cache_key: Optional[str] = None

        def _execute_train_once() -> Optional[Path]:
            train_budget = dict(budget_cfg)
            train_budget["train_max_updates"] = int(train_max_updates)
            train_budget["patience_evals"] = int(
                (runner_cfg.get("early_stopping", {}) or {}).get("patience_evals", 0)
            )
            train_budget["min_delta"] = float(
                (runner_cfg.get("early_stopping", {}) or {}).get("min_delta", 0.0)
            )
            train_res = _try_call_train(
                adapter=adapter,
                train_loader=train_loader,
                val_loader=val_loader,
                budget=train_budget,
                ckpt_dir=(run_dir / "checkpoints"),
            )
            if isinstance(train_res, dict) and train_res.get("ckpt_path"):
                return Path(str(train_res["ckpt_path"])).expanduser().resolve()

            save_res = None
            if hasattr(adapter, "save"):
                save_res = adapter.save(run_dir / "checkpoints")  # type: ignore
            if isinstance(save_res, dict) and save_res.get("ckpt_path"):
                return Path(str(save_res["ckpt_path"])).expanduser().resolve()
            fallback = (run_dir / "checkpoints" / "model.pt")
            if fallback.exists():
                return fallback.resolve()
            return None

        ckpt_path: Optional[Path] = None
        if str(init_id).lower() == "trained":
            train_budget_key = {
                "train_max_updates": int(train_max_updates),
                "patience_evals": int((runner_cfg.get("early_stopping", {}) or {}).get("patience_evals", 0)),
                "min_delta": float((runner_cfg.get("early_stopping", {}) or {}).get("min_delta", 0.0)),
            }
            if cache_enabled and model_cache_dir is not None:
                model_cache_dir.mkdir(parents=True, exist_ok=True)
                data_hash = _dataset_hash([train_path, val_path])
                git_digest = _git_versions_digest(bench_root, run_dir)
                env_hash = _env_digest(device=device, precision=precision, deterministic=deterministic)
                cache_key = _compute_train_cache_key(
                    model_id=model_id,
                    adapter_version=adapter_version_for_cache,
                    task_id=str(task_id),
                    scenario_id=str(scenario_id),
                    seed=int(seed),
                    train_budget=train_budget_key,
                    model_cfg=model,
                    data_hash=data_hash,
                    git_digest=git_digest,
                    env_digest=env_hash,
                )
                cache_paths = _cache_entry_paths(model_cache_dir, model_id=model_id, cache_key=cache_key)
                run_plan["cache"] = {
                    "enabled": True,
                    "model_cache_dir": str(model_cache_dir),
                    "cache_key": str(cache_key),
                    "cache_entry_dir": str(cache_paths["entry_dir"]),
                    "cache_hit": False,
                    "train_skipped": False,
                }
                _write_json(run_dir / "run_plan.json", run_plan)

                if cache_paths["ckpt_path"].exists():
                    train_skipped = True
                    cache_hit = True
                    logger.info("Training cache hit cache_key=%s", cache_key)
                    ckpt_path = (run_dir / "checkpoints" / "model.pt").resolve()
                    _copy_if_exists(cache_paths["ckpt_path"], ckpt_path)
                    _copy_if_exists(cache_paths["train_state_path"], run_dir / "checkpoints" / "train_state.json")
                    _update_ledger_file(
                        ledger_path,
                        {
                            "train_updates_used": 0,
                            "train_outer_updates_used": 0,
                            "train_inner_updates_used": 0,
                            "train_max_updates": int(train_max_updates),
                            "train_skipped": True,
                            "cache_enabled": True,
                            "cache_hit": True,
                            "cache_key": str(cache_key),
                        },
                    )
                else:
                    logger.info("Training cache miss cache_key=%s", cache_key)
                    ckpt_path = _execute_train_once()
                    _update_ledger_file(
                        ledger_path,
                        {
                            "train_skipped": False,
                            "cache_enabled": True,
                            "cache_hit": False,
                            "cache_key": str(cache_key),
                        },
                    )
                    if ckpt_path is not None:
                        cache_paths["entry_dir"].mkdir(parents=True, exist_ok=True)
                        _copy_if_exists(ckpt_path, cache_paths["ckpt_path"])
                        _copy_if_exists(
                            run_dir / "checkpoints" / "train_state.json",
                            cache_paths["train_state_path"],
                        )
                        _write_json(
                            cache_paths["meta_path"],
                            {
                                "cache_key": str(cache_key),
                                "model_id": str(model_id),
                                "adapter_version": str(adapter_version_for_cache),
                                "task_id": str(task_id),
                                "scenario_id": str(scenario_id),
                                "seed": int(seed),
                                "train_budget": train_budget_key,
                                "data_hash": data_hash,
                                "git_digest": git_digest,
                                "env_digest": env_hash,
                            },
                        )
            else:
                logger.info("Running train phase without cache")
                ckpt_path = _execute_train_once()
                _update_ledger_file(
                    ledger_path,
                    {
                        "train_skipped": False,
                        "cache_enabled": False,
                        "cache_hit": False,
                        "cache_key": None,
                    },
                )

            run_plan.setdefault("cache", {})
            run_plan["cache"].update(
                {
                    "enabled": bool(cache_enabled),
                    "model_cache_dir": (str(model_cache_dir) if model_cache_dir is not None else None),
                    "cache_key": (str(cache_key) if cache_key is not None else None),
                    "cache_hit": bool(cache_hit),
                    "train_skipped": bool(train_skipped),
                }
            )
            _write_json(run_dir / "run_plan.json", run_plan)
        elif str(init_id).lower() in ("pretrained", "loaded"):
            if model.get("ckpt_path"):
                ckpt_path = Path(str(model.get("ckpt_path"))).expanduser().resolve()
            elif (run_dir / "checkpoints" / "model.pt").exists():
                ckpt_path = (run_dir / "checkpoints" / "model.pt").resolve()
            run_plan["cache"] = {"enabled": False, "cache_hit": False, "train_skipped": False}
            _write_json(run_dir / "run_plan.json", run_plan)
        else:
            run_plan["cache"] = {"enabled": False, "cache_hit": False, "train_skipped": False}
            _write_json(run_dir / "run_plan.json", run_plan)

        if str(init_id).lower() == "trained" and ckpt_path is None:
            raise FileNotFoundError(
                "io_error: trained plan expected checkpoint at run_dir/checkpoints/model.pt but none was produced."
            )

        eval_ckpt_path = ckpt_path
        if adaptation_enabled:
            should_run_adapt = t0_shift is not None
            if ckpt_path is not None and (str(init_id).lower() in ("pretrained", "loaded") or train_skipped):
                if hasattr(adapter, "load"):
                    adapter.load(str(ckpt_path))  # type: ignore

            adapt_budget = dict(adaptation_budget_cfg)
            adapt_budget["max_updates"] = int(adapt_max_updates)
            adapt_budget["max_updates_per_step"] = int(adapt_max_updates_per_step)
            adapt_budget["allowed_after_t0_only"] = bool(allowed_after_t0_only)

            run_plan["adaptation"]["will_run"] = bool(should_run_adapt)
            _write_json(run_dir / "run_plan.json", run_plan)

            if should_run_adapt:
                logger.info(
                    "Running adapt phase t0=%s max_updates=%s max_updates_per_step=%s allowed_after_t0_only=%s",
                    t0_shift,
                    adapt_max_updates,
                    adapt_max_updates_per_step,
                    allowed_after_t0_only,
                )
                adapt_context = dict(system_info)
                adapt_context["track_id"] = str(track_id)
                adapt_context["init_id"] = str(init_id)
                _try_call_adapt(
                    adapter=adapter,
                    stream_or_loader=test_loader,
                    budget=adapt_budget,
                    t0=t0_shift,
                    allowed_after_t0_only=allowed_after_t0_only,
                    context=adapt_context,
                )
                # Evaluate the adapted in-memory state; do not reload checkpoint.
                eval_ckpt_path = None
            else:
                log_out(
                    "[INFO] budgeted track requested but no shift t0 detected; "
                    "adapt stage skipped by plan rule."
                )
                logger.info("Adapt stage skipped because no shift t0 was detected")

        batch_times_ms: List[float] = []
        eval_res: Any = None

        if hasattr(adapter, "eval"):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_eval0 = time.perf_counter()
            eval_res = _try_call_eval(adapter, test_loader, eval_ckpt_path, track_cfg)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_eval1 = time.perf_counter()
            batch_times_ms = [(t_eval1 - t_eval0) * 1000.0]

            if isinstance(eval_res, dict) and "x_hat" in eval_res:
                x_hat_full = _validate_full_x_hat(eval_res["x_hat"], N=N, T=T, D=Dx)

        if x_hat_full is None:
            x_hat_full, batch_times_ms = _predict_batches(
                adapter=adapter,
                y_full=y_test,
                x_dim=Dx,
                T=T,
                eval_bs=eval_bs,
                device=device,
                context=dict(system_info, debug_every=int(log_cfg["debug_every"])),
            )

        if tuple(x_hat_full.shape) != (N, T, Dx):
            x_hat_full = _validate_full_x_hat(x_hat_full, N=N, T=T, D=Dx)

        x_hat_np = x_hat_full.detach().cpu().numpy()
        if log_cfg["debug_every"] and int(log_cfg["debug_every"]) > 0:
            logger.debug("Eval output %s", format_array_stats("x_hat_full", x_hat_np))
        if not np.isfinite(x_hat_np).all():
            if split_test is not None:
                adapter_runtime = (
                    adapter.get_runtime_diagnostics()  # type: ignore[attr-defined]
                    if hasattr(adapter, "get_runtime_diagnostics")
                    else None
                )
                _write_run_diagnostics(
                    run_dir=run_dir,
                    suite_name=str(suite_name),
                    task_id=str(task_id),
                    scenario_id=str(scenario_id),
                    model_id=str(model_id),
                    track_id=str(track_id),
                    init_id=str(init_id),
                    seed=int(seed),
                    split_test=split_test,
                    x_hat_full=x_hat_full,
                    mse_val=None,
                    mse_db_val=None,
                    mse_t_mean=None,
                    thresholds_hit=["x_hat_nonfinite"],
                    shift_info=shift_info,
                    adapter_meta=adapter_meta_seed,
                    adapter_runtime=adapter_runtime,
                    reason="x_hat_nonfinite",
                )
            raise FloatingPointError("eval_nonfinite: x_hat contains non-finite values.")
        mse_t_mean = mse_per_step(x_hat_np, x_gt)
        if not np.isfinite(mse_t_mean).all():
            if split_test is not None:
                adapter_runtime = (
                    adapter.get_runtime_diagnostics()  # type: ignore[attr-defined]
                    if hasattr(adapter, "get_runtime_diagnostics")
                    else None
                )
                _write_run_diagnostics(
                    run_dir=run_dir,
                    suite_name=str(suite_name),
                    task_id=str(task_id),
                    scenario_id=str(scenario_id),
                    model_id=str(model_id),
                    track_id=str(track_id),
                    init_id=str(init_id),
                    seed=int(seed),
                    split_test=split_test,
                    x_hat_full=x_hat_full,
                    mse_val=None,
                    mse_db_val=None,
                    mse_t_mean=mse_t_mean,
                    thresholds_hit=["mse_t_nonfinite"],
                    shift_info=shift_info,
                    adapter_meta=adapter_meta_seed,
                    adapter_runtime=adapter_runtime,
                    reason="mse_t_nonfinite",
                )
            raise FloatingPointError("eval_nonfinite: mse_t contains non-finite values.")
        mse_val = float(np.mean(mse_t_mean))
        rmse_val = float(np.sqrt(max(mse_val, 0.0)))
        mse_db_val = float(10.0 * np.log10(max(mse_val, 1e-30)))
        adcs_event_metrics = _adcs_event_metrics_if_available(
            x_true=x_gt,
            x_pred=x_hat_np,
            split_extras=split_test.get("extras"),
        )
        _finalize_adapter_evaluation_diagnostics(
            adapter=adapter,
            run_dir=run_dir,
            split_extras=split_test.get("extras"),
            x_true=x_gt,
            x_pred=x_hat_np,
        )
        thresholds_hit: List[str] = []
        if not np.isfinite(np.array([mse_val, mse_db_val], dtype=np.float64)).all():
            thresholds_hit.append("metric_nonfinite")
        if mse_db_val > 100.0:
            thresholds_hit.append("mse_db_gt_100")

        adapter_runtime = (
            adapter.get_runtime_diagnostics()  # type: ignore[attr-defined]
            if hasattr(adapter, "get_runtime_diagnostics")
            else None
        )
        if is_debug_enabled(__name__) or thresholds_hit:
            dump_paths = _write_run_diagnostics(
                run_dir=run_dir,
                suite_name=str(suite_name),
                task_id=str(task_id),
                scenario_id=str(scenario_id),
                model_id=str(model_id),
                track_id=str(track_id),
                init_id=str(init_id),
                seed=int(seed),
                split_test=split_test,
                x_hat_full=x_hat_full,
                mse_val=mse_val,
                mse_db_val=mse_db_val,
                mse_t_mean=mse_t_mean,
                thresholds_hit=thresholds_hit,
                shift_info=shift_info,
                adapter_meta=adapter_meta_seed,
                adapter_runtime=adapter_runtime,
                reason=("anomaly" if thresholds_hit else "debug"),
            )
            logger.debug("Wrote diagnostics: %s", dump_paths)
            if thresholds_hit:
                logger.warning(
                    "Metric anomaly detected mse=%s mse_db=%s thresholds=%s diagnostics=%s",
                    mse_val,
                    mse_db_val,
                    thresholds_hit,
                    dump_paths.get("diagnostics_dir"),
                )

        warmup_batches = 1
        times_used = batch_times_ms[warmup_batches:] if len(batch_times_ms) > warmup_batches else batch_times_ms
        total_ms = float(np.sum(times_used))
        timing_ms_per_step = total_ms / float(max(1, N * T))
        timing_std_ms_per_step = float(
            np.std(np.array(times_used, dtype=np.float64) / float(max(1, eval_bs * T)))
        ) if times_used else 0.0

        recovery = None
        if t0_shift is not None:
            recovery = compute_shift_recovery_k(
                mse_t=mse_t_mean,
                t0=int(t0_shift),
                W=20,
                eps=0.05,
                failure_policy="cap",
            )

        metrics_step_path = run_dir / "metrics_step.csv"
        with metrics_step_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "mse_t", "rmse_t", "mse_db_t"])
            for t in range(T):
                mt = float(mse_t_mean[t])
                w.writerow([t, mt, float(np.sqrt(max(mt, 0.0))), float(10.0 * np.log10(max(mt, 1e-30)))])

        timing_path = run_dir / "timing.csv"
        with timing_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["batch_idx", "batch_size", "ms_predict_whole_seq"])
            if len(batch_times_ms) == 1:
                w.writerow([0, int(N), float(batch_times_ms[0])])
            else:
                for bi, dt in enumerate(batch_times_ms):
                    bs = min(eval_bs, N - bi * eval_bs)
                    w.writerow([bi, bs, float(dt)])

        adapter_meta = {}
        if hasattr(adapter, "get_adapter_meta"):
            try:
                maybe_meta = adapter.get_adapter_meta()  # type: ignore
                if isinstance(maybe_meta, dict):
                    adapter_meta = maybe_meta
            except Exception:
                adapter_meta = {}
        adapter_meta["selected_layout"] = getattr(adapter, "last_layout", None)
        adapter_meta["selected_class"] = getattr(adapter, "last_class", None)
        adapter_extra_metrics = _read_adapter_extra_metrics(adapter)

        if viz_emit:
            diagnostics = _collect_viz_diagnostics(
                adapter=adapter,
                test_loader=test_loader,
                ckpt_path=eval_ckpt_path,
                track_cfg=track_cfg,
                x_hat_np=x_hat_np,
                N=N,
                T=T,
                D=Dx,
            )
            time_s_viz, time_meta_viz = _prediction_time_s(
                task=task,
                split_test=split_test,
                meta=meta,
                n_step=T,
            )
            config_snapshot_path = run_dir / "config_snapshot.yaml"
            config_hash = _sha1_file(config_snapshot_path) if config_snapshot_path.exists() else None
            viz_trajectory_ids = _prediction_trajectory_id(split_test, N)
            viz_trajectory_id_source = (
                "test.npz:trajectory_id"
                if split_test.get("trajectory_id") is not None
                else "test_split_row_index_fallback"
            )
            from viz.io.writer import write_viz_artifacts

            write_viz_artifacts(
                run_dir=run_dir,
                repo_root=bench_root,
                suite_name=str(suite_name),
                task_id=str(task_id),
                task_family=str(task.get("task_family", (meta or {}).get("task_family", ""))),
                scenario_id=str(scenario_id),
                model_id=str(model_id),
                seed=int(seed),
                track_id=str(track_id),
                init_id=str(init_id),
                run_status="ok",
                time_s=time_s_viz,
                time_meta=time_meta_viz,
                x_true=x_gt,
                y_obs=y_test,
                x_hat=x_hat_np,
                split_extras=split_test.get("extras"),
                diagnostics=diagnostics,
                adapter_meta=adapter_meta,
                config_hash=config_hash,
                data_split="test",
                split_source="explicit",
                trajectory_ids=viz_trajectory_ids,
                trajectory_id_source=viz_trajectory_id_source,
                scenario_meta={"display_name": None, "parameters": dict(scenario_settings)},
            )
        logger.info(
            "Eval complete mse=%s rmse=%s mse_db=%s recovery_k=%s",
            mse_val,
            rmse_val,
            mse_db_val,
            (recovery or {}).get("recovery_k", None),
        )

        if save_predictions:
            try:
                time_s, time_meta = _prediction_time_s(
                    task=task,
                    split_test=split_test,
                    meta=meta,
                    n_step=T,
                )
                trajectory_id = _prediction_trajectory_id(split_test, N)
                artifact_meta: Dict[str, Any] = {
                    "suite_name": str(suite_name),
                    "task_id": str(task_id),
                    "scenario_id": str(scenario_id),
                    "seed": int(seed),
                    "model_id": str(model_id),
                    "track_id": str(track_id),
                    "init_id": str(init_id),
                    "source_split": "test",
                    **time_meta,
                }
                artifact_meta.update(
                    _prediction_state_meta(
                        task=task,
                        meta=meta,
                        x_dim=Dx,
                    )
                )
                pred_path, pred_meta_path = save_pred_artifact(
                    run_dir / "artifacts",
                    time_s=time_s,
                    x_true=x_gt,
                    y_obs=y_test,
                    x_hat=x_hat_np,
                    trajectory_id=trajectory_id,
                    meta=artifact_meta,
                )
                prediction_artifact_info = {
                    "enabled": True,
                    "status": "ok",
                    "path": str(pred_path),
                    "meta_path": str(pred_meta_path),
                }
                logger.info("Wrote prediction artifact: %s", pred_path)
            except Exception as artifact_exc:
                artifact_error = f"{type(artifact_exc).__name__}: {artifact_exc}"
                prediction_artifact_info = {
                    "enabled": True,
                    "status": "failed",
                    "error": artifact_error,
                }
                warning_obj = {
                    "type": "prediction_artifact_failed",
                    "message": artifact_error,
                }
                run_warnings.append(warning_obj)
                log_err(f"[WARN] prediction_artifact_failed: {artifact_error}")
                logger.warning("Prediction artifact generation failed: %s", artifact_error)
                if visualization_enabled:
                    raise RuntimeError(
                        "prediction_artifact_failed: visualization.enabled=true requires "
                        f"a valid prediction artifact ({artifact_error})"
                    ) from artifact_exc

        ledger_obj = _read_json_if_exists(ledger_path)
        if not ledger_obj:
            ledger_obj = {
                "train_updates_used": int(getattr(adapter, "train_updates_used", 0)),
                "train_outer_updates_used": int(
                    getattr(adapter, "train_outer_updates_used", getattr(adapter, "train_updates_used", 0))
                ),
                "train_inner_updates_used": int(getattr(adapter, "train_inner_updates_used", 0)),
                "adapt_updates_used": int(getattr(adapter, "adapt_updates_used", 0)),
                "train_max_updates": int(train_max_updates),
                "track_id": str(track_id),
                "init_id": str(init_id),
            }
            _write_json(ledger_path, ledger_obj)

        ledger_obj = _normalize_train_update_accounting(
            ledger_obj=ledger_obj,
            adapter=adapter,
            train_max_updates=int(train_max_updates),
            train_skipped=bool(train_skipped),
        )
        train_outer_updates_used = int(ledger_obj.get("train_outer_updates_used", 0))
        train_skipped_flag = bool(ledger_obj.get("train_skipped", train_skipped))
        ledger_obj["cache_enabled"] = bool(cache_enabled)
        ledger_obj["cache_hit"] = bool(cache_hit)
        ledger_obj["cache_key"] = (str(cache_key) if cache_key is not None else None)

        if str(init_id).lower() == "trained":
            if train_outer_updates_used > int(train_max_updates):
                raise RuntimeError(
                    "budget_overflow: train_outer_updates_used exceeded train_max_updates "
                    f"({train_outer_updates_used} > {train_max_updates})"
                )
            if (not train_skipped_flag) and train_outer_updates_used <= 0:
                raise RuntimeError(
                    "policy_violation: trained plan requires positive train_outer_updates_used unless train_skipped=true."
                )

        adapt_updates_used = int(ledger_obj.get("adapt_updates_used", getattr(adapter, "adapt_updates_used", 0)))
        adapt_updates_per_step = _normalize_adapt_updates_per_step(
            ledger_obj.get("adapt_updates_per_step", getattr(adapter, "adapt_updates_per_step", {}))
        )
        if adapt_updates_per_step:
            ledger_obj["adapt_updates_per_step"] = {str(k): int(v) for k, v in adapt_updates_per_step.items()}
        ledger_obj["adapt_updates_used"] = int(adapt_updates_used)

        if str(track_id) == "frozen" and adapt_updates_used != 0:
            raise RuntimeError(
                f"budget_overflow: frozen track requires adapt_updates_used=0, got {adapt_updates_used}"
            )
        if adaptation_enabled:
            if adapt_updates_used > int(adapt_max_updates):
                raise RuntimeError(
                    "budget_overflow: adapt_updates_used exceeded max_updates "
                    f"({adapt_updates_used} > {adapt_max_updates})"
                )
            if any(c > int(adapt_max_updates_per_step) for c in adapt_updates_per_step.values()):
                raise RuntimeError(
                    "budget_overflow: max per-step adaptation updates exceeded "
                    f"(limit={adapt_max_updates_per_step})"
                )
            if allowed_after_t0_only and t0_shift is not None:
                pre_t0_updates = [
                    t_idx for t_idx, count in adapt_updates_per_step.items()
                    if t_idx < int(t0_shift) and int(count) > 0
                ]
                if pre_t0_updates:
                    first_bad = min(pre_t0_updates)
                    raise RuntimeError(
                        "budget_overflow: adaptation updates observed before t0 "
                        f"(first_bad_t={first_bad}, t0={t0_shift})"
                    )

        _write_json(ledger_path, ledger_obj)

        metrics_obj: Dict[str, Any] = {
            "status": "ok",
            "suite_name": suite_name,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "cache_scenario_id": cache_scenario_id,
            "scenario_id_resolved_from_cache": bool(resolved_from_cache),
            "seed": int(seed),
            "model_id": model_id,
            "track_id": track_id,
            "dims": {"x_dim": int(Dx), "y_dim": int(Dy), "T": int(T)},
            "accuracy": {"mse": mse_val, "rmse": rmse_val, "mse_db": mse_db_val},
            "timing": {
                "timing_ms_per_step": float(timing_ms_per_step),
                "timing_std_ms_per_step": float(timing_std_ms_per_step),
                "warmup_batches_excluded": int(warmup_batches),
                "eval_batch_size": int(eval_bs),
            },
            "nll": {"value": None, "policy": "NA_if_no_cov"},
            "shift_recovery": recovery,
            "t0_used": int(t0_shift) if t0_shift is not None else None,
            "scenario_settings": scenario_settings,
            "scenario_cfg_basis": scenario_cfg_basis,
            "adapter_info": {
                "selected_layout": getattr(adapter, "last_layout", None),
                "selected_class": getattr(adapter, "last_class", None),
            },
            "adapter_meta": adapter_meta,
            "prediction_artifact": prediction_artifact_info,
            "warnings": run_warnings,
            "run_plan": run_plan,
            "budgets": ledger_obj,
            "run_dir": str(run_dir),
        }
        if adcs_event_metrics is not None:
            metrics_obj["adcs_event"] = adcs_event_metrics
        for key, value in adapter_extra_metrics.items():
            if key in metrics_obj:
                logger.warning(
                    "Ignoring adapter extra metric with reserved key=%s for model_id=%s",
                    key,
                    model_id,
                )
                continue
            metrics_obj[key] = value
        _write_json(run_dir / "metrics.json", metrics_obj)
        stale_failure = run_dir / "failure.json"
        if stale_failure.exists():
            stale_failure.unlink()

        log_out(f"[OK] wrote metrics to {run_dir}")
        clear_logging_context()

        return {
            "status": "ok",
            "run_dir": str(run_dir),
            "suite": suite_name,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "seed": seed,
            "model_id": model_id,
            "track_id": track_id,
            "init_id": str(init_id),
            "mse": mse_val,
            "rmse": rmse_val,
            "mse_db": mse_db_val,
            "timing_ms_per_step": float(timing_ms_per_step),
            "recovery_k": (recovery or {}).get("recovery_k", None),
            "prediction_artifact": prediction_artifact_info,
            "warnings": run_warnings,
        }

    except Exception as e:
        status = "failed"
        err_msg = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        failure_type = _classify_failure(e)
        phase = _classify_phase_from_traceback(tb)
        context = {
            "suite_name": suite_name,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "cache_scenario_id": cache_scenario_id,
            "scenario_id_resolved_from_cache": bool(resolved_from_cache),
            "seed": int(seed),
            "model_id": model_id,
            "track_id": track_id,
            "init_id": str(init_id),
            "scenario_settings": scenario_settings,
            "scenario_cfg_basis": scenario_cfg_basis,
        }

        failure = {
            "status": status,
            "failure_type": failure_type,
            "phase": phase,
            "failure_stage": phase,
            "message": err_msg,
            "traceback": tb,
            "context": context,
            # retained for compatibility with existing artifacts/tools
            "error": err_msg,
        }
        _write_json(run_dir / "failure.json", failure)
        log_err(err_msg)
        log_err(tb)
        logger.error("Run failed phase=%s failure_type=%s error=%s", phase, failure_type, err_msg)
        if split_test is not None and (is_debug_enabled(__name__) or failure_type in {"shape_mismatch", "runtime_error", "train_nan"}):
            try:
                dump_paths = _write_run_diagnostics(
                    run_dir=run_dir,
                    suite_name=str(suite_name),
                    task_id=str(task_id),
                    scenario_id=str(scenario_id),
                    model_id=str(model_id),
                    track_id=str(track_id),
                    init_id=str(init_id),
                    seed=int(seed),
                    split_test=split_test,
                    x_hat_full=x_hat_full,
                    mse_val=mse_val,
                    mse_db_val=mse_db_val,
                    mse_t_mean=mse_t_mean,
                    thresholds_hit=[failure_type],
                    shift_info=shift_info,
                    adapter_meta=adapter_meta,
                    adapter_runtime=adapter_runtime,
                    reason=f"failure:{failure_type}",
                )
                logger.debug("Failure diagnostics written: %s", dump_paths)
            except Exception as dump_exc:
                logger.error("Failed to write diagnostics after error: %s", dump_exc)
        clear_logging_context()

        return {
            "status": status,
            "run_dir": str(run_dir),
            "suite": suite_name,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "seed": seed,
            "model_id": model_id,
            "track_id": track_id,
            "init_id": str(init_id),
            "failure_type": failure_type,
            "error": err_msg,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-yaml", type=str, required=True)
    ap.add_argument("--models", nargs="*", default=None, help="model_id list. If omitted, run all models in suite.")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="override suite seeds")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu (default from suite.runner.device)")
    ap.add_argument("--precision", type=str, default=None, help="fp32 / amp (default from suite.runner.precision)")
    ap.add_argument("--track", type=str, default="frozen", help="frozen / budgeted (MVP is frozen)")
    ap.add_argument("--init-id", type=str, default="untrained", help="init plan id: trained / untrained / pretrained")
    ap.add_argument(
        "--plans",
        nargs="*",
        default=None,
        help="explicit plan list using '<init_id>:<track_id>' (e.g. trained:frozen trained:budgeted)",
    )
    ap.add_argument(
        "--keep-going",
        action="store_true",
        help="compatibility flag (run_suite already continues across run combinations).",
    )
    ap.add_argument("--tasks", nargs="*", default=None, help="task_id allowlist (optional)")
    ap.add_argument(
        "--scenario-ids",
        nargs="*",
        default=None,
        help="scenario_id allowlist after sweep expansion (optional; useful for resumable queued runs)",
    )
    ap.add_argument(
        "--log-level",
        type=str,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=None,
        help="log level (default from suite.runner.logging.level or INFO)",
    )
    ap.add_argument("--log-to-file", action="store_true", help="also write per-run bench.log files")
    ap.add_argument("--log-file", type=str, default=None, help="optional explicit log file path")
    ap.add_argument("--debug-every", type=int, default=None, help="periodic debug summary interval")
    ap.add_argument(
        "--emit-viz-artifacts",
        action="store_true",
        help="opt-in diagnostic visualization artifact emission (default: off)",
    )
    args = ap.parse_args()

    suite_path = Path(args.suite_yaml).expanduser().resolve()
    suite = load_suite_yaml(suite_path)
    suite_name = str((suite.get("suite", {}) or {}).get("name", "unknown"))

    runner_cfg = suite.get("runner", {}) or {}
    log_cfg = _runner_logging_cfg(
        runner_cfg,
        log_level=args.log_level,
        log_to_file=(True if args.log_to_file else None),
        log_file=args.log_file,
        debug_every=args.debug_every,
    )
    configure_logging(
        log_cfg["level"],
        run_dir=None,
        log_to_file=bool(log_cfg["log_file"]),
        log_file=(Path(str(log_cfg["log_file"])) if log_cfg["log_file"] else None),
    )
    enabled_policy = runner_cfg.get("enabled_policy", {}) or {}
    task_default = bool(enabled_policy.get("task_default", True))
    model_default = bool(enabled_policy.get("model_default", True))
    skip_if_disabled = bool(enabled_policy.get("skip_if_disabled", True))

    device_str = args.device or runner_cfg.get("device", "cpu")
    precision = args.precision or runner_cfg.get("precision", "fp32")

    seeds = args.seeds if args.seeds is not None else suite.get("seeds", [])
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError("No seeds provided (suite.seeds is empty and no --seeds).")

    plan_specs = _resolve_plans(args, runner_cfg)
    plan_isolation = len(plan_specs) > 1

    models: List[Dict[str, Any]] = suite.get("models", []) or []
    if args.models:
        wanted = set(args.models)
        models = [m for m in models if m.get("model_id") in wanted]

    tasks: List[Dict[str, Any]] = suite.get("tasks", []) or []
    if args.tasks:
        wanted_t = set(args.tasks)
        tasks = [t for t in tasks if t.get("task_id") in wanted_t]
    scenario_id_filter = set(str(x) for x in (args.scenario_ids or []))

    summary_rel = suite.get("reporting", {}).get("tables", {}).get("summary_csv", "reports/summary.csv")
    summary_csv = (_bench_root() / summary_rel).resolve()
    summary_fields = [
        "status", "suite", "task_id", "scenario_id", "seed", "model_id", "track_id", "init_id",
        "mse", "rmse", "mse_db", "timing_ms_per_step", "recovery_k",
        "run_dir", "error",
    ]

    total = 0
    for task in tasks:
        if skip_if_disabled and not _enabled(task, task_default):
            continue
        scenarios = _expand_sweep(task.get("sweep"))
        if scenario_id_filter:
            scenarios = [s for s in scenarios if _scenario_id_for_settings(task, s) in scenario_id_filter]
        total += len(scenarios) * len(seeds) * len(models) * len(plan_specs)

    print(f"[run_suite] plan: ~{total} runs (after enabled filtering)")
    logger.info(
        "Suite plan suite=%s total_runs_estimate=%s log_level=%s log_to_file=%s debug_every=%s",
        suite_name,
        total,
        log_cfg["level"],
        bool(log_cfg["log_to_file"] or log_cfg["log_file"]),
        log_cfg["debug_every"],
    )
    produced_run_dirs: List[str] = []

    for task in tasks:
        if skip_if_disabled and not _enabled(task, task_default):
            print(f"[skip task] {task.get('task_id')} enabled=false")
            continue

        scenario_list = _expand_sweep(task.get("sweep"))
        if scenario_id_filter:
            scenario_list = [s for s in scenario_list if _scenario_id_for_settings(task, s) in scenario_id_filter]
        for scenario_settings in scenario_list:
            for seed in seeds:
                for model in models:
                    if skip_if_disabled and not _enabled(model, model_default):
                        print(f"[skip model] {model.get('model_id')} enabled=false")
                        continue

                    for init_id, track_id in plan_specs:
                        res = run_one(
                            suite=suite,
                            task=task,
                            model=model,
                            scenario_settings=scenario_settings,
                            seed=int(seed),
                            track_id=str(track_id),
                            device_str=str(device_str),
                            precision=str(precision),
                            init_id=str(init_id),
                            plan_isolation=plan_isolation,
                            log_level=str(log_cfg["level"]),
                            log_to_file=bool(log_cfg["log_to_file"]),
                            log_file=log_cfg["log_file"],
                            debug_every=int(log_cfg["debug_every"]),
                            emit_viz_artifacts=bool(args.emit_viz_artifacts),
                        )
                        _append_summary_row(summary_csv, res, summary_fields)
                        rd = res.get("run_dir")
                        if isinstance(rd, str) and rd.strip():
                            produced_run_dirs.append(rd)

    manifest_path = _write_run_manifest(
        bench_root=_bench_root(),
        suite_name=suite_name,
        suite_yaml=suite_path,
        run_dirs=produced_run_dirs,
    )
    logger.info("Suite complete summary_csv=%s manifest=%s", summary_csv, manifest_path)
    print(f"[run_suite] done. summary_csv={summary_csv}")
    print(f"[run_suite] manifest={manifest_path}")


if __name__ == "__main__":
    main()
