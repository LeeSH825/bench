from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("PyYAML(yaml) is required for run_suite. Install pyyaml.") from e

try:
    import torch
except Exception as e:
    raise RuntimeError("PyTorch(torch) is required for run_suite.") from e

from bench.metrics.core import (
    mse_per_step,
    compute_shift_recovery_k,
)


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
    if not sweep:
        return [{}]
    keys = sorted(sweep.keys())
    values_list: List[List[Any]] = []
    for k in keys:
        v = sweep[k]
        values_list.append(v if isinstance(v, list) else [v])

    out: List[Dict[str, Any]] = []
    for combo in product(*values_list):
        d = {k: combo[i] for i, k in enumerate(keys)}
        out.append(d)
    return out


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


def _build_scenario_cfg_basis(task: Dict[str, Any], scenario_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step4 cache의 scenario_id는 (task_id + scenario_cfg_basis)로 canonicalize 된다.
    scenario_cfg_basis = task.noise 복사 + sweep 오버라이드(dotted key) 적용 결과.
    """
    noise = copy.deepcopy(task.get("noise", {}) or {})
    for k, v in (scenario_settings or {}).items():
        if k.startswith("noise."):
            _dotted_set(noise, k[len("noise."):], v)
        else:
            _dotted_set(noise, k, v)

    def _normalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _normalize(_to_jsonable(v)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_normalize(_to_jsonable(x)) for x in obj]
        return _to_jsonable(obj)

    return _normalize(noise)


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


def _coerce_to_btd(x_hat: torch.Tensor, B: int, T: int, D: int) -> torch.Tensor:
    """
    repo별 output layout 차이로 인한 크래시를 줄이기 위한 최소 보정.
    허용:
      - [B,T,D]
      - [T,B,D]
      - [B,D,T]
    그 외는 에러로 상세 shape를 남긴다.
    """
    if x_hat.ndim != 3:
        raise ValueError(f"x_hat must be 3D. got shape={tuple(x_hat.shape)}")

    if tuple(x_hat.shape) == (B, T, D):
        return x_hat
    if tuple(x_hat.shape) == (T, B, D):
        return x_hat.permute(1, 0, 2).contiguous()
    if tuple(x_hat.shape) == (B, D, T):
        return x_hat.permute(0, 2, 1).contiguous()

    raise ValueError(
        f"Unsupported x_hat shape={tuple(x_hat.shape)}; expected one of "
        f"[(B,T,D)=({B},{T},{D}), (T,B,D)=({T},{B},{D}), (B,D,T)=({B},{D},{T})]"
    )


def run_one(
    suite: Dict[str, Any],
    task: Dict[str, Any],
    model: Dict[str, Any],
    scenario_settings: Dict[str, Any],
    seed: int,
    track_id: str,
    device_str: str,
    precision: str,
) -> Dict[str, Any]:
    bench_root = _bench_root()
    cache_root = _cache_root(bench_root)

    suite_name = suite["suite"]["name"]
    task_id = task["task_id"]
    model_id = model["model_id"]

    scenario_cfg_basis = _build_scenario_cfg_basis(task, scenario_settings)
    scenario_id = _canonicalize_scenario_id(task_id, scenario_cfg_basis)

    test_path = _npz_path(cache_root, suite_name, task_id, scenario_id, int(seed), "test")
    resolved_from_cache = False
    if not test_path.exists():
        alt = _resolve_scenario_id_from_cache(cache_root, suite_name, task_id, int(seed), scenario_settings)
        if alt is not None:
            scenario_id = alt
            test_path = _npz_path(cache_root, suite_name, task_id, scenario_id, int(seed), "test")
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
    run_dir = (bench_root / run_dir_rel).resolve()
    _ensure_dir(run_dir)

    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"

    io_mod, seed_mod, env_mod = _try_import_utils()

    def log_out(msg: str) -> None:
        with stdout_log.open("a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")

    def log_err(msg: str) -> None:
        with stderr_log.open("a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")

    deterministic = bool(suite.get("runner", {}).get("deterministic", True))
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

    # Early missing_data
    if not test_path.exists():
        failure = {
            "status": "missing_data",
            "missing_path": str(test_path),
            "hint": "Run Step4 smoke_data to generate cache for this suite/task/scenario/seed.",
            "cache_root": str(cache_root),
            "scenario_settings": {k: _to_jsonable(v) for k, v in (scenario_settings or {}).items()},
            "scenario_cfg_basis": scenario_cfg_basis,
            "scenario_id_method": "bench_generated.canonicalize_scenario_id(task_id, scenario_cfg_basis) (+ fallback scan)",
            "resolved_from_cache": bool(resolved_from_cache),
        }
        _write_json(run_dir / "failure.json", failure)
        log_err(json.dumps(failure, indent=2, ensure_ascii=False))
        return {
            "status": "missing_data",
            "run_dir": str(run_dir),
            "suite": suite_name,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "seed": seed,
            "model_id": model_id,
            "track_id": track_id,
            "error": "missing_data",
        }

    npz = np.load(test_path, allow_pickle=True)
    x_gt = npz["x"]  # [N,T,Dx]
    y = npz["y"]     # [N,T,Dy]
    meta = None
    if "meta_json" in npz:
        try:
            meta = json.loads(npz["meta_json"].item())
        except Exception:
            meta = None

    F = npz["F"] if "F" in npz else None
    H = npz["H"] if "H" in npz else None

    N, T, Dx = x_gt.shape
    _, _, Dy = y.shape

    track_cfg = _track_cfg(suite.get("runner", {}) or {}, track_id)
    cfg_snapshot = {
        "suite": suite.get("suite", {}),
        "task": task,
        "model": model,
        "scenario_settings": scenario_settings,
        "scenario_cfg_basis": scenario_cfg_basis,
        "scenario_id": scenario_id,
        "scenario_id_resolved_from_cache": bool(resolved_from_cache),
        "seed": int(seed),
        "track_id": track_id,
        "track_cfg": track_cfg,
        "runner_overrides": {"device": device_str, "precision": precision},
        "data": {
            "cache_root": str(cache_root),
            "test_npz": str(test_path),
            "canonical_layout": "NTD",  # D15
        },
    }
    _write_yaml(run_dir / "config_snapshot.yaml", cfg_snapshot)

    # env artifacts best-effort
    try:
        if io_mod and hasattr(io_mod, "write_env_snapshot"):
            io_mod.write_env_snapshot(run_dir)  # type: ignore
        if io_mod and hasattr(io_mod, "write_git_snapshot"):
            io_mod.write_git_snapshot(run_dir)  # type: ignore
    except Exception as e:
        log_err(f"[WARN] write_env_snapshot/write_git_snapshot failed: {e}")

    env_txt_lines = []
    env_txt_lines.append(f"python: {sys.version.replace(os.linesep, ' ')}")
    env_txt_lines.append(f"platform: {sys.platform}")
    env_txt_lines.append(f"torch: {getattr(torch, '__version__', 'unknown')}")
    env_txt_lines.append(f"cuda_available: {torch.cuda.is_available()}")
    env_txt_lines.append(f"device_used: {device.type}")
    _write_text(run_dir / "env.txt", "\n".join(env_txt_lines) + "\n")

    # Adapter
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
    }

    try:
        adapter.setup(model, system_info)  # type: ignore
    except TypeError:
        adapter.setup(model)  # type: ignore

    eval_bs = int(suite.get("runner", {}).get("budget", {}).get("eval_batch_size", 32))
    num_batches = (N + eval_bs - 1) // eval_bs

    mse_t_sum = np.zeros((T,), dtype=np.float64)
    mse_t_count = 0

    batch_times_ms: List[float] = []
    warmup_batches = 1

    status = "ok"
    try:
        with torch.no_grad():
            for bi in range(num_batches):
                s = bi * eval_bs
                e = min(N, (bi + 1) * eval_bs)
                y_b = _tensorize(y[s:e], device)
                x_b = _tensorize(x_gt[s:e], device)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred = adapter.predict(y_b, context=system_info, return_cov=False)  # type: ignore
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                dt_ms = (t1 - t0) * 1000.0

                x_hat = pred
                if isinstance(pred, tuple) and len(pred) >= 1:
                    x_hat = pred[0]
                if isinstance(pred, dict) and "x_hat" in pred:
                    x_hat = pred["x_hat"]

                if not isinstance(x_hat, torch.Tensor):
                    raise TypeError(f"adapter.predict must return Tensor/tuple/dict. Got {type(x_hat)}")

                # Coerce to [B,T,D] robustly
                x_hat = _coerce_to_btd(x_hat, B=(e - s), T=T, D=Dx)

                x_hat_np = x_hat.detach().float().cpu().numpy()
                x_b_np = x_b.detach().float().cpu().numpy()

                mse_t = mse_per_step(x_hat_np, x_b_np)
                mse_t_sum += mse_t * float(e - s)
                mse_t_count += (e - s)

                batch_times_ms.append(dt_ms)

        mse_t_mean = mse_t_sum / max(1, mse_t_count)

        mse_val = float(np.mean(mse_t_mean))
        rmse_val = float(np.sqrt(max(mse_val, 0.0)))
        mse_db_val = float(10.0 * np.log10(max(mse_val, 1e-30)))

        times_used = batch_times_ms[warmup_batches:] if len(batch_times_ms) > warmup_batches else batch_times_ms
        total_ms = float(np.sum(times_used))
        timing_ms_per_step = total_ms / float(max(1, N * T))
        timing_std_ms_per_step = float(
            np.std(np.array(times_used, dtype=np.float64) / float(max(1, eval_bs * T)))
        ) if times_used else 0.0

        t0_shift = _extract_t0(task, meta)
        recovery = None
        if t0_shift is not None:
            recovery = compute_shift_recovery_k(
                mse_t=mse_t_mean,
                t0=int(t0_shift),
                W=20,
                eps=0.05,
                failure_policy="cap",  # DECISIONS D7
            )

        # metrics_step.csv
        metrics_step_path = run_dir / "metrics_step.csv"
        with metrics_step_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "mse_t", "rmse_t", "mse_db_t"])
            for t in range(T):
                mt = float(mse_t_mean[t])
                w.writerow([t, mt, float(np.sqrt(max(mt, 0.0))), float(10.0 * np.log10(max(mt, 1e-30)))])

        # timing.csv
        timing_path = run_dir / "timing.csv"
        with timing_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["batch_idx", "batch_size", "ms_predict_whole_seq"])
            for bi, dt in enumerate(batch_times_ms):
                bs = min(eval_bs, N - bi * eval_bs)
                w.writerow([bi, bs, float(dt)])

        adapter_info = {
            "selected_layout": getattr(adapter, "last_layout", None),
            "selected_class": getattr(adapter, "last_class", None),
        }
        metrics_obj: Dict[str, Any] = {
            "status": status,
            "suite_name": suite_name,
            "task_id": task_id,
            "scenario_id": scenario_id,
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
            "adapter_info": adapter_info,
            "run_dir": str(run_dir),
        }
        _write_json(run_dir / "metrics.json", metrics_obj)

        log_out(f"[OK] wrote metrics to {run_dir}")

        return {
            "status": "ok",
            "run_dir": str(run_dir),
            "suite": suite_name,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "seed": seed,
            "model_id": model_id,
            "track_id": track_id,
            "mse": mse_val,
            "rmse": rmse_val,
            "mse_db": mse_db_val,
            "timing_ms_per_step": float(timing_ms_per_step),
            "recovery_k": (recovery or {}).get("recovery_k", None),
        }

    except Exception as e:
        status = "failed"
        err_msg = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()

        failure = {
            "status": status,
            "error": err_msg,
            "traceback": tb,
            "suite_name": suite_name,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "scenario_id_resolved_from_cache": bool(resolved_from_cache),
            "seed": int(seed),
            "model_id": model_id,
            "track_id": track_id,
            "scenario_settings": scenario_settings,
            "scenario_cfg_basis": scenario_cfg_basis,
        }
        _write_json(run_dir / "failure.json", failure)
        log_err(err_msg)
        log_err(tb)

        return {
            "status": status,
            "run_dir": str(run_dir),
            "suite": suite_name,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "seed": seed,
            "model_id": model_id,
            "track_id": track_id,
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
    ap.add_argument("--tasks", nargs="*", default=None, help="task_id allowlist (optional)")
    args = ap.parse_args()

    suite_path = Path(args.suite_yaml).expanduser().resolve()
    suite = load_suite_yaml(suite_path)

    runner_cfg = suite.get("runner", {}) or {}
    enabled_policy = runner_cfg.get("enabled_policy", {}) or {}
    task_default = bool(enabled_policy.get("task_default", True))
    model_default = bool(enabled_policy.get("model_default", True))
    skip_if_disabled = bool(enabled_policy.get("skip_if_disabled", True))

    device_str = args.device or runner_cfg.get("device", "cpu")
    precision = args.precision or runner_cfg.get("precision", "fp32")

    seeds = args.seeds if args.seeds is not None else suite.get("seeds", [])
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError("No seeds provided (suite.seeds is empty and no --seeds).")

    track_id = args.track
    track_cfg = _track_cfg(runner_cfg, track_id)
    if track_cfg.get("adaptation_enabled", False):
        raise NotImplementedError("Step6 MVP implements frozen track only. Use --track frozen.")

    models: List[Dict[str, Any]] = suite.get("models", []) or []
    if args.models:
        wanted = set(args.models)
        models = [m for m in models if m.get("model_id") in wanted]

    tasks: List[Dict[str, Any]] = suite.get("tasks", []) or []
    if args.tasks:
        wanted_t = set(args.tasks)
        tasks = [t for t in tasks if t.get("task_id") in wanted_t]

    summary_rel = suite.get("reporting", {}).get("tables", {}).get("summary_csv", "reports/summary.csv")
    summary_csv = (_bench_root() / summary_rel).resolve()
    summary_fields = [
        "status", "suite", "task_id", "scenario_id", "seed", "model_id", "track_id",
        "mse", "rmse", "mse_db", "timing_ms_per_step", "recovery_k",
        "run_dir", "error",
    ]

    total = 0
    for task in tasks:
        if skip_if_disabled and not _enabled(task, task_default):
            continue
        scenarios = _expand_sweep(task.get("sweep"))
        total += len(scenarios) * len(seeds) * len(models)

    print(f"[run_suite] plan: ~{total} runs (after enabled filtering)")

    for task in tasks:
        if skip_if_disabled and not _enabled(task, task_default):
            print(f"[skip task] {task.get('task_id')} enabled=false")
            continue

        scenario_list = _expand_sweep(task.get("sweep"))
        for scenario_settings in scenario_list:
            for seed in seeds:
                for model in models:
                    if skip_if_disabled and not _enabled(model, model_default):
                        print(f"[skip model] {model.get('model_id')} enabled=false")
                        continue

                    res = run_one(
                        suite=suite,
                        task=task,
                        model=model,
                        scenario_settings=scenario_settings,
                        seed=int(seed),
                        track_id=track_id,
                        device_str=str(device_str),
                        precision=str(precision),
                    )
                    _append_summary_row(summary_csv, res, summary_fields)

    print(f"[run_suite] done. summary_csv={summary_csv}")


if __name__ == "__main__":
    main()

