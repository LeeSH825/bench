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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
from bench.utils.sweep import expand_sweep_grid


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


def _coerce_full_to_btd(x_hat: Any, N: int, T: int, D: int) -> torch.Tensor:
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
    if x_hat.ndim != 3:
        raise ValueError(f"x_hat must be rank-3, got shape={tuple(x_hat.shape)}")

    shape = tuple(x_hat.shape)
    if shape == (N, T, D):
        return x_hat.contiguous()
    if shape == (T, N, D):
        return x_hat.permute(1, 0, 2).contiguous()
    if shape == (N, D, T):
        return x_hat.permute(0, 2, 1).contiguous()

    raise ValueError(
        f"Unsupported full x_hat shape={shape}; expected [(N,T,D)=({N},{T},{D}), (T,N,D)=({T},{N},{D}), (N,D,T)=({N},{D},{T})]"
    )


def _load_split_npz(npz_path: Path) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as z:
        x = z["x"].astype(np.float32, copy=False)
        y = z["y"].astype(np.float32, copy=False)
        u = z["u"].astype(np.float32, copy=False) if "u" in z.files else None
        F = z["F"] if "F" in z.files else None
        H = z["H"] if "H" in z.files else None
        meta = None
        if "meta_json" in z.files:
            try:
                meta = json.loads(z["meta_json"].item())
            except Exception:
                meta = None
    return {"x": x, "y": y, "u": u, "F": F, "H": H, "meta": meta}


class _SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, u: Optional[np.ndarray] = None):
        self._x = x
        self._y = y
        self._u = u

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(self._x[idx])
        y = torch.from_numpy(self._y[idx])
        if self._u is None:
            return {"x": x, "y": y}
        u = torch.from_numpy(self._u[idx])
        return {"x": x, "y": y, "u": u}


def _make_loader(
    *,
    x: np.ndarray,
    y: np.ndarray,
    u: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    ds = _SeqDataset(x=x, y=y, u=u)
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
            if isinstance(x_hat, np.ndarray):
                x_hat = torch.from_numpy(x_hat)
            if not isinstance(x_hat, torch.Tensor):
                raise TypeError(f"adapter.predict must return Tensor/tuple/dict. Got {type(x_hat)}")

            x_hat = _coerce_to_btd(x_hat, B=(e - s), T=T, D=x_dim)
            out_batches.append(x_hat.detach().cpu())

    if not out_batches:
        raise RuntimeError("runtime_error: no predictions produced.")
    return torch.cat(out_batches, dim=0).contiguous(), batch_times_ms


def _try_call_setup(adapter: Any, model_cfg: Dict[str, Any], system_info: Dict[str, Any], run_ctx: Dict[str, Any]) -> None:
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


def _try_call_train(
    adapter: Any,
    train_loader: Any,
    val_loader: Any,
    budget: Dict[str, Any],
    ckpt_dir: Path,
) -> Any:
    try:
        return adapter.train(train_loader, val_loader, budget=budget, ckpt_dir=ckpt_dir)  # type: ignore
    except TypeError:
        return adapter.train(train_loader, val_loader)  # type: ignore


def _try_call_eval(
    adapter: Any,
    test_loader: Any,
    ckpt_path: Optional[Path],
    track_cfg: Dict[str, Any],
) -> Any:
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
) -> Dict[str, Any]:
    bench_root = _bench_root()
    cache_root = _cache_root(bench_root)
    runner_cfg = suite.get("runner", {}) or {}

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
    if deterministic and device.type == "cuda":
        if not str(os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")).strip():
            # Required by CUDA deterministic mode for cuBLAS-backed ops (e.g., batched matmul/bmm).
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            log_out("[INFO] set CUBLAS_WORKSPACE_CONFIG=:4096:8 (deterministic CUDA)")

    # Early missing_data
    if not test_path.exists():
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
        "budgets": {
            "train_max_updates": int(train_max_updates),
            "adapt_max_updates": int(adapt_max_updates) if adaptation_enabled else 0,
            "adapt_max_updates_per_step": int(adapt_max_updates_per_step) if adaptation_enabled else 0,
            "t0_gate_enabled": bool(allowed_after_t0_only),
            "adaptation_budget": _normalize_jsonish(track_cfg.get("adaptation_budget", {})),
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
        "runner_overrides": {"device": device_str, "precision": precision},
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

        N, T, Dx = x_gt.shape
        _, _, Dy = y_test.shape

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

        train_loader = _make_loader(
            x=split_train["x"],
            y=split_train["y"],
            u=split_train.get("u"),
            batch_size=train_bs,
            shuffle=True,
            seed=int(seed),
        )
        val_loader = _make_loader(
            x=split_val["x"],
            y=split_val["y"],
            u=split_val.get("u"),
            batch_size=eval_bs,
            shuffle=False,
            seed=int(seed),
        )
        test_loader = _make_loader(
            x=split_test["x"],
            y=split_test["y"],
            u=split_test.get("u"),
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
        }
        _try_call_setup(adapter, model, system_info, run_ctx)

        adapter_meta_seed = _read_adapter_meta(adapter)
        adapter_version_for_cache = str(adapter_meta_seed.get("adapter_version", "unknown"))
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

        x_hat_full: Optional[torch.Tensor] = None
        batch_times_ms: List[float] = []

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
                x_hat_full = _coerce_full_to_btd(eval_res["x_hat"], N=N, T=T, D=Dx)

        if x_hat_full is None:
            x_hat_full, batch_times_ms = _predict_batches(
                adapter=adapter,
                y_full=y_test,
                x_dim=Dx,
                T=T,
                eval_bs=eval_bs,
                device=device,
                context=system_info,
            )

        if tuple(x_hat_full.shape) != (N, T, Dx):
            x_hat_full = _coerce_full_to_btd(x_hat_full, N=N, T=T, D=Dx)

        x_hat_np = x_hat_full.detach().cpu().numpy()
        if not np.isfinite(x_hat_np).all():
            raise FloatingPointError("eval_nonfinite: x_hat contains non-finite values.")
        mse_t_mean = mse_per_step(x_hat_np, x_gt)
        if not np.isfinite(mse_t_mean).all():
            raise FloatingPointError("eval_nonfinite: mse_t contains non-finite values.")
        mse_val = float(np.mean(mse_t_mean))
        rmse_val = float(np.sqrt(max(mse_val, 0.0)))
        mse_db_val = float(10.0 * np.log10(max(mse_val, 1e-30)))

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

        train_outer_updates_used = int(
            ledger_obj.get(
                "train_outer_updates_used",
                ledger_obj.get("train_updates_used", getattr(adapter, "train_updates_used", 0)),
            )
        )
        train_inner_updates_used = int(
            ledger_obj.get("train_inner_updates_used", getattr(adapter, "train_inner_updates_used", 0))
        )
        train_skipped_flag = bool(ledger_obj.get("train_skipped", train_skipped))
        ledger_obj["train_outer_updates_used"] = int(train_outer_updates_used)
        ledger_obj["train_inner_updates_used"] = int(train_inner_updates_used)
        # Backward-compatible alias: keep train_updates_used as outer updates.
        ledger_obj["train_updates_used"] = int(train_outer_updates_used)
        ledger_obj["train_max_updates"] = int(train_max_updates)
        ledger_obj["train_skipped"] = bool(train_skipped_flag)
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
            "run_plan": run_plan,
            "budgets": ledger_obj,
            "run_dir": str(run_dir),
        }
        _write_json(run_dir / "metrics.json", metrics_obj)
        stale_failure = run_dir / "failure.json"
        if stale_failure.exists():
            stale_failure.unlink()

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
            "init_id": str(init_id),
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
    args = ap.parse_args()

    suite_path = Path(args.suite_yaml).expanduser().resolve()
    suite = load_suite_yaml(suite_path)
    suite_name = str((suite.get("suite", {}) or {}).get("name", "unknown"))

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
        total += len(scenarios) * len(seeds) * len(models) * len(plan_specs)

    print(f"[run_suite] plan: ~{total} runs (after enabled filtering)")
    produced_run_dirs: List[str] = []

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
    print(f"[run_suite] done. summary_csv={summary_csv}")
    print(f"[run_suite] manifest={manifest_path}")


if __name__ == "__main__":
    main()
