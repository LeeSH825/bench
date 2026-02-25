from __future__ import annotations

import argparse
import copy
import csv
import fnmatch
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from bench.runners.run_suite import load_suite_yaml, run_one
from bench.tasks.bench_generated import (
    default_cache_root,
    expand_scenarios_from_sweep,
    merge_scenario_overrides,
    prepare_bench_generated_v0,
)
from bench.tasks.generator.contract import resolve_task_family
from bench.tasks.generator.datasets.common import DatasetMissingError, dataset_root_is_available


NEURAL_MODELS: Tuple[str, ...] = (
    "kalmannet_tsp",
    "adaptive_knet",
    "maml_knet",
    "split_knet",
)
PLOT_MODELS: Tuple[str, ...] = ("mb_kf",) + NEURAL_MODELS

SUPPORTED_FAMILIES: Tuple[str, ...] = (
    "linear",
    "linear_gaussian",
    "linear_gaussian_v0",
    "linear_canonical_v0",
    "default",
    "linear_mismatch",
    "linear_mismatch_v0",
    "ucm",
    "ucm_v0",
    "uniform_circular_motion",
    "uniform_circular_motion_v0",
    "sine_poly",
    "sine_poly_v0",
    "synthetic_nonlinear",
    "synthetic_nonlinear_v0",
    "switching",
    "switching_v0",
    "switching_dynamics",
    "switching_dynamics_v0",
    "lorenz",
    "lorenz_v0",
    "lorenz_dt",
    "lorenz_dt_v0",
    "nclt",
    "nclt_v0",
    "nclt_segway",
    "nclt_segway_v0",
    "uzh_fpv",
    "uzh_fpv_v0",
    "uzh",
    "uzh_v0",
    "uzh_fpv_ca",
    "uzh_fpv_ca_v0",
)
TG7_FAMILIES: Tuple[str, ...] = (
    "nclt",
    "nclt_v0",
    "nclt_segway",
    "nclt_segway_v0",
    "uzh_fpv",
    "uzh_fpv_v0",
    "uzh",
    "uzh_v0",
    "uzh_fpv_ca",
    "uzh_fpv_ca_v0",
)
KF_BASELINE_IDS: Tuple[str, ...] = (
    "mb_kf_nominal",
    "mb_kf_oracle",
    "oracle_kf",
    "nominal_kf",
    "oracle_shift_kf",
)

TRAIN_RUNS_MIN = 1
TRAIN_RUNS_MAX = 10


@dataclass(frozen=True)
class TaskEntry:
    source_suite: str
    source_path: Path
    task: Dict[str, Any]
    task_key: str
    task_family: str


@dataclass(frozen=True)
class ProtocolScenario:
    task_key: str
    task_id: str
    source_suite: str
    source_path: Path
    task_cfg: Dict[str, Any]
    base_flat: Dict[str, Any]
    base_nested: Dict[str, Any]
    r_value: float
    scenario_flat: Dict[str, Any]
    scenario_nested: Dict[str, Any]


def _bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _split_csv_values(values: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    for raw in values or []:
        for part in str(raw).split(","):
            s = part.strip()
            if s:
                out.append(s)
    return out


def _safe_json(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _safe_json(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe_json(x) for x in v]
    if isinstance(v, (np.integer, np.int32, np.int64)):
        return int(v)
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    return v


def _flatten_nested(d: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = str(k)
        dotted = f"{prefix}.{kk}" if prefix else kk
        if isinstance(v, Mapping):
            out.update(_flatten_nested(v, dotted))
        else:
            out[dotted] = _safe_json(v)
    return out


def _normalized_family(task_cfg: Mapping[str, Any]) -> str:
    return str(resolve_task_family(task_cfg)).strip().lower()


def _is_tg7_family(fam: str) -> bool:
    return fam in TG7_FAMILIES


def _is_supported_family(fam: str) -> bool:
    return fam in SUPPORTED_FAMILIES


def _task_signature(task_cfg: Mapping[str, Any]) -> str:
    payload = json.dumps(_safe_json(dict(task_cfg)), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return str(abs(hash(payload)))[:12]


def _load_suite_paths(args: argparse.Namespace) -> List[Path]:
    root = _bench_root()
    provided = [Path(p).expanduser().resolve() for p in _split_csv_values(args.include_suites)]

    if args.all_tasks:
        if provided:
            paths = provided
        else:
            paths = sorted((root / "bench" / "configs").glob("suite*.yaml"))
    else:
        if provided:
            paths = provided
        else:
            paths = [root / "bench" / "configs" / "suite_full_compare.yaml"]

    if not paths:
        raise ValueError("no suite yaml files selected")
    return paths


def _should_exclude(task_key: str, task_id: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    for pat in patterns:
        if fnmatch.fnmatch(task_key, pat) or fnmatch.fnmatch(task_id, pat):
            return True
    return False


def _discover_tasks(args: argparse.Namespace, suite_paths: Sequence[Path]) -> List[TaskEntry]:
    exclude_patterns = _split_csv_values(args.exclude)
    found: List[TaskEntry] = []
    seen_task_keys: set[str] = set()

    for suite_path in suite_paths:
        suite = load_suite_yaml(suite_path)
        suite_name = str((suite.get("suite") or {}).get("name", suite_path.stem))
        for task in (suite.get("tasks") or []):
            t = dict(task)
            task_id = str(t.get("task_id", "")).strip()
            if not task_id:
                continue
            enabled = bool(t.get("enabled", True))
            if (not args.include_disabled) and (not enabled):
                continue
            fam = _normalized_family(t)
            if not _is_supported_family(fam):
                continue
            if (not args.include_tg7) and _is_tg7_family(fam):
                continue

            task_key = f"{task_id}__sig_{_task_signature(t)}"
            if _should_exclude(task_key, task_id, exclude_patterns):
                continue
            if task_key in seen_task_keys:
                continue
            seen_task_keys.add(task_key)

            # Filter obviously incomplete placeholders when family falls back to default linear.
            if fam in {"linear", "linear_gaussian", "linear_gaussian_v0", "linear_canonical_v0", "default"}:
                x_dim = int(t.get("x_dim") or 0)
                y_dim = int(t.get("y_dim") or 0)
                T = int(t.get("sequence_length_T") or 0)
                if x_dim <= 0 or y_dim <= 0 or T <= 0:
                    continue

            found.append(
                TaskEntry(
                    source_suite=suite_name,
                    source_path=suite_path,
                    task=t,
                    task_key=task_key,
                    task_family=fam,
                )
            )
    return found


def _parse_r_grid_spec(spec: str) -> List[float]:
    s = str(spec).strip()
    if not s:
        raise ValueError("empty --r-grid spec")
    if s.startswith("list:"):
        raw = s[len("list:") :].strip()
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
        if not vals:
            raise ValueError("list grid must have at least one value")
        return vals
    if s.startswith("logspace:"):
        parts = [p.strip() for p in s[len("logspace:") :].split(":")]
        if len(parts) != 3:
            raise ValueError("logspace spec must be: logspace:<start_exp>:<stop_exp>:<num>")
        a, b, n = float(parts[0]), float(parts[1]), int(parts[2])
        if n <= 0:
            raise ValueError("logspace num must be > 0")
        return [float(v) for v in np.logspace(a, b, n)]
    if s.startswith("linspace:"):
        parts = [p.strip() for p in s[len("linspace:") :].split(":")]
        if len(parts) != 3:
            raise ValueError("linspace spec must be: linspace:<start>:<stop>:<num>")
        a, b, n = float(parts[0]), float(parts[1]), int(parts[2])
        if n <= 0:
            raise ValueError("linspace num must be > 0")
        return [float(v) for v in np.linspace(a, b, n)]
    raise ValueError(f"unsupported --r-grid spec: {spec}")


def _resolve_r_grid(args: argparse.Namespace) -> List[float]:
    if args.r_grid:
        vals = _parse_r_grid_spec(args.r_grid)
    elif (args.r_min is not None) and (args.r_max is not None) and (args.r_steps is not None):
        if args.r_steps <= 0:
            raise ValueError("--r-steps must be > 0")
        if args.r_mode == "r2":
            vals = [float(v) for v in np.logspace(math.log10(args.r_min), math.log10(args.r_max), int(args.r_steps))]
        else:
            vals = [float(v) for v in np.linspace(float(args.r_min), float(args.r_max), int(args.r_steps))]
    else:
        if args.r_mode == "r2":
            vals = [float(v) for v in np.logspace(-4.0, -1.0, 9)]
        else:
            vals = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    if any(v < 0.0 for v in vals):
        raise ValueError("all r-grid values must be >= 0")
    return vals


def _resolve_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        vals = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
        if not vals:
            raise ValueError("empty --seeds")
        return vals
    if not (TRAIN_RUNS_MIN <= int(args.train_runs) <= TRAIN_RUNS_MAX):
        raise ValueError(f"--train-runs must be in [{TRAIN_RUNS_MIN}, {TRAIN_RUNS_MAX}]")
    base = int(args.seed_base)
    return [base + i for i in range(int(args.train_runs))]


def _resolve_plans(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if not args.plans:
        return [("trained", "frozen")]
    out: List[Tuple[str, str]] = []
    for raw in args.plans:
        s = str(raw).strip()
        if ":" not in s:
            raise ValueError(f"invalid --plans entry '{raw}'. expected '<init_id>:<track_id>'")
        init_id, track_id = [x.strip() for x in s.split(":", 1)]
        if not init_id or not track_id:
            raise ValueError(f"invalid --plans entry '{raw}'")
        out.append((init_id, track_id))
    return out


def _collect_model_cfgs() -> Dict[str, Dict[str, Any]]:
    root = _bench_root()
    cfg_paths = sorted((root / "bench" / "configs").glob("suite*.yaml"))
    out: Dict[str, Dict[str, Any]] = {}
    for p in cfg_paths:
        suite = load_suite_yaml(p)
        for model in (suite.get("models") or []):
            m = dict(model)
            model_id = str(m.get("model_id", "")).strip()
            if not model_id:
                continue
            if model_id not in NEURAL_MODELS:
                continue
            # Prefer enabled entries, then first seen.
            if model_id not in out:
                out[model_id] = m
            elif bool(m.get("enabled", True)) and (not bool(out[model_id].get("enabled", True))):
                out[model_id] = m
    return out


def _mb_kf_model_cfg(mode: str) -> Dict[str, Any]:
    m = str(mode).strip().lower()
    if m not in {"auto", "nominal", "oracle"}:
        raise ValueError(f"unsupported --mb-kf-mode: {mode}")
    if m == "oracle":
        return {
            "model_id": "mb_kf_oracle",
            "display_name": "Model-Based KF (Oracle)",
            "baseline_mode": "oracle",
            "enabled": True,
        }
    # auto => nominal by policy.
    return {
        "model_id": "mb_kf_nominal",
        "display_name": "Model-Based KF (Nominal)",
        "baseline_mode": "nominal",
        "enabled": True,
    }


def _nested_get(d: Mapping[str, Any], path: Sequence[str]) -> Optional[Any]:
    cur: Any = d
    for k in path:
        if not isinstance(cur, Mapping):
            return None
        if k not in cur:
            return None
        cur = cur[k]
    return cur


def _build_r_override(task_cfg: Mapping[str, Any], r_mode: str, r_value: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rv = float(r_value)
    noise = task_cfg.get("noise", {})
    ns = task_cfg.get("noise_schedule", {})

    def _scale(base: float) -> float:
        return rv if r_mode == "r2" else float(base) * rv

    # Stationary / shifted linear noise blocks.
    if isinstance(noise, Mapping):
        r_stationary = _nested_get(noise, ("R", "r2"))
        r_pre = _nested_get(noise, ("pre_shift", "R", "r2"))
        if r_stationary is not None:
            out = {"noise": {"R": {"r2": float(_scale(float(r_stationary)))}}}
            return out
        if r_pre is not None:
            out = {"noise": {"pre_shift": {"R": {"r2": float(_scale(float(r_pre)))}}}}
            r_post = _nested_get(noise, ("shift", "post_shift", "r2"))
            if r_post is not None:
                out = merge_scenario_overrides(
                    out,
                    {"noise": {"shift": {"post_shift": {"r2": float(_scale(float(r_post)))}}}},
                )
            return out

    # TG1 schedule blocks.
    if isinstance(ns, Mapping):
        params = copy.deepcopy(dict(ns.get("params") or {}))
        candidates = [k for k in ("base_r2", "r2_pre", "r2_post", "r2") if k in params]
        if not candidates:
            fallback = 1.0e-3
            if isinstance(noise, Mapping):
                n0 = _nested_get(noise, ("R", "r2")) or _nested_get(noise, ("pre_shift", "R", "r2"))
                if n0 is not None:
                    fallback = float(n0)
            params["base_r2"] = float(fallback)
            candidates = ["base_r2"]
        for k in candidates:
            params[k] = float(_scale(float(params[k])))
        return {"noise_schedule": {"params": params}}

    raise ValueError(
        "cannot apply r override: no recognizable measurement-noise field found "
        "(expected noise.R.r2, noise.pre_shift.R.r2, or noise_schedule.params.*r2*)"
    )


def _protocols_for_task(task_entry: TaskEntry, r_grid: Sequence[float], r_mode: str) -> List[ProtocolScenario]:
    task_cfg = dict(task_entry.task)
    task_id = str(task_cfg.get("task_id"))
    base_nested_list = expand_scenarios_from_sweep(task_cfg)
    if not base_nested_list:
        base_nested_list = [{}]

    protocols: List[ProtocolScenario] = []
    for base_nested in base_nested_list:
        base_flat = _flatten_nested(base_nested)
        for rv in r_grid:
            r_override_nested = _build_r_override(task_cfg, r_mode=r_mode, r_value=float(rv))
            scenario_nested = merge_scenario_overrides(copy.deepcopy(base_nested), r_override_nested)
            scenario_flat = _flatten_nested(scenario_nested)
            protocols.append(
                ProtocolScenario(
                    task_key=task_entry.task_key,
                    task_id=task_id,
                    source_suite=task_entry.source_suite,
                    source_path=task_entry.source_path,
                    task_cfg=task_cfg,
                    base_flat=base_flat,
                    base_nested=base_nested,
                    r_value=float(rv),
                    scenario_flat=scenario_flat,
                    scenario_nested=scenario_nested,
                )
            )
    return protocols


def _is_tg7_task(task_entry: TaskEntry) -> bool:
    return _is_tg7_family(task_entry.task_family)


def _check_tg7_roots(task_entry: TaskEntry) -> Optional[str]:
    fam = task_entry.task_family
    if fam in {"nclt", "nclt_v0", "nclt_segway", "nclt_segway_v0"}:
        if not dataset_root_is_available("NCLT_ROOT"):
            return "io_error: dataset missing (NCLT_ROOT unset or path missing)"
    if fam in {"uzh_fpv", "uzh_fpv_v0", "uzh", "uzh_v0", "uzh_fpv_ca", "uzh_fpv_ca_v0"}:
        if not dataset_root_is_available("UZH_FPV_ROOT"):
            return "io_error: dataset missing (UZH_FPV_ROOT unset or path missing)"
    return None


def _runner_cfg(device: str, deterministic: bool, run_id: str, train_max_updates: int) -> Dict[str, Any]:
    return {
        "device": str(device),
        "precision": "fp32",
        "deterministic": bool(deterministic),
        "enabled_policy": {
            "task_default": True,
            "model_default": True,
            "skip_if_disabled": True,
        },
        "data_mode": "bench_generated",
        "model_cache_dir": f"runs/_model_cache/r_sweep_verify/{run_id}",
        "budget": {
            "train_max_updates": int(train_max_updates),
            "train_batch_size": 32,
            "eval_batch_size": 64,
        },
        "early_stopping": {
            "monitor": "val/mse_db",
            "mode": "min",
            "patience_evals": 4,
            "min_delta": 0.0,
        },
        "tracks": [
            {"track_id": "frozen", "adaptation_enabled": False},
            {
                "track_id": "budgeted",
                "adaptation_enabled": True,
                "adaptation_budget": {
                    "max_updates": 200,
                    "max_updates_per_step": 1,
                    "allowed_after_t0_only": True,
                    "overflow_policy": "fail",
                },
            },
        ],
    }


def _task_n_train(task_cfg: Mapping[str, Any]) -> Optional[int]:
    sizes = task_cfg.get("dataset_sizes")
    if isinstance(sizes, Mapping):
        n_train = sizes.get("N_train", None)
        if n_train is not None:
            n = int(n_train)
            if n > 0:
                return n

    splits = task_cfg.get("splits")
    if isinstance(splits, Mapping):
        train = splits.get("train", {})
        if isinstance(train, Mapping):
            n_train = train.get("N", None)
            if n_train is not None:
                n = int(n_train)
                if n > 0:
                    return n
    return None


def _effective_train_updates(
    *,
    task_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    default_train_max_updates: int,
    train_epochs: int,
    default_train_batch_size: int,
) -> int:
    if int(train_epochs) <= 0:
        return int(default_train_max_updates)

    n_train = _task_n_train(task_cfg)
    if n_train is None:
        return int(default_train_max_updates)

    batch_size = int(model_cfg.get("batch_size", default_train_batch_size))
    batch_size = max(1, batch_size)
    steps_per_epoch = int(math.ceil(float(n_train) / float(batch_size)))
    return max(1, steps_per_epoch * int(train_epochs))


def _model_label(model_id: str) -> str:
    mid = str(model_id).strip().lower()
    if mid in KF_BASELINE_IDS:
        return "mb_kf"
    return mid


def _git_hash(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or None
    except Exception:
        return None


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _plot_all(
    rows: Sequence[Dict[str, Any]],
    *,
    out_dir: Path,
    r_mode: str,
    r_grid: Sequence[float],
) -> None:
    ok_rows = [dict(r) for r in rows if str(r.get("status")) == "ok"]

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in ok_rows:
        key = (str(r.get("task_plot_key")), str(r.get("plan")))
        grouped.setdefault(key, []).append(r)

    if not grouped:
        raise RuntimeError("no successful runs were produced; cannot build plots")

    plot_dir = out_dir / "per_task"
    plot_dir.mkdir(parents=True, exist_ok=True)
    expected_r = {float(v) for v in r_grid}
    failures: List[str] = []

    for (task_plot_key, plan), g in sorted(grouped.items()):
        # model -> r -> [mse]
        table: Dict[str, Dict[float, List[float]]] = {m: {} for m in PLOT_MODELS}
        for rr in g:
            m = str(rr.get("model"))
            rv = float(rr.get("r_value"))
            mse = float(rr.get("mse"))
            table.setdefault(m, {}).setdefault(rv, []).append(mse)

        for model in PLOT_MODELS:
            have = set(table.get(model, {}).keys())
            missing_r = sorted(expected_r - have)
            if missing_r:
                failures.append(
                    f"task={task_plot_key} plan={plan} model={model} missing_r={missing_r}"
                )

        if failures:
            continue

        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        x_sorted = sorted(expected_r)
        for model in PLOT_MODELS:
            means: List[float] = []
            errs: List[float] = []
            for rv in x_sorted:
                arr = np.asarray(table[model][rv], dtype=np.float64)
                means.append(float(np.mean(arr)))
                if arr.size > 1:
                    errs.append(float(np.std(arr, ddof=1) / math.sqrt(float(arr.size))))
                else:
                    errs.append(0.0)
            ax.errorbar(x_sorted, means, yerr=errs, marker="o", linewidth=1.4, markersize=3.8, label=model)

        if min(x_sorted) > 0.0:
            ax.set_xscale("log")
        else:
            ax.set_xscale("linear")
        if r_mode == "r2":
            ax.set_xlabel("r (measurement noise variance r2)")
        else:
            ax.set_xlabel("r (R scale factor)")
        ax.set_ylabel("MSE")
        ax.set_title(f"{task_plot_key} [{plan}]")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{task_plot_key}__{plan}.png", dpi=170)
        plt.close(fig)

        summary = {
            "task_plot_key": task_plot_key,
            "plan": plan,
            "models": list(PLOT_MODELS),
            "r_mode": r_mode,
            "r_grid": x_sorted,
        }
        (plot_dir / f"{task_plot_key}__{plan}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if failures:
        raise RuntimeError(
            "missing required model curves for one or more plots:\n" + "\n".join(failures)
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Generate all runnable tasks, sweep measurement-noise r, train/eval "
            "MB_KF + 4 neural models, and emit Fig.5(a)-style MSE-vs-r plots."
        )
    )
    ap.add_argument("--all-tasks", action="store_true", help="discover tasks from selected suites (or all suite*.yaml).")
    ap.add_argument(
        "--include-suites",
        nargs="*",
        default=[],
        help="suite yaml paths (comma-separated or repeated). default: suite_full_compare.yaml unless --all-tasks.",
    )
    ap.add_argument("--include-disabled", action="store_true", help="include tasks with enabled:false in discovery.")
    ap.add_argument("--include-tg7", dest="include_tg7", action="store_true", default=True, help="include TG7 dataset tasks (default: true).")
    ap.add_argument("--no-include-tg7", dest="include_tg7", action="store_false", help="exclude TG7 dataset tasks.")
    ap.add_argument("--exclude", nargs="*", default=[], help="glob patterns for task_id/task_key exclusion.")
    ap.add_argument("--max-tasks", type=int, default=0, help="optional hard cap on number of discovered tasks.")

    ap.add_argument("--r-mode", choices=("r2", "R_scale"), default="r2")
    ap.add_argument(
        "--r-grid",
        type=str,
        default="",
        help=(
            "r grid spec: 'list:1e-4,3e-4,1e-3', 'logspace:-4:-1:9', "
            "or 'linspace:0.5:10:6'. default: logspace(-4,-1,9) for r2; [0.5,1,2,3,5,10] for R_scale."
        ),
    )
    ap.add_argument("--r-min", type=float, default=None, help="alternative grid min (used with --r-max --r-steps).")
    ap.add_argument("--r-max", type=float, default=None, help="alternative grid max (used with --r-min --r-steps).")
    ap.add_argument("--r-steps", type=int, default=None, help="alternative grid steps (used with --r-min --r-max).")

    ap.add_argument(
        "--train-runs",
        type=int,
        default=1,
        help=f"number of training repeats. allowed range: {TRAIN_RUNS_MIN}..{TRAIN_RUNS_MAX}",
    )
    ap.add_argument("--seed-base", type=int, default=0, help="base seed used when --seeds is not set.")
    ap.add_argument("--seeds", type=str, default="", help="explicit comma-separated seeds (overrides --train-runs).")
    ap.add_argument(
        "--plans",
        nargs="*",
        default=["trained:frozen"],
        help="plan list '<init_id>:<track_id>' e.g. trained:frozen trained:budgeted",
    )
    ap.add_argument("--mb-kf-mode", choices=("auto", "nominal", "oracle"), default="auto")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--train-max-updates", type=int, default=200, help="Route-B train_max_updates")
    ap.add_argument(
        "--train-epochs",
        type=int,
        default=0,
        help=(
            "if >0, override train_max_updates per task/model using "
            "ceil(N_train / batch_size) * train_epochs"
        ),
    )
    ap.add_argument("--deterministic", action="store_true", default=True, help="enforce deterministic mode (default: true)")
    ap.add_argument("--run-id", type=str, default="", help="optional run id for reports/r_sweep/<run_id>")
    ap.add_argument(
        "--yes-i-know-this-is-expensive",
        action="store_true",
        help="without this flag, run only a small sanity subset (first 2 tasks).",
    )
    args = ap.parse_args()

    if not str(args.device).startswith("cuda"):
        raise RuntimeError(
            "This harness requires CUDA training for neural models. "
            "Use --device cuda (or cuda:<index>)."
        )
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for neural model training in this harness, but torch.cuda.is_available() is False. "
            "Use a CUDA-capable environment and rerun with --device cuda."
        )

    suite_paths = _load_suite_paths(args)
    tasks = _discover_tasks(args, suite_paths)
    if not tasks:
        raise RuntimeError("no runnable tasks discovered")

    if args.max_tasks and int(args.max_tasks) > 0:
        tasks = tasks[: int(args.max_tasks)]

    if not args.yes_i_know_this_is_expensive:
        tasks = tasks[:2]
        print("[safety] --yes-i-know-this-is-expensive not set -> running sanity subset (first 2 tasks).")

    r_grid = _resolve_r_grid(args)
    seeds = _resolve_seeds(args)
    plans = _resolve_plans(args)
    train_runs_used = len(seeds)

    model_cfg_map = _collect_model_cfgs()
    missing_models = [m for m in NEURAL_MODELS if m not in model_cfg_map]
    if missing_models:
        raise RuntimeError(f"missing neural model configs in suite yamls: {missing_models}")

    mb_cfg = _mb_kf_model_cfg(args.mb_kf_mode)
    run_models: List[Dict[str, Any]] = [mb_cfg] + [copy.deepcopy(model_cfg_map[m]) for m in NEURAL_MODELS]
    for m in run_models:
        m["enabled"] = True

    total_protocols = 0
    protocol_map: Dict[str, List[ProtocolScenario]] = {}
    for task in tasks:
        ps = _protocols_for_task(task, r_grid=r_grid, r_mode=str(args.r_mode))
        protocol_map[task.task_key] = ps
        total_protocols += len(ps)

    total_runs_est = total_protocols * len(seeds) * len(run_models) * len(plans)
    budget_mode = (
        f"epochs->updates (train_epochs={int(args.train_epochs)})"
        if int(args.train_epochs) > 0
        else f"fixed_updates(train_max_updates={int(args.train_max_updates)})"
    )
    print("[preflight]")
    print(f"  base_tasks: {len(tasks)}")
    print(f"  derived_protocols(task x r): {total_protocols}")
    print(f"  train_runs(seeds): {train_runs_used} -> {seeds}")
    print(f"  train_budget_mode: {budget_mode}")
    print(f"  r_mode: {args.r_mode}")
    print(f"  r_grid: {r_grid}")
    print(f"  models: {[m['model_id'] for m in run_models]}")
    print(f"  plans: {plans}")
    print(f"  estimated_runs: {total_runs_est}")

    run_id = str(args.run_id).strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = _bench_root() / "reports" / "r_sweep" / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    cache_root = default_cache_root()
    suite_name = f"r_sweep_verify_{run_id}"
    suite_obj = {
        "suite": {"name": suite_name, "version": "0.1", "description": "r-sweep verification harness"},
        "runner": _runner_cfg(
            device=str(args.device),
            deterministic=bool(args.deterministic),
            run_id=run_id,
            train_max_updates=int(args.train_max_updates),
        ),
        "reporting": {
            "output_dir_template": f"runs/r_sweep_verify/{run_id}/" + "{task_id}/{model_id}/{track_id}/seed_{seed}/scenario_{scenario_id}"
        },
    }
    default_train_batch_size = int((suite_obj["runner"].get("budget", {}) or {}).get("train_batch_size", 32))

    rows: List[Dict[str, Any]] = []
    skipped_tasks: List[Dict[str, Any]] = []
    for task_entry in tasks:
        task_cfg_raw = copy.deepcopy(task_entry.task)
        task_id = str(task_cfg_raw.get("task_id"))

        # TG7 conditional skip.
        if _is_tg7_task(task_entry):
            miss = _check_tg7_roots(task_entry)
            if miss is not None:
                skipped_tasks.append(
                    {
                        "task_key": task_entry.task_key,
                        "task_id": task_id,
                        "reason": miss,
                        "source_suite": task_entry.source_suite,
                    }
                )
                continue

        task_cfg_gen = copy.deepcopy(task_cfg_raw)
        task_cfg_gen["sweep"] = {}

        for proto in protocol_map[task_entry.task_key]:
            scenario_settings = dict(proto.scenario_flat)
            scenario_nested = dict(proto.scenario_nested)
            task_plot_key = f"{task_id}__from_{task_entry.source_suite}"

            for seed in seeds:
                try:
                    prepare_bench_generated_v0(
                        suite_name=suite_name,
                        task_cfg=task_cfg_gen,
                        seed=int(seed),
                        cache_root=cache_root,
                        scenario_overrides=scenario_nested,
                    )
                except DatasetMissingError as exc:
                    skipped_tasks.append(
                        {
                            "task_key": task_entry.task_key,
                            "task_id": task_id,
                            "reason": str(exc),
                            "source_suite": task_entry.source_suite,
                        }
                    )
                    continue

                for init_id, track_id in plans:
                    for model_cfg in run_models:
                        model_run_cfg = copy.deepcopy(model_cfg)
                        eff_updates = _effective_train_updates(
                            task_cfg=task_cfg_raw,
                            model_cfg=model_run_cfg,
                            default_train_max_updates=int(args.train_max_updates),
                            train_epochs=int(args.train_epochs),
                            default_train_batch_size=default_train_batch_size,
                        )
                        model_run_cfg["train_max_updates"] = int(eff_updates)

                        model_id = str(model_run_cfg["model_id"])
                        run_res = run_one(
                            suite=suite_obj,
                            task=task_cfg_raw,
                            model=model_run_cfg,
                            scenario_settings=scenario_settings,
                            seed=int(seed),
                            track_id=str(track_id),
                            device_str=str(args.device),
                            precision="fp32",
                            init_id=str(init_id),
                            plan_isolation=(len(plans) > 1),
                        )
                        row = {
                            "task_key": task_entry.task_key,
                            "task_id": task_id,
                            "task_plot_key": task_plot_key,
                            "source_suite": task_entry.source_suite,
                            "source_suite_path": str(task_entry.source_path),
                            "task_family": task_entry.task_family,
                            "r_mode": str(args.r_mode),
                            "r_value": float(proto.r_value),
                            "scenario_settings_json": json.dumps(_safe_json(scenario_settings), ensure_ascii=False, sort_keys=True),
                            "seed": int(seed),
                            "plan": f"{init_id}:{track_id}",
                            "init_id": str(init_id),
                            "track_id": str(track_id),
                            "model_id": model_id,
                            "model": _model_label(model_id),
                            "status": str(run_res.get("status", "")),
                            "train_updates_budget": int(eff_updates),
                            "mse": run_res.get("mse", ""),
                            "mse_db": run_res.get("mse_db", ""),
                            "rmse": run_res.get("rmse", ""),
                            "run_dir": str(run_res.get("run_dir", "")),
                            "error": str(run_res.get("error", "")),
                            "failure_type": str(run_res.get("failure_type", "")),
                        }
                        rows.append(row)

    results_csv = out_root / "results.csv"
    _write_csv(
        results_csv,
        rows,
        fieldnames=[
            "task_key",
            "task_id",
            "task_plot_key",
            "source_suite",
            "source_suite_path",
            "task_family",
            "r_mode",
            "r_value",
            "scenario_settings_json",
            "seed",
            "plan",
            "init_id",
            "track_id",
            "model_id",
            "model",
            "status",
            "train_updates_budget",
            "mse",
            "mse_db",
            "rmse",
            "run_dir",
            "failure_type",
            "error",
        ],
    )

    skipped_csv = out_root / "skipped_tasks.csv"
    _write_csv(
        skipped_csv,
        skipped_tasks,
        fieldnames=["task_key", "task_id", "source_suite", "reason"],
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(_bench_root()),
        "git_hash": _git_hash(_bench_root()),
        "args": vars(args),
        "suite_paths": [str(p) for p in suite_paths],
        "device": str(args.device),
        "cuda_available": bool(torch.cuda.is_available()),
        "seeds": seeds,
        "r_grid": r_grid,
        "models": [m.get("model_id") for m in run_models],
        "plans": [f"{i}:{t}" for i, t in plans],
        "rows": len(rows),
        "skipped_tasks": len(skipped_tasks),
    }
    (out_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _plot_all(rows, out_dir=out_root, r_mode=str(args.r_mode), r_grid=r_grid)
    print(f"[done] results: {results_csv}")
    print(f"[done] plots:   {out_root / 'per_task'}")
    print(f"[done] manifest:{out_root / 'run_manifest.json'}")
    if skipped_tasks:
        print(f"[done] skipped dataset tasks: {len(skipped_tasks)} (see {skipped_csv})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
