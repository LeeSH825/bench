#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.tasks.data_format import load_npz_split_v0


REPORTS = ROOT / "reports"
PLOTS = ROOT / "plots"


def _cache_root() -> Path:
    env = os.environ.get("BENCH_DATA_CACHE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return ROOT / "bench_data_cache"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _metric_float(metrics: Dict[str, Any], key: str) -> float:
    for obj in (metrics, metrics.get("accuracy", {})):
        if isinstance(obj, dict) and key in obj:
            try:
                return float(obj[key])
            except Exception:
                return float("nan")
    return float("nan")


def _run_plan_value(run_plan: Dict[str, Any], key: str) -> Any:
    if key in run_plan:
        return run_plan.get(key)
    nested = run_plan.get("run_plan")
    if isinstance(nested, dict):
        return nested.get(key)
    return None


def _recursive_find(mapping: Any, keys: Sequence[str]) -> Any:
    if isinstance(mapping, dict):
        for key in keys:
            if key in mapping:
                return mapping[key]
        for value in mapping.values():
            found = _recursive_find(value, keys)
            if found is not None:
                return found
    elif isinstance(mapping, list):
        for value in mapping:
            found = _recursive_find(value, keys)
            if found is not None:
                return found
    return None


def _profile_id(metrics: Dict[str, Any], run_plan: Dict[str, Any]) -> str:
    for obj in (metrics.get("scenario_settings", {}), metrics, run_plan):
        found = _recursive_find(obj, ("profile_id", "corruption_profile", "corruption_severity"))
        if found is not None:
            return str(found)
    return ""


def _stats_value(stats: Dict[str, Any], key: str, stat: str = "mean") -> float:
    obj = stats.get("adapter_runtime_stats", {}).get(f"adapter_runtime.{key}")
    if isinstance(obj, dict) and stat in obj:
        try:
            return float(obj[stat])
        except Exception:
            return float("nan")
    return float("nan")


def _residual_norm(stats: Dict[str, Any]) -> float:
    obj = stats.get("residual_stats")
    if isinstance(obj, dict):
        try:
            return float(obj.get("norm", float("nan")))
        except Exception:
            return float("nan")
    return float("nan")


def _corruption_ratio(suite_name: str, task_id: str, scenario_id: str, seed: int) -> Tuple[float, str]:
    path = _cache_root() / suite_name / task_id / f"scenario_{scenario_id}" / f"seed_{seed}" / "test.npz"
    if not path.exists():
        return float("nan"), ""
    loaded = load_npz_split_v0(path)
    corruption = loaded.meta.get("corruption", {}) if isinstance(loaded.meta.get("corruption"), dict) else {}
    stats = corruption.get("stats", {}) if isinstance(corruption.get("stats"), dict) else {}
    try:
        ratio = float(stats.get("total_corruption_to_clean_ratio", float("nan")))
    except Exception:
        ratio = float("nan")
    return ratio, str(corruption.get("profile_id", ""))


def _correction_stats(run_dir: Path) -> Tuple[float, float]:
    dump = run_dir / "diagnostics" / "first_batch_dump.npz"
    if not dump.exists():
        return float("nan"), float("nan")
    try:
        with np.load(dump, allow_pickle=False) as z:
            if not {"adapter_delta_applied_btd", "adapter_y_raw_btd", "adapter_y_enh_btd", "adapter_x_ref_btd"}.issubset(set(z.files)):
                return float("nan"), float("nan")
            delta = np.asarray(z["adapter_delta_applied_btd"], dtype=np.float64)
            y_raw = np.asarray(z["adapter_y_raw_btd"], dtype=np.float64)
            y_enh = np.asarray(z["adapter_y_enh_btd"], dtype=np.float64)
            x_ref = np.asarray(z["adapter_x_ref_btd"], dtype=np.float64)
    except Exception:
        return float("nan"), float("nan")
    corruption = y_raw - x_ref
    denom = float(np.linalg.norm(delta.ravel()) * np.linalg.norm(corruption.ravel()))
    alignment = float(np.dot(delta.ravel(), (-corruption).ravel()) / denom) if denom > 0.0 else float("nan")
    raw_mse = float(np.mean((y_raw - x_ref) ** 2))
    enh_mse = float(np.mean((y_enh - x_ref) ** 2))
    return alignment, raw_mse - enh_mse


def _iter_rows(*, suite_name: str, task_id: str) -> Iterable[Dict[str, Any]]:
    run_root = ROOT / "runs" / suite_name
    for metrics_path in sorted(run_root.glob("**/metrics.json")):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path)
        run_plan = _read_json(run_dir / "run_plan.json")
        ledger = _read_json(run_dir / "budget_ledger.json")
        stats = _read_json(run_dir / "diagnostics" / "stats.json")
        model_id = str(metrics.get("model_id") or _run_plan_value(run_plan, "model_id") or "")
        init_id = str(metrics.get("init_id") or _run_plan_value(run_plan, "init_id") or "trained")
        track_id = str(metrics.get("track_id") or _run_plan_value(run_plan, "track_id") or "frozen")
        scenario_id = str(metrics.get("scenario_id") or _run_plan_value(run_plan, "scenario_id") or run_dir.name.replace("scenario_", ""))
        seed = int(metrics.get("seed", _run_plan_value(run_plan, "seed") or -1))
        profile = _profile_id(metrics, run_plan)
        corr_ratio, meta_profile = _corruption_ratio(suite_name, task_id, scenario_id, seed)
        if not profile:
            profile = meta_profile
        mse = _metric_float(metrics, "mse")
        mse_db = _metric_float(metrics, "mse_db")
        expected_db = 10.0 * math.log10(max(mse, 1.0e-300)) if math.isfinite(mse) and mse > 0 else float("nan")
        correction_alignment, correction_mse_reduction = _correction_stats(run_dir)
        yield {
            "run_dir": str(run_dir),
            "task_id": str(metrics.get("task_id") or task_id),
            "scenario_id": scenario_id,
            "profile_id": profile,
            "seed": seed,
            "model_id": model_id,
            "plan": f"{init_id}:{track_id}",
            "mse": mse,
            "mse_db": mse_db,
            "expected_mse_db": expected_db,
            "db_abs_err": abs(mse_db - expected_db) if math.isfinite(mse_db) and math.isfinite(expected_db) else float("nan"),
            "device_requested": str(metrics.get("device_requested", _run_plan_value(run_plan, "device_requested") or "")),
            "device_resolved": str(metrics.get("device_resolved", _run_plan_value(run_plan, "device_resolved") or "")),
            "train_updates_used": int(ledger.get("train_updates_used", 0) or 0),
            "train_outer_updates_used": int(ledger.get("train_outer_updates_used", ledger.get("train_updates_used", 0)) or 0),
            "enhancer_updates_used": int(ledger.get("enhancer_updates_used", 0) or 0),
            "split_updates_used": int(ledger.get("split_updates_used", 0) or 0),
            "adapt_updates_used": int(ledger.get("adapt_updates_used", -1)),
            "failure_json": (run_dir / "failure.json").exists(),
            "diagnostics_present": (run_dir / "diagnostics" / "stats.json").exists(),
            "residual_norm": _residual_norm(stats),
            "delta_to_raw_ratio_mean": _stats_value(stats, "delta_to_raw_ratio_mean", "mean"),
            "delta_to_raw_ratio_max": _stats_value(stats, "delta_to_raw_ratio_max", "max"),
            "y_enh_to_raw_norm_ratio_mean": _stats_value(stats, "y_enh_to_raw_norm_ratio_mean", "mean"),
            "innovation_collapse_ratio": _stats_value(stats, "innovation_collapse_ratio", "mean"),
            "corruption_to_clean_ratio": corr_ratio,
            "correction_alignment": correction_alignment,
            "correction_mse_reduction": correction_mse_reduction,
        }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _is_valid(row: Dict[str, Any], *, expected_device: str, train_updates: int, models: Sequence[str]) -> Tuple[bool, str]:
    if row["model_id"] not in models:
        return False, "unexpected_model"
    if row["plan"] != "trained:frozen":
        return False, "wrong_plan"
    if expected_device and row["device_resolved"] != expected_device:
        return False, f"not_{expected_device}"
    if row["failure_json"]:
        return False, "failure_json"
    if not (math.isfinite(float(row["db_abs_err"])) and float(row["db_abs_err"]) < 1.0e-8):
        return False, "db_invariant"
    if not row["diagnostics_present"] or not math.isfinite(float(row["residual_norm"])):
        return False, "residual_diagnostics"
    if int(row["adapt_updates_used"]) != 0:
        return False, "adapt_updates_nonzero"
    train_updates_used = int(row["train_updates_used"])
    if train_updates >= 0 and not (0 < train_updates_used <= int(train_updates)):
        return False, "train_updates_out_of_budget"
    if row["model_id"] == "me_split_knet_v0":
        if int(row["enhancer_updates_used"]) + int(row["split_updates_used"]) != train_updates_used:
            return False, "me_update_accounting"
        for key in ("delta_to_raw_ratio_mean", "delta_to_raw_ratio_max", "innovation_collapse_ratio"):
            if not math.isfinite(float(row[key])):
                return False, f"{key}_missing"
    return True, ""


def _summarize(valid: List[Dict[str, Any]], profiles: Sequence[str], models: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in valid:
        grouped[(str(row["profile_id"]), str(row["model_id"]))].append(row)
    out: List[Dict[str, Any]] = []
    for profile in profiles:
        split_rows = grouped.get((profile, "split_knet"), [])
        me_rows = grouped.get((profile, "me_split_knet_v0"), [])
        for model in models:
            rows = grouped.get((profile, model), [])
            vals = [float(r["mse_db"]) for r in rows]
            if vals:
                out.append(
                    {
                        "severity": profile,
                        "model_id": model,
                        "mean_mse_db": mean(vals),
                        "std_mse_db": stdev(vals) if len(vals) > 1 else 0.0,
                        "n": len(vals),
                        "corruption_to_clean_ratio": mean(
                            [float(r["corruption_to_clean_ratio"]) for r in rows if math.isfinite(float(r["corruption_to_clean_ratio"]))]
                            or [float("nan")]
                        ),
                        "delta_to_raw_ratio": mean(
                            [float(r["delta_to_raw_ratio_mean"]) for r in rows if math.isfinite(float(r["delta_to_raw_ratio_mean"]))]
                            or [float("nan")]
                        ),
                        "innovation_ratio": mean(
                            [float(r["innovation_collapse_ratio"]) for r in rows if math.isfinite(float(r["innovation_collapse_ratio"]))]
                            or [float("nan")]
                        ),
                        "correction_alignment": mean(
                            [float(r["correction_alignment"]) for r in rows if math.isfinite(float(r["correction_alignment"]))]
                            or [float("nan")]
                        ),
                        "correction_mse_reduction": mean(
                            [float(r["correction_mse_reduction"]) for r in rows if math.isfinite(float(r["correction_mse_reduction"]))]
                            or [float("nan")]
                        ),
                    }
                )
        if split_rows and me_rows:
            split_mean = mean([float(r["mse_db"]) for r in split_rows])
            me_mean = mean([float(r["mse_db"]) for r in me_rows])
            out.append(
                {
                    "severity": profile,
                    "model_id": "improvement",
                    "mean_mse_db": split_mean - me_mean,
                    "std_mse_db": 0.0,
                    "n": min(len(split_rows), len(me_rows)),
                    "corruption_to_clean_ratio": mean(
                        [float(r["corruption_to_clean_ratio"]) for r in me_rows if math.isfinite(float(r["corruption_to_clean_ratio"]))]
                        or [float("nan")]
                    ),
                    "delta_to_raw_ratio": mean(
                        [float(r["delta_to_raw_ratio_mean"]) for r in me_rows if math.isfinite(float(r["delta_to_raw_ratio_mean"]))]
                        or [float("nan")]
                    ),
                    "innovation_ratio": mean(
                        [float(r["innovation_collapse_ratio"]) for r in me_rows if math.isfinite(float(r["innovation_collapse_ratio"]))]
                        or [float("nan")]
                    ),
                    "correction_alignment": mean(
                        [float(r["correction_alignment"]) for r in me_rows if math.isfinite(float(r["correction_alignment"]))]
                        or [float("nan")]
                    ),
                    "correction_mse_reduction": mean(
                        [float(r["correction_mse_reduction"]) for r in me_rows if math.isfinite(float(r["correction_mse_reduction"]))]
                        or [float("nan")]
                    ),
                }
            )
    return out


def _summary_lookup(summary: List[Dict[str, Any]], profile: str, model_id: str) -> Dict[str, Any]:
    for row in summary:
        if row["severity"] == profile and row["model_id"] == model_id:
            return row
    return {}


def _plot(summary: List[Dict[str, Any]], profiles: Sequence[str], prefix: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    x = list(range(len(profiles)))
    labels = list(profiles)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model in ("split_knet", "me_split_knet_v0"):
        ys = [float(_summary_lookup(summary, p, model).get("mean_mse_db", float("nan"))) for p in profiles]
        yerr = [float(_summary_lookup(summary, p, model).get("std_mse_db", 0.0)) for p in profiles]
        ax.errorbar(x, ys, yerr=yerr, marker="o", capsize=3, label=f"{model} | trained:frozen")
    ax.set_xticks(x, labels)
    ax.set_xlabel("corruption severity")
    ax.set_ylabel("mean mse_db")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / f"{prefix}_mse_db.png", dpi=180)
    fig.savefig(PLOTS / f"{prefix}_mse_db.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    imps = [float(_summary_lookup(summary, p, "improvement").get("mean_mse_db", float("nan"))) for p in profiles]
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.plot(x, imps, marker="o")
    ax.set_xticks(x, labels)
    ax.set_xlabel("corruption severity")
    ax.set_ylabel("improvement_db")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{prefix}_improvement_db.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        x,
        [float(_summary_lookup(summary, p, "improvement").get("delta_to_raw_ratio", float("nan"))) for p in profiles],
        marker="o",
        label="delta_to_raw_ratio",
    )
    ax.plot(
        x,
        [float(_summary_lookup(summary, p, "improvement").get("innovation_ratio", float("nan"))) for p in profiles],
        marker="s",
        label="innovation_ratio",
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel("corruption severity")
    ax.set_ylabel("ratio")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / f"{prefix}_delta_ratio.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        x,
        [float(_summary_lookup(summary, p, "improvement").get("corruption_to_clean_ratio", float("nan"))) for p in profiles],
        marker="o",
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel("corruption severity")
    ax.set_ylabel("corruption_to_clean_ratio")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{prefix}_corruption_ratio.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.plot(
        x,
        [float(_summary_lookup(summary, p, "improvement").get("correction_alignment", float("nan"))) for p in profiles],
        marker="o",
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel("corruption severity")
    ax.set_ylabel("cos(delta_applied, -corruption)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{prefix}_correction_alignment.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--task-id", default="Basilisk_ADCS_structured_corruption_v0")
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--models", nargs="+", default=["split_knet", "me_split_knet_v0"])
    parser.add_argument("--expected-device", default="cuda")
    parser.add_argument("--train-updates", type=int, default=500)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    rows = list(_iter_rows(suite_name=args.suite_name, task_id=args.task_id))
    metrics_csv = REPORTS / f"{args.prefix}_metrics.csv"
    summary_csv = REPORTS / f"{args.prefix}_summary.csv"
    audit_json = REPORTS / f"{args.prefix}_acceptance_audit.json"
    analysis_md = REPORTS / f"{args.prefix}_analysis.md"
    _write_csv(metrics_csv, rows)

    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    for row in rows:
        ok, reason = _is_valid(row, expected_device=args.expected_device, train_updates=args.train_updates, models=args.models)
        if ok:
            valid.append(row)
        else:
            bad = dict(row)
            bad["invalid_reason"] = reason
            invalid.append(bad)

    present = {(str(r["profile_id"]), int(r["seed"]), str(r["model_id"])) for r in valid}
    missing = []
    for profile in args.profiles:
        for seed in args.seeds:
            for model in args.models:
                if (profile, seed, model) not in present:
                    missing.append({"profile_id": profile, "seed": seed, "model_id": model, "reason": "missing_or_invalid"})

    summary = _summarize(valid, args.profiles, args.models)
    _write_csv(summary_csv, summary)
    if not missing:
        _plot(summary, args.profiles, args.prefix)

    ratio_by_profile = {
        p: float(_summary_lookup(summary, p, "improvement").get("corruption_to_clean_ratio", float("nan")))
        for p in args.profiles
    }
    finite_ratios = [ratio_by_profile[p] for p in args.profiles if math.isfinite(ratio_by_profile[p])]
    monotonic = all(a <= b for a, b in zip(finite_ratios, finite_ratios[1:])) if len(finite_ratios) >= 2 else True
    improvements = {p: float(_summary_lookup(summary, p, "improvement").get("mean_mse_db", float("nan"))) for p in args.profiles}
    hard_profiles = [p for p in args.profiles if p in {"moderate", "severe"}]
    hard_positive = any(math.isfinite(improvements.get(p, float("nan"))) and improvements[p] > 0.0 for p in hard_profiles)
    easy_degrade = any(
        math.isfinite(improvements.get(p, float("nan"))) and improvements[p] < -0.1
        for p in args.profiles
        if p in {"clean_gaussian", "mild"}
    )
    if hard_positive and not easy_degrade:
        decision = "A"
        decision_reason = "ME helps structured corruption"
    elif len(valid) == len(args.profiles) * len(args.seeds) * len(args.models):
        decision = "B"
        decision_reason = "ME safe but inconclusive"
    else:
        decision = "C"
        decision_reason = "pilot incomplete or failed"

    audit = {
        "expected_metrics": len(args.profiles) * len(args.seeds) * len(args.models),
        "actual_metrics": len(rows),
        "valid_metrics": len(valid),
        "failure_json_count": sum(1 for r in rows if r.get("failure_json")),
        "max_db_abs_err": max([float(r["db_abs_err"]) for r in valid if math.isfinite(float(r["db_abs_err"]))] or [float("nan")]),
        "residual_diagnostics_finite": all(math.isfinite(float(r["residual_norm"])) for r in valid),
        "enhancer_diagnostics_finite": all(
            math.isfinite(float(r["delta_to_raw_ratio_mean"]))
            for r in valid
            if str(r["model_id"]) == "me_split_knet_v0"
        ),
        "innovation_diagnostics_finite": all(
            math.isfinite(float(r["innovation_collapse_ratio"]))
            for r in valid
            if str(r["model_id"]) == "me_split_knet_v0"
        ),
        "corruption_severity_monotonic": monotonic,
        "missing_or_invalid": missing,
        "invalid_rows": invalid,
        "decision_category": decision,
        "decision_reason": decision_reason,
        "metrics_csv": str(metrics_csv.resolve()),
        "summary_csv": str(summary_csv.resolve()),
        "analysis_md": str(analysis_md.resolve()),
        "plots": {
            "mse_db": str((PLOTS / f"{args.prefix}_mse_db.png").resolve()),
            "mse_db_pdf": str((PLOTS / f"{args.prefix}_mse_db.pdf").resolve()),
            "improvement_db": str((PLOTS / f"{args.prefix}_improvement_db.png").resolve()),
            "delta_ratio": str((PLOTS / f"{args.prefix}_delta_ratio.png").resolve()),
            "corruption_ratio": str((PLOTS / f"{args.prefix}_corruption_ratio.png").resolve()),
            "correction_alignment": str((PLOTS / f"{args.prefix}_correction_alignment.png").resolve()),
        },
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ME-Split-KalmanNet v0 Structured Corruption Analysis",
        "",
        f"- valid metrics: {len(valid)}/{audit['expected_metrics']}",
        f"- failure_json_count: {audit['failure_json_count']}",
        f"- max_db_abs_err: {audit['max_db_abs_err']}",
        f"- corruption_severity_monotonic: {monotonic}",
        f"- decision_category: {decision}",
        f"- decision_reason: {decision_reason}",
        "",
        "## Per-Severity Summary",
        "",
    ]
    for profile in args.profiles:
        split = _summary_lookup(summary, profile, "split_knet")
        me = _summary_lookup(summary, profile, "me_split_knet_v0")
        imp = _summary_lookup(summary, profile, "improvement")
        if split and me and imp:
            lines.append(
                f"- {profile}: split={float(split['mean_mse_db']):.4f} dB, "
                f"ME={float(me['mean_mse_db']):.4f} dB, improvement={float(imp['mean_mse_db']):.4f} dB, "
                f"corruption_ratio={float(imp['corruption_to_clean_ratio']):.4f}, "
                f"delta_ratio={float(imp['delta_to_raw_ratio']):.4f}, "
                f"innovation_ratio={float(imp['innovation_ratio']):.4f}"
            )
    analysis_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if not missing and not invalid and monotonic else 1


if __name__ == "__main__":
    raise SystemExit(main())
