#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "gpu_basilisk_imu_pilot"
REPORTS = ROOT / "reports"
PLOTS = ROOT / "plots"
METRICS_CSV = REPORTS / "basilisk_imu_gpu_pilot_metrics.csv"
SUMMARY_CSV = REPORTS / "basilisk_imu_gpu_pilot_summary.csv"
AUDIT_JSON = REPORTS / "basilisk_imu_gpu_pilot_acceptance_audit.json"
PLOT_MSE = PLOTS / "basilisk_imu_gpu_pilot_mse_db.png"
PLOT_IMPROVEMENT = PLOTS / "basilisk_imu_gpu_pilot_improvement_db.png"
PLOT_DELTA = PLOTS / "basilisk_imu_gpu_pilot_delta_ratio.png"
PLOT_IMU_REDUCTION = PLOTS / "basilisk_imu_gpu_pilot_imu_mse_reduction.png"
PLOT_ALIGNMENT = PLOTS / "basilisk_imu_gpu_pilot_alignment.png"

TASK_ID = "Basilisk_IMU_ADCS_pilot_v0"
PROFILES = ("clean_imu", "noisy_imu", "biased_imu", "low_cost_imu")
MODELS = ("split_knet", "me_split_knet_v0")
EXPECTED_COUNT = len(PROFILES) * len(MODELS)
REQUIRE_IMU_ENHANCER_DIAG = False


def _cache_root() -> Path:
    value = os.environ.get("BENCH_DATA_CACHE", "").strip()
    if value:
        return Path(value).expanduser().resolve()
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


def _profile_from_metrics(metrics: Dict[str, Any]) -> str:
    settings = metrics.get("scenario_settings")
    if isinstance(settings, dict):
        for key in ("imu.profile_id", "profile_id", "imu_severity", "severity"):
            value = settings.get(key)
            if value is not None:
                return str(value)
    basis = metrics.get("scenario_cfg_basis")
    if isinstance(basis, dict):
        imu = basis.get("imu")
        if isinstance(imu, dict) and imu.get("profile_id") is not None:
            return str(imu["profile_id"])
    return ""


def _cache_meta(suite_name: str, task_id: str, scenario_id: str, seed: int) -> Dict[str, Any]:
    path = _cache_root() / suite_name / task_id / f"scenario_{scenario_id}" / f"seed_{seed}" / "test.npz"
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as z:
            return json.loads(str(z["meta_json"]))
    except Exception:
        return {}


def _stats_value(stats: Dict[str, Any], key: str, stat: str = "mean") -> float:
    obj = stats.get("adapter_runtime_stats", {}).get(f"adapter_runtime.{key}")
    if isinstance(obj, dict) and stat in obj:
        try:
            return float(obj[stat])
        except Exception:
            return float("nan")
    return float("nan")


def _residual_finite(stats: Dict[str, Any]) -> bool:
    residual = stats.get("residual_stats")
    return isinstance(residual, dict) and bool(residual.get("finite")) and int(residual.get("nan_count", 1)) == 0 and int(residual.get("inf_count", 1)) == 0


def _residual_norm(stats: Dict[str, Any]) -> float:
    residual = stats.get("residual_stats")
    if isinstance(residual, dict):
        try:
            return float(residual.get("norm", float("nan")))
        except Exception:
            return float("nan")
    return float("nan")


def _iter_rows() -> Iterable[Dict[str, Any]]:
    for metrics_path in sorted(RUN_ROOT.glob(f"{TASK_ID}/**/metrics.json")):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path)
        run_plan = _read_json(run_dir / "run_plan.json")
        ledger = _read_json(run_dir / "budget_ledger.json")
        stats = _read_json(run_dir / "diagnostics" / "stats.json")
        suite_name = str(metrics.get("suite_name") or _run_plan_value(run_plan, "suite_name") or "gpu_basilisk_imu_pilot")
        task_id = str(metrics.get("task_id") or _run_plan_value(run_plan, "task_id") or TASK_ID)
        scenario_id = str(metrics.get("scenario_id") or _run_plan_value(run_plan, "scenario_id") or run_dir.name.replace("scenario_", ""))
        seed = int(metrics.get("seed", _run_plan_value(run_plan, "seed") or -1))
        cache_meta = _cache_meta(suite_name, task_id, scenario_id, seed)
        imu_meta = cache_meta.get("imu", {}) if isinstance(cache_meta.get("imu"), dict) else {}
        profile = _profile_from_metrics(metrics) or str(imu_meta.get("profile_id") or "")
        model_id = str(metrics.get("model_id") or _run_plan_value(run_plan, "model_id") or "")
        init_id = str(metrics.get("init_id") or _run_plan_value(run_plan, "init_id") or "trained")
        track_id = str(metrics.get("track_id") or _run_plan_value(run_plan, "track_id") or "frozen")
        dims = metrics.get("dims") if isinstance(metrics.get("dims"), dict) else {}
        mse = _metric_float(metrics, "mse")
        mse_db = _metric_float(metrics, "mse_db")
        expected_db = 10.0 * math.log10(max(mse, 1.0e-300)) if math.isfinite(mse) and mse > 0.0 else float("nan")
        fake_marker = bool(cache_meta.get("fake_marker")) if cache_meta else False
        if cache_meta:
            meta_text = json.dumps(cache_meta, sort_keys=True).lower()
            fake_marker = fake_marker or ("synthetic fallback" in meta_text) or ("fake_data" in meta_text)
        yield {
            "run_dir": str(run_dir),
            "suite_name": suite_name,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "profile_id": profile,
            "seed": seed,
            "model_id": model_id,
            "plan": f"{init_id}:{track_id}",
            "x_dim": int(dims.get("x_dim", cache_meta.get("dims", {}).get("x_dim", -1) if isinstance(cache_meta.get("dims"), dict) else -1)),
            "y_dim": int(dims.get("y_dim", cache_meta.get("dims", {}).get("y_dim", -1) if isinstance(cache_meta.get("dims"), dict) else -1)),
            "mse": mse,
            "mse_db": mse_db,
            "expected_mse_db": expected_db,
            "db_abs_err": abs(mse_db - expected_db) if math.isfinite(mse_db) and math.isfinite(expected_db) else float("nan"),
            "device_requested": str(metrics.get("run_plan", {}).get("device_requested") or _run_plan_value(run_plan, "device_requested") or ""),
            "device_resolved": str(metrics.get("run_plan", {}).get("device_resolved") or _run_plan_value(run_plan, "device_resolved") or ""),
            "train_updates_used": int(ledger.get("train_updates_used", metrics.get("budgets", {}).get("train_updates_used", 0)) or 0),
            "train_outer_updates_used": int(ledger.get("train_outer_updates_used", metrics.get("budgets", {}).get("train_outer_updates_used", 0)) or 0),
            "enhancer_updates_used": int(ledger.get("enhancer_updates_used", metrics.get("budgets", {}).get("enhancer_updates_used", 0)) or 0),
            "split_updates_used": int(ledger.get("split_updates_used", metrics.get("budgets", {}).get("split_updates_used", 0)) or 0),
            "adapt_updates_used": int(ledger.get("adapt_updates_used", metrics.get("budgets", {}).get("adapt_updates_used", -1)) or 0),
            "failure_json": (run_dir / "failure.json").exists(),
            "diagnostics_present": (run_dir / "diagnostics" / "stats.json").exists(),
            "residual_finite": _residual_finite(stats),
            "residual_norm": _residual_norm(stats),
            "delta_to_raw_ratio_mean": _stats_value(stats, "delta_to_raw_ratio_mean", "mean"),
            "delta_to_raw_ratio_max": _stats_value(stats, "delta_to_raw_ratio_max", "max"),
            "y_enh_to_raw_norm_ratio_mean": _stats_value(stats, "y_enh_to_raw_norm_ratio_mean", "mean"),
            "innovation_collapse_ratio": _stats_value(stats, "innovation_collapse_ratio", "mean"),
            "imu_y_raw_to_clean_mse": _stats_value(stats, "imu_y_raw_to_clean_mse", "mean"),
            "imu_y_enh_to_clean_mse": _stats_value(stats, "imu_y_enh_to_clean_mse", "mean"),
            "imu_mse_reduction": _stats_value(stats, "imu_mse_reduction", "mean"),
            "imu_correction_alignment": _stats_value(stats, "imu_correction_alignment", "mean"),
            "delta_to_imu_error_ratio_mean": _stats_value(stats, "delta_to_imu_error_ratio_mean", "mean"),
            "imu_error_to_clean_ratio": float(imu_meta.get("stats", {}).get("imu_error_to_clean_ratio", float("nan"))) if isinstance(imu_meta.get("stats"), dict) else float("nan"),
            "measurement_mode": str(imu_meta.get("measurement_mode") or ""),
            "fake_marker": fake_marker,
        }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _is_valid(row: Dict[str, Any]) -> Tuple[bool, str]:
    if row["model_id"] not in MODELS:
        return False, "unexpected_model"
    if row["plan"] != "trained:frozen":
        return False, "wrong_plan"
    if row["device_resolved"] != "cuda":
        return False, "not_cuda"
    if row["failure_json"]:
        return False, "failure_json"
    if int(row["train_updates_used"]) != 500:
        return False, "train_updates_not_500"
    if int(row["adapt_updates_used"]) != 0:
        return False, "adapt_updates_nonzero"
    if not (math.isfinite(float(row["db_abs_err"])) and float(row["db_abs_err"]) < 1.0e-8):
        return False, "db_invariant"
    if not row["diagnostics_present"] or not row["residual_finite"]:
        return False, "residual_diagnostics"
    if row["fake_marker"]:
        return False, "fake_marker"
    if row["model_id"] == "me_split_knet_v0":
        for key in ("delta_to_raw_ratio_mean", "delta_to_raw_ratio_max", "innovation_collapse_ratio"):
            if not math.isfinite(float(row[key])):
                return False, f"{key}_missing"
        if REQUIRE_IMU_ENHANCER_DIAG:
            for key in ("imu_y_raw_to_clean_mse", "imu_y_enh_to_clean_mse", "imu_mse_reduction", "imu_correction_alignment"):
                if not math.isfinite(float(row[key])):
                    return False, f"{key}_missing"
        if int(row["enhancer_updates_used"]) + int(row["split_updates_used"]) != int(row["train_updates_used"]):
            return False, "me_update_accounting"
    return True, ""


def _summarize(valid: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in valid:
        grouped[(str(row["profile_id"]), str(row["model_id"]))].append(row)
    summary: List[Dict[str, Any]] = []
    split_means: Dict[str, float] = {}
    for profile in PROFILES:
        split_vals = [float(r["mse_db"]) for r in grouped.get((profile, "split_knet"), [])]
        if split_vals:
            split_means[profile] = mean(split_vals)
        for model in MODELS:
            rows = grouped.get((profile, model), [])
            vals = [float(r["mse_db"]) for r in rows]
            if not vals:
                continue
            delta_vals = [float(r["delta_to_raw_ratio_mean"]) for r in rows if math.isfinite(float(r["delta_to_raw_ratio_mean"]))]
            innov_vals = [float(r["innovation_collapse_ratio"]) for r in rows if math.isfinite(float(r["innovation_collapse_ratio"]))]
            imu_reduction_vals = [float(r["imu_mse_reduction"]) for r in rows if math.isfinite(float(r["imu_mse_reduction"]))]
            alignment_vals = [float(r["imu_correction_alignment"]) for r in rows if math.isfinite(float(r["imu_correction_alignment"]))]
            summary.append(
                {
                    "severity": profile,
                    "model_id": model,
                    "plan": "trained:frozen",
                    "mean_mse_db": mean(vals),
                    "std_mse_db": stdev(vals) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                    "y_dim": int(rows[0]["y_dim"]) if rows else -1,
                    "delta_to_raw_ratio": mean(delta_vals) if delta_vals else float("nan"),
                    "innovation_ratio": mean(innov_vals) if innov_vals else float("nan"),
                    "imu_mse_reduction": mean(imu_reduction_vals) if imu_reduction_vals else float("nan"),
                    "correction_alignment": mean(alignment_vals) if alignment_vals else float("nan"),
                    "imu_error_to_clean_ratio": mean(
                        [float(r["imu_error_to_clean_ratio"]) for r in rows if math.isfinite(float(r["imu_error_to_clean_ratio"]))]
                        or [float("nan")]
                    ),
                    "improvement_db": "",
                }
            )
    for row in summary:
        if row["model_id"] == "me_split_knet_v0" and row["severity"] in split_means:
            row["improvement_db"] = split_means[str(row["severity"])] - float(row["mean_mse_db"])
    return summary


def _plot(summary: List[Dict[str, Any]]) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    x = list(range(len(PROFILES)))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for model in MODELS:
        rows = {str(r["severity"]): r for r in summary if r["model_id"] == model}
        vals = [float(rows[p]["mean_mse_db"]) if p in rows else float("nan") for p in PROFILES]
        errs = [float(rows[p]["std_mse_db"]) if p in rows else 0.0 for p in PROFILES]
        ax.errorbar(x, vals, yerr=errs, marker="o", capsize=3, label=f"{model} | trained:frozen")
    ax.set_xticks(x)
    ax.set_xticklabels(PROFILES, rotation=20, ha="right")
    ax.set_ylabel("mean mse_db")
    ax.set_xlabel("IMU severity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_MSE, dpi=180)
    plt.close(fig)

    me_rows = {str(r["severity"]): r for r in summary if r["model_id"] == "me_split_knet_v0"}
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    improvements = [float(me_rows[p]["improvement_db"]) if p in me_rows and me_rows[p]["improvement_db"] != "" else float("nan") for p in PROFILES]
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.plot(x, improvements, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(PROFILES, rotation=20, ha="right")
    ax.set_ylabel("split_knet - me_split_knet_v0 (dB)")
    ax.set_xlabel("IMU severity")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_IMPROVEMENT, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    delta_vals = [float(me_rows[p]["delta_to_raw_ratio"]) if p in me_rows else float("nan") for p in PROFILES]
    innov_vals = [float(me_rows[p]["innovation_ratio"]) if p in me_rows else float("nan") for p in PROFILES]
    ax.plot(x, delta_vals, marker="o", label="delta_to_raw_ratio")
    ax.plot(x, innov_vals, marker="s", label="innovation_ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(PROFILES, rotation=20, ha="right")
    ax.set_ylabel("ratio")
    ax.set_xlabel("IMU severity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DELTA, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    reduction_vals = [float(me_rows[p]["imu_mse_reduction"]) if p in me_rows else float("nan") for p in PROFILES]
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.plot(x, reduction_vals, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(PROFILES, rotation=20, ha="right")
    ax.set_ylabel("MSE(y_raw, y_clean) - MSE(y_enh, y_clean)")
    ax.set_xlabel("IMU severity")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_IMU_REDUCTION, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    alignment_vals = [float(me_rows[p]["correction_alignment"]) if p in me_rows else float("nan") for p in PROFILES]
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.plot(x, alignment_vals, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(PROFILES, rotation=20, ha="right")
    ax.set_ylabel("cos(delta_applied, -imu_error)")
    ax.set_xlabel("IMU severity")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_ALIGNMENT, dpi=180)
    plt.close(fig)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = list(_iter_rows())
    for row in rows:
        valid, reason = _is_valid(row)
        row["run_valid"] = valid
        row["invalid_reason"] = reason
    valid_rows = [r for r in rows if r["run_valid"]]
    summary = _summarize(valid_rows)
    _write_csv(METRICS_CSV, rows)
    _write_csv(SUMMARY_CSV, summary)
    _plot(summary)
    db_errors = [float(r["db_abs_err"]) for r in rows if math.isfinite(float(r["db_abs_err"]))]
    acceptance = {
        "expected_metrics_count": EXPECTED_COUNT,
        "metrics_found_count": len(rows),
        "valid_cuda_metrics_count": len(valid_rows),
        "complete": len(valid_rows) == EXPECTED_COUNT,
        "failure_json_count": sum(1 for r in rows if r["failure_json"]),
        "max_mse_db_abs_err": max(db_errors) if db_errors else None,
        "mse_db_invariant_ok": bool(db_errors) and max(db_errors) < 1.0e-8,
        "residual_diagnostics_finite": all(bool(r["residual_finite"]) for r in valid_rows) if valid_rows else False,
        "enhancer_diagnostics_finite": all(
            math.isfinite(float(r["delta_to_raw_ratio_mean"])) and math.isfinite(float(r["innovation_collapse_ratio"]))
            for r in valid_rows
            if r["model_id"] == "me_split_knet_v0"
        ),
        "imu_enhancer_diagnostics_finite": all(
            math.isfinite(float(r["imu_mse_reduction"])) and math.isfinite(float(r["imu_correction_alignment"]))
            for r in valid_rows
            if r["model_id"] == "me_split_knet_v0"
        ),
        "device_resolved_cuda": all(r["device_resolved"] == "cuda" for r in valid_rows) if valid_rows else False,
        "fake_marker_present": any(r["fake_marker"] for r in rows),
        "missing_or_invalid": [
            {
                "severity": r["profile_id"],
                "seed": r["seed"],
                "model_id": r["model_id"],
                "reason": r["invalid_reason"],
                "run_dir": r["run_dir"],
            }
            for r in rows
            if not r["run_valid"]
        ],
    }
    AUDIT_JSON.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    return 0 if acceptance["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
