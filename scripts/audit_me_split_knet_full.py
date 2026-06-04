#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "gpu_basilisk_me_split_full"
REPORTS = ROOT / "reports"
PLOTS = ROOT / "plots"
METRICS_CSV = REPORTS / "me_split_knet_full_metrics.csv"
SUMMARY_CSV = REPORTS / "me_split_knet_full_summary.csv"
AUDIT_JSON = REPORTS / "me_split_knet_full_acceptance_audit.json"
ANALYSIS_MD = REPORTS / "me_split_knet_full_analysis.md"
PLOT_MSE_PNG = PLOTS / "me_split_knet_full_mse_db.png"
PLOT_MSE_PDF = PLOTS / "me_split_knet_full_mse_db.pdf"
PLOT_IMP = PLOTS / "me_split_knet_full_improvement_db.png"
PLOT_DELTA = PLOTS / "me_split_knet_full_delta_ratio.png"

SEVERITIES = [-10.0, 0.0, 10.0, 20.0, 30.0]
SEEDS = [0, 1, 2]
MODELS = ["split_knet", "me_split_knet_v0"]


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


def _extract_severity(run_dir: Path, metrics: Dict[str, Any], run_plan: Dict[str, Any]) -> float:
    settings = metrics.get("scenario_settings", {})
    if isinstance(settings, dict) and "sensor_noise_scale_db" in settings:
        try:
            return float(settings["sensor_noise_scale_db"])
        except Exception:
            pass
    scenario = run_plan.get("scenario", {}) if isinstance(run_plan.get("scenario"), dict) else {}
    settings = scenario.get("settings", {}) if isinstance(scenario.get("settings"), dict) else {}
    if "sensor_noise_scale_db" in settings:
        try:
            return float(settings["sensor_noise_scale_db"])
        except Exception:
            pass
    return float("nan")


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


def _iter_rows() -> Iterable[Dict[str, Any]]:
    for metrics_path in sorted(RUN_ROOT.glob("**/metrics.json")):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path)
        run_plan = _read_json(run_dir / "run_plan.json")
        ledger = _read_json(run_dir / "budget_ledger.json")
        stats = _read_json(run_dir / "diagnostics" / "stats.json")
        model_id = str(metrics.get("model_id") or _run_plan_value(run_plan, "model_id") or "")
        init_id = str(metrics.get("init_id") or _run_plan_value(run_plan, "init_id") or "trained")
        track_id = str(metrics.get("track_id") or _run_plan_value(run_plan, "track_id") or "frozen")
        mse = _metric_float(metrics, "mse")
        mse_db = _metric_float(metrics, "mse_db")
        expected_db = 10.0 * math.log10(max(mse, 1.0e-300)) if math.isfinite(mse) and mse > 0 else float("nan")
        yield {
            "run_dir": str(run_dir),
            "sensor_noise_scale_db": _extract_severity(run_dir, metrics, run_plan),
            "seed": int(metrics.get("seed", _run_plan_value(run_plan, "seed") or -1)),
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
            "residual_norm_mean": _residual_norm(stats),
            "delta_norm_mean": _stats_value(stats, "delta_norm_mean", "mean"),
            "delta_to_raw_ratio_mean": _stats_value(stats, "delta_to_raw_ratio_mean", "mean"),
            "delta_to_raw_ratio_max": _stats_value(stats, "delta_to_raw_ratio_max", "max"),
            "y_enh_to_raw_norm_ratio_mean": _stats_value(stats, "y_enh_to_raw_norm_ratio_mean", "mean"),
            "innovation_collapse_ratio": _stats_value(stats, "innovation_collapse_ratio", "mean"),
            "diagnostics_present": (run_dir / "diagnostics" / "stats.json").exists(),
        }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _is_valid(row: Dict[str, Any]) -> Tuple[bool, str]:
    if row["model_id"] not in MODELS:
        return False, "non_requested_model"
    if row["plan"] != "trained:frozen":
        return False, "wrong_plan"
    if row["device_resolved"] != "cuda":
        return False, "not_cuda"
    if row["failure_json"]:
        return False, "failure_json_present"
    if not (math.isfinite(float(row["db_abs_err"])) and float(row["db_abs_err"]) < 1.0e-8):
        return False, "db_invariant"
    if not row["diagnostics_present"] or not math.isfinite(float(row["residual_norm_mean"])):
        return False, "residual_diagnostics"
    if int(row["adapt_updates_used"]) != 0:
        return False, "adapt_updates_nonzero"
    if int(row["train_updates_used"]) != 500 or int(row["train_outer_updates_used"]) != 500:
        return False, "train_budget_mismatch"
    if row["model_id"] == "me_split_knet_v0":
        if int(row["enhancer_updates_used"]) + int(row["split_updates_used"]) != 500:
            return False, "me_update_accounting"
        for key in ("delta_to_raw_ratio_mean", "delta_to_raw_ratio_max", "y_enh_to_raw_norm_ratio_mean", "innovation_collapse_ratio"):
            if not math.isfinite(float(row[key])):
                return False, f"{key}_missing"
    return True, ""


def _strict_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid = []
    invalid = []
    for row in rows:
        ok, reason = _is_valid(row)
        if ok:
            valid.append(row)
        else:
            bad = dict(row)
            bad["invalid_reason"] = reason
            invalid.append(bad)
    return valid, invalid


def _missing(valid: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    present = {
        (float(r["sensor_noise_scale_db"]), int(r["seed"]), str(r["model_id"]))
        for r in valid
    }
    out = []
    for sev in SEVERITIES:
        for seed in SEEDS:
            for model_id in MODELS:
                if (sev, seed, model_id) not in present:
                    out.append({"sensor_noise_scale_db": sev, "seed": seed, "model_id": model_id, "reason": "missing_or_invalid"})
    return out


def _summary(valid: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[float, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in valid:
        grouped[(float(row["sensor_noise_scale_db"]), str(row["model_id"]))].append(row)

    out: List[Dict[str, Any]] = []
    for sev in SEVERITIES:
        split_rows = grouped.get((sev, "split_knet"), [])
        me_rows = grouped.get((sev, "me_split_knet_v0"), [])
        split_vals = [float(r["mse_db"]) for r in split_rows]
        me_vals = [float(r["mse_db"]) for r in me_rows]
        if not split_vals or not me_vals:
            continue
        me_delta = [float(r["delta_to_raw_ratio_mean"]) for r in me_rows]
        me_delta_max = [float(r["delta_to_raw_ratio_max"]) for r in me_rows]
        me_y_ratio = [float(r["y_enh_to_raw_norm_ratio_mean"]) for r in me_rows]
        me_innov = [float(r["innovation_collapse_ratio"]) for r in me_rows]
        residuals = [float(r["residual_norm_mean"]) for r in me_rows]
        split_mean = mean(split_vals)
        me_mean = mean(me_vals)
        out.append(
            {
                "sensor_noise_scale_db": sev,
                "split_mean_db": split_mean,
                "split_std_db": stdev(split_vals) if len(split_vals) > 1 else 0.0,
                "me_mean_db": me_mean,
                "me_std_db": stdev(me_vals) if len(me_vals) > 1 else 0.0,
                "improvement_db": split_mean - me_mean,
                "n": min(len(split_vals), len(me_vals)),
                "delta_to_raw_ratio_mean": mean(me_delta),
                "delta_to_raw_ratio_max": max(me_delta_max),
                "y_enh_to_raw_norm_ratio_mean": mean(me_y_ratio),
                "innovation_collapse_ratio": mean(me_innov),
                "residual_norm_mean": mean(residuals),
            }
        )
    return out


def _plot(summary: List[Dict[str, Any]]) -> None:
    if not summary:
        return
    PLOTS.mkdir(parents=True, exist_ok=True)
    xs = [float(r["sensor_noise_scale_db"]) for r in summary]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, [float(r["split_mean_db"]) for r in summary], yerr=[float(r["split_std_db"]) for r in summary], marker="o", capsize=3, label="split_knet | trained:frozen")
    ax.errorbar(xs, [float(r["me_mean_db"]) for r in summary], yerr=[float(r["me_std_db"]) for r in summary], marker="o", capsize=3, label="me_split_knet_v0 | trained:frozen")
    ax.set_xlabel("sensor_noise_scale_db")
    ax.set_ylabel("mean mse_db")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_MSE_PNG, dpi=180)
    fig.savefig(PLOT_MSE_PDF)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.plot(xs, [float(r["improvement_db"]) for r in summary], marker="o")
    ax.set_xlabel("sensor_noise_scale_db")
    ax.set_ylabel("improvement_db")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_IMP, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, [float(r["delta_to_raw_ratio_mean"]) for r in summary], marker="o", label="mean")
    ax.plot(xs, [float(r["delta_to_raw_ratio_max"]) for r in summary], marker="o", label="max")
    ax.axhline(0.25, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("sensor_noise_scale_db")
    ax.set_ylabel("delta_to_raw_ratio")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DELTA, dpi=180)
    plt.close(fig)


def _decision(summary: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not summary:
        return "incomplete", "Strict-valid metrics are incomplete."
    by_sev = {float(r["sensor_noise_scale_db"]): r for r in summary}
    easy_ok = all(float(by_sev[s]["improvement_db"]) >= -0.1 for s in (-10.0, 0.0, 10.0) if s in by_sev)
    hard_positive = any(float(by_sev[s]["improvement_db"]) > 0.0 for s in (20.0, 30.0) if s in by_sev)
    hard_material = any(float(by_sev[s]["improvement_db"]) >= 0.1 for s in (20.0, 30.0) if s in by_sev)
    safety_ok = all(
        float(r["delta_to_raw_ratio_mean"]) <= 0.25
        and float(r["y_enh_to_raw_norm_ratio_mean"]) >= 0.5
        and float(r["innovation_collapse_ratio"]) >= 0.5
        for r in summary
    )
    consistent_degrade = all(float(r["improvement_db"]) < 0.0 for r in summary)
    if safety_ok and easy_ok and hard_positive and hard_material:
        return "A", "demonstrated v0 benefit"
    if safety_ok and easy_ok and hard_positive:
        return "B", "marginal / inconclusive: hard-severity improvement is positive but below 0.1 dB"
    if consistent_degrade or not safety_ok:
        return "C", "failed: degradation or unsafe diagnostics"
    return "B", "marginal / inconclusive: mixed signs or small effects"


def _analysis(summary: List[Dict[str, Any]], audit: Dict[str, Any], category: str, reason: str) -> None:
    lines = [
        "# ME-Split-KalmanNet v0 Full Pilot Analysis",
        "",
        f"- valid CUDA metrics: {audit['valid_cuda_metrics']}/{audit['expected_metrics']}",
        f"- failure_json_count: {audit['failure_json_count']}",
        f"- max_db_abs_err: {audit['max_db_abs_err']}",
        f"- decision_category: {category}",
        f"- decision_reason: {reason}",
        "",
        "## Per-Severity Summary",
        "",
    ]
    for row in summary:
        lines.append(
            f"- severity={row['sensor_noise_scale_db']}: split={float(row['split_mean_db']):.4f} dB, "
            f"ME={float(row['me_mean_db']):.4f} dB, improvement={float(row['improvement_db']):.4f} dB, "
            f"delta_ratio_mean={float(row['delta_to_raw_ratio_mean']):.4f}, "
            f"innovation_ratio={float(row['innovation_collapse_ratio']):.4f}"
        )
    if category == "A":
        lines.extend(["", "## Recommended Next Action", "", "- Add basilisk_mrp_ekf and kalmannet_tsp to a final comparison, then design the structured low-cost IMU corruption benchmark."])
    elif category == "B":
        lines.extend(["", "## Recommended Next Action", "", "- Do not claim success yet. Treat this as a marginal v0 signal and move to a structured corruption benchmark before G2 modification."])
    elif category == "C":
        lines.extend(["", "## Recommended Next Action", "", "- Stop v0 for the Gaussian-only task and add structured corruption before more ME model work."])
    ANALYSIS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = list(_iter_rows())
    _write_csv(METRICS_CSV, rows)
    valid, invalid = _strict_rows(rows)
    missing = _missing(valid)
    summary = _summary(valid)
    _write_csv(SUMMARY_CSV, summary)
    if not missing:
        _plot(summary)
    category, reason = _decision(summary if not missing else [])
    expected = len(SEVERITIES) * len(SEEDS) * len(MODELS)
    db_errors = [float(r["db_abs_err"]) for r in rows if math.isfinite(float(r["db_abs_err"]))]
    audit = {
        "expected_metrics": expected,
        "actual_metrics": len(rows),
        "valid_cuda_metrics": len(valid),
        "failure_json_count": sum(1 for r in rows if bool(r["failure_json"])),
        "max_db_abs_err": max(db_errors) if db_errors else float("nan"),
        "residual_diagnostics_finite": all(math.isfinite(float(r["residual_norm_mean"])) for r in valid),
        "enhancer_diagnostics_finite": all(
            math.isfinite(float(r["delta_to_raw_ratio_mean"]))
            and math.isfinite(float(r["y_enh_to_raw_norm_ratio_mean"]))
            for r in valid
            if r["model_id"] == "me_split_knet_v0"
        ),
        "innovation_diagnostics_finite": all(
            math.isfinite(float(r["innovation_collapse_ratio"]))
            for r in valid
            if r["model_id"] == "me_split_knet_v0"
        ),
        "device_resolved_cuda_count": sum(1 for r in rows if str(r["device_resolved"]) == "cuda"),
        "missing_or_invalid": missing,
        "invalid_rows": invalid,
        "decision_category": category,
        "decision_reason": reason,
        "metrics_csv": str(METRICS_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "analysis_md": str(ANALYSIS_MD),
        "plots": {
            "mse_db_png": str(PLOT_MSE_PNG),
            "mse_db_pdf": str(PLOT_MSE_PDF),
            "improvement_db": str(PLOT_IMP),
            "delta_ratio": str(PLOT_DELTA),
        },
    }
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    _analysis(summary, audit, category, reason)
    print(json.dumps({k: v for k, v in audit.items() if k not in {"invalid_rows"}}, indent=2))
    return 0 if not missing and len(valid) == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
