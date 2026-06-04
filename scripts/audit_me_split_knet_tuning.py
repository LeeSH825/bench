#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "gpu_basilisk_me_split_tuning"
REPORTS = ROOT / "reports"
PLOTS = ROOT / "plots"
METRICS_CSV = REPORTS / "me_split_knet_tuning_metrics.csv"
SUMMARY_CSV = REPORTS / "me_split_knet_tuning_summary.csv"
AUDIT_JSON = REPORTS / "me_split_knet_tuning_acceptance_audit.json"
ANALYSIS_MD = REPORTS / "me_split_knet_tuning_analysis.md"
PLOT_MSE = PLOTS / "me_split_knet_tuning_mse_db.png"
PLOT_DELTA = PLOTS / "me_split_knet_tuning_delta_ratio.png"
PLOT_INNOV = PLOTS / "me_split_knet_tuning_innovation_ratio.png"

SETTING_MAP = {
    "split_knet": "split_knet_raw",
    "me_split_knet_v0_ds100": "A_baseline_old",
    "me_split_knet_v0_ds025": "B_weak_correction",
    "me_split_knet_v0_ds010": "C_very_weak_correction",
    "me_split_knet_v0_small": "D_small_enhancer",
    "me_split_knet_v0_regstrong": "E_strong_regularization",
    "me_split_knet_v0_clip025": "F_clipped_correction",
}


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


def _extract_severity(run_dir: Path, metrics: Dict[str, Any]) -> float:
    settings = metrics.get("scenario_settings", {})
    if isinstance(settings, dict):
        for key in ("sensor_noise_scale_db", "severity", "invR2db"):
            if key in settings:
                try:
                    return float(settings[key])
                except Exception:
                    pass
    run_plan = _read_json(run_dir / "run_plan.json")
    scenario = run_plan.get("scenario", {}) if isinstance(run_plan.get("scenario"), dict) else {}
    settings = scenario.get("settings", {}) if isinstance(scenario.get("settings"), dict) else {}
    for key in ("sensor_noise_scale_db", "severity", "invR2db"):
        if key in settings:
            try:
                return float(settings[key])
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
            "sensor_noise_scale_db": _extract_severity(run_dir, metrics),
            "seed": int(metrics.get("seed", _run_plan_value(run_plan, "seed") or -1)),
            "model_id": model_id,
            "setting": SETTING_MAP.get(model_id, model_id),
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
            "innovation_raw_norm_mean": _stats_value(stats, "innovation_raw_norm_mean", "mean"),
            "innovation_enh_norm_mean": _stats_value(stats, "innovation_enh_norm_mean", "mean"),
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


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    split = {
        float(r["sensor_noise_scale_db"]): float(r["mse_db"])
        for r in rows
        if str(r["model_id"]) == "split_knet" and math.isfinite(float(r["sensor_noise_scale_db"]))
    }
    out: List[Dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: (float(x["sensor_noise_scale_db"]), str(x["setting"]))):
        sev = float(r["sensor_noise_scale_db"])
        base = split.get(sev, float("nan"))
        improvement = base - float(r["mse_db"]) if str(r["model_id"]) != "split_knet" and math.isfinite(base) else ""
        classification = ""
        if str(r["model_id"]) != "split_knet":
            ratio = float(r["delta_to_raw_ratio_mean"])
            enh_ratio = float(r["y_enh_to_raw_norm_ratio_mean"])
            innov_ratio = float(r["innovation_collapse_ratio"])
            imp = float(improvement)
            if (enh_ratio < 0.5 or innov_ratio < 0.5) and imp <= 0:
                classification = "over_smoothing"
            elif ratio < 0.02 and imp <= 0:
                classification = "too_weak"
            elif imp > 0 and ratio <= 0.25 and innov_ratio >= 0.5:
                classification = "promising"
            else:
                classification = "neutral"
        out.append(
            {
                "sensor_noise_scale_db": sev,
                "setting": str(r["setting"]),
                "model_id": str(r["model_id"]),
                "split_mse_db": base if str(r["model_id"]) != "split_knet" else "",
                "me_mse_db": float(r["mse_db"]) if str(r["model_id"]) != "split_knet" else "",
                "split_raw_mse_db": float(r["mse_db"]) if str(r["model_id"]) == "split_knet" else "",
                "improvement_db": improvement,
                "delta_norm_mean": r["delta_norm_mean"],
                "delta_to_raw_ratio_mean": r["delta_to_raw_ratio_mean"],
                "delta_to_raw_ratio_max": r["delta_to_raw_ratio_max"],
                "y_enh_to_raw_norm_ratio_mean": r["y_enh_to_raw_norm_ratio_mean"],
                "innovation_collapse_ratio": r["innovation_collapse_ratio"],
                "residual_norm_mean": r["residual_norm_mean"],
                "classification": classification,
            }
        )
    return out


def _plot(summary: List[Dict[str, Any]]) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    me_rows = [r for r in summary if r["model_id"] != "split_knet"]
    split_rows = [r for r in summary if r["model_id"] == "split_knet"]

    fig, ax = plt.subplots(figsize=(9, 5))
    split_by_sev = {float(r["sensor_noise_scale_db"]): float(r["split_raw_mse_db"]) for r in split_rows}
    ax.plot(sorted(split_by_sev), [split_by_sev[s] for s in sorted(split_by_sev)], marker="o", label="split_knet_raw")
    for setting in sorted({str(r["setting"]) for r in me_rows}):
        rows = sorted([r for r in me_rows if str(r["setting"]) == setting], key=lambda r: float(r["sensor_noise_scale_db"]))
        ax.plot([float(r["sensor_noise_scale_db"]) for r in rows], [float(r["me_mse_db"]) for r in rows], marker="o", label=setting)
    ax.set_xlabel("sensor_noise_scale_db")
    ax.set_ylabel("mse_db")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_MSE, dpi=180)
    plt.close(fig)

    for path, key, ylabel in [
        (PLOT_DELTA, "delta_to_raw_ratio_mean", "delta_to_raw_ratio_mean"),
        (PLOT_INNOV, "innovation_collapse_ratio", "innovation_collapse_ratio"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for setting in sorted({str(r["setting"]) for r in me_rows}):
            rows = sorted([r for r in me_rows if str(r["setting"]) == setting], key=lambda r: float(r["sensor_noise_scale_db"]))
            ax.plot([float(r["sensor_noise_scale_db"]) for r in rows], [float(r[key]) for r in rows], marker="o", label=setting)
        ax.set_xlabel("sensor_noise_scale_db")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def _analysis(summary: List[Dict[str, Any]], audit: Dict[str, Any]) -> None:
    me_rows = [r for r in summary if r["model_id"] != "split_knet"]
    by_setting: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in me_rows:
        by_setting[str(row["setting"])].append(row)

    candidates: List[Dict[str, Any]] = []
    for setting, rows in by_setting.items():
        by_sev = {float(r["sensor_noise_scale_db"]): r for r in rows}
        if not all(sev in by_sev for sev in (0.0, 20.0, 30.0)):
            continue
        imp0 = float(by_sev[0.0]["improvement_db"])
        imp20 = float(by_sev[20.0]["improvement_db"])
        imp30 = float(by_sev[30.0]["improvement_db"])
        max_ratio = max(float(r["delta_to_raw_ratio_mean"]) for r in rows)
        min_innov = min(float(r["innovation_collapse_ratio"]) for r in rows)
        mean_imp = (imp0 + imp20 + imp30) / 3.0
        hard_imp = max(imp20, imp30)
        qualifies = (
            hard_imp > 0.0
            and imp0 >= -0.1
            and max_ratio <= 0.25
            and min_innov >= 0.5
        )
        candidates.append(
            {
                "setting": setting,
                "qualifies": qualifies,
                "mean_improvement_db": mean_imp,
                "hard_improvement_db": hard_imp,
                "improvement_0_db": imp0,
                "improvement_20_db": imp20,
                "improvement_30_db": imp30,
                "max_delta_to_raw_ratio": max_ratio,
                "min_innovation_collapse_ratio": min_innov,
            }
        )

    qualified = [c for c in candidates if c["qualifies"]]
    best = max(
        qualified,
        key=lambda c: (float(c["hard_improvement_db"]), float(c["mean_improvement_db"])),
        default=None,
    )
    lines = [
        "# ME-Split-KalmanNet v0 Tuning Analysis",
        "",
        f"- valid CUDA metrics: {audit['valid_cuda_metrics']}/{audit['expected_metrics']}",
        f"- failure_json_count: {audit['failure_json_count']}",
        f"- max_db_abs_err: {audit['max_db_abs_err']}",
        "",
        "## Decision",
        "",
    ]
    if best:
        lines.append(
            f"- Selected setting: {best['setting']}."
        )
        lines.append(
            f"- It improves at hard severity with hard_improvement_db={best['hard_improvement_db']:.4f}, "
            f"does not degrade severity 0 by more than 0.1 dB "
            f"(improvement_0_db={best['improvement_0_db']:.4f}), "
            f"keeps max_delta_to_raw_ratio={best['max_delta_to_raw_ratio']:.4f}, "
            f"and has min_innovation_collapse_ratio={best['min_innovation_collapse_ratio']:.4f}."
        )
        lines.append("- Next step: run a full 3-seed pilot for this setting before changing the G1/G2 path.")
    else:
        lines.append("- No setting met the promising criteria in this tuning sweep.")
        lines.append("- Next step: add structured low-cost IMU corruption before a full ME-Split run, or tune further with stronger anti-collapse constraints.")
    lines.extend(["", "## Setting Criteria", ""])
    for c in sorted(candidates, key=lambda x: str(x["setting"])):
        lines.append(
            f"- {c['setting']}: qualifies={c['qualifies']} "
            f"imp0={c['improvement_0_db']:.4f} "
            f"imp20={c['improvement_20_db']:.4f} "
            f"imp30={c['improvement_30_db']:.4f} "
            f"max_delta_ratio={c['max_delta_to_raw_ratio']:.4f} "
            f"min_innovation_ratio={c['min_innovation_collapse_ratio']:.4f}"
        )
    lines.extend(["", "## Classification Counts", ""])
    counts: Dict[str, int] = defaultdict(int)
    for r in me_rows:
        counts[str(r["classification"])] += 1
    for k in sorted(counts):
        lines.append(f"- {k}: {counts[k]}")
    ANALYSIS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = list(_iter_rows())
    _write_csv(METRICS_CSV, rows)
    summary = _summarize(rows)
    _write_csv(SUMMARY_CSV, summary)
    _plot(summary)

    expected = 21
    valid = [
        r for r in rows
        if str(r["device_resolved"]) == "cuda"
        and str(r["plan"]) == "trained:frozen"
        and float(r["db_abs_err"]) < 1.0e-8
        and not bool(r["failure_json"])
        and bool(r["diagnostics_present"])
        and math.isfinite(float(r["residual_norm_mean"]))
        and int(r["adapt_updates_used"]) == 0
    ]
    me_rows = [r for r in rows if str(r["model_id"]) != "split_knet"]
    audit = {
        "expected_metrics": expected,
        "actual_metrics": len(rows),
        "valid_cuda_metrics": len(valid),
        "failure_json_count": sum(1 for r in rows if bool(r["failure_json"])),
        "max_db_abs_err": max([float(r["db_abs_err"]) for r in rows if math.isfinite(float(r["db_abs_err"]))] or [float("nan")]),
        "residual_diagnostics_finite": all(math.isfinite(float(r["residual_norm_mean"])) for r in rows),
        "enhancer_diagnostics_finite": all(
            math.isfinite(float(r["delta_to_raw_ratio_mean"]))
            and math.isfinite(float(r["y_enh_to_raw_norm_ratio_mean"]))
            for r in me_rows
        ),
        "innovation_diagnostics_finite": all(math.isfinite(float(r["innovation_collapse_ratio"])) for r in me_rows),
        "device_resolved_cuda_count": sum(1 for r in rows if str(r["device_resolved"]) == "cuda"),
        "metrics_csv": str(METRICS_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "analysis_md": str(ANALYSIS_MD),
        "plots": {
            "mse_db": str(PLOT_MSE),
            "delta_ratio": str(PLOT_DELTA),
            "innovation_ratio": str(PLOT_INNOV),
        },
    }
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    _analysis(summary, audit)
    print(json.dumps(audit, indent=2))
    return 0 if len(valid) == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
