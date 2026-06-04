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
RUN_ROOT = ROOT / "runs" / "gpu_basilisk_me_split_pilot"
REPORTS = ROOT / "reports"
PLOTS = ROOT / "plots"
SUMMARY_CSV = REPORTS / "me_split_knet_ablation_summary.csv"
METRICS_CSV = REPORTS / "me_split_knet_gpu_pilot_metrics.csv"
AUDIT_JSON = REPORTS / "me_split_knet_gpu_pilot_acceptance_audit.json"
NOTES_MD = REPORTS / "me_split_knet_ablation_notes.md"
PLOT_MSE = PLOTS / "me_split_knet_gpu_pilot_mse_db.png"
PLOT_DELTA = PLOTS / "me_split_knet_delta_norm.png"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _extract_severity(run_dir: Path, metrics: Dict[str, Any]) -> float:
    for key in ("sensor_noise_scale_db", "severity", "invR2db"):
        if key in metrics:
            try:
                return float(metrics[key])
            except Exception:
                pass
    scenario_settings = metrics.get("scenario_settings", {})
    if isinstance(scenario_settings, dict):
        for key in ("sensor_noise_scale_db", "severity", "invR2db"):
            if key in scenario_settings:
                try:
                    return float(scenario_settings[key])
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
    name = str(metrics.get("scenario_id") or run_dir.name)
    if "sensor_noise_scale_db_" in name:
        tail = name.rsplit("sensor_noise_scale_db_", 1)[-1]
        try:
            return float(tail.replace("m", "-"))
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
    if isinstance(run_plan.get("run_plan"), dict) and key in run_plan["run_plan"]:
        return run_plan["run_plan"].get(key)
    return None


def _residual_norm(stats: Dict[str, Any]) -> float:
    residual = stats.get("residual_stats")
    if isinstance(residual, dict):
        for key in ("norm", "mean"):
            if key in residual:
                try:
                    return float(residual[key])
                except Exception:
                    pass
    return float("nan")


def _iter_metric_rows() -> Iterable[Dict[str, Any]]:
    for metrics_path in sorted(RUN_ROOT.glob("**/metrics.json")):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path)
        run_plan = _read_json(run_dir / "run_plan.json")
        ledger = _read_json(run_dir / "budget_ledger.json")
        stats = _read_json(run_dir / "diagnostics" / "stats.json")
        model_id = str(metrics.get("model_id") or _run_plan_value(run_plan, "model_id") or run_dir.parts[-5])
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
            "failure_json": (run_dir / "failure.json").exists(),
            "residual_norm": _residual_norm(stats),
            "delta_norm_mean": _stats_value(stats, "delta_norm_mean", "mean"),
            "delta_norm_max": _stats_value(stats, "delta_norm_max", "max"),
            "y_raw_norm_mean": _stats_value(stats, "y_raw_norm_mean", "mean"),
            "y_enh_norm_mean": _stats_value(stats, "y_enh_norm_mean", "mean"),
            "diagnostics_present": (run_dir / "diagnostics" / "stats.json").exists(),
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


def _plot_summary(summary_rows: List[Dict[str, Any]]) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_model[str(row["model_id"])].append(row)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model_id, rows in sorted(by_model.items()):
        rows = sorted(rows, key=lambda r: float(r["sensor_noise_scale_db"]))
        ax.errorbar(
            [float(r["sensor_noise_scale_db"]) for r in rows],
            [float(r["mean_mse_db"]) for r in rows],
            yerr=[float(r["std_mse_db"]) for r in rows],
            marker="o",
            capsize=3,
            label=f"{model_id} | trained:frozen",
        )
    ax.set_xlabel("sensor_noise_scale_db")
    ax.set_ylabel("mean mse_db")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_MSE, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = [r for r in summary_rows if str(r["model_id"]) == "me_split_knet_v0"]
    if rows:
        rows = sorted(rows, key=lambda r: float(r["sensor_noise_scale_db"]))
        ax.plot(
            [float(r["sensor_noise_scale_db"]) for r in rows],
            [float(r["delta_norm_mean"]) for r in rows],
            marker="o",
            label="delta_norm_mean",
        )
        ax.plot(
            [float(r["sensor_noise_scale_db"]) for r in rows],
            [float(r["delta_norm_max"]) for r in rows],
            marker="s",
            label="delta_norm_max",
        )
    ax.set_xlabel("sensor_noise_scale_db")
    ax.set_ylabel("measurement correction norm")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DELTA, dpi=180)
    plt.close(fig)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = list(_iter_metric_rows())
    _write_csv(METRICS_CSV, rows)

    grouped: Dict[Tuple[float, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["sensor_noise_scale_db"]), str(row["model_id"]), str(row["plan"]))].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (severity, model_id, plan), group in sorted(grouped.items()):
        vals = [float(r["mse_db"]) for r in group if math.isfinite(float(r["mse_db"]))]
        delta_mean_vals = [float(r["delta_norm_mean"]) for r in group if math.isfinite(float(r["delta_norm_mean"]))]
        delta_max_vals = [float(r["delta_norm_max"]) for r in group if math.isfinite(float(r["delta_norm_max"]))]
        summary_rows.append(
            {
                "sensor_noise_scale_db": severity,
                "model_id": model_id,
                "plan": plan,
                "mean_mse_db": sum(vals) / len(vals) if vals else float("nan"),
                "std_mse_db": 0.0 if len(vals) <= 1 else float((sum((v - sum(vals) / len(vals)) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5),
                "n": len(vals),
                "residual_norm_mean": sum(float(r["residual_norm"]) for r in group if math.isfinite(float(r["residual_norm"]))) / max(1, sum(1 for r in group if math.isfinite(float(r["residual_norm"])))),
                "delta_norm_mean": sum(delta_mean_vals) / len(delta_mean_vals) if delta_mean_vals else float("nan"),
                "delta_norm_max": max(delta_max_vals) if delta_max_vals else float("nan"),
            }
        )

    split_by_severity = {
        float(r["sensor_noise_scale_db"]): float(r["mean_mse_db"])
        for r in summary_rows
        if str(r["model_id"]) == "split_knet"
    }
    for row in summary_rows:
        if str(row["model_id"]) == "me_split_knet_v0":
            base = split_by_severity.get(float(row["sensor_noise_scale_db"]), float("nan"))
            row["improvement_db"] = base - float(row["mean_mse_db"]) if math.isfinite(base) else float("nan")
        else:
            row["improvement_db"] = ""

    _write_csv(SUMMARY_CSV, summary_rows)
    _plot_summary(summary_rows)

    expected = 6
    valid = [
        r for r in rows
        if str(r["device_resolved"]) == "cuda"
        and str(r["plan"]) == "trained:frozen"
        and str(r["model_id"]) in {"split_knet", "me_split_knet_v0"}
        and float(r["db_abs_err"]) < 1.0e-8
        and not bool(r["failure_json"])
        and bool(r["diagnostics_present"])
        and math.isfinite(float(r["residual_norm"]))
    ]
    me_rows = [r for r in rows if str(r["model_id"]) == "me_split_knet_v0"]
    audit = {
        "expected_metrics": expected,
        "actual_metrics": len(rows),
        "valid_cuda_metrics": len(valid),
        "failure_json_count": sum(1 for r in rows if bool(r["failure_json"])),
        "max_db_abs_err": max([float(r["db_abs_err"]) for r in rows if math.isfinite(float(r["db_abs_err"]))] or [float("nan")]),
        "residual_diagnostics_finite": all(math.isfinite(float(r["residual_norm"])) for r in rows),
        "enhancer_delta_diagnostics_finite": all(
            math.isfinite(float(r["delta_norm_mean"])) and math.isfinite(float(r["delta_norm_max"]))
            for r in me_rows
        ),
        "device_resolved_cuda_count": sum(1 for r in rows if str(r["device_resolved"]) == "cuda"),
        "plots": {
            "mse_db": str(PLOT_MSE),
            "delta_norm": str(PLOT_DELTA),
        },
        "metrics_csv": str(METRICS_CSV),
        "summary_csv": str(SUMMARY_CSV),
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    NOTES_MD.write_text(
        "\n".join(
            [
                "# ME-Split-KalmanNet v0 Ablation Notes",
                "",
                "- B2: split_knet | trained:frozen uses raw Basilisk measurements.",
                "- B4: me_split_knet_v0 | trained:frozen uses y_raw + causal residual measurement enhancement.",
                "- Positive improvement_db means ME-Split-KNet has lower mse_db than raw Split-KalmanNet.",
                "- Innovation collapse is not directly instrumented for Split-KalmanNet v0; safety is assessed by residual norms and delta/y norm ratios.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))
    return 0 if len(valid) == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
