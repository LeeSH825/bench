from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/dss-pc-05/bench")
CACHE_ROOT = ROOT / "bench_data_cache"
SUITE = "gpu_basilisk_adcs_official"
TASK_ID = "Basilisk_ADCS_sensor_noise_sweep_v0"
RUN_ROOT = ROOT / "runs" / SUITE / TASK_ID
REPORTS = ROOT / "reports"
PLOTS = ROOT / "plots"

SCENARIOS = {
    "37bd751afdc0": -10,
    "4c70131fb32a": 0,
    "b2da60a0830d": 10,
    "4d7e2a5202c0": 20,
    "ef19946c6066": 30,
}
MODELS = ["kalmannet_tsp", "split_knet"]
SEEDS = [0, 1, 2]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_meta(scenario_id: str, seed: int) -> Dict[str, Any]:
    path = CACHE_ROOT / SUITE / TASK_ID / f"scenario_{scenario_id}" / f"seed_{seed}" / "test.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta_json"]))


def collect_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario_id, severity in SCENARIOS.items():
        for seed in SEEDS:
            meta = _cache_meta(scenario_id, seed)
            meta_text = json.dumps(meta, sort_keys=True).lower() if meta else ""
            framework = meta.get("ssm", {}).get("true", {}).get("framework") if meta else None
            for model_id in MODELS:
                run_dir = RUN_ROOT / model_id / "frozen" / f"seed_{seed}" / f"scenario_{scenario_id}"
                metrics_path = run_dir / "metrics.json"
                failure_path = run_dir / "failure.json"
                stats_path = run_dir / "diagnostics" / "stats.json"
                row: Dict[str, Any] = {
                    "suite": SUITE,
                    "task_id": TASK_ID,
                    "sensor_noise_scale_db": severity,
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "model_id": model_id,
                    "plan": "trained:frozen",
                    "run_dir": str(run_dir),
                    "metrics_exists": metrics_path.exists(),
                    "failure_json": failure_path.exists(),
                    "diagnostics_exists": stats_path.exists(),
                    "cache_framework": framework,
                    "fake_marker_present": bool("fake" in meta_text or "fallback" in meta_text),
                }
                if metrics_path.exists():
                    metrics = _read_json(metrics_path)
                    accuracy = metrics.get("accuracy", {})
                    budgets = metrics.get("budgets", {})
                    plan = metrics.get("run_plan", {})
                    mse = float(accuracy["mse"])
                    mse_db = float(accuracy["mse_db"])
                    expected = 10.0 * math.log10(mse)
                    row.update(
                        {
                            "mse": mse,
                            "mse_db": mse_db,
                            "mse_db_expected": expected,
                            "mse_db_abs_err": abs(mse_db - expected),
                            "train_updates_used": budgets.get("train_updates_used"),
                            "train_outer_updates_used": budgets.get("train_outer_updates_used"),
                            "train_max_updates": budgets.get("train_max_updates"),
                            "adapt_updates_used": budgets.get("adapt_updates_used"),
                            "device_requested": plan.get("device_requested"),
                            "device_resolved": plan.get("device_resolved"),
                            "init_id": plan.get("init_id"),
                            "track_id": plan.get("track_id"),
                        }
                    )
                if stats_path.exists():
                    stats = _read_json(stats_path)
                    residual = stats.get("residual_stats") or {}
                    row.update(
                        {
                            "residual_finite": residual.get("finite"),
                            "residual_norm": residual.get("norm"),
                            "residual_nan_count": residual.get("nan_count"),
                            "residual_inf_count": residual.get("inf_count"),
                        }
                    )
                rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for severity in sorted(set(SCENARIOS.values())):
        for model_id in MODELS:
            vals = [
                float(r["mse_db"])
                for r in rows
                if r.get("run_valid") and r.get("sensor_noise_scale_db") == severity and r.get("model_id") == model_id
            ]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            out.append(
                {
                    "sensor_noise_scale_db": severity,
                    "model_id": model_id,
                    "plan": "trained:frozen",
                    "mean_mse_db": float(np.mean(arr)),
                    "std_mse_db": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "n": int(len(arr)),
                }
            )
    return out


def write_plot(summary_rows: List[Dict[str, Any]]) -> None:
    if not summary_rows:
        return
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model_id in MODELS:
        rows = [r for r in summary_rows if r["model_id"] == model_id]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["sensor_noise_scale_db"])
        x = [float(r["sensor_noise_scale_db"]) for r in rows]
        y = [float(r["mean_mse_db"]) for r in rows]
        err = [float(r["std_mse_db"]) for r in rows]
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=f"{model_id} | trained:frozen")
    ax.set_xlabel("sensor_noise_scale_db")
    ax.set_ylabel("mse_db mean")
    ax.set_title("Basilisk ADCS GPU official, train_max_updates=500")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "basilisk_gpu_official_500_mse_db.png", dpi=180)
    fig.savefig(PLOTS / "basilisk_gpu_official_500_mse_db.pdf")
    plt.close(fig)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    for row in rows:
        residual_ok = (
            bool(row.get("residual_finite"))
            and int(row.get("residual_nan_count") or 0) == 0
            and int(row.get("residual_inf_count") or 0) == 0
        )
        row["run_valid"] = (
            bool(row.get("metrics_exists"))
            and not bool(row.get("failure_json"))
            and row.get("model_id") in set(MODELS)
            and row.get("plan") == "trained:frozen"
            and row.get("init_id") == "trained"
            and row.get("track_id") == "frozen"
            and row.get("device_requested") == "cuda"
            and row.get("device_resolved") == "cuda"
            and int(row.get("train_max_updates") or -1) == 500
            and int(row.get("adapt_updates_used") or 0) == 0
            and float(row.get("mse_db_abs_err") or 0.0) < 1e-8
            and residual_ok
            and not bool(row.get("fake_marker_present"))
        )
    summary_rows = summarize(rows)
    write_csv(REPORTS / "basilisk_gpu_official_500_metrics_audit.csv", rows)
    write_csv(REPORTS / "basilisk_gpu_official_500_summary.csv", summary_rows)
    write_plot(summary_rows)

    completed = [r for r in rows if r.get("run_valid")]
    metrics_present = [r for r in rows if r.get("metrics_exists")]
    failures = [r for r in rows if r.get("failure_json")]
    residual_ok = [
        bool(r.get("residual_finite"))
        and int(r.get("residual_nan_count") or 0) == 0
        and int(r.get("residual_inf_count") or 0) == 0
        for r in metrics_present
    ]
    db_abs = [float(r.get("mse_db_abs_err") or 0.0) for r in metrics_present]
    acceptance = {
        "expected_metrics_count": len(SCENARIOS) * len(SEEDS) * len(MODELS),
        "metrics_json_count": len(metrics_present),
        "valid_metrics_count": len(completed),
        "complete": len(completed) == len(SCENARIOS) * len(SEEDS) * len(MODELS),
        "failure_json_count": len(failures),
        "max_mse_db_abs_err": max(db_abs) if db_abs else None,
        "mse_db_invariant_ok": bool(db_abs) and max(db_abs) < 1e-8,
        "residual_diagnostics_finite": all(residual_ok) if residual_ok else False,
        "device_resolved_cuda": all(r.get("device_resolved") == "cuda" for r in metrics_present) if metrics_present else False,
        "train_max_updates_500": all(int(r.get("train_max_updates") or -1) == 500 for r in metrics_present) if metrics_present else False,
        "official_models_only": all(r.get("model_id") in set(MODELS) and r.get("plan") == "trained:frozen" for r in metrics_present),
        "fake_basilisk_data_generated": any(r.get("fake_marker_present") for r in rows),
        "failed_or_missing": [
            {
                "sensor_noise_scale_db": r.get("sensor_noise_scale_db"),
                "scenario_id": r.get("scenario_id"),
                "seed": r.get("seed"),
                "model_id": r.get("model_id"),
                "run_dir": r.get("run_dir"),
                "metrics_exists": r.get("metrics_exists"),
                "failure_json": r.get("failure_json"),
                "device_resolved": r.get("device_resolved"),
                "train_max_updates": r.get("train_max_updates"),
                "residual_finite": r.get("residual_finite"),
            }
            for r in rows
            if not r.get("run_valid")
        ],
    }
    (REPORTS / "basilisk_gpu_official_500_acceptance_audit.json").write_text(
        json.dumps(acceptance, indent=2), encoding="utf-8"
    )
    (REPORTS / "basilisk_gpu_official_500_legend_audit.txt").write_text(
        "Expected legend labels:\\n"
        "- kalmannet_tsp | trained:frozen\\n"
        "- split_knet | trained:frozen\\n"
        "\\nForbidden labels/models: mb_kf_oracle, adaptive_knet, maml_knet, untrained:frozen, trained:budgeted\\n",
        encoding="utf-8",
    )
    print(json.dumps(acceptance, indent=2))
    return 0 if acceptance["complete"] and acceptance["failure_json_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
