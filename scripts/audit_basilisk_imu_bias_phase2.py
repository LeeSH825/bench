from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.diagnostics.imu_measurement_model_audit import (
    analytic_H_imu_bias,
    finite_difference_H_imu_bias,
    h_imu_bias,
)
from bench.tasks.bench_generated import (
    _scenario_cfg_basis_for_id,
    canonicalize_scenario_id,
    expand_scenarios_from_sweep,
)


def _load_suite(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _split_paths(cache_root: Path, suite_name: str, task_id: str, scenario_id: str, seed: int) -> Dict[str, Path]:
    base = cache_root / suite_name / task_id / f"scenario_{scenario_id}" / f"seed_{int(seed)}"
    return {split: base / f"{split}.npz" for split in ("train", "val", "test")}


def _load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        out = {key: z[key] for key in z.files}
    out["meta"] = json.loads(str(out["meta_json"].tolist() if hasattr(out["meta_json"], "tolist") else out["meta_json"]))
    return out


def _finite(x: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(x)).all())


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(d * d))


def _scenario_rows(suite: Mapping[str, Any]) -> List[Dict[str, Any]]:
    tasks = list(suite.get("tasks", []) or [])
    if len(tasks) != 1:
        raise ValueError("This audit script expects a single-task suite.")
    task = dict(tasks[0])
    scenarios = expand_scenarios_from_sweep(task) or [{}]
    rows: List[Dict[str, Any]] = []
    for scenario_cfg in scenarios:
        basis = _scenario_cfg_basis_for_id(task, scenario_cfg)
        scenario_id = canonicalize_scenario_id(str(task["task_id"]), basis)
        rows.append({"task": task, "scenario_cfg": scenario_cfg, "scenario_id": scenario_id})
    return rows


def _profile_from_meta(meta: Mapping[str, Any]) -> str:
    bias_state = meta.get("bias_state", {}) if isinstance(meta.get("bias_state", {}), Mapping) else {}
    return str(bias_state.get("profile_id", bias_state.get("severity", "unknown")))


def audit_data(*, suite_yaml: Path, cache_root: Path, reports_dir: Path) -> Path:
    suite = _load_suite(suite_yaml)
    suite_name = str(suite["suite"]["name"])
    seeds = [int(v) for v in suite.get("seeds", [0])]
    out_name = (
        "basilisk_imu_bias_gpu_pilot_data_audit.csv"
        if "gpu" in suite_name
        else "basilisk_imu_bias_data_smoke_audit.csv"
    )
    out_path = reports_dir / out_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for scenario in _scenario_rows(suite):
        task = scenario["task"]
        task_id = str(task["task_id"])
        scenario_id = str(scenario["scenario_id"])
        for seed in seeds:
            for split, path in _split_paths(cache_root, suite_name, task_id, scenario_id, seed).items():
                row: Dict[str, Any] = {
                    "suite_name": suite_name,
                    "task_id": task_id,
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "split": split,
                    "path": str(path),
                    "exists": path.exists(),
                }
                if not path.exists():
                    rows.append(row)
                    continue
                payload = _load_npz(path)
                x = np.asarray(payload["x"], dtype=np.float64)
                y = np.asarray(payload["y"], dtype=np.float64)
                meta = payload["meta"]
                dt = float(meta["ssm"]["true"]["dt"])
                h = np.asarray(payload.get("H", meta["ssm"]["assumed"]["H"]), dtype=np.float64)
                y_model = h_imu_bias(x, dt=dt)
                y_clean_bias = np.asarray(payload.get("imu_bias_clean_y_seq", y_model), dtype=np.float64)
                bias = np.asarray(payload.get("gyro_bias_seq"), dtype=np.float64)
                relation = np.asarray(payload.get("bias_component_seq"), dtype=np.float64) + np.asarray(
                    payload.get("noise_component_seq"), dtype=np.float64
                )
                y_relation_err = _mse(y - np.asarray(payload["imu_clean_y_seq"], dtype=np.float64), relation)
                h_fd = finite_difference_H_imu_bias(x[0, min(1, x.shape[1] - 1)], dt=dt)
                h_true = analytic_H_imu_bias(dt)
                row.update(
                    {
                        "profile_id": _profile_from_meta(meta),
                        "x_shape": "x".join(str(v) for v in x.shape),
                        "y_shape": "x".join(str(v) for v in y.shape),
                        "x_dtype": str(payload["x"].dtype),
                        "y_dtype": str(payload["y"].dtype),
                        "finite": _finite(x) and _finite(y),
                        "fake_marker": bool(meta.get("fake_marker", True)),
                        "H_shape": "x".join(str(v) for v in h.shape),
                        "H_rank": int(np.linalg.matrix_rank(h)),
                        "H_is_identity": bool(h.shape[0] == h.shape[1] and np.allclose(h, np.eye(h.shape[0]))),
                        "H_fd_error": float(np.max(np.abs(h_true - h_fd))),
                        "h_model_mse": _mse(y_model, y_clean_bias),
                        "relation_mse": y_relation_err,
                        "bias_norm_mean": float(np.mean(np.linalg.norm(bias, axis=2))),
                        "bias_to_omega_ratio": float(meta["bias_state"]["stats"].get("bias_to_omega_ratio", float("nan"))),
                        "imu_error_to_clean_ratio": float(
                            meta["bias_state"]["stats"].get("imu_error_to_clean_ratio", float("nan"))
                        ),
                    }
                )
                rows.append(row)
    _write_csv(out_path, rows)
    hh_rows = [
        {
            "suite_name": r.get("suite_name"),
            "task_id": r.get("task_id"),
            "scenario_id": r.get("scenario_id"),
            "profile_id": r.get("profile_id"),
            "split": r.get("split"),
            "H_shape": r.get("H_shape"),
            "H_rank": r.get("H_rank"),
            "H_is_identity": r.get("H_is_identity"),
            "finite_diff_error": r.get("H_fd_error"),
            "h_model_mse": r.get("h_model_mse"),
        }
        for r in rows
        if r.get("exists")
    ]
    hh_path = reports_dir / "basilisk_imu_bias_hH_audit.csv"
    _write_csv(hh_path, hh_rows)
    md = [
        "# Basilisk IMU Bias h/H Audit",
        "",
        "Analytic model:",
        "",
        "```text",
        "h_bias(x) = [omega + b_g, (omega + b_g) * dt]",
        "H_bias = [0 I I; 0 dt*I dt*I]",
        "```",
        "",
        "The expected H shape is `[6,9]`. Rank is expected to be 3 for this gyro/delta-angle packet.",
        "",
    ]
    if hh_rows:
        max_fd = max(float(r["finite_diff_error"]) for r in hh_rows if r.get("finite_diff_error") not in {None, ""})
        max_h = max(float(r["h_model_mse"]) for r in hh_rows if r.get("h_model_mse") not in {None, ""})
        ranks = sorted({str(r.get("H_rank")) for r in hh_rows})
        shapes = sorted({str(r.get("H_shape")) for r in hh_rows})
        md.extend(
            [
                f"- rows: {len(hh_rows)}",
                f"- H shapes: {', '.join(shapes)}",
                f"- H ranks: {', '.join(ranks)}",
                f"- max finite-difference error: {max_fd:.6g}",
                f"- max h-model MSE: {max_h:.6g}",
            ]
        )
    (reports_dir / "basilisk_imu_bias_hH_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out_path


def _run_dir(suite_name: str, task_id: str, model_id: str, seed: int, scenario_id: str) -> Path:
    return Path("runs") / suite_name / task_id / model_id / "frozen" / f"seed_{int(seed)}" / f"scenario_{scenario_id}"


def _load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _diagnostics_finite(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    text = json.dumps(obj, allow_nan=True)
    return all(token not in text for token in ("NaN", "Infinity", "-Infinity"))


def audit_runs(*, suite_yaml: Path, cache_root: Path, reports_dir: Path, plots_dir: Path, require_cuda: bool) -> Dict[str, Path]:
    suite = _load_suite(suite_yaml)
    suite_name = str(suite["suite"]["name"])
    task = dict((suite.get("tasks") or [])[0])
    task_id = str(task["task_id"])
    seeds = [int(v) for v in suite.get("seeds", [0])]
    models = [str(m["model_id"]) for m in suite.get("models", []) if str(m.get("model_id")) in {"split_knet", "me_split_knet_v0"}]
    rows: List[Dict[str, Any]] = []
    valid_count = 0
    failure_count = 0
    max_db_err = 0.0
    for scenario in _scenario_rows(suite):
        scenario_id = str(scenario["scenario_id"])
        for seed in seeds:
            test_path = _split_paths(cache_root, suite_name, task_id, scenario_id, seed)["test"]
            test_payload = _load_npz(test_path) if test_path.exists() else None
            meta = test_payload["meta"] if test_payload else {}
            profile_id = _profile_from_meta(meta)
            x_true = np.asarray(test_payload["x"], dtype=np.float64) if test_payload else None
            for model_id in models:
                rd = _run_dir(suite_name, task_id, model_id, seed, scenario_id)
                metrics_path = rd / "metrics.json"
                failure_path = rd / "failure.json"
                failure_count += int(failure_path.exists())
                row: Dict[str, Any] = {
                    "suite_name": suite_name,
                    "task_id": task_id,
                    "scenario_id": scenario_id,
                    "profile_id": profile_id,
                    "seed": seed,
                    "model_id": model_id,
                    "plan": "trained:frozen",
                    "run_dir": str(rd),
                    "metrics_exists": metrics_path.exists(),
                    "failure_exists": failure_path.exists(),
                }
                if not metrics_path.exists() or test_payload is None:
                    rows.append(row)
                    continue
                metrics = _load_metrics(metrics_path)
                acc = metrics.get("accuracy", {})
                budgets = metrics.get("budgets", {})
                run_plan = metrics.get("run_plan", {})
                mse = float(acc.get("mse", float("nan")))
                mse_db = float(acc.get("mse_db", float("nan")))
                expected_db = 10.0 * math.log10(mse) if mse > 0.0 and math.isfinite(mse) else float("nan")
                db_err = abs(mse_db - expected_db) if math.isfinite(expected_db) else float("inf")
                max_db_err = max(max_db_err, db_err if math.isfinite(db_err) else 0.0)
                preds_path = rd / "artifacts" / "preds_test.npz"
                x_hat = None
                if preds_path.exists():
                    with np.load(preds_path, allow_pickle=False) as z:
                        x_hat = np.asarray(z["x_hat"], dtype=np.float64)
                mse_sigma = mse_omega = mse_bias = bias_rmse = float("nan")
                if x_hat is not None and x_true is not None and x_hat.shape == x_true.shape:
                    err = x_hat - x_true
                    mse_sigma = float(np.mean(err[..., 0:3] ** 2))
                    mse_omega = float(np.mean(err[..., 3:6] ** 2))
                    mse_bias = float(np.mean(err[..., 6:9] ** 2))
                    bias_rmse = float(math.sqrt(mse_bias))
                diag_finite = _diagnostics_finite(rd / "diagnostics" / "stats.json")
                device_ok = str(run_plan.get("device_resolved", "")).startswith("cuda") if require_cuda else True
                valid = (
                    metrics_path.exists()
                    and not failure_path.exists()
                    and device_ok
                    and int(budgets.get("train_max_updates", -1)) == int((suite.get("runner", {}).get("budget", {}) or {}).get("train_max_updates", -1))
                    and db_err < 1.0e-8
                    and bool(diag_finite)
                    and int(budgets.get("adapt_updates_used", -1)) == 0
                    and x_hat is not None
                    and x_true is not None
                    and tuple(x_hat.shape) == tuple(x_true.shape)
                )
                valid_count += int(valid)
                row.update(
                    {
                        "mse": mse,
                        "mse_db": mse_db,
                        "expected_mse_db": expected_db,
                        "db_err": db_err,
                        "device_requested": run_plan.get("device_requested"),
                        "device_resolved": run_plan.get("device_resolved"),
                        "train_updates_used": budgets.get("train_updates_used"),
                        "train_max_updates": budgets.get("train_max_updates"),
                        "adapt_updates_used": budgets.get("adapt_updates_used"),
                        "diagnostics_finite": diag_finite,
                        "x_hat_shape": "x".join(str(v) for v in x_hat.shape) if x_hat is not None else "",
                        "mse_sigma": mse_sigma,
                        "mse_omega": mse_omega,
                        "mse_bias": mse_bias,
                        "bias_rmse": bias_rmse,
                        "delta_to_raw_ratio": _adapter_diag_scalar(rd, "delta_to_raw_ratio_mean"),
                        "correction_alignment": _adapter_diag_scalar(rd, "imu_correction_alignment"),
                        "valid": valid,
                    }
                )
                rows.append(row)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "basilisk_imu_bias_gpu_pilot_metrics.csv"
    if not require_cuda:
        metrics_path = reports_dir / "basilisk_imu_bias_cpu_smoke_audit.csv"
    _write_csv(metrics_path, rows)
    summary = _summary_rows(rows)
    summary_path = reports_dir / "basilisk_imu_bias_gpu_pilot_summary.csv"
    if not require_cuda:
        summary_path = reports_dir / "basilisk_imu_bias_cpu_smoke_summary.csv"
    _write_csv(summary_path, summary)
    acceptance = {
        "suite_name": suite_name,
        "expected_metrics": len(_scenario_rows(suite)) * len(seeds) * len(models),
        "valid_metrics": int(valid_count),
        "failure_json_count": int(failure_count),
        "max_db_err": float(max_db_err),
        "require_cuda": bool(require_cuda),
    }
    acceptance_path = reports_dir / "basilisk_imu_bias_gpu_pilot_acceptance_audit.json"
    if not require_cuda:
        acceptance_path = reports_dir / "basilisk_imu_bias_cpu_acceptance_audit.json"
    acceptance_path.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if require_cuda and summary:
        _plot_summary(summary, plots_dir)
    return {"metrics": metrics_path, "summary": summary_path, "acceptance": acceptance_path}


def _adapter_diag_scalar(run_dir: Path, key: str) -> float:
    path = run_dir / "diagnostics" / "stats.json"
    if not path.exists():
        return float("nan")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return float("nan")
    adapter = obj.get("adapter_diagnostics", {}) if isinstance(obj, Mapping) else {}
    candidates: List[Any] = []
    if isinstance(adapter, Mapping):
        candidates.extend([adapter.get(key), adapter.get(f"adapter_runtime.{key}")])
    runtime = obj.get("adapter_runtime_stats", {}) if isinstance(obj, Mapping) else {}
    if isinstance(runtime, Mapping):
        candidates.extend([runtime.get(key), runtime.get(f"adapter_runtime.{key}")])
    for value in candidates:
        if isinstance(value, Mapping):
            for stat_key in ("mean", "min", "max"):
                if value.get(stat_key) is not None:
                    try:
                        return float(value[stat_key])
                    except Exception:
                        pass
        try:
            return float(value)
        except Exception:
            pass
    return float("nan")


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_profile: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {}
    for row in rows:
        if not row.get("valid"):
            continue
        by_profile.setdefault(str(row.get("profile_id")), {}).setdefault(str(row.get("model_id")), []).append(row)
    out: List[Dict[str, Any]] = []
    order = ["clean_bias", "mild_bias", "moderate_bias", "low_cost_bias"]
    for profile in sorted(by_profile, key=lambda p: order.index(p) if p in order else 99):
        split_rows = by_profile[profile].get("split_knet", [])
        me_rows = by_profile[profile].get("me_split_knet_v0", [])
        if not split_rows and not me_rows:
            continue
        split_db = _mean([float(r["mse_db"]) for r in split_rows])
        me_db = _mean([float(r["mse_db"]) for r in me_rows])
        me_first = me_rows[0] if me_rows else {}
        out.append(
            {
                "severity": profile,
                "split_mean_db": split_db,
                "me_mean_db": me_db,
                "improvement_db": split_db - me_db if math.isfinite(split_db) and math.isfinite(me_db) else float("nan"),
                "mse_sigma": _mean([float(r.get("mse_sigma", float("nan"))) for r in me_rows or split_rows]),
                "mse_omega": _mean([float(r.get("mse_omega", float("nan"))) for r in me_rows or split_rows]),
                "mse_bias": _mean([float(r.get("mse_bias", float("nan"))) for r in me_rows or split_rows]),
                "bias_rmse": _mean([float(r.get("bias_rmse", float("nan"))) for r in me_rows or split_rows]),
                "delta_to_raw_ratio": float(me_first.get("delta_to_raw_ratio", float("nan"))),
                "correction_alignment": float(me_first.get("correction_alignment", float("nan"))),
                "n": max(len(split_rows), len(me_rows)),
            }
        )
    return out


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _plot_summary(summary: Sequence[Mapping[str, Any]], plots_dir: Path) -> None:
    labels = [str(r["severity"]) for r in summary]
    x = np.arange(len(labels))
    split = [float(r["split_mean_db"]) for r in summary]
    me = [float(r["me_mean_db"]) for r in summary]
    imp = [float(r["improvement_db"]) for r in summary]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, split, marker="o", label="split_knet | trained:frozen")
    ax.plot(x, me, marker="o", label="me_split_knet_v0 | trained:frozen")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("mse_db")
    ax.set_xlabel("bias severity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "basilisk_imu_bias_gpu_pilot_mse_db.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.bar(x, imp, color="#4c78a8")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("improvement_db")
    ax.set_xlabel("bias severity")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "basilisk_imu_bias_gpu_pilot_improvement_db.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.plot(x, [float(r["bias_rmse"]) for r in summary], marker="o", label="bias_rmse")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("bias RMSE")
    ax.set_xlabel("bias severity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "basilisk_imu_bias_gpu_pilot_bias_rmse.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.plot(x, [float(r["delta_to_raw_ratio"]) for r in summary], marker="o", label="delta_to_raw_ratio")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("ratio")
    ax.set_xlabel("bias severity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "basilisk_imu_bias_gpu_pilot_delta_ratio.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.plot(x, [float(r["correction_alignment"]) for r in summary], marker="o", label="correction_alignment")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("cosine")
    ax.set_xlabel("bias severity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "basilisk_imu_bias_gpu_pilot_correction_alignment.png", dpi=160)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-yaml", required=True)
    parser.add_argument("--cache-root", default="/home/dss-pc-05/bench/bench_data_cache")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--runs", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    suite_yaml = Path(args.suite_yaml)
    reports_dir = Path(args.reports_dir)
    plots_dir = Path(args.plots_dir)
    cache_root = Path(args.cache_root)
    outputs: Dict[str, str] = {}
    if args.data or not args.runs:
        outputs["data"] = str(audit_data(suite_yaml=suite_yaml, cache_root=cache_root, reports_dir=reports_dir))
    if args.runs:
        outputs.update(
            {k: str(v) for k, v in audit_runs(
                suite_yaml=suite_yaml,
                cache_root=cache_root,
                reports_dir=reports_dir,
                plots_dir=plots_dir,
                require_cuda=bool(args.require_cuda),
            ).items()}
        )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
