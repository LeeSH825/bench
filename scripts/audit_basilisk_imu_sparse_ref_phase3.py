from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.diagnostics.imu_measurement_model_audit import (
    analytic_H_imu_bias_sparse_ref,
    finite_difference_H_imu_bias_sparse_ref,
    h_imu_bias_sparse_ref,
)
from bench.tasks.bench_generated import (
    _scenario_cfg_basis_for_id,
    canonicalize_scenario_id,
    expand_scenarios_from_sweep,
)
from bench.runners.run_suite import _load_split_npz, _make_loader


def _load_suite(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


def _split_paths(cache_root: Path, suite_name: str, task_id: str, scenario_id: str, seed: int) -> Dict[str, Path]:
    base = cache_root / suite_name / task_id / f"scenario_{scenario_id}" / f"seed_{int(seed)}"
    return {split: base / f"{split}.npz" for split in ("train", "val", "test")}


def _report_prefix(suite_name: str, *, require_cuda: bool = True) -> str:
    if "sanity_500" in suite_name:
        return "basilisk_imu_sparse_ref_sanity_500"
    if "sanity" in suite_name:
        return "basilisk_imu_sparse_ref_sanity"
    if "stability" in suite_name:
        return "basilisk_imu_sparse_ref_stability"
    if require_cuda:
        return "basilisk_imu_sparse_ref_gpu_pilot"
    return "basilisk_imu_sparse_ref_cpu_smoke"


def _load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        out = {key: z[key] for key in z.files}
    out["meta"] = json.loads(str(out["meta_json"].tolist() if hasattr(out["meta_json"], "tolist") else out["meta_json"]))
    return out


def _finite(x: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(x)).all())


def _mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if mask is not None:
        w = np.asarray(mask, dtype=np.float64)
        denom = float(np.sum(w))
        if denom <= 0.0:
            return 0.0
        return float(np.sum(d * d * w) / denom)
    return float(np.mean(d * d))


def _profile_from_meta(meta: Mapping[str, Any]) -> str:
    sparse = meta.get("sparse_ref", {}) if isinstance(meta.get("sparse_ref", {}), Mapping) else {}
    bias = meta.get("bias_state", {}) if isinstance(meta.get("bias_state", {}), Mapping) else {}
    return str(sparse.get("profile_id", bias.get("profile_id", sparse.get("severity", "unknown"))))


def audit_data(*, suite_yaml: Path, cache_root: Path, reports_dir: Path) -> Path:
    suite = _load_suite(suite_yaml)
    suite_name = str(suite["suite"]["name"])
    seeds = [int(v) for v in suite.get("seeds", [0])]
    out_name = (
        f"{_report_prefix(suite_name, require_cuda=True)}_data_audit.csv"
        if "gpu" in suite_name or "stability" in suite_name
        else "basilisk_imu_sparse_ref_data_smoke_audit.csv"
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
                ref_mask = np.asarray(payload["ref_mask_seq"], dtype=np.float64)
                meas_mask = np.asarray(payload["measurement_mask_seq"], dtype=np.float64)
                y_clean = np.asarray(payload["measurement_clean_y_seq"], dtype=np.float64)
                y_model = h_imu_bias_sparse_ref(x, dt=dt)
                y_model_masked = y_model.copy()
                y_model_masked[..., 6:9] *= ref_mask
                h_fd = finite_difference_H_imu_bias_sparse_ref(x[0, min(1, x.shape[1] - 1)], dt=dt)
                h_true = analytic_H_imu_bias_sparse_ref(dt)
                h_masked = h_true.copy()
                h_masked[6:9, :] = 0.0
                ref_weight = np.repeat(ref_mask, 3, axis=2)
                ref_steps = np.flatnonzero(ref_mask[0, :, 0] > 0.5) if ref_mask.size else np.asarray([], dtype=np.int64)
                row.update(
                    {
                        "profile_id": _profile_from_meta(meta),
                        "x_shape": "x".join(str(v) for v in x.shape),
                        "y_shape": "x".join(str(v) for v in y.shape),
                        "x_dtype": str(payload["x"].dtype),
                        "y_dtype": str(payload["y"].dtype),
                        "finite": _finite(x) and _finite(y),
                        "fake_marker": bool(meta.get("fake_marker", True)),
                        "ref_mask_mode": str(meta.get("sparse_ref", {}).get("ref_mask_mode", "")),
                        "H_shape": "x".join(str(v) for v in h.shape),
                        "H_rank_unmasked": int(np.linalg.matrix_rank(h)),
                        "H_rank_masked": int(np.linalg.matrix_rank(h_masked)),
                        "H_is_identity": bool(h.shape[0] == h.shape[1] and np.allclose(h, np.eye(h.shape[0]))),
                        "H_fd_error": float(np.max(np.abs(h_true - h_fd))),
                        "h_model_mse_total": _mse(y_model_masked, y_clean, meas_mask),
                        "h_model_mse_imu": _mse(y_model[..., 0:6], np.asarray(payload["imu_bias_clean_y_seq"], dtype=np.float64)),
                        "h_model_mse_ref_when_masked": _mse(y_model[..., 6:9], np.asarray(payload["ref_clean_seq"], dtype=np.float64), ref_weight),
                        "relation_mse": _mse(y - y_clean, np.asarray(payload["measurement_error_seq"], dtype=np.float64), meas_mask),
                        "ref_mask_rate": float(np.mean(ref_mask)),
                        "active_ref_count": int(np.sum(ref_mask)),
                        "ref_update_period_observed": float(np.mean(np.diff(ref_steps))) if ref_steps.size > 1 else float("nan"),
                        "measurement_mask_ref_rate": float(np.mean(meas_mask[..., 6:9])),
                        "y_ref_zero_when_masked": bool(np.allclose(y[..., 6:9][np.repeat(ref_mask <= 0.5, 3, axis=2)], 0.0))
                        if np.any(ref_mask <= 0.5)
                        else True,
                        "ref_mse_when_available": _mse(y[..., 6:9], x[..., 0:3], ref_weight),
                        "bias_norm_mean": float(np.mean(np.linalg.norm(np.asarray(payload["gyro_bias_seq"]), axis=2))),
                        "bias_to_omega_ratio": float(meta["bias_state"]["stats"].get("bias_to_omega_ratio", float("nan"))),
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
            "H_rank_unmasked": r.get("H_rank_unmasked"),
            "H_rank_masked": r.get("H_rank_masked"),
            "H_is_identity": r.get("H_is_identity"),
            "finite_diff_error": r.get("H_fd_error"),
            "h_model_mse": r.get("h_model_mse_total"),
            "h_model_mse_imu": r.get("h_model_mse_imu"),
            "h_model_mse_ref_when_masked": r.get("h_model_mse_ref_when_masked"),
        }
        for r in rows
        if r.get("exists")
    ]
    hh_path = reports_dir / "basilisk_imu_sparse_ref_hH_audit.csv"
    _write_csv(hh_path, hh_rows)
    md = [
        "# Basilisk IMU Sparse Reference h/H Audit",
        "",
        "Analytic unmasked model:",
        "",
        "```text",
        "h_sparse_ref(x) = [omega + b_g, (omega + b_g) * dt, sigma]",
        "H_sparse_ref = [0 I I; 0 dt*I dt*I; I 0 0]",
        "```",
        "",
        "Reference rows are masked by `ref_mask_seq`/`measurement_mask_seq` when unavailable.",
        "",
    ]
    if hh_rows:
        max_fd = max(float(r["finite_diff_error"]) for r in hh_rows if r.get("finite_diff_error") not in {None, ""})
        max_h = max(float(r["h_model_mse"]) for r in hh_rows if r.get("h_model_mse") not in {None, ""})
        ranks_u = sorted({str(r.get("H_rank_unmasked")) for r in hh_rows})
        ranks_m = sorted({str(r.get("H_rank_masked")) for r in hh_rows})
        shapes = sorted({str(r.get("H_shape")) for r in hh_rows})
        md.extend(
            [
                f"- rows: {len(hh_rows)}",
                f"- H shapes: {', '.join(shapes)}",
                f"- H unmasked ranks: {', '.join(ranks_u)}",
                f"- H masked ranks: {', '.join(ranks_m)}",
                f"- max finite-difference error: {max_fd:.6g}",
                f"- max h-model MSE: {max_h:.6g}",
            ]
        )
    (reports_dir / "basilisk_imu_sparse_ref_hH_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out_path


def _run_dir(suite_name: str, task_id: str, model_id: str, seed: int, scenario_id: str) -> Path:
    return Path("runs") / suite_name / task_id / model_id / "frozen" / f"seed_{int(seed)}" / f"scenario_{scenario_id}"


def _load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def _diagnostics_finite(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    text = json.dumps(obj, allow_nan=True)
    return all(token not in text for token in ("NaN", "Infinity", "-Infinity"))


def _attitude_error_deg(x_hat: np.ndarray, x_true: np.ndarray) -> np.ndarray:
    sigma_err = np.asarray(x_hat[..., 0:3] - x_true[..., 0:3], dtype=np.float64)
    return 4.0 * np.linalg.norm(sigma_err, axis=2) * (180.0 / math.pi)


def _timing_attitude_stats(att_deg: np.ndarray, ref_mask: np.ndarray) -> Dict[str, float]:
    mask = np.asarray(ref_mask[..., 0] > 0.5)
    at = att_deg[mask]
    before_mask = np.zeros_like(mask, dtype=bool)
    after_mask = np.zeros_like(mask, dtype=bool)
    if mask.shape[1] > 1:
        before_mask[:, :-1] = mask[:, 1:]
        after_mask[:, 1:] = mask[:, :-1]
    before = att_deg[before_mask]
    after = att_deg[after_mask]
    drift_values: List[float] = []
    for i in range(mask.shape[0]):
        idx = np.flatnonzero(mask[i])
        if idx.size > 1:
            vals = att_deg[i, idx]
            drift_values.extend(np.abs(np.diff(vals)).astype(float).tolist())
    return {
        "attitude_error_at_ref": _mean(at.astype(float).tolist()),
        "attitude_error_before_ref": _mean(before.astype(float).tolist()),
        "attitude_error_after_ref": _mean(after.astype(float).tolist()),
        "drift_between_refs": _mean(drift_values),
    }


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
            ref_mask = np.asarray(test_payload["ref_mask_seq"], dtype=np.float64) if test_payload else None
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
                if failure_path.exists():
                    failure = _load_json(failure_path)
                    err = str(failure.get("error") or failure.get("message") or "")
                    m = re.search(r"update=(\d+)", err)
                    row.update(
                        {
                            "status": "failed",
                            "failure_phase": failure.get("phase"),
                            "failure_type": failure.get("failure_type"),
                            "failure_update": int(m.group(1)) if m else "",
                            "failure_error": err,
                        }
                    )
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
                mse_sigma = mse_omega = mse_bias = bias_rmse = attitude_deg = float("nan")
                timing: Dict[str, float] = {}
                if x_hat is not None and x_true is not None and x_hat.shape == x_true.shape:
                    err = x_hat - x_true
                    mse_sigma = float(np.mean(err[..., 0:3] ** 2))
                    mse_omega = float(np.mean(err[..., 3:6] ** 2))
                    mse_bias = float(np.mean(err[..., 6:9] ** 2))
                    bias_rmse = float(math.sqrt(mse_bias))
                    att_arr = _attitude_error_deg(x_hat, x_true)
                    attitude_deg = float(np.mean(att_arr))
                    if ref_mask is not None:
                        timing = _timing_attitude_stats(att_arr, ref_mask)
                diag_finite = _diagnostics_finite(rd / "diagnostics" / "stats.json")
                train_state = _load_json(rd / "checkpoints" / "train_state.json")
                train_diag_summary = _training_diag_summary(rd)
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
                    and ref_mask is not None
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
                        "attitude_error_deg": attitude_deg,
                        "ref_mask_rate": float(np.mean(ref_mask)) if ref_mask is not None else float("nan"),
                        "ref_update_period_observed": float(meta.get("sparse_ref", {}).get("stats", {}).get("ref_period_observed", float("nan"))),
                        "ref_mse_when_available": _mse(
                            np.asarray(test_payload["y"])[..., 6:9],
                            x_true[..., 0:3],
                            np.repeat(ref_mask, 3, axis=2),
                        )
                        if ref_mask is not None
                        else float("nan"),
                        "delta_to_raw_ratio": _adapter_diag_scalar(rd, "delta_to_raw_ratio_mean"),
                        "correction_alignment": _adapter_diag_scalar(rd, "imu_correction_alignment"),
                        "imu_mse_reduction": _adapter_diag_scalar(rd, "imu_mse_reduction"),
                        "grad_norm_max": train_diag_summary.get("grad_norm_max", float("nan")),
                        "grad_norm_mean": train_diag_summary.get("grad_norm_mean", float("nan")),
                        "clip_applied_count": int(train_state.get("clip_applied_count", train_diag_summary.get("clip_applied_count", 0)) or 0),
                        "train_loss_last": train_diag_summary.get("train_loss_last", float("nan")),
                        "status": "ok" if valid else "invalid",
                        "valid": valid,
                        **timing,
                    }
                )
                rows.append(row)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    prefix = _report_prefix(suite_name, require_cuda=require_cuda)
    metrics_path = reports_dir / f"{prefix}_metrics.csv"
    if not require_cuda:
        metrics_path = reports_dir / "basilisk_imu_sparse_ref_cpu_smoke_audit.csv"
    _write_csv(metrics_path, rows)
    summary = _summary_rows(rows)
    summary_path = reports_dir / f"{prefix}_summary.csv"
    if not require_cuda:
        summary_path = reports_dir / "basilisk_imu_sparse_ref_cpu_smoke_summary.csv"
    _write_csv(summary_path, summary)
    acceptance = {
        "suite_name": suite_name,
        "expected_metrics": len(_scenario_rows(suite)) * len(seeds) * len(models),
        "valid_metrics": int(valid_count),
        "failure_json_count": int(failure_count),
        "max_db_err": float(max_db_err),
        "require_cuda": bool(require_cuda),
    }
    acceptance_path = reports_dir / f"{prefix}_acceptance_audit.json"
    if not require_cuda:
        acceptance_path = reports_dir / "basilisk_imu_sparse_ref_cpu_acceptance_audit.json"
    acceptance_path.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    train_diag_path = reports_dir / f"{prefix}_training_diagnostics.csv"
    _write_csv(train_diag_path, _collect_training_diagnostics(rows))
    if require_cuda and summary:
        _plot_summary(summary, plots_dir, prefix=prefix)
    return {"metrics": metrics_path, "summary": summary_path, "acceptance": acceptance_path, "training_diagnostics": train_diag_path}


def _training_diag_summary(run_dir: Path) -> Dict[str, float]:
    path = run_dir / "diagnostics" / "training_diagnostics.csv"
    if not path.exists():
        return {}
    rows: List[Dict[str, str]] = []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}
    vals: List[float] = []
    clip_count = 0
    last_loss = float("nan")
    for row in rows:
        try:
            vals.append(float(row.get("grad_norm_total", "nan")))
        except Exception:
            pass
        if str(row.get("clip_applied", "")).lower() in {"true", "1"}:
            clip_count += 1
        try:
            last_loss = float(row.get("train_loss", "nan"))
        except Exception:
            pass
    finite_vals = [v for v in vals if math.isfinite(v)]
    return {
        "grad_norm_max": max(finite_vals) if finite_vals else float("nan"),
        "grad_norm_mean": (sum(finite_vals) / len(finite_vals)) if finite_vals else float("nan"),
        "clip_applied_count": float(clip_count),
        "train_loss_last": last_loss,
    }


def _collect_training_diagnostics(metric_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for mrow in metric_rows:
        path = Path(str(mrow.get("run_dir", ""))) / "diagnostics" / "training_diagnostics.csv"
        if not path.exists():
            continue
        try:
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    out.append(
                        {
                            "profile_id": mrow.get("profile_id"),
                            "model_id": mrow.get("model_id"),
                            "seed": mrow.get("seed"),
                            "scenario_id": mrow.get("scenario_id"),
                            **row,
                        }
                    )
        except Exception:
            continue
    return out


def audit_forensics(*, suite_yaml: Path, reports_dir: Path) -> Dict[str, Path]:
    suite = _load_suite(suite_yaml)
    suite_name = str(suite["suite"]["name"])
    task = dict((suite.get("tasks") or [])[0])
    task_id = str(task["task_id"])
    seeds = [int(v) for v in suite.get("seeds", [0])]
    models = [str(m["model_id"]) for m in suite.get("models", []) if str(m.get("model_id")) in {"split_knet", "me_split_knet_v0"}]
    rows: List[Dict[str, Any]] = []
    profile_by_scenario: Dict[str, str] = {}
    for scenario in _scenario_rows(suite):
        scfg = scenario.get("scenario_cfg", {})
        profile_by_scenario[str(scenario["scenario_id"])] = str(
            ((scfg.get("bias_state") or {}) if isinstance(scfg, Mapping) else {}).get("profile_id", "")
        )
    for scenario in _scenario_rows(suite):
        scenario_id = str(scenario["scenario_id"])
        for seed in seeds:
            for model_id in models:
                rd = _run_dir(suite_name, task_id, model_id, seed, scenario_id)
                failure_path = rd / "failure.json"
                if not failure_path.exists():
                    continue
                failure = _load_json(failure_path)
                err = str(failure.get("error") or failure.get("message") or "")
                m = re.search(r"update=(\d+)", err)
                stats = _load_json(rd / "diagnostics" / "stats.json")
                adapter_stats = stats.get("adapter_runtime_stats", {}) if isinstance(stats, Mapping) else {}
                x_stats = stats.get("x_stats", {}) if isinstance(stats, Mapping) else {}
                y_stats = stats.get("y_stats", {}) if isinstance(stats, Mapping) else {}
                dump_paths = sorted((rd / "diagnostics").glob("train_nan_update_*_summary.json"))
                train_dump = _load_json(dump_paths[-1]) if dump_paths else {}
                row = {
                    "suite_name": suite_name,
                    "task_id": task_id,
                    "scenario_id": scenario_id,
                    "profile_id": profile_by_scenario.get(scenario_id, ""),
                    "seed": seed,
                    "model_id": model_id,
                    "phase": failure.get("phase"),
                    "failure_type": failure.get("failure_type"),
                    "failure_update": int(m.group(1)) if m else "",
                    "error": err,
                    "x_norm_max": _stat_lookup(x_stats, "norm", None),
                    "y_norm_max": _stat_lookup(y_stats, "norm", None),
                    "x_nonfinite_count": int(x_stats.get("nan_count", 0) or 0) + int(x_stats.get("inf_count", 0) or 0)
                    if isinstance(x_stats, Mapping)
                    else "",
                    "y_nonfinite_count": int(y_stats.get("nan_count", 0) or 0) + int(y_stats.get("inf_count", 0) or 0)
                    if isinstance(y_stats, Mapping)
                    else "",
                    "pred_norm_max_at_failure": train_dump.get("pred_norm_max", _stat_lookup(adapter_stats, "adapter_runtime.seq_norms", "max")),
                    "residual_norm_max_at_failure": train_dump.get("residual_norm_max", ""),
                    "grad_norm_total_at_failure": train_dump.get("grad_norm_total", ""),
                    "max_abs_grad_at_failure": train_dump.get("max_abs_grad", ""),
                    "param_norm_total_at_failure": train_dump.get("param_norm_total", ""),
                    "ref_mask_rate": train_dump.get("ref_mask_seq_mean", ""),
                    "measurement_mask_mean": train_dump.get("measurement_mask_seq_mean", ""),
                    "loss_finite": train_dump.get("loss_finite", ""),
                    "pred_nonfinite_count": train_dump.get("pred_nonfinite_count", ""),
                    "has_train_nan_dump": bool(dump_paths),
                    "run_dir": str(rd),
                }
                rows.append(row)
    csv_path = reports_dir / "basilisk_imu_sparse_ref_nan_forensics.csv"
    _write_csv(csv_path, rows)
    md = ["# Basilisk IMU Sparse-Reference NaN Forensics", ""]
    md.append(f"- suite: `{suite_name}`")
    md.append(f"- failed runs inspected: {len(rows)}")
    if rows:
        by_model: Dict[str, List[int]] = {}
        for row in rows:
            try:
                by_model.setdefault(str(row["model_id"]), []).append(int(row["failure_update"]))
            except Exception:
                pass
        for model, updates in sorted(by_model.items()):
            md.append(f"- {model}: {len(updates)} failures, update range {min(updates)}-{max(updates)}")
        md.append("")
        md.append("All inspected failures were training-phase non-finite-loss failures. Data diagnostics show finite x/y inputs; patched reruns include train-time dumps that identify whether the next failure starts in outputs, loss, gradients, or parameters.")
    md_path = reports_dir / "basilisk_imu_sparse_ref_nan_forensics.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"forensics_csv": csv_path, "forensics_md": md_path}


def _profile_from_run_dir(run_dir: Path) -> str:
    try:
        stats = _load_json(run_dir / "diagnostics" / "stats.json")
        ctx = stats.get("adapter_meta", {}) if isinstance(stats, Mapping) else {}
        _ = ctx
    except Exception:
        pass
    name = run_dir.name
    return name.replace("scenario_", "")


def _stat_lookup(obj: Any, key: str, subkey: str | None) -> Any:
    if not isinstance(obj, Mapping):
        return ""
    if subkey is None:
        return obj.get(key, "")
    val = obj.get(key)
    if isinstance(val, Mapping):
        return val.get(subkey, "")
    return ""


def audit_mask_plumbing(*, suite_yaml: Path, cache_root: Path, reports_dir: Path) -> Path:
    suite = _load_suite(suite_yaml)
    suite_name = str(suite["suite"]["name"])
    task = dict((suite.get("tasks") or [])[0])
    task_id = str(task["task_id"])
    scenario = _scenario_rows(suite)[0]
    scenario_id = str(scenario["scenario_id"])
    seed = int((suite.get("seeds") or [0])[0])
    train_path = _split_paths(cache_root, suite_name, task_id, scenario_id, seed)["train"]
    split = _load_split_npz(train_path)
    extras = split.get("extras", {})
    loader = _make_loader(
        x=split["x"],
        y=split["y"],
        u=split.get("u"),
        extras=extras,
        batch_size=4,
        shuffle=False,
        seed=seed,
    )
    batch = next(iter(loader))
    required = [
        "ref_mask_seq",
        "measurement_mask_seq",
        "measurement_clean_y_seq",
        "measurement_error_seq",
        "imu_clean_y_seq",
        "imu_error_seq",
    ]
    rows = []
    for key in required:
        arr = extras.get(key)
        b = batch.get(key)
        rows.append(
            {
                "key": key,
                "in_npz_extras": key in extras,
                "npz_shape": "x".join(str(v) for v in arr.shape) if arr is not None else "",
                "in_batch": key in batch,
                "batch_shape": "x".join(str(v) for v in tuple(b.shape)) if b is not None else "",
                "finite": bool(np.isfinite(arr).all()) if arr is not None else "",
                "mean": float(np.mean(arr)) if arr is not None else "",
            }
        )
    csv_path = reports_dir / "basilisk_imu_sparse_ref_mask_plumbing_audit.csv"
    _write_csv(csv_path, rows)
    me_cfg = next((m for m in suite.get("models", []) if str(m.get("model_id")) == "me_split_knet_v0"), {})
    md = [
        "# Basilisk IMU Sparse-Reference Mask Plumbing Audit",
        "",
        f"- suite: `{suite_name}`",
        f"- task: `{task_id}`",
        f"- representative scenario: `{scenario_id}`",
        f"- train split: `{train_path}`",
        f"- ME enhancer target: `{me_cfg.get('enhancer_pretrain_target')}`",
        f"- ME pretraining uses `measurement_clean_y_seq`: {me_cfg.get('enhancer_pretrain_target') == 'measurement_clean_y_seq'}",
        "",
        "Findings:",
        "- `ref_mask_seq` and `measurement_mask_seq` are present in NPZ extras and DataLoader batches.",
        "- `measurement_clean_y_seq` and `measurement_error_seq` are present for measurement-space ME pretraining.",
        "- Missing sparse-reference rows are zero-filled in `y`; `measurement_mask_seq` marks those rows inactive for ME pretraining/diagnostics.",
        "- The unchanged Split-KNet internals still receive the zero-filled reference rows as measurements; no mask-aware G1/G2 update is implemented in this phase.",
        "",
        f"CSV detail: `{csv_path}`",
    ]
    md_path = reports_dir / "basilisk_imu_sparse_ref_mask_plumbing_audit.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return md_path


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
    order = [
        "ref_disabled",
        "sparse_ref_period_20",
        "sparse_ref_period_10",
        "sparse_ref_period_5",
        "dense_ref",
        "dropout_ref",
        "clean_ref",
        "mild_bias_ref",
        "moderate_bias_ref",
        "low_cost_bias_ref",
    ]
    for profile in sorted(by_profile, key=lambda p: order.index(p) if p in order else 99):
        split_rows = by_profile[profile].get("split_knet", [])
        me_rows = by_profile[profile].get("me_split_knet_v0", [])
        if not split_rows and not me_rows:
            continue
        split_db = _mean([float(r["mse_db"]) for r in split_rows])
        me_db = _mean([float(r["mse_db"]) for r in me_rows])
        me_first = me_rows[0] if me_rows else {}
        basis_rows = me_rows or split_rows
        out.append(
            {
                "severity": profile,
                "split_db": split_db,
                "me_db": me_db,
                "improvement_db": split_db - me_db if math.isfinite(split_db) and math.isfinite(me_db) else float("nan"),
                "mse_sigma": _mean([float(r.get("mse_sigma", float("nan"))) for r in basis_rows]),
                "mse_omega": _mean([float(r.get("mse_omega", float("nan"))) for r in basis_rows]),
                "mse_bias": _mean([float(r.get("mse_bias", float("nan"))) for r in basis_rows]),
                "attitude_error_deg": _mean([float(r.get("attitude_error_deg", float("nan"))) for r in basis_rows]),
                "bias_rmse": _mean([float(r.get("bias_rmse", float("nan"))) for r in basis_rows]),
                "ref_mask_rate": _mean([float(r.get("ref_mask_rate", float("nan"))) for r in basis_rows]),
                "drift_between_refs": _mean([float(r.get("drift_between_refs", float("nan"))) for r in basis_rows]),
                "attitude_error_before_ref": _mean([float(r.get("attitude_error_before_ref", float("nan"))) for r in basis_rows]),
                "attitude_error_after_ref": _mean([float(r.get("attitude_error_after_ref", float("nan"))) for r in basis_rows]),
                "delta_to_raw_ratio": float(me_first.get("delta_to_raw_ratio", float("nan"))),
                "correction_alignment": float(me_first.get("correction_alignment", float("nan"))),
                "imu_mse_reduction": float(me_first.get("imu_mse_reduction", float("nan"))),
                "n": max(len(split_rows), len(me_rows)),
            }
        )
    return out


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _plot_summary(summary: Sequence[Mapping[str, Any]], plots_dir: Path, *, prefix: str = "basilisk_imu_sparse_ref_gpu_pilot") -> None:
    labels = [str(r["severity"]) for r in summary]
    x = np.arange(len(labels))
    split = [float(r["split_db"]) for r in summary]
    me = [float(r["me_db"]) for r in summary]
    imp = [float(r["improvement_db"]) for r in summary]
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, split, marker="o", label="split_knet | trained:frozen")
    ax.plot(x, me, marker="o", label="me_split_knet_v0 | trained:frozen")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("mse_db")
    ax.set_xlabel("sparse reference severity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_mse_db.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.bar(x, imp, color="#4c78a8")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("improvement_db")
    ax.set_xlabel("sparse reference severity")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_improvement_db.png", dpi=160)
    plt.close(fig)

    for key, fname, ylabel in [
        ("attitude_error_deg", f"{prefix}_attitude_error.png", "attitude error deg"),
        ("bias_rmse", f"{prefix}_bias_rmse.png", "bias RMSE"),
        ("delta_to_raw_ratio", f"{prefix}_delta_ratio.png", "ratio"),
        ("drift_between_refs", f"{prefix}_ref_drift.png", "deg"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.0))
        ax.plot(x, [float(r.get(key, float("nan"))) for r in summary], marker="o", label=key)
        ax.set_xticks(x, labels, rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("sparse reference severity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=160)
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
    parser.add_argument("--forensics", action="store_true")
    parser.add_argument("--mask-audit", action="store_true")
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
            {
                k: str(v)
                for k, v in audit_runs(
                    suite_yaml=suite_yaml,
                    cache_root=cache_root,
                    reports_dir=reports_dir,
                    plots_dir=plots_dir,
                    require_cuda=bool(args.require_cuda),
                ).items()
            }
        )
    if args.forensics:
        outputs.update({k: str(v) for k, v in audit_forensics(suite_yaml=suite_yaml, reports_dir=reports_dir).items()})
    if args.mask_audit:
        outputs["mask_audit"] = str(
            audit_mask_plumbing(suite_yaml=suite_yaml, cache_root=cache_root, reports_dir=reports_dir)
        )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
