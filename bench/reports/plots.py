from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from .aggregate import RunRecord, safe_float, safe_int


def read_metrics_step_csv(path: Path) -> Optional[Dict[str, List[float]]]:
    if not path.exists():
        return None
    try:
        out: Dict[str, List[float]] = {"t": [], "mse_t": [], "rmse_t": [], "mse_db_t": []}
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                out["t"].append(float(row["t"]))
                out["mse_t"].append(float(row["mse_t"]))
                out["rmse_t"].append(float(row["rmse_t"]))
                out["mse_db_t"].append(float(row["mse_db_t"]))
        return out
    except Exception:
        return None


def _group_key(r: RunRecord) -> Tuple[str, str, str]:
    # group by (model_id, scenario_id, track_id)
    return (r.model_id, r.scenario_id, r.track_id)


def _mean_curve(curves: List[List[float]]) -> List[float]:
    if not curves:
        return []
    T = min(len(c) for c in curves)
    out = []
    for i in range(T):
        out.append(sum(c[i] for c in curves) / float(len(curves)))
    return out


def plot_shift_recovery_curves(
    task_id: str,
    records: List[RunRecord],
    out_path: Path,
    t0: Optional[int] = None,
    metric: str = "mse_t",
) -> None:
    """
    Plot mean curve over seeds: (model_id, scenario_id, track_id) groups.
    - Expects metrics_step.csv present for ok runs.
    """
    # collect curves
    groups: Dict[Tuple[str, str, str], List[List[float]]] = {}
    T_max = 0
    t_axis: Optional[List[float]] = None

    for r in records:
        if r.status != "ok":
            continue
        ms = r.run_dir / "metrics_step.csv"
        data = read_metrics_step_csv(ms)
        if not data:
            continue
        if metric not in data:
            continue
        curve = data[metric]
        groups.setdefault(_group_key(r), []).append(curve)
        T_max = max(T_max, len(curve))
        if t_axis is None:
            t_axis = data["t"]

    if not groups:
        # nothing to plot
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for (model_id, scenario_id, track_id), curves in sorted(groups.items()):
        mean_c = _mean_curve(curves)
        if not mean_c:
            continue
        xs = t_axis[: len(mean_c)] if t_axis else list(range(len(mean_c)))
        label = f"{model_id} | scen={scenario_id} | {track_id}"
        plt.plot(xs, mean_c, label=label)

    if t0 is not None:
        plt.axvline(float(t0), linestyle="--")
        plt.text(float(t0) + 1.0, 0.0, "t0", rotation=90, va="bottom")

    plt.xlabel("timestep")
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"Shift recovery curve: {task_id} ({metric})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_severity_sweep(
    task_id: str,
    records: List[RunRecord],
    out_path: Path,
    severity_key: str = "shift.post_shift.R_scale",
) -> None:
    """
    Plot mean scalar MSE vs severity parameter for shift tasks.
    Requires that scenario_id maps to severity levels; we approximate by reading config_snapshot.yaml if available.
    """
    # severity extraction: from config_snapshot.yaml if present
    def get_severity(r: RunRecord) -> Optional[float]:
        cfg = r.run_dir / "config_snapshot.yaml"
        if not cfg.exists():
            return None
        try:
            import yaml  # type: ignore

            obj = yaml.safe_load(cfg.read_text(encoding="utf-8"))
            scen = obj.get("scenario_settings", {}) or {}
            v = scen.get(severity_key, None)
            return safe_float(v)
        except Exception:
            return None

    # group by (model, track) => (severity -> list[mse])
    groups: Dict[Tuple[str, str], Dict[float, List[float]]] = {}
    for r in records:
        if r.status != "ok":
            continue
        sev = get_severity(r)
        if sev is None:
            continue
        if r.mse is None:
            continue
        groups.setdefault((r.model_id, r.track_id), {}).setdefault(float(sev), []).append(float(r.mse))

    if not groups:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()

    for (model_id, track_id), sev_map in sorted(groups.items()):
        xs = sorted(sev_map.keys())
        ys = []
        for x in xs:
            vals = sev_map[x]
            ys.append(sum(vals) / float(len(vals)))
        plt.plot(xs, ys, marker="o", label=f"{model_id} | {track_id}")

    plt.xlabel(severity_key)
    plt.ylabel("mse (mean over seeds)")
    plt.title(f"Mismatch severity sweep: {task_id}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / float(len(xs)))


def plot_track_comparison(
    task_id: str,
    records: List[RunRecord],
    out_path: Path,
) -> Optional[str]:
    """
    Compare trained,frozen vs trained,budgeted.
    - y-axis: recovery_k if available, else mse_db.
    Returns the metric key used or None if no plottable data exists.
    """
    by_model_track: Dict[Tuple[str, str], List[RunRecord]] = {}
    for r in records:
        if r.status != "ok":
            continue
        if r.init_id != "trained":
            continue
        if r.track_id not in ("frozen", "budgeted"):
            continue
        by_model_track.setdefault((r.model_id, r.track_id), []).append(r)

    if not by_model_track:
        return None

    has_recovery = any(r.recovery_k is not None for rs in by_model_track.values() for r in rs)
    metric_key = "recovery_k" if has_recovery else "mse_db"

    models = sorted({m for (m, _t) in by_model_track.keys()})
    if not models:
        return None

    frozen_vals: List[float] = []
    budgeted_vals: List[float] = []
    for m in models:
        rs_f = by_model_track.get((m, "frozen"), [])
        rs_b = by_model_track.get((m, "budgeted"), [])
        if metric_key == "recovery_k":
            vf = _mean([float(r.recovery_k) for r in rs_f if r.recovery_k is not None])
            vb = _mean([float(r.recovery_k) for r in rs_b if r.recovery_k is not None])
        else:
            vf = _mean([float(r.mse_db) for r in rs_f if r.mse_db is not None])
            vb = _mean([float(r.mse_db) for r in rs_b if r.mse_db is not None])
        frozen_vals.append(float("nan") if vf is None else float(vf))
        budgeted_vals.append(float("nan") if vb is None else float(vb))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = list(range(len(models)))
    width = 0.38
    plt.figure()
    plt.bar([x - width / 2.0 for x in xs], frozen_vals, width=width, label="trained,frozen")
    plt.bar([x + width / 2.0 for x in xs], budgeted_vals, width=width, label="trained,budgeted")
    plt.xticks(xs, models, rotation=20)
    plt.xlabel("model_id")
    plt.ylabel(metric_key)
    plt.title(f"Track comparison ({task_id})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return metric_key


def plot_budget_curve(
    task_id: str,
    records: List[RunRecord],
    out_path: Path,
) -> Optional[str]:
    """
    Plot budget curve:
    - x-axis: adapt_updates_used
    - y-axis: mse_db (or recovery_k if mse_db unavailable)
    - grouped by severity_r_scale
    """
    points: Dict[str, List[Tuple[float, float]]] = {}
    has_mse_db = any(r.mse_db is not None for r in records if r.status == "ok")
    metric_key = "mse_db" if has_mse_db else "recovery_k"

    for r in records:
        if r.status != "ok":
            continue
        if r.track_id != "budgeted":
            continue
        if r.adapt_updates_used is None:
            continue
        yv = r.mse_db if metric_key == "mse_db" else r.recovery_k
        if yv is None:
            continue
        sev = "na"
        if r.severity_r_scale is not None:
            sev = f"R_scale={r.severity_r_scale:g}"
        points.setdefault(sev, []).append((float(r.adapt_updates_used), float(yv)))

    if not points:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for sev, vals in sorted(points.items(), key=lambda kv: kv[0]):
        vals_sorted = sorted(vals, key=lambda x: x[0])
        xs = [x for x, _y in vals_sorted]
        ys = [y for _x, y in vals_sorted]
        plt.plot(xs, ys, marker="o", linestyle="-", label=sev)

    plt.xlabel("adapt_updates_used")
    plt.ylabel(metric_key)
    plt.title(f"Budget curve ({task_id})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return metric_key


def plot_ops_tradeoff(
    suite_name: str,
    records: List[RunRecord],
    out_path: Path,
) -> bool:
    """
    Ops scatter:
    - x-axis: total_time_s (fallback eval_time_s)
    - y-axis: mse_db
    - grouped by plan (init_id,track_id)
    """
    groups: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for r in records:
        if r.status != "ok":
            continue
        if r.mse_db is None:
            continue
        x = r.total_time_s if r.total_time_s is not None else r.eval_time_s
        if x is None:
            continue
        groups.setdefault((r.init_id, r.track_id), []).append((float(x), float(r.mse_db)))

    if not groups:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for (init_id, track_id), vals in sorted(groups.items()):
        xs = [x for x, _ in vals]
        ys = [y for _, y in vals]
        plt.scatter(xs, ys, label=f"{init_id},{track_id}", alpha=0.8)

    plt.xlabel("total_time_s")
    plt.ylabel("mse_db")
    plt.title(f"Ops tradeoff ({suite_name})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_mse_db_by_model(
    *,
    task_id: str,
    records: List[RunRecord],
    out_path: Path,
) -> bool:
    """
    Per-task comparison plot:
    - x-axis: model_id
    - y-axis: mse_db
    - grouped by plan (init_id,track_id)
    """
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    model_ids: set[str] = set()

    for r in records:
        if r.status != "ok":
            continue
        if r.mse_db is None:
            continue
        plan = (str(r.init_id), str(r.track_id))
        mid = str(r.model_id)
        grouped.setdefault(plan, {}).setdefault(mid, []).append(float(r.mse_db))
        model_ids.add(mid)

    if not grouped or not model_ids:
        return False

    models = sorted(model_ids)
    plans = sorted(grouped.keys())
    xs = list(range(len(models)))
    width = 0.8 / max(1, len(plans))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()

    for pi, plan in enumerate(plans):
        yvals: List[float] = []
        for mid in models:
            vals = grouped.get(plan, {}).get(mid, [])
            if not vals:
                yvals.append(float("nan"))
            else:
                yvals.append(sum(vals) / float(len(vals)))
        xoff = [x - 0.4 + (pi + 0.5) * width for x in xs]
        label = f"{plan[0]},{plan[1]}"
        plt.bar(xoff, yvals, width=width, label=label)

    plt.xticks(xs, models, rotation=20)
    plt.xlabel("model_id")
    plt.ylabel("mse_db")
    plt.title(f"MSE[dB] by model ({task_id})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_fig5a_mse_vs_inv_r2(
    *,
    records: List[RunRecord],
    out_path: Path,
) -> bool:
    """
    Fig.5(a)-style plot:
      x-axis: inv_r2_db = 10*log10(1/r^2)
      y-axis: mse_db
      lines grouped by (model_id, x_dim, y_dim, T)
    """
    groups: Dict[Tuple[int, int, int, str], Dict[float, List[float]]] = {}
    for r in records:
        if r.status != "ok":
            continue
        if r.mse_db is None or r.inv_r2_db is None:
            continue
        if r.x_dim is None or r.y_dim is None or r.T is None:
            continue
        key = (int(r.x_dim), int(r.y_dim), int(r.T), str(r.model_id))
        x = float(r.inv_r2_db)
        y = float(r.mse_db)
        groups.setdefault(key, {}).setdefault(x, []).append(y)

    if not groups:
        return False

    style_by_model: Dict[str, Dict[str, str]] = {
        "kalmannet_tsp": {"linestyle": "-", "marker": "o"},
        "oracle_kf": {"linestyle": "--", "marker": "s"},
        "nominal_kf": {"linestyle": ":", "marker": "^"},
        "oracle_shift_kf": {"linestyle": "-.", "marker": "D"},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for (x_dim, y_dim, t_len, model_id), x_map in sorted(groups.items()):
        xs = sorted(x_map.keys())
        ys = []
        for x in xs:
            vals = x_map[x]
            ys.append(sum(vals) / float(len(vals)))
        style = style_by_model.get(str(model_id), {"linestyle": "-", "marker": "o"})
        label = f"{x_dim}x{y_dim}, T={t_len} | {model_id}"
        plt.plot(
            xs,
            ys,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.6,
            label=label,
        )

    plt.xlabel("inv_r2_db = 10*log10(1/r^2)")
    plt.ylabel("mse_db")
    plt.title("Fig5a-style: MSE[dB] vs (1/r^2)[dB]")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True
