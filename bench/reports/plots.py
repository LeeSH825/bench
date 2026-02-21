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
