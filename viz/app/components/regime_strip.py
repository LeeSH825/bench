from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from viz.analysis import regime


@dataclass(frozen=True)
class RegimeStripResult:
    figure: go.Figure
    intervals: dict[str, list[dict[str, Any]]]
    disabled_reason: str | None = None
    empty_reason: str | None = None


def _flags_from_traj(traj: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in ("eclipse_flag", "event_flag"):
        if key in traj:
            out[key] = np.asarray(traj[key], dtype=bool).reshape(-1)
    return out


def build_regime_strip(meta: Mapping[str, Any], traj: Mapping[str, np.ndarray]) -> RegimeStripResult:
    fig = go.Figure()
    if not bool(meta.get("capabilities", {}).get("regime_labels")):
        return RegimeStripResult(fig, {}, disabled_reason="Regime strip disabled: capabilities.regime_labels=false for this artifact.")
    flags = _flags_from_traj(traj)
    if not flags:
        return RegimeStripResult(fig, {}, disabled_reason="Regime strip disabled: no event_flag or eclipse_flag series is present.")
    t = np.asarray(traj["t"], dtype=np.float64)
    z = np.vstack([values.astype(np.int8) for values in flags.values()])
    fig.add_trace(
        go.Heatmap(
            x=t,
            y=list(flags.keys()),
            z=z,
            showscale=False,
            colorscale=[[0.0, "#e8edf3"], [1.0, "#d1495b"]],
        )
    )
    intervals: dict[str, list[dict[str, Any]]] = {}
    for name, values in flags.items():
        items = []
        for item in regime.true_intervals(values):
            start = int(item["start"])
            end = int(item["end"])
            if t.size == 0:
                start_time = np.nan
                end_time = np.nan
            else:
                start_time = float(t[min(start, t.size - 1)])
                if end < t.size:
                    end_time = float(t[end])
                elif t.size >= 2:
                    end_time = float(t[-1] + (t[-1] - t[-2]))
                else:
                    end_time = float(t[-1])
            items.append(
                {
                    "start": start,
                    "end": end,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
        intervals[name] = items
    empty_reason = None
    if not any(intervals.values()):
        empty_reason = "Regime data present, but no eclipse/event interval occurred."
    fig.update_layout(
        title="Regime Timeline",
        height=150,
        margin={"l": 56, "r": 24, "t": 42, "b": 36},
        template="plotly_white",
        xaxis_title="time [s]",
    )
    return RegimeStripResult(fig, intervals, empty_reason=empty_reason)


def render_regime_strip(meta: Mapping[str, Any], traj: Mapping[str, np.ndarray]) -> None:
    result = build_regime_strip(meta, traj)
    if result.disabled_reason:
        st.info(result.disabled_reason)
        return
    if result.empty_reason:
        st.info(result.empty_reason)
    st.plotly_chart(result.figure, width="stretch")
