from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz.analysis import attitude, comparison, consistency, decomposition, gain, units


AXIS_MODES = ("split", "overlay", "norm")
AXIS_LABELS = ("x", "y", "z")
MAX_RENDER_POINTS = 5000
EMP_STD_WARNING_THRESHOLD = 0.15
TRANSIENT_OPTIONS = ("all", "exclude_10", "exclude_20", "custom")

GAIN_SEMANTIC_LABELS = {
    "learned_combined_kalman_gain": "Learned combined Kalman gain",
    "learned_split_factor_g1": "Learned G1 factor",
    "learned_split_factor_g2": "Learned G2 factor",
}


@dataclass(frozen=True)
class PanelResult:
    figure: go.Figure
    disabled_reason: Optional[str] = None
    downsample_notice: Optional[str] = None


def gain_source_label(meta: Mapping[str, Any], gain_key: str) -> str:
    semantics = meta.get("diagnostic_semantics")
    semantic_value = semantics.get(gain_key) if isinstance(semantics, Mapping) else None
    if semantic_value in GAIN_SEMANTIC_LABELS:
        return GAIN_SEMANTIC_LABELS[str(semantic_value)]
    adapter_meta = meta.get("adapter_meta")
    if (
        gain_key == "gain"
        and isinstance(adapter_meta, Mapping)
        and adapter_meta.get("gain_semantics") == "learned_kalman_gain"
    ):
        return "Learned Kalman gain"
    return {"gain": "Kalman gain", "gain_g1": "G1 factor", "gain_g2": "G2 factor"}.get(
        gain_key,
        gain_key,
    )


def _as_f64(arr: Any) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def _time(traj: Mapping[str, np.ndarray]) -> np.ndarray:
    return _as_f64(traj["t"])


def normalize_analysis_window(n_steps: int, analysis_window: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    total = max(0, int(n_steps))
    if total == 0:
        return {"mode": "all", "label": "All", "start_idx": 0, "n_steps": 0}
    window = analysis_window if isinstance(analysis_window, Mapping) else {}
    start_idx = int(window.get("start_idx", 0))
    start_idx = min(max(start_idx, 0), total - 1)
    mode = str(window.get("mode", "custom" if start_idx else "all"))
    if mode not in TRANSIENT_OPTIONS:
        mode = "custom" if start_idx else "all"
    label = str(window.get("label") or ("All" if start_idx == 0 else f"Exclude first {start_idx} steps"))
    return {"mode": mode, "label": label, "start_idx": start_idx, "n_steps": total - start_idx}


def _window_start(n_steps: int, analysis_window: Optional[Mapping[str, Any]]) -> int:
    return int(normalize_analysis_window(n_steps, analysis_window)["start_idx"])


def _axis_name(idx: int) -> str:
    return AXIS_LABELS[idx] if idx < len(AXIS_LABELS) else f"ch{idx}"


def _wrap_annotation(text: str, width: int = 54) -> str:
    words = str(text).split()
    if not words:
        return ""
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current += " " + word
    lines.append(current)
    return "<br>".join(lines)


def _panel_placeholder(title: str, reason: str) -> PanelResult:
    fig = go.Figure()
    fig.add_annotation(
        text=_wrap_annotation(reason),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        align="center",
        font={"size": 15},
    )
    fig.update_layout(
        title=title,
        height=260,
        margin={"l": 48, "r": 24, "t": 48, "b": 42},
        xaxis={"visible": False},
        yaxis={"visible": False},
        template="plotly_white",
    )
    return PanelResult(fig, disabled_reason=reason)


def _minmax_downsample_indices(y: np.ndarray, max_points: int = MAX_RENDER_POINTS) -> tuple[np.ndarray, bool]:
    values = _as_f64(y)
    n = int(values.shape[0])
    limit = int(max_points)
    if n <= limit or limit <= 0:
        return np.arange(n), False
    bins = max(1, limit // 2)
    edges = np.linspace(0, n, bins + 1, dtype=int)
    indices: list[int] = [0, n - 1]
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        segment = values[start:end]
        finite = np.isfinite(segment)
        if not bool(np.any(finite)):
            indices.append(int(start))
            continue
        local = np.flatnonzero(finite)
        finite_values = segment[local]
        indices.append(int(start + local[int(np.argmin(finite_values))]))
        indices.append(int(start + local[int(np.argmax(finite_values))]))
    return np.array(sorted(set(indices)), dtype=int), True


def _trace_xy(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    indices, downsampled = _minmax_downsample_indices(y)
    return t[indices], y[indices], downsampled


def _add_excluded_region(fig: go.Figure, t: np.ndarray, start_idx: int) -> None:
    if start_idx <= 0 or t.size == 0:
        return
    boundary = min(int(start_idx), int(t.size) - 1)
    kwargs = {
        "x0": float(t[0]),
        "x1": float(t[boundary]),
        "fillcolor": "#6b7280",
        "opacity": 0.14,
        "line_width": 0,
        "layer": "below",
        "annotation_text": "excluded",
        "annotation_position": "top left",
    }
    try:
        fig.add_vrect(row="all", col=1, **kwargs)
    except Exception:
        fig.add_vrect(**kwargs)


def _finish_figure(
    fig: go.Figure,
    title: str,
    rows: int,
    downsampled: bool,
    yaxis_title: str,
    *,
    t: Optional[np.ndarray] = None,
    analysis_window: Optional[Mapping[str, Any]] = None,
) -> PanelResult:
    if t is not None:
        _add_excluded_region(fig, _as_f64(t), _window_start(len(t), analysis_window))
    if downsampled:
        fig.add_annotation(
            text="Downsampled: deterministic min/max envelope per trace",
            xref="paper",
            yref="paper",
            x=1.0,
            y=1.14,
            showarrow=False,
            xanchor="right",
            font={"size": 11, "color": "#8a4b00"},
        )
    fig.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left"},
        height=max(260, 190 * int(rows)),
        margin={"l": 56, "r": 132, "t": 70, "b": 42},
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.01,
            "font": {"size": 10},
        },
        template="plotly_white",
    )
    fig.update_xaxes(title_text="time [s]", matches="x")
    fig.update_yaxes(title_text=yaxis_title)
    notice = "Downsampled: deterministic min/max envelope per trace" if downsampled else None
    return PanelResult(fig, downsample_notice=notice)


def _vector_figure(
    *,
    title: str,
    t: np.ndarray,
    series: Mapping[str, np.ndarray],
    axis_mode: str,
    yaxis_title: str,
    bands: Optional[np.ndarray] = None,
    analysis_window: Optional[Mapping[str, Any]] = None,
) -> PanelResult:
    mode = axis_mode if axis_mode in AXIS_MODES else AXIS_MODES[0]
    prepared = {name: _as_f64(value) for name, value in series.items()}
    dims = max(int(value.shape[1]) for value in prepared.values() if value.ndim == 2)
    downsampled = False

    if mode == "split":
        fig = make_subplots(rows=dims, cols=1, shared_xaxes=True, subplot_titles=[_axis_name(i) for i in range(dims)])
        for dim in range(dims):
            for name, values in prepared.items():
                if dim >= values.shape[1]:
                    continue
                x_ds, y_ds, did = _trace_xy(t, values[:, dim])
                downsampled = downsampled or did
                fig.add_trace(go.Scatter(x=x_ds, y=y_ds, mode="lines", name=f"{name} {_axis_name(dim)}"), row=dim + 1, col=1)
            if bands is not None and dim < bands.shape[1]:
                band = _as_f64(bands[:, dim])
                for sign, label in ((1.0, "+3 sigma"), (-1.0, "-3 sigma")):
                    x_ds, y_ds, did = _trace_xy(t, sign * band)
                    downsampled = downsampled or did
                    fig.add_trace(
                        go.Scatter(x=x_ds, y=y_ds, mode="lines", name=f"{label} {_axis_name(dim)}", line={"dash": "dot"}),
                        row=dim + 1,
                        col=1,
                    )
        return _finish_figure(fig, title, dims, downsampled, yaxis_title, t=t, analysis_window=analysis_window)

    fig = go.Figure()
    if mode == "norm":
        for name, values in prepared.items():
            y = np.linalg.norm(values, axis=1)
            x_ds, y_ds, did = _trace_xy(t, y)
            downsampled = downsampled or did
            fig.add_trace(go.Scatter(x=x_ds, y=y_ds, mode="lines", name=f"{name} norm"))
        if bands is not None:
            y = np.linalg.norm(_as_f64(bands), axis=1)
            for sign, label in ((1.0, "+3 sigma norm"), (-1.0, "-3 sigma norm")):
                x_ds, y_ds, did = _trace_xy(t, sign * y)
                downsampled = downsampled or did
                fig.add_trace(go.Scatter(x=x_ds, y=y_ds, mode="lines", name=label, line={"dash": "dot"}))
    else:
        for name, values in prepared.items():
            for dim in range(values.shape[1]):
                x_ds, y_ds, did = _trace_xy(t, values[:, dim])
                downsampled = downsampled or did
                fig.add_trace(go.Scatter(x=x_ds, y=y_ds, mode="lines", name=f"{name} {_axis_name(dim)}"))
        if bands is not None:
            for dim in range(bands.shape[1]):
                band = _as_f64(bands[:, dim])
                for sign, label in ((1.0, "+3 sigma"), (-1.0, "-3 sigma")):
                    x_ds, y_ds, did = _trace_xy(t, sign * band)
                    downsampled = downsampled or did
                    fig.add_trace(go.Scatter(x=x_ds, y=y_ds, mode="lines", name=f"{label} {_axis_name(dim)}", line={"dash": "dot"}))
    return _finish_figure(fig, title, 1, downsampled, yaxis_title, t=t, analysis_window=analysis_window)


def _has_state_kind(meta: Mapping[str, Any], kind: str) -> bool:
    return _state_slice(meta, kind=kind) is not None


def attitude_rpy_panel(
    meta: Mapping[str, Any],
    traj: Mapping[str, np.ndarray],
    axis_mode: str,
    analysis_window: Optional[Mapping[str, Any]] = None,
) -> PanelResult:
    if not _has_state_kind(meta, "attitude"):
        return _panel_placeholder("A. Attitude RPY Overlay", "Disabled: state_spec has no attitude state")
    if "q_true" not in traj or "q_hat" not in traj:
        return _panel_placeholder("A. Attitude RPY Overlay", "q_true/q_hat not available")
    q_true = _as_f64(traj["q_true"])
    q_hat = _as_f64(traj["q_hat"])
    rpy_true = units.rad_to_deg(np.unwrap(attitude.euler321_from_quat(q_true), axis=0))
    rpy_hat = units.rad_to_deg(np.unwrap(attitude.euler321_from_quat(q_hat), axis=0))
    return _vector_figure(
        title="A. Attitude RPY Overlay",
        t=_time(traj),
        series={"truth": rpy_true, "estimate": rpy_hat},
        axis_mode=axis_mode,
        yaxis_title="deg",
        analysis_window=analysis_window,
    )


def attitude_error_panel(
    meta: Mapping[str, Any],
    traj: Mapping[str, np.ndarray],
    axis_mode: str,
    analysis_window: Optional[Mapping[str, Any]] = None,
) -> PanelResult:
    if not _has_state_kind(meta, "attitude"):
        return _panel_placeholder("B. Attitude Error + 3 sigma", "Disabled: state_spec has no attitude state")
    if "x_true" not in traj or "x_hat" not in traj:
        return _panel_placeholder("B. Attitude Error + 3 sigma", "x_true/x_hat not available")
    err_axis = units.rad_to_deg(attitude.mrp_axis_error_rad(_as_f64(traj["x_true"])[:, :3], _as_f64(traj["x_hat"])[:, :3]))
    bands = None
    if bool(meta.get("capabilities", {}).get("covariance")) and "P" in traj:
        covariance_space = meta.get("state_spec", {}).get("covariance_space")
        bands = units.covariance_axis_band_deg(_as_f64(traj["P"])[:, :3, :3], covariance_space)
    return _vector_figure(
        title="B. Attitude Error + 3 sigma",
        t=_time(traj),
        series={"delta theta": err_axis},
        axis_mode=axis_mode,
        yaxis_title="deg",
        bands=bands,
        analysis_window=analysis_window,
    )


def _state_slice(meta: Mapping[str, Any], *, kind: str) -> Optional[slice]:
    offset = 0
    for item in meta.get("state_spec", {}).get("layout", []):
        dim = int(item.get("dim", 0))
        if item.get("kind") == kind:
            return slice(offset, offset + dim)
        offset += dim
    return None


def bias_panel(
    meta: Mapping[str, Any],
    traj: Mapping[str, np.ndarray],
    axis_mode: str,
    analysis_window: Optional[Mapping[str, Any]] = None,
) -> PanelResult:
    if not bool(meta.get("capabilities", {}).get("bias_state")):
        return _panel_placeholder(
            "C. Bias + 3 sigma",
            "Disabled: capabilities.bias_state=false; no bias state truth is provided for this artifact",
        )
    bias_slice = _state_slice(meta, kind="bias")
    if bias_slice is None or "b_true" not in traj:
        return _panel_placeholder("C. Bias + 3 sigma", "Disabled: bias state or b_true is not present")
    bias_err = units.rad_s_to_deg_h(_as_f64(traj["x_hat"])[:, bias_slice] - _as_f64(traj["b_true"]))
    bands = None
    if bool(meta.get("capabilities", {}).get("covariance")) and "P" in traj:
        bias_sigma = np.sqrt(np.maximum(np.diagonal(_as_f64(traj["P"])[:, bias_slice, bias_slice], axis1=-2, axis2=-1), 0.0))
        bands = units.rad_s_to_deg_h(3.0 * bias_sigma)
    return _vector_figure(
        title="C. Bias + 3 sigma",
        t=_time(traj),
        series={"bias error": bias_err},
        axis_mode=axis_mode,
        yaxis_title="deg/h",
        bands=bands,
        analysis_window=analysis_window,
    )


def innovation_panel(
    meta: Mapping[str, Any],
    traj: Mapping[str, np.ndarray],
    axis_mode: str,
    analysis_window: Optional[Mapping[str, Any]] = None,
) -> PanelResult:
    if not bool(meta.get("capabilities", {}).get("innovation")) or "innov" not in traj:
        return _panel_placeholder("D. Innovation", "Disabled: innovation is not available")
    innov = _as_f64(traj["innov"])
    valid = np.asarray(traj.get("innov_valid", np.ones(innov.shape[0], dtype=bool)), dtype=bool)
    shown = np.where(valid[:, None], innov, np.nan)
    return _vector_figure(
        title="D. Innovation",
        t=_time(traj),
        series={"innovation": shown},
        axis_mode=axis_mode,
        yaxis_title="measurement units",
        analysis_window=analysis_window,
    )


def _mean_judgement(values: np.ndarray, dim: int) -> tuple[float, np.ndarray, bool]:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.nan, np.asarray([np.nan, np.nan], dtype=np.float64), False
    bounds = consistency.chi2_mean_bounds(n_samples=int(finite_values.size), dim=int(dim))
    mean = float(np.nanmean(finite_values))
    return mean, bounds, bool(bounds[0] <= mean <= bounds[1])


def consistency_panel(
    meta: Mapping[str, Any],
    traj: Mapping[str, np.ndarray],
    analysis_window: Optional[Mapping[str, Any]] = None,
    *,
    empirical_available: bool = False,
) -> PanelResult:
    t = _time(traj)
    window = normalize_analysis_window(len(t), analysis_window)
    start_idx = int(window["start_idx"])
    fig = go.Figure()
    downsampled = False
    judgement = [
        f"Window: {window['label']} ({window['n_steps']} steps)",
    ]
    caps = meta.get("capabilities", {})
    has_p = bool(caps.get("covariance")) and "P" in traj
    has_s = bool(caps.get("innovation_cov")) and "S" in traj and "innov" in traj
    if has_p:
        err = _as_f64(traj["x_hat"]) - _as_f64(traj["x_true"])
        nees = consistency.nees(err, _as_f64(traj["P"]))
        x_ds, y_ds, downsampled = _trace_xy(t, nees)
        fig.add_trace(go.Scatter(x=x_ds, y=y_ds, mode="lines", name="NEES"))
        bounds = consistency.chi2_mean_bounds(n_samples=1, dim=err.shape[1])
        for value, name in zip(bounds, ("NEES lower", "NEES upper")):
            fig.add_hline(y=float(value), line_dash="dot", annotation_text=name)
        nees_mean, nees_mean_bounds, nees_ok = _mean_judgement(nees[start_idx:], err.shape[1])
        if np.isfinite(nees_mean):
            fig.add_hline(y=nees_mean, line_dash="solid", line_color="#1f77b4", annotation_text="NEES window mean")
        judgement.append(
            f"NEES mean {nees_mean:.3g} in [{nees_mean_bounds[0]:.3g}, {nees_mean_bounds[1]:.3g}] {'PASS' if nees_ok else 'FAIL'}"
        )
    else:
        judgement.append("NEES unavailable: physical state covariance P is not provided")
        if empirical_available:
            judgement.append("Empirical ensemble uncertainty is available separately; it is not physical covariance P")
    if has_s:
        nis = consistency.nis(_as_f64(traj["innov"]), _as_f64(traj["S"]), valid=traj.get("innov_valid"))
        x_ds, y_ds, did = _trace_xy(t, nis)
        downsampled = downsampled or did
        fig.add_trace(go.Scatter(x=x_ds, y=y_ds, mode="lines", name="NIS"))
        nis_bounds = consistency.chi2_mean_bounds(n_samples=1, dim=traj["innov"].shape[1])
        for value, name in zip(nis_bounds, ("NIS lower", "NIS upper")):
            fig.add_hline(y=float(value), line_dash="dash", annotation_text=name)
        nis_mean, nis_mean_bounds, nis_ok = _mean_judgement(nis[start_idx:], traj["innov"].shape[1])
        if np.isfinite(nis_mean):
            fig.add_hline(y=nis_mean, line_dash="solid", line_color="#d62728", annotation_text="NIS window mean")
        judgement.append(
            f"NIS mean {nis_mean:.3g} in [{nis_mean_bounds[0]:.3g}, {nis_mean_bounds[1]:.3g}] {'PASS' if nis_ok else 'FAIL'}"
        )
    else:
        if "innov" not in traj:
            judgement.append("NIS unavailable: innovation is not available")
        else:
            judgement.append("NIS unavailable: innovation covariance S is not provided")
    if not fig.data:
        return _panel_placeholder("E. NEES / NIS + chi-square", ". ".join(judgement))
    fig.add_annotation(
        text="<br>".join(judgement),
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.26,
        xanchor="left",
        showarrow=False,
        align="left",
        font={"size": 11},
    )
    return _finish_figure(fig, "E. NEES / NIS + chi-square", 1, downsampled, "statistic", t=t, analysis_window=window)


def gain_panel(
    meta: Mapping[str, Any],
    traj: Mapping[str, np.ndarray],
    axis_mode: str,
    analysis_window: Optional[Mapping[str, Any]] = None,
    gain_display: Optional[Mapping[str, Any]] = None,
) -> PanelResult:
    _ = axis_mode
    display = gain_display if isinstance(gain_display, Mapping) else {}
    gain_key = str(display.get("gain_key", "gain"))
    if gain_key not in {"gain", "gain_g1", "gain_g2"}:
        return _panel_placeholder("F. Gain", f"Disabled: unsupported gain key {gain_key!r}")
    if not bool(meta.get("capabilities", {}).get("gain")):
        return _panel_placeholder("F. Gain", "Disabled: Kalman gain is not available")
    if gain_key not in traj:
        return _panel_placeholder("F. Gain", f"Disabled: {gain_key} is not present")
    arr = _as_f64(traj[gain_key])
    if arr.ndim != 3:
        return _panel_placeholder("F. Gain", f"Disabled: {gain_key} must have shape [T,n,m], got {arr.shape}")
    mode = str(display.get("mode", "frobenius"))
    t = _time(traj)
    gain_label = gain_source_label(meta, gain_key)
    shape_label = f"{arr.shape[1]} x {arr.shape[2]}"
    if mode == "element":
        row = min(max(int(display.get("row", 0)), 0), arr.shape[1] - 1)
        col = min(max(int(display.get("col", 0)), 0), arr.shape[2] - 1)
        y = arr[:, row, col]
        name = f"{gain_key}[{row},{col}]"
        title = f"F. {gain_label} ({shape_label}) Element [{row},{col}]"
        yaxis_title = "gain element"
    elif mode == "frobenius":
        y = gain.gain_norm(arr)
        name = f"{gain_key} Frobenius norm"
        title = f"F. {gain_label} ({shape_label}) Frobenius Norm"
        yaxis_title = "gain norm"
    else:
        return _panel_placeholder("F. Gain", f"Disabled: unsupported gain display mode {mode!r}")
    fig = go.Figure()
    x_ds, y_ds, downsampled = _trace_xy(t, y)
    fig.add_trace(go.Scatter(x=x_ds, y=y_ds, mode="lines", name=name))
    return _finish_figure(fig, title, 1, downsampled, yaxis_title, t=t, analysis_window=analysis_window)


def decomposition_panel(traj: Mapping[str, np.ndarray]) -> PanelResult:
    required = ("bias_component", "noise_component", "imu_error")
    if any(key not in traj for key in required):
        return _panel_placeholder("Error Component Decomposition", "Disabled: bias_component/noise_component/imu_error are not present in this artifact")
    residual = decomposition.decomposition_residual(traj["bias_component"], traj["noise_component"], traj["imu_error"])
    frac = decomposition.contribution_fractions(traj["bias_component"], traj["noise_component"])
    fig = go.Figure(
        data=[
            go.Bar(
                x=["deterministic", "stochastic"],
                y=[frac["deterministic"], frac["stochastic"]],
                text=[f"residual max {float(np.nanmax(np.abs(residual))):.3e}", ""],
            )
        ]
    )
    fig.update_layout(title="Error Component Decomposition", height=260, template="plotly_white", margin={"l": 48, "r": 24, "t": 48, "b": 42})
    return PanelResult(fig)


def innovation_acf_panel(meta: Mapping[str, Any], traj: Mapping[str, np.ndarray]) -> PanelResult:
    if not bool(meta.get("capabilities", {}).get("innovation")) or "innov" not in traj:
        return _panel_placeholder("Innovation ACF", "Disabled: innovation is not available")
    innov = _as_f64(traj["innov"])
    valid = np.asarray(traj.get("innov_valid", np.ones(innov.shape[0], dtype=bool)), dtype=bool)
    innov_valid = innov[valid]
    if innov_valid.shape[0] < 3:
        return _panel_placeholder("Innovation ACF", "Disabled: fewer than three valid innovation samples")
    max_lag = min(50, max(1, innov_valid.shape[0] // 10))
    acf = consistency.innovation_acf(innov_valid, max_lag=max_lag)
    lags = np.arange(acf.shape[0])
    return _vector_figure(
        title="Innovation ACF",
        t=lags,
        series={"ACF": acf},
        axis_mode="overlay",
        yaxis_title="correlation",
    )


def _format_metric(value: float, precision: int = 4) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{float(value):.{int(precision)}g}"


def dataset_summary_items(
    meta: Mapping[str, Any],
    aggregate: Optional[Mapping[str, np.ndarray]],
    metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    data_spec = meta.get("data_spec", {})
    n_samples = int(data_spec.get("num_trajectories", 0) or 0)
    n_stored = int(data_spec.get("num_stored_trajectories", 0) or 0)
    split = str(data_spec.get("split", "unknown"))
    split_source = str(data_spec.get("split_source", "legacy_unknown"))
    out = {
        "data_split": split,
        "split_source": split_source,
        "num_trajectories": str(n_samples),
        "num_stored_trajectories": str(n_stored),
        "aggregate_attitude_rmse_deg": "NA",
        "aggregate_generic_state_rmse": "NA",
        "aggregate_bias_rmse_deg_h": "Disabled",
        "aggregate_consistency": "Unavailable",
        "run_status": str(meta.get("run_status") or "unknown"),
    }
    metrics_obj = metrics if isinstance(metrics, Mapping) else {}
    adcs_event = metrics_obj.get("adcs_event")
    if _has_state_kind(meta, "attitude") and isinstance(adcs_event, Mapping):
        value = adcs_event.get("attitude_rmse_deg")
        try:
            out["aggregate_attitude_rmse_deg"] = _format_metric(float(value))
        except (TypeError, ValueError):
            pass
    if aggregate and "err_mean" in aggregate and n_samples > 0:
        mean_error = _as_f64(aggregate["err_mean"])
        mean_square = mean_error * mean_error
        if "emp_std" in aggregate and n_samples >= 2:
            sample_variance = _as_f64(aggregate["emp_std"]) ** 2
            mean_square = mean_square + sample_variance * ((n_samples - 1) / n_samples)
        out["aggregate_generic_state_rmse"] = _format_metric(float(np.sqrt(np.nanmean(mean_square))))
    caps = meta.get("capabilities", {})
    if bool(caps.get("covariance")) and bool(caps.get("innovation_cov")):
        out["aggregate_consistency"] = "Physical P/S available"
    elif bool(caps.get("covariance")):
        out["aggregate_consistency"] = "NIS unavailable: physical S missing"
    elif bool(caps.get("innovation_cov")):
        out["aggregate_consistency"] = "NEES unavailable: physical P missing"
    else:
        out["aggregate_consistency"] = "Physical P/S unavailable"
    return out


def summary_items(
    meta: Mapping[str, Any],
    traj: Mapping[str, np.ndarray],
    analysis_window: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    t = _time(traj)
    window = normalize_analysis_window(len(t), analysis_window)
    start_idx = int(window["start_idx"])
    out: Dict[str, str] = {
        "window": f"{window['label']} ({window['n_steps']} steps)",
        "attitude_rmse_deg": "NA",
        "generic_state_rmse": "NA",
        "bias_rmse_deg_h": "Disabled",
        "coverage_pct": "NA",
    }
    has_attitude = _has_state_kind(meta, "attitude")
    if has_attitude and "q_true" in traj and "q_hat" in traj:
        geodesic = units.rad_to_deg(attitude.geodesic_angle_rad(traj["q_true"], traj["q_hat"]))
        selected = geodesic[start_idx:]
        out["attitude_rmse_deg"] = _format_metric(float(np.sqrt(np.nanmean(selected * selected))))
    elif "x_true" in traj and "x_hat" in traj:
        state_error = _as_f64(traj["x_hat"])[start_idx:] - _as_f64(traj["x_true"])[start_idx:]
        out["generic_state_rmse"] = _format_metric(float(np.sqrt(np.nanmean(state_error * state_error))))
    if has_attitude and "x_true" in traj and "x_hat" in traj and "P" in traj:
        err_axis = units.rad_to_deg(attitude.mrp_axis_error_rad(_as_f64(traj["x_true"])[:, :3], _as_f64(traj["x_hat"])[:, :3]))
        covariance_space = meta.get("state_spec", {}).get("covariance_space")
        band = units.covariance_axis_band_deg(_as_f64(traj["P"])[:, :3, :3], covariance_space)
        covered = np.abs(err_axis[start_idx:]) <= band[start_idx:]
        out["coverage_pct"] = f"{float(np.nanmean(covered) * 100.0):.1f}%"
    bias_slice = _state_slice(meta, kind="bias")
    if bool(meta.get("capabilities", {}).get("bias_state")) and bias_slice is not None and "b_true" in traj:
        bias_err = units.rad_s_to_deg_h(_as_f64(traj["x_hat"])[:, bias_slice] - _as_f64(traj["b_true"]))
        selected_bias = bias_err[start_idx:]
        out["bias_rmse_deg_h"] = _format_metric(float(np.sqrt(np.nanmean(selected_bias * selected_bias))))
    return out


def empirical_sigma_status(meta: Mapping[str, Any], aggregate: Optional[Mapping[str, np.ndarray]]) -> Dict[str, Any]:
    if not aggregate or "emp_std" not in aggregate:
        return {
            "available": False,
            "reason": "aggregate.emp_std is not available",
            "physical_covariance_available": bool(meta.get("capabilities", {}).get("covariance")),
        }
    n_samples = int(meta.get("data_spec", {}).get("num_trajectories", meta.get("N_test", 0)))
    if n_samples < 2:
        return {
            "available": False,
            "reason": "at least two trajectories are required for empirical ensemble uncertainty",
            "n_samples": n_samples,
            "physical_covariance_available": bool(meta.get("capabilities", {}).get("covariance")),
        }
    relative_standard_error = consistency.ensemble_relative_standard_error(n_samples)
    emp = _as_f64(aggregate["emp_std"])
    confidence_interval = consistency.ensemble_sigma_confidence_interval(emp, n_samples=n_samples)
    physical_covariance_available = bool(meta.get("capabilities", {}).get("covariance"))
    status: Dict[str, Any] = {
        "available": True,
        "label": "Empirical ensemble uncertainty",
        "source": "sample standard deviation of trajectory estimation errors",
        "n_samples": n_samples,
        "relative_standard_error": relative_standard_error,
        "emp_std_mean": float(np.nanmean(emp)),
        "confidence_interval_mean": [
            float(np.nanmean(confidence_interval[0])),
            float(np.nanmean(confidence_interval[1])),
        ],
        "warning": relative_standard_error > EMP_STD_WARNING_THRESHOLD,
        "physical_covariance_available": physical_covariance_available,
    }
    if physical_covariance_available and "pred_sigma_mean" in aggregate:
        status["pred_sigma_mean"] = float(np.nanmean(_as_f64(aggregate["pred_sigma_mean"])))
    elif "pred_sigma_mean" in aggregate:
        status["ignored_pred_sigma_reason"] = (
            "aggregate.pred_sigma_mean is ignored because capabilities.covariance=false"
        )
    return status


def add_overlay_traces(
    base: PanelResult,
    overlay: PanelResult,
    *,
    overlay_label: str,
) -> PanelResult:
    if base.disabled_reason or overlay.disabled_reason:
        return base
    figure = go.Figure(base.figure)
    for trace in overlay.figure.data:
        copied = trace.to_plotly_json()
        copied["name"] = f"{overlay_label} · {copied.get('name') or 'trace'}"
        if copied.get("type") == "scatter":
            line = dict(copied.get("line") or {})
            line["dash"] = "dash"
            copied["line"] = line
            copied["opacity"] = 0.72
        figure.add_trace(copied)
    notice_parts = [value for value in (base.downsample_notice, overlay.downsample_notice) if value]
    return PanelResult(
        figure=figure,
        downsample_notice="; ".join(dict.fromkeys(notice_parts)) if notice_parts else None,
    )


def _comparison_payload(
    model: Mapping[str, Any],
    metric: str,
    *,
    gain_row: int,
    gain_col: int,
) -> tuple[np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray]]:
    meta = model["meta"]
    traj = model["traj"]
    aggregate = model.get("aggregate")
    physical_band = None
    empirical_spread = None
    if metric == "attitude_rpy":
        values = comparison.attitude_rpy_deg(meta, traj, estimate=True)
        yaxis = "deg"
    elif metric == "attitude_geodesic_error":
        values = comparison.attitude_geodesic_error_deg(meta, traj)
        yaxis = "deg"
    elif metric in {"attitude_error_components", "attitude_uncertainty"}:
        values = comparison.attitude_error_components_deg(meta, traj)
        yaxis = "deg"
        if metric == "attitude_uncertainty":
            physical_band = comparison.physical_attitude_band_deg(meta, traj)
            if aggregate is not None and "emp_std" in aggregate:
                empirical_spread = comparison.empirical_spread(meta, aggregate, kind="attitude")
    elif metric == "gyro_bias":
        values = comparison.bias_estimate_deg_h(meta, traj)
        yaxis = "deg/h"
    elif metric in {"gyro_bias_error", "gyro_bias_uncertainty"}:
        values = comparison.bias_error_deg_h(meta, traj)
        yaxis = "deg/h"
        if metric == "gyro_bias_uncertainty":
            physical_band = comparison.physical_bias_band_deg_h(meta, traj)
            if aggregate is not None and "emp_std" in aggregate:
                empirical_spread = comparison.empirical_spread(meta, aggregate, kind="bias")
    elif metric == "empirical_attitude_spread":
        if aggregate is None:
            raise ValueError("aggregate empirical uncertainty is unavailable")
        values = comparison.empirical_spread(meta, aggregate, kind="attitude")
        yaxis = "empirical sigma [deg]"
    elif metric == "empirical_bias_spread":
        if aggregate is None:
            raise ValueError("aggregate empirical uncertainty is unavailable")
        values = comparison.empirical_spread(meta, aggregate, kind="bias")
        yaxis = "empirical sigma [deg/h]"
    elif metric == "attitude_correction":
        values = comparison.physical_correction(meta, traj, kind="attitude")
        yaxis = "reconstructed correction [deg]"
    elif metric == "bias_correction":
        values = comparison.physical_correction(meta, traj, kind="bias")
        yaxis = "reconstructed correction [deg/h]"
    else:
        values = comparison.strict_metric_series(
            meta,
            traj,
            metric,
            row=gain_row,
            col=gain_col,
        )
        yaxis = {
            "innovation": "measurement units",
            "innovation_norm": "innovation norm",
            "gain_norm": "gain norm",
            "gain_element": "gain element",
            "attitude_gain_block": "gain block norm",
            "bias_gain_block": "gain block norm",
            "nees": "NEES",
            "nis": "NIS",
            "p_diagonal": "state covariance",
            "s_diagonal": "innovation covariance",
        }.get(metric, "value")
    return _as_f64(values), yaxis, physical_band, empirical_spread


def _comparison_truth(
    model: Mapping[str, Any],
    metric: str,
) -> Optional[np.ndarray]:
    if metric == "attitude_rpy":
        return comparison.attitude_rpy_deg(model["meta"], model["traj"], estimate=False)
    if metric == "gyro_bias":
        return comparison.bias_truth_deg_h(model["meta"], model["traj"])
    return None


def _transparent_hex_color(color: str, alpha: float) -> str:
    value = str(color).lstrip("#")
    if len(value) != 6:
        return "rgba(31,119,180,0.08)"
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{float(alpha):.2f})"


def _comparison_event_markers(traj: Mapping[str, Any]) -> list[tuple[str, np.ndarray]]:
    markers: list[tuple[str, np.ndarray]] = []
    for key, label in (
        ("event_flag", "Event start"),
        ("eclipse_flag", "Eclipse start"),
        ("ref_mask", "Reference update"),
    ):
        if key not in traj:
            continue
        flag = np.asarray(traj[key], dtype=bool)
        if flag.ndim != 1 or flag.size == 0:
            continue
        starts = flag & np.concatenate((np.ones(1, dtype=bool), ~flag[:-1]))
        indices = np.flatnonzero(starts)
        if indices.size:
            markers.append((label, indices))
    return markers


def cross_model_comparison_panel(
    metric: str,
    models: Sequence[Mapping[str, Any]],
    *,
    axis_mode: str = "split",
    show_truth: bool = True,
    show_empirical: bool = False,
    gain_row: int = 0,
    gain_col: int = 0,
) -> PanelResult:
    if not models:
        return _panel_placeholder("Cross-Model Comparison", "No compatible models are selected")
    labels = dict(comparison.PHYSICAL_METRIC_LABELS)
    labels.update(comparison.STRICT_METRIC_LABELS)
    if metric not in labels:
        return _panel_placeholder("Cross-Model Comparison", f"Unsupported comparison metric {metric!r}")

    prepared: list[tuple[Mapping[str, Any], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]] = []
    yaxis_title = "value"
    for model in models:
        values, yaxis_title, physical_band, empirical_spread = _comparison_payload(
            model,
            metric,
            gain_row=gain_row,
            gain_col=gain_col,
        )
        prepared.append((model, values, physical_band, empirical_spread))
    first_t = _as_f64(models[0]["traj"]["t"])
    if any(not np.array_equal(first_t, _as_f64(model["traj"]["t"])) for model in models[1:]):
        return _panel_placeholder("Cross-Model Comparison", "Time axis mismatch; interpolation is not allowed")

    dimensions = max(1, max(1 if values.ndim == 1 else int(values.shape[1]) for _, values, _, _ in prepared))
    mode = axis_mode if axis_mode in AXIS_MODES else "split"
    if mode == "norm" and dimensions > 1:
        rows = 1
    elif mode == "split" and dimensions > 1:
        rows = dimensions
    else:
        rows = 1
    if rows > 1:
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[_axis_name(index) for index in range(rows)],
        )
    else:
        fig = go.Figure()

    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#7f7f7f")
    downsampled = False

    def add_trace(trace: go.Scatter, row: int = 1) -> None:
        if rows > 1:
            fig.add_trace(trace, row=row, col=1)
        else:
            fig.add_trace(trace)

    if show_truth:
        truth = _comparison_truth(models[0], metric)
        if truth is not None:
            truth_arr = _as_f64(truth)
            truth_dims = 1 if truth_arr.ndim == 1 else int(truth_arr.shape[1])
            truth_plot_dims = 1 if mode == "norm" or truth_dims == 1 else truth_dims
            for dim in range(truth_plot_dims):
                values = truth_arr if truth_arr.ndim == 1 else (
                    np.linalg.norm(truth_arr, axis=1) if mode == "norm" else truth_arr[:, dim]
                )
                x_ds, y_ds, did = _trace_xy(first_t, values)
                downsampled = downsampled or did
                suffix = "" if truth_dims == 1 or mode == "norm" else f" {_axis_name(dim)}"
                add_trace(
                    go.Scatter(
                        x=x_ds,
                        y=y_ds,
                        mode="lines",
                        name=f"Truth{suffix}",
                        line={"color": "#111827", "width": 2.2},
                    ),
                    row=dim + 1,
                )

    for model_index, (model, values, physical_band, empirical_spread) in enumerate(prepared):
        label = str(model.get("label") or model["meta"].get("model_id") or f"model {model_index + 1}")
        color = colors[model_index % len(colors)]
        value_dims = 1 if values.ndim == 1 else int(values.shape[1])
        plot_dims = 1 if mode == "norm" or value_dims == 1 else value_dims
        for dim in range(plot_dims):
            if values.ndim == 1:
                y = values
            elif mode == "norm":
                y = np.linalg.norm(values, axis=1)
            else:
                y = values[:, dim]
            x_ds, y_ds, did = _trace_xy(first_t, y)
            downsampled = downsampled or did
            suffix = "" if value_dims == 1 or mode == "norm" else f" {_axis_name(dim)}"
            add_trace(
                go.Scatter(
                    x=x_ds,
                    y=y_ds,
                    mode="lines",
                    name=f"{label}{suffix}",
                    legendgroup=label,
                    line={"color": color, "width": 1.8},
                ),
                row=dim + 1,
            )

            if physical_band is not None:
                band = _as_f64(physical_band)
                band_y = np.linalg.norm(band, axis=1) if mode == "norm" else band[:, dim]
                x_upper, y_upper, did_upper = _trace_xy(first_t, band_y)
                x_lower, y_lower, did_lower = _trace_xy(first_t, -band_y)
                downsampled = downsampled or did_upper or did_lower
                add_trace(
                    go.Scatter(
                        x=x_upper,
                        y=y_upper,
                        mode="lines",
                        name=f"{label} physical +3 sigma{suffix}",
                        legendgroup=f"{label}-physical",
                        line={"color": color, "dash": "dash", "width": 1},
                        showlegend=dim == 0,
                    ),
                    row=dim + 1,
                )
                add_trace(
                    go.Scatter(
                        x=x_lower,
                        y=y_lower,
                        mode="lines",
                        name=f"{label} physical -3 sigma{suffix}",
                        legendgroup=f"{label}-physical",
                        line={"color": color, "dash": "dash", "width": 1},
                        fill="tonexty",
                        fillcolor=_transparent_hex_color(color, 0.08),
                        showlegend=False,
                    ),
                    row=dim + 1,
                )

            if show_empirical and empirical_spread is not None:
                spread = _as_f64(empirical_spread)
                spread_y = np.linalg.norm(spread, axis=1) if mode == "norm" else spread[:, dim]
                for sign, sign_label in ((1.0, "+"), (-1.0, "-")):
                    x_ds, y_ds, did = _trace_xy(first_t, sign * spread_y)
                    downsampled = downsampled or did
                    add_trace(
                        go.Scatter(
                            x=x_ds,
                            y=y_ds,
                            mode="lines",
                            name=f"{label} empirical {sign_label}1 sigma (ensemble){suffix}",
                            legendgroup=f"{label}-empirical",
                            line={"color": color, "dash": "dot", "width": 1.4},
                            showlegend=dim == 0 and sign > 0,
                        ),
                        row=dim + 1,
                    )

    if metric in {"innovation", "innovation_norm"}:
        base_values = prepared[0][1]
        if base_values.ndim > 1:
            base_values = (
                np.linalg.norm(base_values, axis=1)
                if mode == "norm"
                else base_values[:, 0]
            )
        finite_values = base_values[np.isfinite(base_values)]
        marker_y = float(np.max(finite_values)) if finite_values.size else 0.0
        for marker_index, (label, indices) in enumerate(
            _comparison_event_markers(models[0]["traj"])
        ):
            add_trace(
                go.Scatter(
                    x=first_t[indices],
                    y=np.full(indices.shape, marker_y, dtype=np.float64),
                    mode="markers",
                    name=label,
                    marker={
                        "symbol": "diamond-open",
                        "size": 8,
                        "color": colors[(marker_index + len(prepared)) % len(colors)],
                    },
                )
            )

    title = f"Cross-Model · {labels[metric]}"
    result = _finish_figure(
        fig,
        title,
        rows,
        downsampled,
        yaxis_title,
        t=first_t,
    )
    return result
