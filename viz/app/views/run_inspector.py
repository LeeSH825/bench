from __future__ import annotations

import os
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import streamlit as st

from viz.app.components.axis_toggle import AXIS_MODE_OPTIONS, render_axis_toggle
from viz.app.components.comparison_picker import comparison_candidates, comparison_run_label
from viz.app.components.overlay_picker import discover_run_index, render_run_picker
from viz.app.components.regime_strip import render_regime_strip
from viz.analysis import comparison
from viz.figures import panels
from viz.io.loader import TrajectoryInfo, VizRun, assert_overlay_compatible, load_run


WINDOW_MODE_OPTIONS: Mapping[str, str] = {
    "all": "All",
    "exclude_10": "Exclude first 10%",
    "exclude_20": "Exclude first 20%",
    "custom": "Custom",
}


def _query_value(name: str) -> Optional[str]:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _default_axis_mode() -> str:
    query = _query_value("axis_mode") or os.environ.get("VIZ_AXIS_MODE", "split")
    return query if query in AXIS_MODE_OPTIONS else "split"


def _default_window_mode() -> str:
    query = _query_value("window") or os.environ.get("VIZ_WINDOW_MODE", "all")
    return query if query in WINDOW_MODE_OPTIONS else "all"


def _selected_traj_index(run: VizRun) -> int | None:
    query = _query_value("traj")
    if query is not None:
        if query.lower() == "aggregate":
            return None
        try:
            requested = int(query)
            run.trajectory_by_stored_index(requested)
            return requested
        except ValueError:
            pass
        except KeyError:
            pass
    return run.trajectories[0].stored_index if run.trajectories else None


def trajectory_option_label(info: TrajectoryInfo | None) -> str:
    if info is None:
        return "Aggregate summary (no trajectory panels)"
    source = "Unknown" if info.source_trajectory_id is None else str(info.source_trajectory_id)
    length = "Unknown" if info.length_T is None else str(info.length_T)
    event = "Unknown" if info.has_event is None else ("Yes" if info.has_event else "No")
    eclipse = "Unknown" if info.has_eclipse is None else ("Yes" if info.has_eclipse else "No")
    return (
        f"Stored #{info.stored_index} · Source ID {source} · T={length} · "
        f"Event={event} · Eclipse={eclipse}"
    )


def preferred_trajectory_info(
    run: VizRun,
    *,
    query_stored_index: int | None,
    preserve_previous: bool = False,
    previous_was_aggregate: bool = False,
    previous_source_trajectory_id: Any | None = None,
) -> TrajectoryInfo | None:
    if preserve_previous:
        if previous_was_aggregate:
            return None
        if previous_source_trajectory_id is not None:
            try:
                return run.trajectory_by_source_id(previous_source_trajectory_id)
            except KeyError as exc:
                raise KeyError(
                    f"Source ID {previous_source_trajectory_id!r} is not stored by the selected run"
                ) from exc
    if query_stored_index is not None:
        try:
            return run.trajectory_by_stored_index(query_stored_index)
        except KeyError:
            pass
    return run.trajectories[0] if run.trajectories else None


def _window_start_for_mode(mode: str, n_steps: int) -> int:
    total = max(0, int(n_steps))
    if total <= 1:
        return 0
    if mode == "exclude_10":
        return int(total * 0.10)
    if mode == "exclude_20":
        return int(total * 0.20)
    if mode == "custom":
        query = _query_value("window_start")
        try:
            default = int(query) if query is not None else 0
        except ValueError:
            default = 0
        return int(st.number_input("Excluded first steps", min_value=0, max_value=total - 1, value=min(max(default, 0), total - 1), step=1))
    return 0


def _render_window_control(n_steps: int) -> Dict[str, Any]:
    keys = list(WINDOW_MODE_OPTIONS.keys())
    labels = [WINDOW_MODE_OPTIONS[key] for key in keys]
    default = _default_window_mode()
    selected = st.radio(
        "Transient window",
        labels,
        index=keys.index(default),
        horizontal=True,
        key="transient_window",
    )
    mode = keys[labels.index(selected)]
    start_idx = _window_start_for_mode(mode, n_steps)
    if mode == "custom":
        label = f"Exclude first {start_idx} steps"
    else:
        label = WINDOW_MODE_OPTIONS[mode]
    return panels.normalize_analysis_window(n_steps, {"mode": mode, "label": label, "start_idx": start_idx})


def _badge(label: str, value: Any, tone: str = "neutral") -> str:
    colors = {
        "neutral": ("#eef2f7", "#263241"),
        "ok": ("#e7f6ec", "#1f6f3d"),
        "warn": ("#fff3d6", "#805600"),
        "bad": ("#fde7e9", "#9b1c31"),
    }
    bg, fg = colors.get(tone, colors["neutral"])
    return (
        f"<span style='display:inline-block;padding:0.18rem 0.48rem;margin:0.08rem;"
        f"border-radius:0.35rem;background:{bg};color:{fg};font-size:0.82rem;'>"
        f"{label}: {value}</span>"
    )


def _status_tone(value: Any) -> str:
    if value == "ok":
        return "ok"
    if value == "train_nan":
        return "bad"
    return "warn"


def _capability_badges(meta: Mapping[str, Any]) -> str:
    caps = meta.get("capabilities", {})
    badges = []
    for key in ("covariance", "gain", "innovation", "innovation_cov", "bias_state", "regime_labels"):
        tone = "ok" if bool(caps.get(key)) else "warn"
        badges.append(_badge(key, bool(caps.get(key)), tone))
    return "".join(badges)


def build_run_inspector_bundle(
    run: VizRun,
    *,
    traj_idx: int | None = None,
    source_trajectory_id: Any | None = None,
    axis_mode: str,
    analysis_window: Optional[Mapping[str, Any]] = None,
    gain_display: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    start = time.perf_counter()
    if (traj_idx is None) == (source_trajectory_id is None):
        raise ValueError("provide exactly one trajectory selector")
    info = (
        run.trajectory_by_stored_index(int(traj_idx))
        if traj_idx is not None
        else run.trajectory_by_source_id(source_trajectory_id)
    )
    traj = run.load_trajectory(stored_index=info.stored_index)
    window = panels.normalize_analysis_window(len(traj["t"]), analysis_window)
    trajectory_summary = panels.summary_items(run.meta, traj, window)
    bundle = {
        "traj": traj,
        "trajectory_info": info,
        "analysis_window": window,
        "dataset_summary": panels.dataset_summary_items(run.meta, run.aggregate, run.metrics),
        "trajectory_summary": trajectory_summary,
        # Compatibility alias for callers written before dataset/trajectory separation.
        "summary": trajectory_summary,
        "empirical_sigma": panels.empirical_sigma_status(run.meta, run.aggregate),
        "figures": {
            "attitude_rpy": panels.attitude_rpy_panel(run.meta, traj, axis_mode, window),
            "attitude_error": panels.attitude_error_panel(run.meta, traj, axis_mode, window),
            "bias": panels.bias_panel(run.meta, traj, axis_mode, window),
            "innovation": panels.innovation_panel(run.meta, traj, axis_mode, window),
            "consistency": panels.consistency_panel(
                run.meta,
                traj,
                window,
                empirical_available=bool(run.aggregate and "emp_std" in run.aggregate),
            ),
            "gain": panels.gain_panel(run.meta, traj, axis_mode, window, gain_display),
            "decomposition": panels.decomposition_panel(traj),
            "innovation_acf": panels.innovation_acf_panel(run.meta, traj),
        },
    }
    bundle["build_seconds"] = time.perf_counter() - start
    return bundle


def _render_badges(run: VizRun) -> None:
    meta = run.meta
    commit = str(meta.get("commit") or "unknown")[:12]
    parts = [
        _badge("commit", commit),
        _badge("worktree_dirty", bool(meta.get("worktree_dirty")), "warn" if meta.get("worktree_dirty") else "ok"),
        _badge("run_status", meta.get("run_status"), _status_tone(meta.get("run_status"))),
        _badge("sanity_benchmark_only", bool(meta.get("sanity_benchmark_only")), "warn" if meta.get("sanity_benchmark_only") else "neutral"),
        _badge("artifact_version", meta.get("artifact_version")),
        _badge("covariance_space", meta.get("state_spec", {}).get("covariance_space")),
    ]
    st.markdown("".join(parts), unsafe_allow_html=True)
    st.markdown(_capability_badges(meta), unsafe_allow_html=True)
    if meta.get("run_status") != "ok":
        st.error(f"Run status is {meta.get('run_status')}. Failed runs are shown explicitly and should not be used as successful comparisons.")


def _render_data_source(run: VizRun) -> None:
    data_spec = run.meta.get("data_spec", {})
    split = str(data_spec.get("split", "unknown"))
    split_source = str(data_spec.get("split_source", "legacy_unknown"))
    split_label = "Unknown (legacy artifact)" if split == "unknown" and split_source == "legacy_unknown" else split.title()
    st.markdown(
        "".join(
            (
                _badge("Mode", "Offline artifact", "ok"),
                _badge("Data split", split_label, "warn" if split == "unknown" else "ok"),
                _badge("Split source", split_source),
            )
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Generated at: {run.meta.get('created_at') or 'unknown'} · "
        f"Artifact path: {run.run_dir}"
    )


def _render_dataset_summary(summary: Mapping[str, str]) -> None:
    st.subheader(
        f"Dataset Summary · {summary.get('data_split', 'unknown').title()} · "
        f"N={summary.get('num_trajectories', '0')} trajectories"
    )
    if summary.get("aggregate_attitude_rmse_deg", "NA") != "NA":
        aggregate_error = (
            "Dataset aggregate attitude geodesic RMSE [deg]",
            summary.get("aggregate_attitude_rmse_deg", "NA"),
        )
    else:
        aggregate_error = (
            "Dataset aggregate generic-state RMSE [state units]",
            summary.get("aggregate_generic_state_rmse", "NA"),
        )
    cards = {
        "Split provenance": summary.get("split_source", "unknown"),
        "Total / stored": f"{summary.get('num_trajectories', '0')} / {summary.get('num_stored_trajectories', '0')}",
        aggregate_error[0]: aggregate_error[1],
        "Aggregate consistency": summary.get("aggregate_consistency", "Unavailable"),
        "Run-level status": summary.get("run_status", "unknown"),
    }
    _render_metric_cards(cards)


def _render_metric_cards(items: Mapping[str, Any]) -> None:
    columns = st.columns(min(4, max(1, len(items))))
    for index, (label, value) in enumerate(items.items()):
        columns[index % len(columns)].metric(label, value)


def _render_summary(summary: Mapping[str, str]) -> None:
    st.markdown(
        """
        <style>
        .viz-summary-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.45rem;
            margin-bottom: 0.65rem;
        }
        .viz-summary-card {
            border: 1px solid #d9e0ea;
            border-radius: 6px;
            padding: 0.42rem 0.5rem;
            background: #ffffff;
            min-height: 4.15rem;
        }
        .viz-summary-label {
            color: #5b6675;
            font-size: 0.72rem;
            line-height: 1.05rem;
            overflow-wrap: anywhere;
        }
        .viz-summary-value {
            color: #111827;
            font-size: 1.05rem;
            line-height: 1.35rem;
            font-weight: 650;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if summary.get("attitude_rmse_deg", "NA") != "NA":
        primary_error = ("Attitude RMSE [deg]", summary.get("attitude_rmse_deg", "NA"))
    else:
        primary_error = ("Generic state RMSE [state units]", summary.get("generic_state_rmse", "NA"))
    items = [
        ("Window", summary.get("window", "NA")),
        primary_error,
        ("Bias RMSE [deg/h]", summary.get("bias_rmse_deg_h", "NA")),
        ("3 sigma coverage", summary.get("coverage_pct", "NA")),
    ]
    html = ["<div class='viz-summary-grid'>"]
    for label, value in items:
        html.append(
            "<div class='viz-summary-card'>"
            f"<div class='viz-summary-label'>{escape(label)}</div>"
            f"<div class='viz-summary-value'>{escape(str(value))}</div>"
            "</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_trajectory_header(info: TrajectoryInfo) -> None:
    source = "Unknown (legacy artifact)" if info.source_trajectory_id is None else str(info.source_trajectory_id)
    st.subheader(f"Selected Trajectory · Source ID {source}")
    _render_metric_cards(
        {
            "Stored index": info.stored_index,
            "Length T": info.length_T if info.length_T is not None else "Unknown",
            "Event": "Unknown" if info.has_event is None else ("Yes" if info.has_event else "No"),
            "Eclipse": "Unknown" if info.has_eclipse is None else ("Yes" if info.has_eclipse else "No"),
            "Trajectory status": info.run_status,
        }
    )


def _render_empirical_sigma(status: Mapping[str, Any]) -> None:
    if not status.get("available"):
        covariance_text = (
            "Physical covariance available."
            if status.get("physical_covariance_available")
            else "Physical covariance unavailable."
        )
        st.info(f"Empirical ensemble uncertainty unavailable: {status.get('reason')}. {covariance_text}")
        return
    ci_low, ci_high = status["confidence_interval_mean"]
    text = (
        f"Empirical ensemble uncertainty: N={status['n_samples']}, "
        f"relative_standard_error={status['relative_standard_error']:.3f}, "
        f"emp_std_mean={status['emp_std_mean']:.3e}, "
        f"approximate mean confidence interval=[{ci_low:.3e}, {ci_high:.3e}]. "
    )
    if status.get("physical_covariance_available"):
        text += "Physical covariance is available separately."
    else:
        text += "Physical covariance unavailable."
    if "pred_sigma_mean" in status:
        text += f" Physical pred_sigma_mean={status['pred_sigma_mean']:.3e}."
    if status.get("warning"):
        st.warning(text + " Sample uncertainty is high; do not interpret empirical spread as model covariance or a filter fault without more trajectories.")
    else:
        st.info(text)


def available_gain_sources(meta: Mapping[str, Any], traj: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (key, panels.gain_source_label(meta, key))
        for key in ("gain", "gain_g1", "gain_g2")
        if key in traj
    ]


def clamp_gain_element(shape: tuple[int, ...], row: int, col: int) -> tuple[int, int]:
    if len(shape) != 3 or shape[1] <= 0 or shape[2] <= 0:
        return 0, 0
    return min(max(int(row), 0), int(shape[1]) - 1), min(max(int(col), 0), int(shape[2]) - 1)


def _render_gain_display_control(meta: Mapping[str, Any], traj: Mapping[str, Any]) -> Dict[str, Any]:
    sources = available_gain_sources(meta, traj)
    available = [key for key, _label in sources]
    labels = dict(sources)
    if not available:
        return {"gain_key": "gain", "mode": "frobenius"}
    gain_key = st.selectbox(
        "Gain source",
        available,
        index=0,
        key="gain_source",
        format_func=lambda key: labels[key],
    )
    display_labels = ["Frobenius norm", "Matrix element"]
    selected = st.radio("Gain display", display_labels, index=0, horizontal=True, key="gain_display")
    if selected != "Matrix element":
        return {"gain_key": gain_key, "mode": "frobenius"}
    arr = traj[gain_key]
    n_rows = int(arr.shape[1]) if getattr(arr, "ndim", 0) == 3 else 1
    n_cols = int(arr.shape[2]) if getattr(arr, "ndim", 0) == 3 else 1
    current_row, current_col = clamp_gain_element(
        tuple(getattr(arr, "shape", ())),
        int(st.session_state.get("gain_row", 0)),
        int(st.session_state.get("gain_col", 0)),
    )
    st.session_state["gain_row"] = current_row
    st.session_state["gain_col"] = current_col
    row = st.number_input("Gain row", min_value=0, max_value=max(0, n_rows - 1), step=1, key="gain_row")
    col = st.number_input("Gain col", min_value=0, max_value=max(0, n_cols - 1), step=1, key="gain_col")
    return {"gain_key": gain_key, "mode": "element", "row": int(row), "col": int(col)}


def _plot(result: panels.PanelResult) -> None:
    if result.downsample_notice:
        st.caption(result.downsample_notice)
    st.plotly_chart(result.figure, width="stretch")


def _with_overlay(
    bundle: Dict[str, Any],
    overlay_bundle: Dict[str, Any],
    *,
    overlay_label: str,
) -> None:
    for key in bundle["figures"]:
        bundle["figures"][key] = panels.add_overlay_traces(
            bundle["figures"][key],
            overlay_bundle["figures"][key],
            overlay_label=overlay_label,
        )


_PANEL_COMPARISON_METRICS = {
    "attitude_rpy": ("physical", "attitude_rpy"),
    "attitude_error": ("physical", "attitude_geodesic_error"),
    "bias": ("physical", "gyro_bias"),
    "innovation": ("strict", "innovation"),
    "consistency": ("strict", "nees"),
    "gain": ("strict", "gain_norm"),
}


def _panel_model_toggles(
    panel_id: str,
    candidates: Sequence[Any],
) -> list[Any]:
    if not candidates:
        return []
    st.caption(f"{panel_id} models · base: current run; toggle compatible overlays")
    selected: list[Any] = []
    columns = st.columns(min(3, len(candidates)))
    for index, item in enumerate(candidates):
        status = item.compatibility
        compatible = bool(status.get("compatible")) and item.trajectory is not None
        reasons = list(status.get("reasons", []))
        label = comparison_run_label(item.run)
        with columns[index % len(columns)]:
            checked = st.toggle(
                label,
                value=compatible,
                disabled=not compatible,
                key=f"panel_model_{panel_id}_{abs(hash(str(item.run.run_dir)))}",
            )
            if not compatible:
                st.caption(f"Unavailable: {reasons[0] if reasons else 'incompatible semantics'}")
            elif checked:
                selected.append(item)
    return selected


def _panel_overlay_bundles(
    *,
    panel_id: str,
    run: VizRun,
    selected_info: TrajectoryInfo,
    selected_traj: Mapping[str, Any],
    runs_root: str | Path,
    axis_mode: str,
    analysis_window: Optional[Mapping[str, Any]],
    gain_display: Optional[Mapping[str, Any]],
    cache: Dict[str, Dict[str, Any]],
) -> list[tuple[str, Dict[str, Any]]]:
    mode, metric = _PANEL_COMPARISON_METRICS[panel_id]
    candidates = comparison_candidates(
        run,
        discover_run_index(runs_root)[0],
        source_trajectory_id=selected_info.source_trajectory_id,
        mode=mode,
        metric=metric,
    )
    selected = _panel_model_toggles(panel_id, candidates)
    evaluator = (
        comparison.evaluate_physical_metric_compatibility
        if mode == "physical"
        else comparison.evaluate_internal_metric_compatibility
    )
    overlays: list[tuple[str, Dict[str, Any]]] = []
    for item in selected:
        run_key = str(item.run.run_dir)
        try:
            loaded = load_run(item.run.run_dir)
            candidate_traj = loaded.load_trajectory(
                source_trajectory_id=selected_info.source_trajectory_id
            )
            exact = evaluator(
                run.meta,
                loaded.meta,
                metric=metric,
                base_source_trajectory_id=selected_info.source_trajectory_id,
                candidate_source_trajectory_id=item.trajectory.source_trajectory_id,
                base_time=selected_traj["t"],
                candidate_time=candidate_traj["t"],
            )
            if not exact["compatible"]:
                st.warning(
                    f"{panel_id}: {loaded.meta.get('model_id')} blocked: "
                    + "; ".join(exact["reasons"])
                )
                continue
            if run_key not in cache:
                cache[run_key] = build_run_inspector_bundle(
                    loaded,
                    source_trajectory_id=selected_info.source_trajectory_id,
                    axis_mode=axis_mode,
                    analysis_window=analysis_window,
                    gain_display=gain_display,
                )
            overlays.append((str(loaded.meta.get("model_id")), cache[run_key]))
        except Exception as exc:
            st.warning(f"{panel_id}: {item.run.meta.get('model_id')} unavailable: {exc}")
    return overlays


def _plot_panel_with_model_toggles(
    *,
    panel_id: str,
    figure_key: str,
    bundle: Dict[str, Any],
    run: VizRun,
    selected_info: TrajectoryInfo,
    selected_traj: Mapping[str, Any],
    runs_root: str | Path,
    axis_mode: str,
    analysis_window: Optional[Mapping[str, Any]],
    gain_display: Optional[Mapping[str, Any]],
    overlay_cache: Dict[str, Dict[str, Any]],
) -> None:
    figure = bundle["figures"][figure_key]
    for label, overlay_bundle in _panel_overlay_bundles(
        panel_id=panel_id,
        run=run,
        selected_info=selected_info,
        selected_traj=selected_traj,
        runs_root=runs_root,
        axis_mode=axis_mode,
        analysis_window=analysis_window,
        gain_display=gain_display,
        cache=overlay_cache,
    ):
        figure = panels.add_overlay_traces(figure, overlay_bundle["figures"][figure_key], overlay_label=label)
    _plot(figure)


def _comparison_model_label(meta: Mapping[str, Any], metric: str) -> str:
    model = str(meta.get("model_id") or "unknown model")
    if metric.startswith("gain") or metric.endswith("gain_block"):
        return f"{model} ({panels.gain_source_label(meta, 'gain')})"
    return model


def _render_cross_model_comparison(
    *,
    runs_root: str | Path,
    run: VizRun,
    selected_info: TrajectoryInfo,
    selected_traj: Mapping[str, Any],
    axis_mode: str,
) -> None:
    st.subheader("Cross-Model Comparison")
    st.caption(
        "Physical Outputs compares canonical physical quantities across compatible formulations. "
        "Strict Internals retains state, measurement, residual, source, and time semantics guards."
    )
    mode_labels = {"physical": "Physical Outputs", "strict": "Strict Internals"}
    query_mode = _query_value("compare_mode")
    default_mode = query_mode if query_mode in mode_labels else "physical"
    selected_mode_label = st.segmented_control(
        "Comparison mode",
        list(mode_labels.values()),
        default=mode_labels[default_mode],
        key="comparison_mode",
    )
    mode = next(key for key, label in mode_labels.items() if label == selected_mode_label)

    indexed_runs, index_errors, scan_seconds = discover_run_index(runs_root)
    rmse_models: list[tuple[str, VizRun]] = [(str(run.meta.get("model_id") or "base"), run)]
    if selected_info.source_trajectory_id is not None and isinstance(run.meta.get("state_spec"), Mapping):
        rmse_candidates = comparison_candidates(
            run,
            indexed_runs,
            source_trajectory_id=selected_info.source_trajectory_id,
            mode="strict",
            metric="state_error",
        )
        for item in rmse_candidates:
            if item.compatibility.get("compatible") and item.run.run_dir != run.run_dir:
                rmse_models.append((str(item.run.meta.get("model_id") or "unknown"), item.run))

    st.markdown("**Dataset-average RMSE**")
    st.caption(
        "Full evaluation dataset RMSE; selected trajectory RMSE is separate. "
        "Only models with matching state semantics are listed."
    )
    rmse_columns = st.columns(min(4, max(1, len(rmse_models))))
    for index, (label, rmse_run) in enumerate(rmse_models):
        title, value = panels.dataset_rmse_metric(
            rmse_run.meta, rmse_run.aggregate, rmse_run.metrics
        )
        rmse_columns[index % len(rmse_columns)].metric(label, value, help=title)

    available = (
        comparison.available_physical_metrics(run.meta)
        if mode == "physical"
        else comparison.available_internal_metrics(run.meta)
    )
    labels = (
        comparison.PHYSICAL_METRIC_LABELS
        if mode == "physical"
        else comparison.STRICT_METRIC_LABELS
    )
    if not available:
        missing_reason = (
            "comparison_spec is absent or no canonical physical output is declared"
            if mode == "physical"
            else "no strictly comparable internal diagnostic semantics are declared"
        )
        st.info(f"Cross-model comparison unavailable: {missing_reason}.")
        return
    query_metric = _query_value("compare_metric")
    metric_index = available.index(query_metric) if query_metric in available else 0
    metric = st.selectbox(
        "Comparison metric",
        available,
        index=metric_index,
        format_func=lambda key: labels[key],
        key=f"comparison_metric_{mode}",
    )

    candidates = comparison_candidates(
        run,
        indexed_runs,
        source_trajectory_id=selected_info.source_trajectory_id,
        mode=mode,
        metric=metric,
    )
    st.caption(
        f"Comparison index scan: {scan_seconds:.3f} s · selected Source ID "
        f"{selected_info.source_trajectory_id!r} · unselected trajectories are not loaded"
    )
    if index_errors:
        st.caption(f"Invalid artifacts excluded from comparison: {len(index_errors)}")

    st.markdown("**Models to display**")
    st.caption("Compatible models are selected by default; uncheck a model to remove its trace.")
    st.checkbox(
        f"{_comparison_model_label(run.meta, metric)} · base",
        value=True,
        disabled=True,
        key="comparison_base_model",
    )
    base_covariance = bool(run.meta.get("comparison_spec", {}).get("covariance", {}).get("physical"))
    base_status = "Compatible" if base_covariance or "uncertainty" not in metric else "No physical covariance"
    st.markdown(_badge("Base", base_status, "ok" if base_status == "Compatible" else "warn"), unsafe_allow_html=True)

    query_models = {
        value.strip()
        for value in str(_query_value("compare_models") or "").split(",")
        if value.strip()
    }
    selected_candidates = []
    selected_count = 0
    for item in candidates:
        status = item.compatibility
        compatible = bool(status.get("compatible")) and item.trajectory is not None
        model_id = str(item.run.meta.get("model_id"))
        key = f"comparison_candidate_{abs(hash(str(item.run.run_dir)))}_{mode}_{metric}"
        auto_select_compatible = not query_models
        requested = (
            model_id in query_models
            or str(item.run.run_dir) in query_models
            or auto_select_compatible
        )
        already_selected = bool(st.session_state.get(key, requested))
        limit_reached = selected_count >= 3 and not already_selected
        checked = st.checkbox(
            comparison_run_label(item.run),
            value=requested,
            disabled=(not compatible) or limit_reached,
            key=key,
        )
        reasons = list(status.get("reasons", []))
        warnings = list(status.get("warnings", []))
        if not compatible:
            detail = reasons[0] if reasons else "selected Source ID is unavailable"
            st.markdown(_badge("Blocked", detail, "bad"), unsafe_allow_html=True)
            if mode == "strict" and (metric.startswith("gain") or metric.endswith("gain_block")):
                st.caption("Raw gain is incompatible. Use Physical Outputs > Attitude correction or Bias correction when available.")
        elif warnings:
            st.markdown(_badge("Physical outputs only", warnings[0], "warn"), unsafe_allow_html=True)
        else:
            st.markdown(_badge("Compatible", "declared semantics match", "ok"), unsafe_allow_html=True)
        if checked and compatible:
            selected_count += 1
            selected_candidates.append(item)
    if len(candidates) > 3:
        st.caption("At most three overlay models can be selected in addition to the base model.")

    model_data: list[Dict[str, Any]] = [
        {
            "label": _comparison_model_label(run.meta, metric),
            "meta": run.meta,
            "traj": selected_traj,
            "aggregate": run.aggregate,
            "metrics": run.metrics,
        }
    ]
    evaluator = (
        comparison.evaluate_physical_metric_compatibility
        if mode == "physical"
        else comparison.evaluate_internal_metric_compatibility
    )
    for item in selected_candidates:
        try:
            loaded = load_run(item.run.run_dir)
            candidate_traj = loaded.load_trajectory(
                source_trajectory_id=selected_info.source_trajectory_id
            )
            exact = evaluator(
                run.meta,
                loaded.meta,
                metric=metric,
                base_source_trajectory_id=selected_info.source_trajectory_id,
                candidate_source_trajectory_id=item.trajectory.source_trajectory_id,
                base_time=selected_traj["t"],
                candidate_time=candidate_traj["t"],
            )
            if not exact["compatible"]:
                st.error(
                    f"Comparison blocked for {loaded.meta.get('model_id')}: "
                    + "; ".join(exact["reasons"])
                )
                continue
            model_data.append(
                {
                    "label": _comparison_model_label(loaded.meta, metric),
                    "meta": loaded.meta,
                    "traj": candidate_traj,
                    "aggregate": loaded.aggregate,
                    "metrics": loaded.metrics,
                }
            )
        except Exception as exc:
            st.error(
                f"Comparison trajectory could not be loaded for {item.run.run_dir}: {exc}. "
                "No fallback trajectory was substituted."
            )

    option_columns = st.columns(3)
    with option_columns[0]:
        show_truth = st.checkbox(
            "Show truth",
            value=metric in {"attitude_rpy", "gyro_bias"},
            disabled=metric not in {"attitude_rpy", "gyro_bias"},
            key=f"comparison_truth_{mode}_{metric}",
        )
    with option_columns[1]:
        show_empirical = st.checkbox(
            "Show empirical ensemble spread",
            value=False,
            disabled=metric not in {"attitude_uncertainty", "gyro_bias_uncertainty"},
            key=f"comparison_empirical_{mode}_{metric}",
        )
    gain_row = 0
    gain_col = 0
    with option_columns[2]:
        if metric == "gain_element":
            gain_shape = run.meta.get("comparison_spec", {}).get("gain", {}).get("shape", [1, 1])
            gain_row = int(st.number_input("Comparison gain row", 0, max(0, int(gain_shape[0]) - 1), 0))
            gain_col = int(st.number_input("Comparison gain col", 0, max(0, int(gain_shape[1]) - 1), 0))
    result = panels.cross_model_comparison_panel(
        metric,
        model_data,
        axis_mode=axis_mode,
        show_truth=show_truth,
        show_empirical=show_empirical,
        gain_row=gain_row,
        gain_col=gain_col,
    )
    if metric in {"attitude_uncertainty", "gyro_bias_uncertainty"}:
        st.caption(
            "Physical bands use filled dashed +/-3 sigma. Empirical ensemble spread uses "
            "unfilled dotted +/-1 sample sigma and is not model-predicted covariance."
        )
    _plot(result)
    provenance = str(run.meta.get("data_spec", {}).get("source_trajectory_id_source", "unknown"))
    if "fallback" in provenance:
        st.warning(
            f"Source ID provenance: {provenance}. Comparison is valid only for the same dataset file and row ordering."
        )


def render_run_inspector(runs_root: str | Path = "runs") -> None:
    st.title("Run Inspector")
    run, overlay, overlay_error = render_run_picker(runs_root)
    if run is None:
        return
    _render_data_source(run)
    _render_badges(run)

    default_stored_index = _selected_traj_index(run)
    trajectory_options: list[TrajectoryInfo | None] = [None, *run.list_trajectories()]
    previous_run_dir = st.session_state.get("viz_selected_run_dir")
    preserve_previous = previous_run_dir is not None and previous_run_dir != str(run.run_dir)
    selection_notice = None
    try:
        default_option = preferred_trajectory_info(
            run,
            query_stored_index=default_stored_index,
            preserve_previous=preserve_previous,
            previous_was_aggregate=bool(st.session_state.get("viz_selected_aggregate", False)),
            previous_source_trajectory_id=st.session_state.get("viz_selected_source_trajectory_id"),
        )
    except KeyError as exc:
        default_option = None
        selection_notice = f"{exc.args[0]}. No different stored index was substituted."
    selected_info = st.selectbox(
        "Trajectory view",
        trajectory_options,
        index=trajectory_options.index(default_option),
        format_func=trajectory_option_label,
        key=f"trajectory_view_{abs(hash(str(run.run_dir)))}",
    )
    st.session_state["viz_selected_run_dir"] = str(run.run_dir)
    st.session_state["viz_selected_aggregate"] = selected_info is None
    st.session_state["viz_selected_source_trajectory_id"] = (
        selected_info.source_trajectory_id if selected_info is not None else None
    )
    if selection_notice:
        st.warning(selection_notice)

    if overlay is not None and not overlay_error and selected_info is not None:
        if selected_info.source_trajectory_id is None:
            overlay_error = "overlay unavailable: selected legacy trajectory has no source trajectory ID"
        else:
            try:
                assert_overlay_compatible(
                    run,
                    overlay,
                    source_trajectory_id=selected_info.source_trajectory_id,
                )
            except Exception as exc:
                overlay_error = str(exc)
    if overlay is not None and overlay_error:
        detail = str(overlay_error)
        prefix = "overlay blocked: "
        if detail.lower().startswith(prefix):
            detail = detail[len(prefix):]
        st.error(f"Overlay blocked: {detail}")
    elif overlay is not None and selected_info is not None:
        st.success(
            f"Overlay compatible · Source ID {selected_info.source_trajectory_id} · "
            f"{overlay.meta.get('model_id')}"
        )

    dataset_summary = panels.dataset_summary_items(run.meta, run.aggregate, run.metrics)
    _render_dataset_summary(dataset_summary)
    _render_empirical_sigma(panels.empirical_sigma_status(run.meta, run.aggregate))
    if selected_info is None:
        st.info(
            "Aggregate statistics remain available. Select a stored representative trajectory to display time-series panels."
            if run.trajectories
            else "No stored representative trajectories. Aggregate statistics remain available."
        )
        return

    _render_trajectory_header(selected_info)

    left, right = st.columns([2, 1])
    with left:
        axis_col, window_col, gain_col = st.columns([1, 2, 2])
        with axis_col:
            axis_mode = render_axis_toggle(_default_axis_mode())
        with window_col:
            analysis_window = _render_window_control(int(run.meta.get("T") or 0))
        try:
            selected_traj = run.load_trajectory(stored_index=selected_info.stored_index)
        except Exception as exc:
            st.error(
                "Selected trajectory artifact could not be loaded. "
                f"Stored index {selected_info.stored_index}, source ID "
                f"{selected_info.source_trajectory_id!r}: {exc}. "
                "No fallback trajectory was substituted."
            )
            return
        with gain_col:
            gain_display = _render_gain_display_control(run.meta, selected_traj)
    bundle = build_run_inspector_bundle(
        run,
        traj_idx=selected_info.stored_index,
        axis_mode=axis_mode,
        analysis_window=analysis_window,
        gain_display=gain_display,
    )
    if overlay is not None and not overlay_error:
        overlay_bundle = build_run_inspector_bundle(
            overlay,
            source_trajectory_id=selected_info.source_trajectory_id,
            axis_mode=axis_mode,
            analysis_window=analysis_window,
            gain_display=gain_display,
        )
        _with_overlay(bundle, overlay_bundle, overlay_label=str(overlay.meta.get("model_id")))
    st.caption(f"Panel build time: {bundle['build_seconds']:.3f} s")
    _render_cross_model_comparison(
        runs_root=runs_root,
        run=run,
        selected_info=selected_info,
        selected_traj=selected_traj,
        axis_mode=axis_mode,
    )

    figures = bundle["figures"]
    overlay_cache: Dict[str, Dict[str, Any]] = {}
    with left:
        _plot_panel_with_model_toggles(
            panel_id="attitude_rpy",
            figure_key="attitude_rpy",
            bundle=bundle,
            run=run,
            selected_info=selected_info,
            selected_traj=selected_traj,
            runs_root=runs_root,
            axis_mode=axis_mode,
            analysis_window=analysis_window,
            gain_display=gain_display,
            overlay_cache=overlay_cache,
        )
        _plot_panel_with_model_toggles(
            panel_id="attitude_error",
            figure_key="attitude_error",
            bundle=bundle,
            run=run,
            selected_info=selected_info,
            selected_traj=selected_traj,
            runs_root=runs_root,
            axis_mode=axis_mode,
            analysis_window=analysis_window,
            gain_display=gain_display,
            overlay_cache=overlay_cache,
        )
        _plot_panel_with_model_toggles(
            panel_id="bias",
            figure_key="bias",
            bundle=bundle,
            run=run,
            selected_info=selected_info,
            selected_traj=selected_traj,
            runs_root=runs_root,
            axis_mode=axis_mode,
            analysis_window=analysis_window,
            gain_display=gain_display,
            overlay_cache=overlay_cache,
        )
        _plot_panel_with_model_toggles(
            panel_id="innovation",
            figure_key="innovation",
            bundle=bundle,
            run=run,
            selected_info=selected_info,
            selected_traj=selected_traj,
            runs_root=runs_root,
            axis_mode=axis_mode,
            analysis_window=analysis_window,
            gain_display=gain_display,
            overlay_cache=overlay_cache,
        )
        _plot_panel_with_model_toggles(
            panel_id="consistency",
            figure_key="consistency",
            bundle=bundle,
            run=run,
            selected_info=selected_info,
            selected_traj=selected_traj,
            runs_root=runs_root,
            axis_mode=axis_mode,
            analysis_window=analysis_window,
            gain_display=gain_display,
            overlay_cache=overlay_cache,
        )
        _plot_panel_with_model_toggles(
            panel_id="gain",
            figure_key="gain",
            bundle=bundle,
            run=run,
            selected_info=selected_info,
            selected_traj=selected_traj,
            runs_root=runs_root,
            axis_mode=axis_mode,
            analysis_window=analysis_window,
            gain_display=gain_display,
            overlay_cache=overlay_cache,
        )
        render_regime_strip(run.meta, bundle["traj"])
    with right:
        st.markdown("**Selected-trajectory metrics**")
        _render_summary(bundle["trajectory_summary"])
        _plot(figures["decomposition"])
        _plot(figures["innovation_acf"])
        st.info("V-6 attitude 3D panel placeholder")


def main() -> None:
    runs_root = os.environ.get("VIZ_RUNS_ROOT", "runs")
    render_run_inspector(runs_root)
