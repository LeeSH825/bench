from __future__ import annotations

import os
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import streamlit as st

from viz.app.components.axis_toggle import AXIS_MODE_OPTIONS, render_axis_toggle
from viz.app.components.comparison_picker import comparison_candidates
from viz.app.components.model_toggle_picker import (
    model_context_key,
    reconcile_selection,
    suite_candidates,
    variant_label,
)
from viz.app.components.overlay_picker import discover_run_index, render_run_picker
from viz.app.components.regime_strip import render_regime_strip
from viz.analysis import comparison
from viz.figures import panels
from viz.io.loader import TrajectoryInfo, VizRun, load_run


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


def _render_dataset_summary(summary: Mapping[str, str], primary_model_id: str) -> None:
    st.subheader(
        f"Dataset Summary · {primary_model_id} · {summary.get('data_split', 'unknown').title()} · "
        f"N={summary.get('num_trajectories', '0')} trajectories"
    )
    st.caption(
        "Reflects the primary navigation model only — unaffected by the Models to display toggles above."
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


def _selected_models_gain_sources(
    meta: Mapping[str, Any],
    traj: Mapping[str, Any],
    indexed_runs: Sequence[VizRun],
    selected_model_dirs: set[str],
) -> list[tuple[str, str]]:
    """Union of gain keys the primary or any *selected* model declares.

    `available_gain_sources` alone only looks at the primary's own loaded
    trajectory. If a selected overlay model provides `gain_g1`/`gain_g2` but
    the primary does not, that key would never appear in the "Gain source"
    dropdown and a user could never pick it — making it impossible to reach
    the "G1 selected, models without G1 excluded" scenario at all. This
    checks each selected model's already-loaded `diagnostic_semantics`
    metadata (no extra NPZ load) to widen the key list; row/col bounds for
    "Matrix element" mode still come only from the primary's own trajectory.
    """
    sources: dict[str, str] = dict(available_gain_sources(meta, traj))
    for candidate in indexed_runs:
        if str(candidate.run_dir) not in selected_model_dirs:
            continue
        semantics = candidate.meta.get("diagnostic_semantics")
        if not isinstance(semantics, Mapping):
            continue
        for key in ("gain", "gain_g1", "gain_g2"):
            if key in semantics and key not in sources:
                sources[key] = panels.gain_source_label(candidate.meta, key)
    return sorted(sources.items(), key=lambda item: item[0])


def _render_gain_display_control(
    meta: Mapping[str, Any],
    traj: Mapping[str, Any],
    indexed_runs: Sequence[VizRun],
    selected_model_dirs: set[str],
) -> Dict[str, Any]:
    sources = _selected_models_gain_sources(meta, traj, indexed_runs, selected_model_dirs)
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
    if gain_key not in traj:
        st.caption(
            f"Matrix element mode needs {gain_key!r} in the primary's own trajectory; "
            "showing Frobenius norm for all selected models instead."
        )
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


def _plot(result: panels.PanelResult | Any) -> None:
    """Render a panel result; tolerate raw Figures from legacy callers."""
    if isinstance(result, panels.PanelResult):
        if result.downsample_notice:
            st.caption(result.downsample_notice)
        figure = result.figure
    else:
        figure = result
    st.plotly_chart(figure, width="stretch")


_PANEL_COMPARISON_METRICS = {
    "attitude_rpy": ("physical", "attitude_rpy"),
    "attitude_error": ("physical", "attitude_geodesic_error"),
    "bias": ("physical", "gyro_bias"),
    "innovation": ("strict", "innovation"),
    "consistency": ("strict", "nees"),
    "gain": ("strict", "gain_norm"),
}


def _checkbox_columns_per_row(count: int) -> int:
    """Column count that avoids a lopsided grid for a given candidate count.

    Fixed modulo-3 wrapping stacks leftover candidates into the first column
    (e.g. 4 candidates -> 2/1/1). This instead picks a row width that divides
    (close to) evenly: 1-3 -> one row, 4 -> 2x2, 5-6 -> 3-wide, 7+ -> 4-wide.
    """
    if count <= 3:
        return max(1, count)
    if count == 4:
        return 2
    if count <= 6:
        return 3
    return 4


def _global_panel_model_toggles(
    run: VizRun,
    selected_info: TrajectoryInfo,
    indexed_runs: Sequence[VizRun],
) -> set[str]:
    """Select overlay models once; every A-F panel consumes this selection.

    The primary navigation run is selected by default but its checkbox is not
    disabled: the user may turn it off as long as at least one other model
    stays selected. Navigation, artifact metadata, and Dataset Summary always
    stay keyed to the primary run regardless of this toggle state.
    """
    if selected_info.source_trajectory_id is None:
        st.caption("Model selection unavailable: selected trajectory has no source ID")
        return {str(run.run_dir)}
    candidates = suite_candidates(
        run,
        indexed_runs,
        source_trajectory_id=selected_info.source_trajectory_id,
    )
    context_key = model_context_key(run.meta)
    state_key = f"viz_model_selection_{abs(hash(context_key))}"
    previous = st.session_state.get(state_key)
    selected = reconcile_selection(
        candidates,
        primary_run_dir=str(run.run_dir),
        previous=previous,
    )

    st.markdown("**Models to display**")
    st.caption(
        "Candidates are limited to the current suite/task/scenario/split/seed/track "
        "context (same evaluation data and protocol). Initialization and training "
        "variants — init=trained/pretrained/untrained/... — are listed separately below "
        "and are not filtered out; toggling them on overlays them like any other model."
    )

    if len(candidates) <= 1:
        only_label = candidates[0].label if candidates else "the primary run"
        st.info(
            f"Only one run artifact is available in the current suite/task/scenario/split/"
            f"seed/track context ({only_label}). Initialization and training variants are "
            "not excluded by this filter. Other runs may be missing the selected Source ID "
            "or another required evaluation-context field."
        )
        # De-duplicate by run_dir: `run` (the primary) is normally also one of
        # the entries in `indexed_runs` (it was selected from that list by
        # the picker), so counting both would double-count it.
        context_match_dirs = {
            str(candidate_run.run_dir)
            for candidate_run in (run, *indexed_runs)
            if model_context_key(candidate_run.meta) == context_key
        }
        with st.expander("Why only one candidate?", expanded=False):
            st.caption(
                f"Indexed runs: {len(indexed_runs)} · "
                f"Matching suite/task/scenario/split/seed/track: {len(context_match_dirs)} · "
                f"Containing Source ID {selected_info.source_trajectory_id!r}: {len(candidates)} · "
                f"Display candidates: {len(candidates)}"
            )

    keys = [f"global_panel_model_v4_{abs(hash((context_key, c.run_dir)))}" for c in candidates]
    warn_key = f"{state_key}__warn"

    def _keep_at_least_one_selected(this_key: str) -> None:
        # Streamlit callbacks run after the widget's own session_state value
        # is updated but before the script reruns, so this is the sanctioned
        # place to revert an interaction (direct reassignment after the
        # widget is instantiated in the main script body raises).
        if not any(st.session_state.get(other_key, False) for other_key in keys):
            st.session_state[this_key] = True
            st.session_state[warn_key] = True

    columns_per_row = _checkbox_columns_per_row(len(candidates))
    new_selected: set[str] = set()
    for row_start in range(0, len(candidates), columns_per_row):
        row = candidates[row_start : row_start + columns_per_row]
        row_columns = st.columns(len(row))
        for offset, (column, candidate) in enumerate(zip(row_columns, row)):
            index = row_start + offset
            run_dir = candidate.run_dir
            key = keys[index]
            label = candidate.label + (" · primary" if run_dir == str(run.run_dir) else "")
            checkbox_kwargs: Dict[str, Any] = {
                "help": "Candidate discovered from the selected Source ID; NPZ is loaded only when selected.",
                "key": key,
                "on_change": _keep_at_least_one_selected,
                "args": (key,),
            }
            # Only seed `value=` on the widget's true first render. Once the
            # key already lives in session_state (including a callback-driven
            # revert on this very rerun), passing `value=` too triggers a
            # Streamlit warning about redundant default vs. session-state value.
            if key not in st.session_state:
                checkbox_kwargs["value"] = run_dir in selected
            with column:
                checked = st.checkbox(label, **checkbox_kwargs)
            if checked:
                new_selected.add(run_dir)

    if st.session_state.pop(warn_key, False):
        st.warning("At least one model must remain selected.")

    st.session_state[state_key] = set(new_selected)
    st.caption(f"Selected models: {len(new_selected)}")
    _render_provenance_notice(candidates, new_selected)
    return new_selected


def _render_provenance_notice(
    candidates: Sequence[Any],
    selected: set[str],
) -> None:
    """Interpretive (non-blocking) notice + table for differing init/training provenance.

    Evaluation-context compatibility (suite/task/scenario/split/seed/track)
    is a hard gate enforced upstream; this is purely informational — it
    never removes a selected run from the toggle set or from any panel.
    """
    selected_candidates = [c for c in candidates if c.run_dir in selected]
    if len(selected_candidates) < 2:
        return
    init_ids = {str(c.metadata.get("init_id") or "unknown") for c in selected_candidates}
    if len(init_ids) > 1:
        st.warning(
            "Selected runs use different initialization/training labels. Overlay is "
            "allowed because the evaluation context (suite/task/scenario/split/seed/track) "
            "matches. Interpret the plot as a baseline, ablation, or adaptation comparison, "
            "not as identical training conditions."
        )
    with st.expander("Run provenance", expanded=False):
        header = "| Run | Model | init_id | track | seed | checkpoint |"
        separator = "|---|---|---|---|---|---|"
        rows = [header, separator]
        for c in selected_candidates:
            model = str(c.metadata.get("model_id") or "unknown model")
            init = str(c.metadata.get("init_id") or "unknown")
            track = str(c.metadata.get("track_id") or "unknown")
            seed = str(c.metadata.get("seed") if c.metadata.get("seed") is not None else "unknown")
            checkpoint = c.metadata.get("checkpoint")
            checkpoint_text = str(checkpoint) if checkpoint else "—"
            run_id = Path(c.run_dir).name
            rows.append(f"| {run_id} | {model} | {init} | {track} | {seed} | {checkpoint_text} |")
        st.markdown("\n".join(rows))


def _panel_overlay_bundles(
    *,
    panel_id: str,
    run: VizRun,
    selected_info: TrajectoryInfo,
    selected_traj: Mapping[str, Any],
    indexed_runs: Sequence[VizRun],
    axis_mode: str,
    analysis_window: Optional[Mapping[str, Any]],
    gain_display: Optional[Mapping[str, Any]],
    cache: Dict[str, Dict[str, Any]],
    selected_model_dirs: set[str],
) -> tuple[list[tuple[str, Dict[str, Any]]], list[tuple[str, str]]]:
    """Build overlay bundles for selected, panel-compatible models.

    Returns (overlays, exclusions). `exclusions` covers every reason a
    selected model does not end up in `overlays` — metadata incompatibility,
    an exact source/time mismatch, a load failure, *and* a case that passes
    those metadata checks but still ends up disabled once the panel is
    actually built (e.g. a selected Gain source key the model does not have).
    Without collecting that last case explicitly, `panels.add_overlay_traces`
    would silently drop the model with no visible explanation.
    """
    mode, metric = _PANEL_COMPARISON_METRICS[panel_id]
    candidates = comparison_candidates(
        run,
        indexed_runs,
        source_trajectory_id=selected_info.source_trajectory_id,
        mode=mode,
        metric=metric,
    )
    evaluator = (
        comparison.evaluate_physical_metric_compatibility
        if mode == "physical"
        else comparison.evaluate_internal_metric_compatibility
    )
    overlays: list[tuple[str, Dict[str, Any]]] = []
    exclusions: list[tuple[str, str]] = []
    for item in candidates:
        if str(item.run.run_dir) not in selected_model_dirs:
            continue
        # variant_label, not bare model_id: once init_id is out of the hard
        # candidate filter, two selected variants can share model_id (e.g.
        # the same model trained vs. untrained) and must stay distinguishable
        # as separate trace/legend/exclusion identities.
        variant_id = variant_label(item.run.meta)
        if not item.compatibility.get("compatible") or item.trajectory is None:
            reasons = "; ".join(item.compatibility.get("reasons", []))
            exclusions.append((variant_id, reasons or "selected Source ID is unavailable"))
            continue
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
                exclusions.append((variant_id, "; ".join(exact["reasons"])))
                continue
            if run_key not in cache:
                cache[run_key] = build_run_inspector_bundle(
                    loaded,
                    source_trajectory_id=selected_info.source_trajectory_id,
                    axis_mode=axis_mode,
                    analysis_window=analysis_window,
                    gain_display=gain_display,
                )
            panel_result = cache[run_key]["figures"][panel_id]
            if panel_result.disabled_reason:
                # Metadata compatibility passed, but the actual panel build
                # for this model came back disabled (e.g. the selected Gain
                # source key is not present for this model). Surface it the
                # same way as the metadata-level exclusions above instead of
                # letting add_overlay_traces drop it silently.
                exclusions.append((variant_id, panel_result.disabled_reason))
                continue
            overlays.append((variant_id, cache[run_key]))
        except Exception as exc:
            exclusions.append((variant_id, f"unavailable: {exc}"))
    return overlays, exclusions


def _plot_panel_with_model_toggles(
    *,
    panel_id: str,
    figure_key: str,
    bundle: Dict[str, Any],
    run: VizRun,
    selected_info: TrajectoryInfo,
    selected_traj: Mapping[str, Any],
    indexed_runs: Sequence[VizRun],
    axis_mode: str,
    analysis_window: Optional[Mapping[str, Any]],
    gain_display: Optional[Mapping[str, Any]],
    overlay_cache: Dict[str, Dict[str, Any]],
    selected_model_dirs: set[str],
) -> Dict[str, str]:
    """Render one A-F panel and return {model_id: "Available" | reason}.

    The return value feeds the Advanced compatibility diagnostics matrix so
    that section can summarize availability without recomputing anything or
    restating the reason text already shown as a caption here.
    """
    model_titles = {
        "attitude_rpy": "A. Attitude RPY Overlay",
        "attitude_error": "B. Attitude Error + 3 sigma",
        "bias": "C. Bias + 3 sigma",
        "innovation": "D. Innovation",
        "consistency": "E. NEES / NIS + chi-square",
        "gain": "F. Gain",
    }
    overlays, exclusions = _panel_overlay_bundles(
        panel_id=panel_id,
        run=run,
        selected_info=selected_info,
        selected_traj=selected_traj,
        indexed_runs=indexed_runs,
        axis_mode=axis_mode,
        analysis_window=analysis_window,
        gain_display=gain_display,
        cache=overlay_cache,
        selected_model_dirs=selected_model_dirs,
    )
    primary_included = str(run.run_dir) in selected_model_dirs
    primary_variant_id = variant_label(run.meta)
    primary_disabled_reason = (
        bundle["figures"][figure_key].disabled_reason if primary_included else None
    )

    # Primary's own missing capability (e.g. the selected Gain source key is
    # not present for the primary model) is reported the same way as an
    # excluded overlay model, instead of just quietly not appearing.
    full_exclusions = list(exclusions)
    if primary_disabled_reason:
        full_exclusions = [(primary_variant_id, primary_disabled_reason)] + full_exclusions

    if len(full_exclusions) == 1:
        model_id, reason = full_exclusions[0]
        st.caption(f"{model_titles[panel_id]}: {model_id} not shown — {reason}")
    elif len(full_exclusions) > 1:
        st.caption(f"{len(full_exclusions)} selected models are not shown in {model_titles[panel_id]}.")
        with st.expander("Why?", expanded=False):
            for model_id, reason in full_exclusions:
                st.caption(f"- {model_id} — {reason}")

    status: Dict[str, str] = {variant_id: reason for variant_id, reason in exclusions}
    status.update({label: "Available" for label, _ in overlays})
    if primary_included:
        status[primary_variant_id] = primary_disabled_reason or "Available"

    if not primary_included and not overlays:
        _plot(panels.no_model_selected_panel(model_titles[figure_key]))
        return status

    ordered_ids = ([primary_variant_id] if primary_included else []) + [label for label, _ in overlays]
    result = (
        panels.label_model_traces(
            bundle["figures"][figure_key],
            model_id=primary_variant_id,
            ordered_model_ids=ordered_ids,
        )
        if primary_included
        else panels.no_model_selected_panel(model_titles[figure_key])
    )
    for label, overlay_bundle in overlays:
        result = panels.add_overlay_traces(
            result,
            panels.label_model_traces(
                overlay_bundle["figures"][figure_key],
                model_id=label,
                ordered_model_ids=ordered_ids,
            ),
            overlay_label="",
            overlay_color=panels.model_color(label, ordered_ids),
        )
    _plot(result)
    return status


_PANEL_MATRIX_COLUMNS = (
    ("attitude_rpy", "Attitude RPY"),
    ("attitude_error", "Attitude Error"),
    ("bias", "Bias"),
    ("innovation", "Innovation"),
    ("gain", "Gain"),
    ("consistency", "NEES/NIS"),
)


def _render_advanced_compatibility_diagnostics(
    *,
    indexed_runs: Sequence[VizRun],
    run: VizRun,
    selected_info: TrajectoryInfo,
    selected_model_dirs: set[str],
    panel_status: Mapping[str, Mapping[str, str]],
) -> None:
    """Dataset-wide RMSE and a per-model x per-panel compatibility matrix.

    This intentionally does not build a comparison time-series figure —
    overlay traces for the selected models already live in the A-F panels
    above — and it intentionally does not restate each panel's full
    exclusion reason text; that already lives as a caption/expander directly
    under the panel it applies to (see `_plot_panel_with_model_toggles`).
    This section only summarizes availability across all panels at once.
    """
    st.caption(
        "Physical Outputs compares canonical physical quantities across compatible formulations. "
        "Strict Internals retains state, measurement, residual, source, and time semantics guards. "
        "Full per-model reasons are shown under each A-F panel above; this section only "
        "summarizes availability so it does not repeat that text."
    )

    rmse_models: list[tuple[str, VizRun]] = [(variant_label(run.meta), run)]
    if selected_info.source_trajectory_id is not None and isinstance(run.meta.get("state_spec"), Mapping):
        rmse_candidates = comparison_candidates(
            run,
            indexed_runs,
            source_trajectory_id=selected_info.source_trajectory_id,
            mode="strict",
            metric="state_error",
        )
        for item in rmse_candidates:
            if (
                str(item.run.run_dir) in selected_model_dirs
                and item.compatibility.get("compatible")
                and item.run.run_dir != run.run_dir
            ):
                rmse_models.append((variant_label(item.run.meta), item.run))

    st.markdown("**Dataset-average RMSE**")
    st.caption(
        "Full evaluation dataset RMSE for the selected models; selected trajectory RMSE is separate. "
        "Only models with matching state semantics are listed."
    )
    rmse_columns = st.columns(min(4, max(1, len(rmse_models))))
    for index, (label, rmse_run) in enumerate(rmse_models):
        title, value = panels.dataset_rmse_metric(
            rmse_run.meta, rmse_run.aggregate, rmse_run.metrics
        )
        rmse_columns[index % len(rmse_columns)].metric(label, value, help=title)

    st.markdown("**Panel compatibility matrix**")
    st.caption(
        "Selected models only. \"Available\" means that model's trace is shown in that panel; "
        "\"Blocked\" means it was excluded — see the caption under that panel above for the reason."
    )
    # Keyed by variant_label (model_id + init_id), not bare model_id — two
    # selected variants of the same model_id must appear as separate rows.
    variant_ids: list[str] = []
    for panel_id, _title in _PANEL_MATRIX_COLUMNS:
        for variant_id in panel_status.get(panel_id, {}):
            if variant_id not in variant_ids:
                variant_ids.append(variant_id)
    if not variant_ids:
        st.caption("No panel status is available yet.")
        return
    header = "| Model | " + " | ".join(title for _, title in _PANEL_MATRIX_COLUMNS) + " |"
    separator = "|" + "---|" * (len(_PANEL_MATRIX_COLUMNS) + 1)
    rows = [header, separator]
    for variant_id in variant_ids:
        cells = []
        for panel_id, _title in _PANEL_MATRIX_COLUMNS:
            status = panel_status.get(panel_id, {}).get(variant_id)
            if status is None:
                cells.append("n/a")
            else:
                cells.append("Available" if status == "Available" else "Blocked")
        rows.append(f"| {variant_id} | " + " | ".join(cells) + " |")
    st.markdown("\n".join(rows))


def render_run_inspector(runs_root: str | Path = "runs") -> None:
    st.title("Run Inspector")
    # Scanned exactly once per rerun and threaded down to every consumer
    # below (the picker, the model toggle, all six A-F panels, and the
    # Advanced compatibility diagnostics section) — see VIZ-R1.3.2. Before
    # this, each of those called discover_run_index(runs_root) independently.
    with st.spinner("Scanning run artifacts..."):
        indexed_runs, index_errors, scan_seconds = discover_run_index(runs_root)
    run = render_run_picker(
        indexed_runs,
        index_errors,
        runs_root=runs_root,
        scan_seconds=scan_seconds,
    )
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

    if selected_info is not None:
        selected_model_dirs = _global_panel_model_toggles(run, selected_info, indexed_runs)
    else:
        selected_model_dirs = {str(run.run_dir)}

    dataset_summary = panels.dataset_summary_items(run.meta, run.aggregate, run.metrics)
    _render_dataset_summary(dataset_summary, str(run.meta.get("model_id") or "unknown model"))
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
            gain_display = _render_gain_display_control(
                run.meta, selected_traj, indexed_runs, selected_model_dirs
            )
    bundle = build_run_inspector_bundle(
        run,
        traj_idx=selected_info.stored_index,
        axis_mode=axis_mode,
        analysis_window=analysis_window,
        gain_display=gain_display,
    )
    st.caption(f"Panel build time: {bundle['build_seconds']:.3f} s")

    figures = bundle["figures"]
    overlay_cache: Dict[str, Dict[str, Any]] = {}
    panel_status: Dict[str, Dict[str, str]] = {}
    with left:
        for panel_id in ("attitude_rpy", "attitude_error", "bias", "innovation", "consistency", "gain"):
            panel_status[panel_id] = _plot_panel_with_model_toggles(
                panel_id=panel_id,
                figure_key=panel_id,
                bundle=bundle,
                run=run,
                selected_info=selected_info,
                selected_traj=selected_traj,
                indexed_runs=indexed_runs,
                axis_mode=axis_mode,
                analysis_window=analysis_window,
                gain_display=gain_display,
                overlay_cache=overlay_cache,
                selected_model_dirs=selected_model_dirs,
            )
        render_regime_strip(run.meta, bundle["traj"])
    with right:
        st.markdown("**Selected-trajectory metrics**")
        _render_summary(bundle["trajectory_summary"])
        _plot(figures["decomposition"])
        _plot(figures["innovation_acf"])
        st.info("V-6 attitude 3D panel placeholder")

    with st.expander("Advanced compatibility diagnostics", expanded=False):
        _render_advanced_compatibility_diagnostics(
            indexed_runs=indexed_runs,
            run=run,
            selected_info=selected_info,
            selected_model_dirs=selected_model_dirs,
            panel_status=panel_status,
        )


def main() -> None:
    runs_root = os.environ.get("VIZ_RUNS_ROOT", "runs")
    render_run_inspector(runs_root)
