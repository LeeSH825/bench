from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import streamlit as st

from viz.io.loader import VizRun, assert_overlay_compatible, load_run


def discover_runs(runs_root: str | Path) -> list[Path]:
    root = Path(runs_root).expanduser().resolve()
    if not root.exists():
        return []
    return sorted(path.parent for path in root.rglob("meta.json"))


def discover_run_index(runs_root: str | Path) -> tuple[list[VizRun], list[str], float]:
    start = time.perf_counter()
    runs: list[VizRun] = []
    errors: list[str] = []
    for run_dir in discover_runs(runs_root):
        try:
            runs.append(load_run(run_dir, load_aggregate=False, load_metrics=False))
        except Exception as exc:
            errors.append(f"{run_dir}: {exc}")
    runs.sort(
        key=lambda run: (
            str(run.meta.get("data_spec", {}).get("split", "unknown")),
            str(run.meta.get("suite", "")),
            str(run.meta.get("task", "")),
            str(run.meta.get("scenario_id", "")),
            str(run.meta.get("model_id", "")),
            str(run.meta.get("seed", "")),
            str(run.meta.get("track_id", "")),
            str(run.run_dir),
        )
    )
    return runs, errors, time.perf_counter() - start


def _run_field(run: VizRun, field: str) -> Any:
    if field == "data_split":
        return run.meta.get("data_spec", {}).get("split", "unknown")
    return run.meta.get(field)


def filter_run_index(runs: Iterable[VizRun], filters: Mapping[str, Any]) -> list[VizRun]:
    return [
        run
        for run in runs
        if all(value is None or _run_field(run, field) == value for field, value in filters.items())
    ]


def scenario_label(run: VizRun) -> str:
    scenario = run.meta.get("scenario", {})
    display_name = scenario.get("display_name") if isinstance(scenario, Mapping) else None
    scenario_id = str(run.meta.get("scenario_id") or "unknown")
    return f"{display_name} · {scenario_id}" if display_name else f"Unnamed scenario · {scenario_id}"


def split_label(run: VizRun) -> str:
    data_spec = run.meta.get("data_spec", {})
    split = str(data_spec.get("split", "unknown"))
    source = str(data_spec.get("split_source", "legacy_unknown"))
    if split == "unknown":
        return "Unknown (legacy artifact)" if source == "legacy_unknown" else "Unknown"
    return split.title()


def run_label(run: VizRun, runs_root: str | Path) -> str:
    root = Path(runs_root).expanduser().resolve()
    try:
        relative = run.run_dir.relative_to(root)
    except ValueError:
        relative = run.run_dir
    return (
        f"{split_label(run)} · {run.meta.get('suite')} · {run.meta.get('task')} · "
        f"{run.meta.get('scenario_id')} · {run.meta.get('model_id')} · seed {run.meta.get('seed')} · {relative}"
    )


def _query_value(name: str) -> Optional[str]:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _run_from_query(runs: Iterable[VizRun], runs_root: str | Path, query_name: str) -> Optional[VizRun]:
    query = _query_value(query_name)
    if not query:
        return None
    root = Path(runs_root).expanduser().resolve()
    for run in runs:
        if query == str(run.run_dir):
            return run
        try:
            if query == str(run.run_dir.relative_to(root)):
                return run
        except ValueError:
            pass
    return None


def _select_filter(
    *,
    label: str,
    field: str,
    candidates: list[VizRun],
    preferred: Any,
    key: str,
    format_func: Any = None,
) -> tuple[Any, list[VizRun]]:
    values = sorted({_run_field(run, field) for run in candidates}, key=lambda value: str(value))
    if not values:
        raise RuntimeError(f"no values available for navigation field {field}")
    current = st.session_state.get(key)
    if current not in values:
        st.session_state.pop(key, None)
        current = preferred if preferred in values else values[0]
    selectbox_options = {
        "index": values.index(current),
        "key": key,
    }
    if format_func is not None:
        selectbox_options["format_func"] = format_func
    value = st.selectbox(label, values, **selectbox_options)
    return value, filter_run_index(candidates, {field: value})


def render_run_picker(runs_root: str | Path) -> tuple[Optional[VizRun], Optional[VizRun], Optional[str]]:
    runs, errors, scan_seconds = discover_run_index(runs_root)
    if not runs:
        st.error(f"No valid visualization runs found under {Path(runs_root).expanduser().resolve()}")
        if errors:
            st.code("\n".join(errors))
        return None, None, None
    target = _run_from_query(runs, runs_root, "run") or runs[0]
    preferred = {
        "data_split": _run_field(target, "data_split"),
        "suite": target.meta.get("suite"),
        "task": target.meta.get("task"),
        "scenario_id": target.meta.get("scenario_id"),
        "model_id": target.meta.get("model_id"),
        "seed": target.meta.get("seed"),
        "track_id": target.meta.get("track_id"),
        "init_id": target.meta.get("init_id"),
    }
    candidates = runs
    nav_columns = st.columns(4)
    with nav_columns[0]:
        split, candidates = _select_filter(
            label="Data split",
            field="data_split",
            candidates=candidates,
            preferred=preferred["data_split"],
            key="nav_data_split",
            format_func=lambda value: (
                "Unknown (legacy artifact)" if value == "unknown" else str(value).title()
            ),
        )
    with nav_columns[1]:
        _, candidates = _select_filter(
            label="Suite", field="suite", candidates=candidates, preferred=preferred["suite"], key="nav_suite"
        )
    with nav_columns[2]:
        _, candidates = _select_filter(
            label="Task", field="task", candidates=candidates, preferred=preferred["task"], key="nav_task"
        )
    scenario_runs = {run.meta.get("scenario_id"): run for run in candidates}
    with nav_columns[3]:
        _, candidates = _select_filter(
            label="Scenario",
            field="scenario_id",
            candidates=candidates,
            preferred=preferred["scenario_id"],
            key="nav_scenario",
            format_func=lambda value: scenario_label(scenario_runs[value]),
        )
    detail_columns = st.columns(4)
    with detail_columns[0]:
        _, candidates = _select_filter(
            label="Model", field="model_id", candidates=candidates, preferred=preferred["model_id"], key="nav_model"
        )
    with detail_columns[1]:
        _, candidates = _select_filter(
            label="Seed", field="seed", candidates=candidates, preferred=preferred["seed"], key="nav_seed"
        )
    with detail_columns[2]:
        _, candidates = _select_filter(
            label="Track", field="track_id", candidates=candidates, preferred=preferred["track_id"], key="nav_track"
        )
    with detail_columns[3]:
        _, candidates = _select_filter(
            label="Init/checkpoint",
            field="init_id",
            candidates=candidates,
            preferred=preferred["init_id"],
            key="nav_init",
        )
    if len(candidates) > 1:
        preferred_run = target if target in candidates else candidates[0]
        base = st.selectbox(
            "Artifact",
            candidates,
            index=candidates.index(preferred_run),
            format_func=lambda run: str(run.run_dir),
            key="nav_artifact",
        )
    else:
        base = candidates[0]
    scenario = base.meta.get("scenario", {})
    parameters = scenario.get("parameters") if isinstance(scenario, Mapping) else None
    st.caption(
        f"Run index scan: {scan_seconds:.3f} s · Scenario parameters: "
        f"{parameters if parameters else 'No named parameters in artifact metadata'}"
    )
    if errors:
        with st.expander(f"Invalid artifacts excluded ({len(errors)})"):
            st.code("\n".join(errors))

    overlay_options: list[Optional[VizRun]] = [None, *[run for run in runs if run.run_dir != base.run_dir]]
    overlay_target = _run_from_query(runs, runs_root, "overlay")
    overlay_index = overlay_options.index(overlay_target) if overlay_target in overlay_options else 0
    overlay = st.selectbox(
        "Overlay artifact",
        overlay_options,
        index=overlay_index,
        format_func=lambda run: "None" if run is None else run_label(run, runs_root),
        key="overlay_run",
    )
    overlay_error = None
    if overlay is not None:
        try:
            assert_overlay_compatible(base, overlay)
        except Exception as exc:
            overlay_error = str(exc)
    base_loaded = load_run(base.run_dir)
    overlay_loaded = load_run(overlay.run_dir) if overlay is not None else None
    return base_loaded, overlay_loaded, overlay_error
