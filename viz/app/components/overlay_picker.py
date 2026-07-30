from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import streamlit as st

from viz.app.help_content import HELP_TEXT
from viz.io.loader import VizRun, load_run


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
    help: Optional[str] = None,
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
    if help is not None:
        selectbox_options["help"] = help
    value = st.selectbox(label, values, **selectbox_options)
    return value, filter_run_index(candidates, {field: value})


def render_run_picker(
    indexed_runs: Sequence[VizRun],
    index_errors: Sequence[str],
    *,
    runs_root: str | Path,
    scan_seconds: float,
) -> Optional[VizRun]:
    """Render the suite/task/scenario/model/seed/track/init navigation filters.

    Takes an already-discovered run index instead of scanning `runs_root`
    itself, so a single `discover_run_index` call (made once by the caller)
    is shared across the picker, the model toggle, the A-F panels, and the
    Advanced compatibility diagnostics section — see VIZ-R1.3.2.
    """
    runs = list(indexed_runs)
    errors = list(index_errors)
    if not runs:
        st.error(f"No valid visualization runs found under {Path(runs_root).expanduser().resolve()}")
        if errors:
            st.code("\n".join(errors))
        return None
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
            help=HELP_TEXT["data_split"],
        )
    with nav_columns[1]:
        _, candidates = _select_filter(
            label="Suite",
            field="suite",
            candidates=candidates,
            preferred=preferred["suite"],
            key="nav_suite",
            help=HELP_TEXT["suite"],
        )
    with nav_columns[2]:
        _, candidates = _select_filter(
            label="Task",
            field="task",
            candidates=candidates,
            preferred=preferred["task"],
            key="nav_task",
            help=HELP_TEXT["task"],
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
            help=HELP_TEXT["scenario"],
        )
    detail_columns = st.columns(4)
    with detail_columns[0]:
        _, candidates = _select_filter(
            label="Model",
            field="model_id",
            candidates=candidates,
            preferred=preferred["model_id"],
            key="nav_model",
            help=HELP_TEXT["model"],
        )
    with detail_columns[1]:
        _, candidates = _select_filter(
            label="Seed",
            field="seed",
            candidates=candidates,
            preferred=preferred["seed"],
            key="nav_seed",
            help=HELP_TEXT["seed"],
        )
    with detail_columns[2]:
        _, candidates = _select_filter(
            label="Track",
            field="track_id",
            candidates=candidates,
            preferred=preferred["track_id"],
            key="nav_track",
            help=HELP_TEXT["track"],
        )
    with detail_columns[3]:
        _, candidates = _select_filter(
            label="Init/checkpoint",
            field="init_id",
            candidates=candidates,
            preferred=preferred["init_id"],
            key="nav_init",
            help=HELP_TEXT["init_checkpoint"],
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

    return load_run(base.run_dir)
