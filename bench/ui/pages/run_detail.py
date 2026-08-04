"""Run detail page: identity, progress, metrics, resources, logs, artifacts.

The page is deliberately explicit about what it does *not* know:

* a legacy run's state is labelled as inferred, with its confidence;
* an adapter with only phase-level instrumentation says so, instead of showing
  an empty chart that looks like a broken run;
* checkpoints are listed as "not resume-certified" rather than offering a
  Resume button that would not work.

There are no Stop / Resume / Warm-start controls anywhere on this page, because
this build implements none of them. A disabled button still advertises a feature.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import plotly.graph_objects as go
from dash import dcc, html

from ..api_client import is_error
from ..components import (
    error_panel,
    fidelity_badge,
    format_bytes,
    info_panel,
    instrumentation_badge,
    key_values,
    resume_badge,
    section,
    state_badge,
)

#: Metric series drawn on the loss chart, in draw order.
LOSS_SERIES = ("loss/train_total", "loss/validation_total")

#: Metric series drawn on the evaluation chart.
EVAL_SERIES = ("metric/test_mse", "metric/test_mse_db", "metric/test_rmse", "metric/test_nll")

_LINE_COLOURS = ("#1d4ed8", "#b91c1c", "#15803d", "#a16207", "#7c3aed")


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message, showarrow=False, font={"size": 13, "color": "#64748b"}, xref="paper", yref="paper", x=0.5, y=0.5
    )
    figure.update_layout(
        xaxis={"visible": False}, yaxis={"visible": False}, margin={"l": 40, "r": 20, "t": 30, "b": 40}, height=280,
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
    )
    return figure


def metric_figure(series: Mapping[str, Sequence[Mapping[str, Any]]], names: Sequence[str], title: str) -> go.Figure:
    """Line chart of the requested metric series.

    Points are plotted against their own ``step`` axis. Missing series are simply
    absent — never zero-filled, which would draw a false flat line.
    """
    present = {name: list(series.get(name) or []) for name in names if series.get(name)}
    if not present:
        return _empty_figure(f"No {title.lower()} events recorded for this run.")
    figure = go.Figure()
    for index, (name, points) in enumerate(present.items()):
        xs = [point.get("step") if point.get("step") is not None else i for i, point in enumerate(points)]
        ys = [point.get("value") for point in points]
        figure.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers", name=name,
                line={"width": 2, "color": _LINE_COLOURS[index % len(_LINE_COLOURS)]},
                marker={"size": 4},
            )
        )
    figure.update_layout(
        title={"text": title, "font": {"size": 13}},
        margin={"l": 55, "r": 20, "t": 36, "b": 44},
        height=300, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        legend={"orientation": "h", "y": -0.22},
        xaxis={"title": "step", "gridcolor": "#e2e8f0"},
        yaxis={"title": "value", "gridcolor": "#e2e8f0"},
    )
    return figure


def resource_figure(samples: Sequence[Mapping[str, Any]]) -> go.Figure:
    """CPU / RAM / GPU chart.

    Whole-device GPU utilization and this process's GPU memory are drawn as
    separate traces and labelled as such, because they answer different
    questions and NVML cannot attribute utilization per process.
    """
    if not samples:
        return _empty_figure("No resource samples recorded for this run.")
    timestamps = [sample.get("timestamp") for sample in samples]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=[sample.get("process_tree_cpu_percent") for sample in samples],
            mode="lines", name="process tree CPU %", line={"color": "#1d4ed8", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=[
                (sample.get("process_tree_rss_bytes") or 0) / (1024**3) if sample.get("process_tree_rss_bytes") else None
                for sample in samples
            ],
            mode="lines", name="process tree RSS (GiB)", line={"color": "#15803d", "width": 2}, yaxis="y2",
        )
    )
    gpu_utilization = [(sample.get("gpu") or {}).get("device_utilization_percent") for sample in samples]
    if any(value is not None for value in gpu_utilization):
        figure.add_trace(
            go.Scatter(
                x=timestamps, y=gpu_utilization, mode="lines",
                name="GPU utilization % (whole device)", line={"color": "#b91c1c", "width": 2},
            )
        )
        process_memory = [
            ((sample.get("gpu") or {}).get("process_memory_used_bytes") or 0) / (1024**3)
            if (sample.get("gpu") or {}).get("process_memory_used_bytes")
            else None
            for sample in samples
        ]
        if any(value is not None for value in process_memory):
            figure.add_trace(
                go.Scatter(
                    x=timestamps, y=process_memory, mode="lines",
                    name="GPU memory this process (GiB)", line={"color": "#a16207", "width": 2, "dash": "dot"}, yaxis="y2",
                )
            )
    figure.update_layout(
        margin={"l": 55, "r": 55, "t": 20, "b": 60}, height=320,
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        legend={"orientation": "h", "y": -0.28},
        xaxis={"title": "time", "gridcolor": "#e2e8f0"},
        yaxis={"title": "percent", "gridcolor": "#e2e8f0"},
        yaxis2={"title": "GiB", "overlaying": "y", "side": "right", "showgrid": False},
    )
    return figure


def identity_block(detail: Mapping[str, Any]) -> html.Div:
    identity = dict(detail.get("identity") or {})
    badges = [
        state_badge(str(detail.get("state"))),
        fidelity_badge(str(identity.get("paper_fidelity_status", "unverified")), str(identity.get("paper_fidelity_note", ""))),
        resume_badge(bool(identity.get("supports_exact_resume"))),
        instrumentation_badge(str(identity.get("event_instrumentation", "none")), str(identity.get("instrumentation_note", ""))),
    ]
    if detail.get("legacy"):
        badges.append(
            html.Span(
                f"legacy import ({detail.get('status_confidence') or 'unknown'} confidence)",
                title=(
                    "This record was reconstructed from a pre-existing runs/ directory. "
                    "Its state was inferred from artifacts, not recorded by a worker."
                ),
                style={
                    "display": "inline-block", "padding": "1px 8px", "borderRadius": "10px",
                    "border": "1px solid #a16207", "color": "#a16207", "fontSize": "0.78rem", "fontWeight": 600,
                },
            )
        )
    return html.Div(
        [
            html.Div(badges, style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"}),
            key_values(
                [
                    ("run_id", detail.get("run_id")),
                    ("model_id", identity.get("model_id")),
                    ("implementation_id", identity.get("implementation_id")),
                    ("init_id", identity.get("init_id")),
                    ("variant_id", identity.get("variant_id_short")),
                    ("task_id", detail.get("task_id")),
                    ("scenario_id", detail.get("scenario_id")),
                    ("seed", detail.get("seed")),
                    ("device", detail.get("device")),
                    ("experiment_id", detail.get("experiment_id")),
                ]
            ),
        ]
    )


def progress_block(detail: Mapping[str, Any]) -> html.Div:
    worker = dict(detail.get("worker") or {})
    return key_values(
        [
            ("phase", detail.get("phase")),
            ("global_step", detail.get("global_step")),
            ("epoch", detail.get("epoch")),
            ("last event id", detail.get("last_event_id")),
            ("heartbeat", str(detail.get("heartbeat_at") or "")[:19].replace("T", " ")),
            ("pid", detail.get("pid")),
            ("process group", detail.get("process_group_id")),
            ("pid alive", worker.get("pid_alive")),
            ("host", detail.get("host")),
            ("started", str(detail.get("started_at") or "")[:19].replace("T", " ")),
            ("ended", str(detail.get("ended_at") or "")[:19].replace("T", " ")),
            ("exit code", detail.get("exit_code")),
        ]
    )


def outcome_block(detail: Mapping[str, Any]) -> Any:
    children: list[Any] = []
    if detail.get("exit_code_description"):
        children.append(info_panel(str(detail["exit_code_description"])))
    if detail.get("error_summary"):
        children.append(
            html.Pre(
                str(detail["error_summary"]),
                style={
                    "background": "#fef2f2", "border": "1px solid #fecaca", "color": "#7f1d1d",
                    "padding": "10px", "borderRadius": "6px", "whiteSpace": "pre-wrap",
                    "fontSize": "0.8rem", "maxHeight": "220px", "overflow": "auto",
                },
            )
        )
    if detail.get("terminal_reason"):
        children.append(html.Div(f"terminal reason: {detail['terminal_reason']}", style={"fontSize": "0.85rem", "color": "#475569"}))
    if not children:
        children.append(info_panel("This run has not recorded a terminal outcome."))
    return html.Div(children)


def checkpoints_block(detail: Mapping[str, Any]) -> Any:
    checkpoints = list(detail.get("checkpoints") or [])
    if not checkpoints:
        return info_panel(
            "No checkpoints are registered in the catalog. This build does not write "
            "checkpoint records (checkpoint v1 is a later phase), so weight files on "
            "disk are listed under Artifacts and are NOT resume-certified."
        )
    return html.Ul(
        [
            html.Li(
                f"{row.get('checkpoint_id')} · {row.get('kind')} · step {row.get('global_step')} · "
                f"{'complete' if row.get('complete') else 'INCOMPLETE'} · "
                f"{'resume-certified' if row.get('exact_resume_certified') else 'not resume-certified'}"
            )
            for row in checkpoints
        ],
        style={"fontSize": "0.85rem"},
    )


def artifacts_block(payload: Mapping[str, Any]) -> Any:
    if is_error(payload):
        return error_panel(payload)
    on_disk = list(payload.get("on_disk") or [])
    if not on_disk:
        return info_panel("No artifact files found in this run directory.")
    rows = [
        html.Tr(
            [
                html.Td(item.get("path"), style={"padding": "3px 8px", "fontFamily": "ui-monospace, monospace", "fontSize": "0.8rem"}),
                html.Td(format_bytes(item.get("bytes")), style={"padding": "3px 8px", "fontSize": "0.8rem", "textAlign": "right"}),
            ]
        )
        for item in on_disk[:200]
    ]
    children: list[Any] = [
        html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"}),
    ]
    if payload.get("on_disk_truncated") or len(on_disk) > 200:
        children.append(info_panel(f"Showing the first 200 of {len(on_disk)} files."))
    if payload.get("failure_present"):
        children.insert(0, info_panel("This run wrote a failure.json record."))
    return html.Div(children)


def provenance_block(detail: Mapping[str, Any]) -> html.Div:
    link = detail.get("inspector_deep_link") or {}
    children: list[Any] = [
        key_values(
            [
                ("structural config hash", (detail.get("structural_config_hash") or "").replace("sha256:", "")[:16]),
                ("operational config hash", (detail.get("operational_config_hash") or "").replace("sha256:", "")[:16]),
                ("resolved spec hash", (detail.get("resolved_spec_hash") or "").replace("sha256:", "")[:16]),
                ("run directory", detail.get("run_dir")),
            ]
        )
    ]
    if link.get("run_path"):
        children.append(
            html.Div(
                [
                    html.Span("Streamlit Run Inspector: ", style={"fontSize": "0.85rem"}),
                    html.Code(f"?run={link['run_path']}", style={"fontSize": "0.78rem"}),
                    html.Div(
                        "Open the Inspector (streamlit run viz/app/main.py) and append that query "
                        "parameter to its URL to select this run.",
                        style={"color": "#64748b", "fontSize": "0.78rem", "marginTop": "3px"},
                    ),
                ],
                style={"marginTop": "8px"},
            )
        )
    else:
        children.append(
            info_panel(
                "No visualization meta.json was found for this run, so the Streamlit Run "
                "Inspector cannot display it. The Inspector only indexes directories that "
                "contain meta.json."
            )
        )
    return html.Div(children)


def transitions_block(detail: Mapping[str, Any]) -> Any:
    transitions = list(detail.get("transitions") or [])
    if not transitions:
        return info_panel("No state transitions recorded.")
    return html.Ol(
        [
            html.Li(
                [
                    html.Code(f"{row.get('from_state') or '∅'} → {row.get('to_state')}"),
                    html.Span(f"  ·  {str(row.get('at') or '')[:19].replace('T', ' ')}  ·  {row.get('actor')}",
                              style={"color": "#64748b", "fontSize": "0.78rem"}),
                    html.Div(str(row.get("reason") or ""), style={"color": "#475569", "fontSize": "0.78rem"}),
                ],
                style={"marginBottom": "4px"},
            )
            for row in transitions
        ],
        style={"fontSize": "0.85rem", "paddingLeft": "20px"},
    )


def log_block(payload: Mapping[str, Any], stream: str) -> Any:
    if is_error(payload):
        return error_panel(payload)
    if not payload.get("present", True):
        return info_panel(f"No {stream} log file exists for this run.")
    text = str(payload.get("text") or "")
    if not text.strip():
        return info_panel(f"The {stream} log is empty.")
    children: list[Any] = []
    if payload.get("truncated"):
        children.append(
            info_panel(
                f"Showing the last {len(text.encode('utf-8')) // 1000} KB of a "
                f"{format_bytes(payload.get('size_bytes'))} log (bounded tail)."
            )
        )
    children.append(
        html.Pre(
            text,
            style={
                "background": "#0f172a", "color": "#e2e8f0", "padding": "10px", "borderRadius": "6px",
                "fontSize": "0.76rem", "maxHeight": "360px", "overflow": "auto", "whiteSpace": "pre-wrap",
            },
        )
    )
    return html.Div(children)


def layout(run_id: str) -> html.Div:
    return html.Div(
        [
            dcc.Store(id="run-detail-id", data=run_id),
            html.Div(
                [
                    dcc.Link("← All runs", href="/runs", style={"color": "#1d4ed8", "textDecoration": "none"}),
                ],
                style={"marginBottom": "8px"},
            ),
            html.H2("Run detail", style={"marginTop": 0}),
            html.Div(
                html.Code(run_id, style={"fontSize": "0.82rem", "color": "#475569"}),
                style={"marginBottom": "12px"},
            ),
            # A stable per-(run, action) idempotency key lives in the browser
            # store, so a double-click or a re-render reuses it instead of
            # minting a second logical request.
            dcc.Store(id="stop-idem-key", data=f"ui-stop-{run_id}"),
            dcc.Store(id="resume-idem-key", data=f"ui-resume-{run_id}"),
            dcc.Store(id="active-action-id", data=None),
            dcc.Loading(html.Div(id="run-detail-identity"), type="dot"),
            html.Div(id="run-detail-controls"),
            # Owned by the action callback, deliberately *outside*
            # run-detail-controls: that block is re-rendered on every poll and
            # would otherwise wipe the panel between refreshes.
            html.Div(id="action-status"),
            html.Div(id="run-detail-progress"),
            html.Div(id="run-detail-charts"),
            html.Div(id="run-detail-logs"),
            html.Div(id="run-detail-rest"),
        ]
    )


# -- write-control panel -----------------------------------------------------
#
# Rendered only when the API reports write mode enabled. Every condition and
# every human-readable reason comes from the API's eligibility read model; this
# module never recomputes them, so UI and backend cannot drift (ADR-WC-020).

STOP_DESCRIPTION = (
    "현재 optimizer update를 완료한 뒤 검증된 interrupt checkpoint를 저장하고 "
    "종료합니다. 즉시 종료되지 않을 수 있습니다."
)
RESUME_DESCRIPTION = (
    "검증된 checkpoint에서 새 child run을 생성해 학습을 이어갑니다. "
    "기존 parent run과 checkpoint는 변경되지 않습니다."
)


def _reason_note(text: str) -> html.Div:
    return html.Div(
        text,
        style={"fontSize": "0.8rem", "color": "#92400e", "background": "#fffbeb",
               "border": "1px solid #fde68a", "borderRadius": "6px", "padding": "8px"},
    )


def controls_block(detail: Mapping[str, Any]) -> Any:
    """Stop safely / Resume training, or the exact reason neither is offered."""
    eligibility = dict(detail.get("action_eligibility") or {})
    if not eligibility.get("write_control_enabled"):
        return None

    stop = dict(eligibility.get("stop_action") or {})
    resume = dict(eligibility.get("resume_action") or {})
    children: list[Any] = []

    # --- Stop safely ---
    if stop.get("eligible"):
        children.append(html.Div([
            html.Button("Stop safely", id="stop-button", n_clicks=0,
                        style={"padding": "8px 14px", "borderRadius": "6px",
                               "border": "1px solid #b45309", "background": "#f59e0b",
                               "color": "#1f2937", "cursor": "pointer",
                               "fontWeight": 600}),
            html.Div(STOP_DESCRIPTION,
                     style={"fontSize": "0.8rem", "color": "#475569", "marginTop": "6px"}),
            html.Div(
                "This is a graceful stop, not a kill. If the interrupt checkpoint "
                "cannot be written the run becomes FAILED (exit 50) and is not resumable.",
                style={"fontSize": "0.78rem", "color": "#64748b", "marginTop": "4px"},
            ),
        ], style={"marginBottom": "12px"}))
    else:
        children.append(_reason_note(
            stop.get("reason") or "Safe stop unavailable."))

    # --- Resume training ---
    if resume.get("eligible"):
        children.append(html.Div([
            html.Button("Resume training", id="resume-button", n_clicks=0,
                        style={"padding": "8px 14px", "borderRadius": "6px",
                               "border": "1px solid #1d4ed8", "background": "#3b82f6",
                               "color": "white", "cursor": "pointer", "fontWeight": 600}),
            html.Div(RESUME_DESCRIPTION,
                     style={"fontSize": "0.8rem", "color": "#475569", "marginTop": "6px"}),
            html.Div(
                f"checkpoint {str(resume.get('checkpoint_id') or '')[:12]}… · "
                "exact resume, not a warm start · creates a new child run",
                style={"fontSize": "0.78rem", "color": "#64748b", "marginTop": "4px"},
            ),
        ], style={"marginBottom": "12px"}))
    else:
        children.append(_reason_note(
            resume.get("reason") or "Exact resume unavailable."))

    return html.Div(children)


def action_status_block(action: Optional[Mapping[str, Any]]) -> Any:
    """Live action state. Durable state is the authority, not the browser."""
    if not action:
        return None
    if action.get("_error"):
        return _reason_note(
            f"{action.get('reason_code') or 'error'}: {action.get('message') or ''}")

    state = str(action.get("state") or "")
    rows = [
        html.Div([html.Strong(f"{action.get('action_type')} "),
                  html.Span(state, style={"fontFamily": "monospace"})]),
        html.Div(f"action {str(action.get('action_id') or '')[:12]}…",
                 style={"fontSize": "0.76rem", "color": "#64748b"}),
    ]
    if action.get("error"):
        rows.append(html.Div(str(action["error"]),
                             style={"fontSize": "0.78rem", "color": "#b91c1c"}))
    if action.get("result_checkpoint_id"):
        rows.append(html.Div(
            f"interrupt checkpoint {str(action['result_checkpoint_id'])[:12]}…",
            style={"fontSize": "0.78rem", "color": "#065f46"}))
    child = action.get("child_run_id")
    if child:
        # Launch completion is not training completion — say so, and link.
        rows.append(html.Div([
            html.Span("child run: ", style={"fontSize": "0.8rem"}),
            dcc.Link(str(child)[:12] + "…", href=f"/runs/{child}",
                     id="child-run-link",
                     style={"color": "#1d4ed8", "fontSize": "0.8rem"}),
        ]))
        rows.append(html.Div(
            "The launch action is complete; the child's own state tells you "
            "whether its training finished.",
            style={"fontSize": "0.76rem", "color": "#64748b"}))
    return html.Div(rows, id="action-status-body",
                    style={"border": "1px solid #cbd5f5", "borderRadius": "6px",
                           "padding": "10px", "background": "#f8fafc"})
