"""System page: health, GPUs, workers, capability matrix, build limitations."""

from __future__ import annotations

from typing import Any, Mapping

from dash import dcc, html

from ..api_client import is_error
from ..components import (
    badge,
    error_panel,
    fidelity_badge,
    format_bytes,
    info_panel,
    instrumentation_badge,
    key_values,
    resume_badge,
    section,
)

_STATUS_COLOURS = {"ok": "#15803d", "degraded": "#a16207", "error": "#b91c1c"}


def health_block(payload: Mapping[str, Any]) -> Any:
    if is_error(payload):
        return error_panel(payload)
    components = dict(payload.get("components") or {})
    rows = []
    for name, component in components.items():
        status = str(component.get("status", "error"))
        rows.append(
            html.Div(
                [
                    badge(f"{name}: {status}", _STATUS_COLOURS.get(status, "#6b7280"), title=str(component.get("detail") or status)),
                    html.Div(
                        str(component.get("detail") or ""),
                        style={"color": "#64748b", "fontSize": "0.78rem", "marginTop": "3px"},
                    ),
                ],
                style={"marginBottom": "8px"},
            )
        )
    return html.Div(
        [
            html.Div(rows),
            key_values(
                [
                    ("overall", payload.get("status")),
                    ("control root", payload.get("control_root")),
                    ("host", payload.get("host")),
                    ("api pid", payload.get("pid")),
                    ("uptime (s)", round(float(payload.get("uptime_seconds") or 0), 1)),
                    ("python", payload.get("python")),
                ]
            ),
        ]
    )


def gpu_block(payload: Mapping[str, Any]) -> Any:
    if is_error(payload):
        return error_panel(payload)
    if not payload.get("available"):
        return info_panel(str(payload.get("note") or "No NVIDIA GPU is visible to the API process."))
    children = []
    for device in payload.get("devices") or []:
        children.append(
            html.Div(
                [
                    html.Strong(f"GPU {device.get('device_index')} — {device.get('device_name') or 'unknown'}"),
                    key_values(
                        [
                            ("backend", device.get("backend")),
                            ("uuid", device.get("device_uuid")),
                            ("utilization (whole device)", f"{device.get('device_utilization_percent')} %"),
                            ("memory used (whole device)", format_bytes(device.get("device_memory_used_bytes"))),
                            ("memory total", format_bytes(device.get("device_memory_total_bytes"))),
                            ("temperature", f"{device.get('temperature_c')} °C" if device.get("temperature_c") is not None else None),
                            ("power", f"{device.get('power_w')} W" if device.get("power_w") is not None else None),
                            ("attribution quality", device.get("attribution_quality")),
                        ]
                    ),
                ],
                style={"marginBottom": "10px"},
            )
        )
    leases = payload.get("leases") or []
    children.append(
        info_panel(
            f"{len(leases)} active GPU lease(s)."
            if leases
            else "No GPU leases are currently held. Leases are enforced by a unique index in the registry."
        )
    )
    return html.Div(children)


def workers_block(payload: Mapping[str, Any]) -> Any:
    if is_error(payload):
        return error_panel(payload)
    workers = list(payload.get("workers") or [])
    if not workers:
        return info_panel("No worker processes have been registered yet.")
    header_style = {"textAlign": "left", "padding": "5px 8px", "borderBottom": "2px solid #cbd5e1", "fontSize": "0.76rem", "color": "#475569"}
    cell_style = {"padding": "5px 8px", "borderBottom": "1px solid #e2e8f0", "fontSize": "0.8rem", "fontFamily": "ui-monospace, monospace"}
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(name, style=header_style) for name in ("Run", "PID", "PGID", "State", "Exit", "Last heartbeat")])),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(dcc.Link(str(worker.get("run_id"))[:8], href=f"/runs/{worker.get('run_id')}"), style=cell_style),
                            html.Td(str(worker.get("pid")), style=cell_style),
                            html.Td(str(worker.get("process_group_id")), style=cell_style),
                            html.Td(str(worker.get("state")), style=cell_style),
                            html.Td(str(worker.get("exit_code")), style=cell_style),
                            html.Td(str(worker.get("last_heartbeat_at") or "")[:19].replace("T", " "), style=cell_style),
                        ]
                    )
                    for worker in workers
                ]
            ),
        ],
        style={"width": "100%", "borderCollapse": "collapse"},
    )


def orphan_block(payload: Mapping[str, Any]) -> Any:
    if is_error(payload):
        return error_panel(payload)
    candidates = list(payload.get("candidates") or [])
    if not candidates:
        return info_panel("No orphan candidates. Every non-terminal run has a live, identity-matched worker.")
    return html.Div(
        [
            info_panel(
                "These runs have a missing, recycled, or stale worker. They are reported, not "
                "acted on: classifying an orphan requires checking PID identity, heartbeat, and "
                "checkpoint integrity, and is a researcher decision. Run "
                "`python -m bench.control.cli reconcile` to record ORPHANED for verified-dead workers."
            ),
            html.Ul(
                [
                    html.Li(
                        [
                            dcc.Link(str(candidate.get("run_id"))[:8], href=f"/runs/{candidate.get('run_id')}"),
                            html.Span(f" · {candidate.get('state')} · {candidate.get('reason')}", style={"fontSize": "0.82rem"}),
                        ]
                    )
                    for candidate in candidates
                ]
            ),
        ]
    )


def capability_block(payload: Mapping[str, Any]) -> Any:
    if is_error(payload):
        return error_panel(payload)
    control = dict(payload.get("control_plane") or {})
    flags = [
        html.Li(
            [
                badge("yes" if value else "no", "#15803d" if value else "#b91c1c"),
                html.Span(f"  {name.replace('_', ' ')}", style={"fontSize": "0.85rem"}),
            ],
            style={"marginBottom": "3px", "listStyle": "none"},
        )
        for name, value in control.items()
        if isinstance(value, bool)
    ]

    models = list(payload.get("models") or [])
    header_style = {"textAlign": "left", "padding": "5px 8px", "borderBottom": "2px solid #cbd5e1", "fontSize": "0.74rem", "color": "#475569"}
    cell_style = {"padding": "5px 8px", "borderBottom": "1px solid #e2e8f0", "fontSize": "0.8rem", "verticalAlign": "top"}
    model_rows = [
        html.Tr(
            [
                html.Td(
                    [
                        html.Div(model.get("display_name"), style={"fontWeight": 600}),
                        html.Div(model.get("model_id"), style={"fontSize": "0.74rem", "color": "#64748b", "fontFamily": "ui-monospace, monospace"}),
                    ],
                    style=cell_style,
                ),
                html.Td(model.get("implementation_id"), style={**cell_style, "fontFamily": "ui-monospace, monospace", "fontSize": "0.74rem"}),
                html.Td("yes" if model.get("trainable") else "no", style=cell_style),
                html.Td(fidelity_badge(str(model.get("paper_fidelity_status")), str(model.get("paper_fidelity_note", ""))), style=cell_style),
                html.Td(resume_badge(bool(model.get("supports_exact_resume"))), style=cell_style),
                html.Td(instrumentation_badge(str(model.get("event_instrumentation")), str(model.get("instrumentation_note", ""))), style=cell_style),
            ]
        )
        for model in models
    ]

    return html.Div(
        [
            html.H4("This build", style={"margin": "4px 0"}),
            info_panel(str(control.get("notes") or "")),
            html.Ul(flags, style={"paddingLeft": 0, "columns": "2", "marginTop": "6px"}),
            html.H4("Schema versions", style={"margin": "12px 0 4px 0"}),
            key_values(list((payload.get("schema_versions") or {}).items())),
            html.H4("Model capability matrix", style={"margin": "12px 0 4px 0"}),
            info_panel(
                "Paper fidelity is independent of whether a model runs. An adapter that "
                "executes is not thereby paper-faithful, and no implementation here is "
                "certified for exact resume."
            ),
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [html.Th(name, style=header_style) for name in ("Model", "Implementation", "Trainable", "Paper fidelity", "Exact resume", "Instrumentation")]
                            )
                        ),
                        html.Tbody(model_rows),
                    ],
                    style={"width": "100%", "borderCollapse": "collapse", "minWidth": "900px"},
                ),
                style={"overflowX": "auto"},
            ),
        ]
    )


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("System", style={"marginTop": 0}),
            section("Health", [dcc.Loading(html.Div(id="system-health"), type="dot")],
                    subtitle="Per-subsystem status; the overall status is the worst component."),
            section("GPUs", [html.Div(id="system-gpus")],
                    subtitle="Whole-device readings. NVML cannot attribute utilization per process."),
            section("Orphan candidates", [html.Div(id="system-orphans")],
                    subtitle="Runs whose worker appears to have vanished. Reported only."),
            section("Workers", [html.Div(id="system-workers")]),
            section("Capabilities", [html.Div(id="system-capabilities")],
                    subtitle="What this build can do, and what each model adapter declares."),
        ]
    )
