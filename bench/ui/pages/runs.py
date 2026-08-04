"""Runs list page.

The table's row key is ``run_id`` and nothing else. Two runs of the same model
with different initialization are separate rows with separate links — the exact
collision the audit called out when `variant_label` was doing duty as an
identity (acceptance C-03/U-01).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dash import dcc, html

from ..api_client import is_error
from ..components import (
    error_panel,
    fidelity_badge,
    info_panel,
    section,
    state_badge,
)

#: Columns rendered in the runs table.
COLUMNS = (
    "State",
    "Model / variant",
    "Task",
    "Seed",
    "Step",
    "Updated",
    "Source",
)


def filter_controls(states: Sequence[str], selected_state: Optional[str], include_legacy: bool) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Label("State", htmlFor="runs-state-filter", style={"fontSize": "0.8rem", "color": "#475569"}),
                    dcc.Dropdown(
                        id="runs-state-filter",
                        options=[{"label": "All states", "value": ""}]
                        + [{"label": state, "value": state} for state in states],
                        value=selected_state or "",
                        clearable=False,
                        style={"width": "220px"},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Source", htmlFor="runs-legacy-filter", style={"fontSize": "0.8rem", "color": "#475569"}),
                    dcc.Dropdown(
                        id="runs-legacy-filter",
                        options=[
                            {"label": "Control-plane and legacy", "value": "all"},
                            {"label": "Control-plane runs only", "value": "new"},
                        ],
                        value="all" if include_legacy else "new",
                        clearable=False,
                        style={"width": "240px"},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Rows", htmlFor="runs-limit", style={"fontSize": "0.8rem", "color": "#475569"}),
                    dcc.Dropdown(
                        id="runs-limit",
                        options=[{"label": str(n), "value": n} for n in (25, 50, 100, 250)],
                        value=50,
                        clearable=False,
                        style={"width": "110px"},
                    ),
                ]
            ),
        ],
        style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "alignItems": "flex-end"},
    )


def _row(run: Mapping[str, Any]) -> html.Tr:
    identity = dict(run.get("identity") or {})
    run_id = str(run.get("run_id"))
    legacy = bool(run.get("legacy"))
    confidence = run.get("status_confidence")

    state_title = None
    if legacy and confidence:
        state_title = (
            f"Imported legacy run; status inferred from artifacts with {confidence} confidence. "
            "No worker recorded this transition."
        )

    variant_short = identity.get("variant_id_short") or ""
    cell_style = {"padding": "6px 8px", "borderBottom": "1px solid #e2e8f0", "fontSize": "0.86rem", "verticalAlign": "top"}

    return html.Tr(
        [
            html.Td(state_badge(str(run.get("state")), title=state_title), style=cell_style),
            html.Td(
                [
                    dcc.Link(
                        identity.get("display_name") or run.get("model_id") or "—",
                        href=f"/runs/{run_id}",
                        style={"fontWeight": 600, "color": "#1d4ed8", "textDecoration": "none"},
                    ),
                    html.Div(
                        f"{run.get('implementation_id') or '—'} · init={run.get('init_id') or '—'}",
                        style={"color": "#64748b", "fontSize": "0.76rem"},
                    ),
                    html.Div(
                        f"variant {variant_short}" if variant_short else "",
                        style={"color": "#94a3b8", "fontSize": "0.72rem", "fontFamily": "ui-monospace, monospace"},
                    ),
                ],
                style=cell_style,
            ),
            html.Td(
                [
                    html.Div(str(run.get("task_id") or "—")),
                    html.Div(
                        str(run.get("scenario_id") or ""),
                        style={"color": "#94a3b8", "fontSize": "0.74rem", "fontFamily": "ui-monospace, monospace"},
                    ),
                ],
                style=cell_style,
            ),
            html.Td(str(run.get("seed", "—")), style=cell_style),
            html.Td(str(run.get("global_step", 0)), style=cell_style),
            html.Td(
                str(run.get("updated_at") or "")[:19].replace("T", " "),
                style={**cell_style, "fontFamily": "ui-monospace, monospace", "fontSize": "0.78rem"},
            ),
            html.Td(
                (
                    html.Span("legacy import", title="Read-only projection of a pre-existing runs/ directory.", style={"color": "#a16207"})
                    if legacy
                    else html.Span("control plane", style={"color": "#15803d"})
                ),
                style=cell_style,
            ),
        ]
    )


def runs_table(payload: Mapping[str, Any]) -> Any:
    if is_error(payload):
        return error_panel(payload)
    runs = list(payload.get("runs") or [])
    if not runs:
        return info_panel(
            "No runs match this filter. Launch one with: "
            "python -m bench.control.cli launch-synthetic --updates 40 — or import existing "
            "artifacts with: python -m bench.control.cli import-legacy"
        )
    header_style = {
        "textAlign": "left",
        "padding": "6px 8px",
        "borderBottom": "2px solid #cbd5e1",
        "fontSize": "0.76rem",
        "textTransform": "uppercase",
        "letterSpacing": "0.03em",
        "color": "#475569",
    }
    return html.Div(
        [
            html.Div(
                f"Showing {payload.get('count', len(runs))} of {payload.get('total', '?')} runs",
                style={"color": "#64748b", "fontSize": "0.8rem", "marginBottom": "6px"},
            ),
            html.Table(
                [
                    html.Thead(html.Tr([html.Th(name, style=header_style) for name in COLUMNS])),
                    html.Tbody([_row(run) for run in runs]),
                ],
                style={"width": "100%", "borderCollapse": "collapse", "background": "#fff"},
            ),
        ]
    )


def layout(states: Sequence[str]) -> html.Div:
    return html.Div(
        [
            html.H2("Runs", style={"marginTop": 0}),
            section(
                "Filters",
                [filter_controls(states, None, True)],
                subtitle="Rows are keyed by run_id. Two runs of the same model with different initialization are separate rows.",
            ),
            section(
                "Run registry",
                [
                    dcc.Loading(html.Div(id="runs-table"), type="dot"),
                ],
                subtitle="Read-only view of the SQLite run registry.",
            ),
        ]
    )
