"""Dash dashboard application.

Multi-page routing over ``dcc.Location``:

    /                 → redirects to /runs
    /runs             → run table
    /runs/<run_id>    → run detail (run_id is the route key, per U-07)
    /system           → health, GPUs, workers, capability matrix

Live updates use **bounded polling** (`dcc.Interval` → HTTP GET with a cursor),
not WebSockets. That is a deliberate MVP choice: the event API already exposes
the cursor/gap-fill semantics a push transport would need, so adding WebSockets
later is additive. Introducing them now — without reconnect and gap tests —
would be the premature optimization design doc 05 §6 warns about.

No callback here starts, stops, or modifies a run. The dashboard is a viewer.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import dash
from dash import Input, Output, State, dcc, html

from .api_client import ApiClient, is_error
from .components import PAGE_STYLE, error_panel, info_panel, section
from .pages import run_detail, runs as runs_page, system as system_page

#: Poll interval for live views. 3 s is fast enough to feel live and slow
#: enough that a 100-run registry is not re-serialized 20 times a second.
DEFAULT_POLL_MS = 3000

NAV_LINK_STYLE = {
    "marginRight": "16px",
    "color": "#1d4ed8",
    "textDecoration": "none",
    "fontWeight": 600,
}


def _navbar() -> html.Div:
    return html.Div(
        [
            html.Span("Benchmark Control Plane", style={"fontWeight": 700, "marginRight": "24px"}),
            dcc.Link("Runs", href="/runs", style=NAV_LINK_STYLE),
            dcc.Link("System", href="/system", style=NAV_LINK_STYLE),
            html.Span(
                "read-only",
                title=(
                    "This dashboard observes runs. It cannot start, stop, or modify them; "
                    "the API it reads has no write endpoints."
                ),
                style={
                    "marginLeft": "auto", "border": "1px solid #64748b", "color": "#64748b",
                    "borderRadius": "10px", "padding": "1px 8px", "fontSize": "0.76rem",
                },
            ),
        ],
        style={
            "display": "flex", "alignItems": "center", "gap": "4px", "padding": "10px 14px",
            "background": "#ffffff", "border": "1px solid #e2e8f0", "borderRadius": "8px",
            "marginBottom": "16px",
        },
    )


def create_dash_app(
    *,
    api_base_url: Optional[str] = None,
    poll_interval_ms: int = DEFAULT_POLL_MS,
    requests_pathname_prefix: Optional[str] = None,
) -> dash.Dash:
    """Build the Dash application."""
    client = ApiClient(api_base_url or os.environ.get("BENCH_CONTROL_API", "http://127.0.0.1:8765"))

    app = dash.Dash(
        __name__,
        title="Benchmark Control Plane",
        update_title=None,
        suppress_callback_exceptions=True,
        requests_pathname_prefix=requests_pathname_prefix,
    )

    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Interval(id="poll", interval=int(poll_interval_ms), n_intervals=0),
            _navbar(),
            html.Div(id="page-content"),
        ],
        style=PAGE_STYLE,
    )

    # -- routing ------------------------------------------------------------

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def render_page(pathname: Optional[str]) -> Any:
        path = (pathname or "/").rstrip("/") or "/"
        if path in ("/", "/runs"):
            machine = client.state_machine()
            states = [] if is_error(machine) else list(machine.get("active_states_this_build") or [])
            return runs_page.layout(states)
        if path.startswith("/runs/"):
            run_id = path.split("/runs/", 1)[1]
            return run_detail.layout(run_id)
        if path == "/system":
            return system_page.layout()
        return section("Not found", [info_panel(f"No page at {path!r}.")])

    # -- runs list ----------------------------------------------------------

    @app.callback(
        Output("runs-table", "children"),
        Input("poll", "n_intervals"),
        Input("runs-state-filter", "value"),
        Input("runs-legacy-filter", "value"),
        Input("runs-limit", "value"),
    )
    def refresh_runs(_ticks: int, state: Optional[str], source: Optional[str], limit: Optional[int]) -> Any:
        payload = client.runs(
            state=state or None,
            include_legacy=str(source != "new").lower(),
            limit=int(limit or 50),
        )
        return runs_page.runs_table(payload)

    # -- run detail ---------------------------------------------------------

    @app.callback(
        Output("run-detail-identity", "children"),
        Output("run-detail-progress", "children"),
        Input("poll", "n_intervals"),
        State("run-detail-id", "data"),
    )
    def refresh_detail(_ticks: int, run_id: Optional[str]) -> tuple[Any, Any]:
        if not run_id:
            return dash.no_update, dash.no_update
        detail = client.run(run_id)
        if is_error(detail):
            return error_panel(detail), None
        identity = section("Identity", [run_detail.identity_block(detail)])
        progress = html.Div(
            [
                section("Progress and worker", [run_detail.progress_block(detail)]),
                section("Outcome", [run_detail.outcome_block(detail)]),
            ]
        )
        return identity, progress

    @app.callback(
        Output("run-detail-charts", "children"),
        Input("poll", "n_intervals"),
        State("run-detail-id", "data"),
    )
    def refresh_charts(_ticks: int, run_id: Optional[str]) -> Any:
        if not run_id:
            return dash.no_update
        metrics = client.metrics(run_id)
        resources = client.resources(run_id, limit=1500)
        if is_error(metrics):
            return error_panel(metrics)
        series = dict(metrics.get("series") or {})
        children: list[Any] = []
        if not series:
            children.append(
                info_panel(
                    "No metric events were recorded for this run. For legacy imports this is "
                    "expected — they predate the event journal. For a control-plane run it means "
                    "the adapter has phase-level instrumentation only; see the badge above."
                )
            )
        children.append(
            dcc.Graph(
                figure=run_detail.metric_figure(series, run_detail.LOSS_SERIES, "Training and validation loss"),
                config={"displayModeBar": False},
            )
        )
        children.append(
            dcc.Graph(
                figure=run_detail.metric_figure(series, run_detail.EVAL_SERIES, "Evaluation metrics"),
                config={"displayModeBar": False},
            )
        )
        resource_children: list[Any] = []
        if is_error(resources):
            resource_children.append(error_panel(resources))
        else:
            samples = list(resources.get("samples") or [])
            resource_children.append(
                dcc.Graph(figure=run_detail.resource_figure(samples), config={"displayModeBar": False})
            )
        return html.Div(
            [
                section("Metrics", children, subtitle="Sourced from structured events, never from parsed stdout."),
                section(
                    "Resources",
                    resource_children,
                    subtitle="Whole-device GPU utilization and per-process GPU memory are separate series.",
                ),
            ]
        )

    @app.callback(
        Output("run-detail-logs", "children"),
        Input("poll", "n_intervals"),
        State("run-detail-id", "data"),
    )
    def refresh_logs(_ticks: int, run_id: Optional[str]) -> Any:
        if not run_id:
            return dash.no_update
        stdout = client.logs(run_id, stream="stdout")
        stderr = client.logs(run_id, stream="stderr")
        return section(
            "Captured output",
            [
                html.H4("stdout", style={"margin": "6px 0 4px 0", "fontSize": "0.9rem"}),
                run_detail.log_block(stdout, "stdout"),
                html.H4("stderr", style={"margin": "12px 0 4px 0", "fontSize": "0.9rem"}),
                run_detail.log_block(stderr, "stderr"),
            ],
            subtitle="Bounded tail of the worker's redirected stdio. Logs are for humans; metrics come from events.",
        )

    @app.callback(
        Output("run-detail-rest", "children"),
        Input("poll", "n_intervals"),
        State("run-detail-id", "data"),
    )
    def refresh_rest(_ticks: int, run_id: Optional[str]) -> Any:
        if not run_id:
            return dash.no_update
        detail = client.run(run_id)
        if is_error(detail):
            return None
        artifacts = client.artifacts(run_id)
        return html.Div(
            [
                section("Checkpoints", [run_detail.checkpoints_block(detail)]),
                section("Artifacts", [run_detail.artifacts_block(artifacts)]),
                section("Provenance and deep links", [run_detail.provenance_block(detail)]),
                section("State transitions", [run_detail.transitions_block(detail)]),
            ]
        )

    # -- system -------------------------------------------------------------

    @app.callback(
        Output("system-health", "children"),
        Output("system-gpus", "children"),
        Output("system-orphans", "children"),
        Output("system-workers", "children"),
        Output("system-capabilities", "children"),
        Input("poll", "n_intervals"),
    )
    def refresh_system(_ticks: int) -> tuple[Any, Any, Any, Any, Any]:
        return (
            system_page.health_block(client.health()),
            system_page.gpu_block(client.gpus()),
            system_page.orphan_block(client.orphan_candidates()),
            system_page.workers_block(client.workers()),
            system_page.capability_block(client.capabilities()),
        )

    return app


def main(argv: Optional[list[str]] = None) -> int:  # pragma: no cover - process entry point
    import argparse

    parser = argparse.ArgumentParser(prog="bench.ui.dash_app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--api", default=None, help="Control-plane API base URL")
    parser.add_argument("--poll-ms", type=int, default=DEFAULT_POLL_MS)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    from ..control.api.app import resolve_bind_host

    host = resolve_bind_host(args.host)
    app = create_dash_app(api_base_url=args.api, poll_interval_ms=args.poll_ms)
    app.run(host=host, port=int(args.port), debug=bool(args.debug))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
