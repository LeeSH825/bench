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

The default dashboard is a viewer.  When the API and dashboard are started
with ``BENCH_CONTROL_ENABLE_WRITES=1`` on loopback, the guarded callbacks expose
configuration launch, graceful stop, and exact resume through the same action
services used by the CLI and API.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

import dash
from dash import Input, Output, State, dcc, html

from .api_client import ApiClient, is_error
from .components import PAGE_STYLE, error_panel, info_panel, section
from .pages import new_run, run_detail, runs as runs_page, system as system_page

#: Poll interval for live views. 3 s is fast enough to feel live and slow
#: enough that a 100-run registry is not re-serialized 20 times a second.
DEFAULT_POLL_MS = 3000

NAV_LINK_STYLE = {
    "marginRight": "16px",
    "color": "#1d4ed8",
    "textDecoration": "none",
    "fontWeight": 600,
}


def _write_enabled() -> bool:
    """The badge must not claim read-only when writes are on."""
    from ..control.api.write_mode import writes_enabled

    return writes_enabled()


def _navbar(write_enabled: bool = False) -> html.Div:
    return html.Div(
        [
            html.Span("Benchmark Control Plane", style={"fontWeight": 700, "marginRight": "24px"}),
            dcc.Link("Runs", href="/runs", style=NAV_LINK_STYLE),
            dcc.Link("New run", href="/new-run", style=NAV_LINK_STYLE),
            dcc.Link("System", href="/system", style=NAV_LINK_STYLE),
            html.Span(
                ("write control enabled" if write_enabled else "read-only"),
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


def _collect_overrides(schema: Any, values: Any, ids: Any) -> dict[str, Any]:
    """Turn form controls into typed overrides for the API.

    Values are coerced using the descriptor's declared type so the API receives
    an int where it expects an int; it re-validates regardless.
    """
    types = {f["path"]: f["type"] for f in (schema or {}).get("fields") or []}
    overrides: dict[str, Any] = {}
    for value, identifier in zip(values or [], ids or []):
        if value is None or value == "":
            continue
        path = identifier.get("path")
        kind = types.get(path)
        try:
            if kind == "integer":
                overrides[path] = int(value)
            elif kind == "number":
                overrides[path] = float(value)
            elif kind == "boolean":
                overrides[path] = str(value).lower() == "true"
            else:
                overrides[path] = value
        except (TypeError, ValueError):
            continue
    return overrides


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
            _navbar(_write_enabled()),
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
        if path == "/new-run":
            return new_run.layout(_write_enabled())
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

    # -- new run wizard -----------------------------------------------------
    #
    # Every callback below goes through ApiClient. Nothing here imports the
    # resolver, registry, adapters or WorkerManager: the wizard only knows what
    # the API tells it, and the API re-validates whatever it submits.

    @app.callback(
        Output("nr-preset", "options"), Output("nr-schema", "data"),
        Input("url", "pathname"),
    )
    def load_presets(pathname: Optional[str]) -> Any:
        if (pathname or "").rstrip("/") != "/new-run":
            return dash.no_update, dash.no_update
        payload = client.presets()
        schema = client.config_schema()
        options = [] if is_error(payload) else new_run.preset_options(payload)
        return options, ({} if is_error(schema) else schema)

    @app.callback(
        Output("nr-preset-entry", "data"), Output("nr-preset-summary", "children"),
        Output("nr-model", "options"), Output("nr-model", "value"),
        Output("nr-yaml", "value"), Output("nr-idem-key", "data"),
        Input("nr-preset", "value"),
        prevent_initial_call=True,
    )
    def choose_preset(preset_id: Optional[str]) -> Any:
        if not preset_id:
            return None, None, [], None, "", None
        detail = client.preset(preset_id)
        if is_error(detail):
            return None, error_panel(detail), [], None, "", None
        models = list(detail.get("launchable_model_ids") or detail.get("model_ids") or [])
        # One stable idempotency key per (preset, session): a double-click or a
        # re-render reuses it rather than launching twice.
        key = f"ui-launch-{preset_id}-{uuid.uuid4()}"
        return (detail, new_run.preset_summary(detail),
                [{"label": m, "value": m} for m in models],
                (models[0] if models else None),
                detail.get("yaml_text") or "", key)

    @app.callback(
        Output("nr-form", "children"), Output("nr-steps", "children"),
        Input("nr-schema", "data"), Input("nr-model", "value"),
        State("nr-preset-entry", "data"),
    )
    def render_form(schema: Any, model_id: Optional[str], entry: Any) -> Any:
        if not schema:
            return dash.no_update, new_run.step_header(0)
        # Trainability comes from the API's launchable model list, not from a
        # local model-name check.
        trainable = bool(model_id) and model_id in (
            (entry or {}).get("launchable_model_ids") or [])
        if trainable and model_id in ("mb_kf",):
            trainable = False
        active = 1 if entry else 0
        return new_run.config_form(schema, entry, trainable), new_run.step_header(active)

    @app.callback(
        Output("nr-form", "style"), Output("nr-yaml-wrap", "style"),
        Input("nr-edit-mode", "value"),
    )
    def toggle_editor(mode: Optional[str]) -> Any:
        if mode == "yaml":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    @app.callback(
        Output("nr-validation", "data"), Output("nr-validation-panel", "children"),
        Output("nr-review", "children"), Output("nr-confirmation", "children"),
        Input("nr-validate", "n_clicks"),
        State("nr-preset-entry", "data"), State("nr-model", "value"),
        State("nr-init", "value"), State("nr-edit-mode", "value"),
        State("nr-yaml", "value"), State("nr-schema", "data"),
        State({"role": "cfg", "path": dash.ALL}, "value"),
        State({"role": "cfg", "path": dash.ALL}, "id"),
        prevent_initial_call=True,
    )
    def run_validation(n_clicks, entry, model_id, init_id, mode, yaml_text,
                       schema, values, ids) -> Any:
        if not n_clicks or not entry:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        overrides = _collect_overrides(schema, values, ids)
        body: dict[str, Any] = {
            "preset_id": entry.get("preset_id"), "model_id": model_id,
            "init_id": init_id, "include_diff": True,
        }
        if mode == "yaml" and yaml_text:
            body["yaml_text"] = yaml_text
        else:
            body["overrides"] = overrides
        result = client.validate_config(**body)
        if is_error(result):
            return None, error_panel(result), None, None
        return (result, new_run.validation_panel(result),
                new_run.review_panel(result),
                new_run.launch_confirmation(entry, result, model_id))

    @app.callback(
        Output("nr-action-id", "data"), Output("nr-launch", "disabled"),
        Input("nr-launch", "n_clicks"),
        State("nr-preset-entry", "data"), State("nr-validation", "data"),
        State("nr-model", "value"), State("nr-init", "value"),
        State("nr-idem-key", "data"), State("nr-schema", "data"),
        State({"role": "cfg", "path": dash.ALL}, "value"),
        State({"role": "cfg", "path": dash.ALL}, "id"),
        prevent_initial_call=True,
    )
    def do_launch(n_clicks, entry, validation, model_id, init_id, key,
                  schema, values, ids) -> Any:
        if not n_clicks or not entry or not validation or not validation.get("valid"):
            return dash.no_update, dash.no_update
        if not (validation.get("launch_eligibility") or {}).get("eligible"):
            return dash.no_update, True
        result = client.launch_run(
            idempotency_key=str(key),
            preset_id=entry.get("preset_id"),
            preset_digest=entry.get("content_digest"),
            model_id=model_id, init_id=init_id,
            overrides=_collect_overrides(schema, values, ids),
            expected_structural_config_hash=validation.get("structural_config_hash"),
            expected_operational_config_hash=validation.get("operational_config_hash"),
        )
        return (result.get("action_id") or result), True

    @app.callback(
        Output("nr-launch-status", "children"),
        Input("poll", "n_intervals"), Input("nr-action-id", "data"),
        prevent_initial_call=True,
    )
    def refresh_launch(_ticks: int, action: Any) -> Any:
        if not action:
            return dash.no_update
        if isinstance(action, dict):
            return new_run.launch_status(action)  # immediate error body
        return new_run.launch_status(client.action(str(action)))

    # -- write control ------------------------------------------------------
    #
    # These callbacks run server-side and talk only to the API client. No
    # registry, adapter, trainer or WorkerManager is imported here, and the
    # browser never calls the API directly (ADR-WC-019).

    @app.callback(
        Output("run-detail-controls", "children"),
        Input("poll", "n_intervals"),
        State("run-detail-id", "data"),
    )
    def refresh_controls(_ticks: int, run_id: Optional[str]) -> Any:
        if not run_id:
            return dash.no_update
        detail = client.run(run_id)
        if is_error(detail):
            return None
        block = run_detail.controls_block(detail)
        return section("Control actions", [block]) if block is not None else None

    @app.callback(
        Output("active-action-id", "data"),
        Output("stop-button", "disabled"),
        Input("stop-button", "n_clicks"),
        State("run-detail-id", "data"),
        State("stop-idem-key", "data"),
        prevent_initial_call=True,
    )
    def on_stop(n_clicks: Optional[int], run_id: Optional[str], key: Optional[str]) -> Any:
        if not n_clicks or not run_id:
            return dash.no_update, dash.no_update
        detail = client.run(run_id)
        version = detail.get("state_version") if not is_error(detail) else None
        result = client.request_stop(run_id, idempotency_key=str(key),
                                     expected_state_version=version)
        # Disable immediately either way: a retry must reuse the same key, not
        # race a second request.
        return result.get("action_id"), True

    @app.callback(
        Output("active-action-id", "data", allow_duplicate=True),
        Output("resume-button", "disabled"),
        Input("resume-button", "n_clicks"),
        State("run-detail-id", "data"),
        State("resume-idem-key", "data"),
        prevent_initial_call=True,
    )
    def on_resume(n_clicks: Optional[int], run_id: Optional[str], key: Optional[str]) -> Any:
        if not n_clicks or not run_id:
            return dash.no_update, dash.no_update
        detail = client.run(run_id)
        if is_error(detail):
            return dash.no_update, True
        eligibility = (detail.get("action_eligibility") or {}).get("resume_action") or {}
        checkpoint_id = eligibility.get("checkpoint_id")
        if not checkpoint_id:
            return dash.no_update, True
        result = client.request_resume(
            str(checkpoint_id), idempotency_key=str(key),
            expected_parent_state_version=detail.get("state_version"))
        return result.get("action_id"), True

    @app.callback(
        Output("action-status", "children"),
        Input("poll", "n_intervals"),
        Input("active-action-id", "data"),
        prevent_initial_call=True,
    )
    def refresh_action(_ticks: int, action_id: Optional[str]) -> Any:
        if not action_id:
            return dash.no_update
        # Durable state is the authority; the browser only displays it.
        return run_detail.action_status_block(client.action(str(action_id)))

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
