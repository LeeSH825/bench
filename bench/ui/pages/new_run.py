"""New Run wizard.

Everything here goes through ``ApiClient``. This module imports no registry,
no resolver, no adapter and no WorkerManager — the browser and the Dash server
only ever see what the API chose to expose, and the API re-validates whatever
the form submits.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dash import dcc, html

STEPS = ("Choose preset", "Configure", "Validate", "Review", "Launch")

LAUNCH_DESCRIPTION = (
    "선택한 preset에서 새 immutable run을 생성합니다. "
    "원본 preset 파일은 변경되지 않습니다."
)


def _muted(text: str, size: str = "0.8rem") -> html.Div:
    return html.Div(text, style={"fontSize": size, "color": "#64748b"})


def _warn(text: str) -> html.Div:
    return html.Div(text, style={
        "fontSize": "0.8rem", "color": "#92400e", "background": "#fffbeb",
        "border": "1px solid #fde68a", "borderRadius": "6px", "padding": "8px",
        "marginTop": "6px"})


def _error(text: str) -> html.Div:
    return html.Div(text, style={
        "fontSize": "0.8rem", "color": "#b91c1c", "background": "#fef2f2",
        "border": "1px solid #fecaca", "borderRadius": "6px", "padding": "8px",
        "marginTop": "6px"})


def step_header(active: int) -> html.Div:
    chips = []
    for index, name in enumerate(STEPS):
        done = index < active
        current = index == active
        chips.append(html.Div(
            f"{index + 1}. {name}",
            style={
                "padding": "4px 10px", "borderRadius": "999px", "fontSize": "0.76rem",
                "background": "#1d4ed8" if current else ("#dcfce7" if done else "#e2e8f0"),
                "color": "white" if current else "#334155",
            }))
    return html.Div(chips, style={"display": "flex", "gap": "6px",
                                  "flexWrap": "wrap", "marginBottom": "12px"})


def preset_options(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Dropdown options. Unsupported presets stay visible but disabled."""
    options = []
    for entry in payload.get("presets") or []:
        supported = bool(entry.get("launch_supported"))
        label = entry.get("display_name") or entry.get("preset_id")
        models = ", ".join(entry.get("launchable_model_ids") or entry.get("model_ids") or [])
        suffix = f" — {models}" if models else ""
        if not supported:
            suffix += "  (not launchable)"
        options.append({
            "label": f"{label}{suffix}",
            "value": entry.get("preset_id"),
            "disabled": not supported,
        })
    return options


def preset_summary(entry: Optional[Mapping[str, Any]]) -> Any:
    if not entry:
        return _muted("Choose a preset to continue.")
    rows = [
        html.Div([html.Strong(entry.get("display_name") or ""),
                  html.Span(f"  ({entry.get('relative_path')})",
                            style={"fontSize": "0.76rem", "color": "#64748b"})]),
        _muted(f"digest {str(entry.get('content_digest') or '')[:26]}…"),
        _muted(f"suite {entry.get('suite_name')} {entry.get('suite_version') or ''}"),
        _muted(f"tasks: {', '.join(entry.get('task_ids') or [])}"),
        _muted(f"models: {', '.join(entry.get('model_ids') or [])}"),
    ]
    if not entry.get("launch_supported"):
        rows.append(_warn(entry.get("unsupported_reason") or "This preset is not launchable."))
    return html.Div(rows)


def config_form(schema: Mapping[str, Any], entry: Optional[Mapping[str, Any]],
                model_trainable: bool) -> Any:
    """Schema-driven form. Fields come from the API descriptor, not from here."""
    if not schema:
        return _muted("Schema unavailable.")
    groups = {g["id"]: g for g in schema.get("groups") or []}
    by_group: dict[str, list[Any]] = {}

    for field in schema.get("fields") or []:
        if field.get("read_only"):
            continue
        # Conditional visibility, evaluated from the API's own predicate name.
        if field.get("visible_when") == "model_trainable" and not model_trainable:
            continue
        path = field["path"]
        control: Any
        if field["type"] == "boolean":
            control = dcc.Dropdown(
                id={"role": "cfg", "path": path},
                options=[{"label": "true", "value": "true"},
                         {"label": "false", "value": "false"}],
                value=str(bool(field.get("default"))).lower(), clearable=False,
                style={"fontSize": "0.82rem"})
        elif field.get("enum"):
            control = dcc.Dropdown(
                id={"role": "cfg", "path": path},
                options=[{"label": str(v), "value": str(v)} for v in field["enum"]],
                value=str(field.get("default") or field["enum"][0]), clearable=False,
                style={"fontSize": "0.82rem"})
        else:
            control = dcc.Input(
                id={"role": "cfg", "path": path},
                type="number" if field["type"] in ("integer", "number") else "text",
                value=field.get("default"),
                min=field.get("minimum"), max=field.get("maximum"),
                debounce=True, style={"width": "100%", "fontSize": "0.82rem"})

        tag = field.get("classification")
        by_group.setdefault(field.get("group", "experiment"), []).append(html.Div([
            html.Div([
                html.Label(field["label"], style={"fontSize": "0.82rem", "fontWeight": 600}),
                html.Span(tag, style={
                    "marginLeft": "6px", "fontSize": "0.68rem", "padding": "1px 6px",
                    "borderRadius": "999px",
                    "background": "#ede9fe" if tag == "structural" else "#e2e8f0",
                    "color": "#4c1d95" if tag == "structural" else "#475569"}),
            ]),
            control,
            _muted(field.get("help") or "", "0.74rem"),
        ], style={"marginBottom": "10px"}))

    sections = []
    for group_id, controls in by_group.items():
        group = groups.get(group_id, {"label": group_id, "help": ""})
        sections.append(html.Div([
            html.H4(group["label"], style={"margin": "8px 0 2px"}),
            _muted(group.get("help", ""), "0.76rem"),
            html.Div(controls, style={"marginTop": "8px"}),
        ], style={"marginBottom": "14px"}))
    if not model_trainable:
        sections.insert(0, _muted(
            "Training fields are hidden: this model has no learning lifecycle, so it "
            "runs evaluation only and offers no Stop or Resume."))
    return html.Div(sections)


def validation_panel(result: Optional[Mapping[str, Any]]) -> Any:
    if not result:
        return _muted("Run validation to see issues and the resolved spec.")
    children: list[Any] = []
    if result.get("valid"):
        children.append(html.Div("Configuration is valid.",
                                 style={"color": "#065f46", "fontWeight": 600}))
    else:
        children.append(html.Div("Configuration is not valid.",
                                 style={"color": "#b91c1c", "fontWeight": 600}))
    for issue in result.get("issues") or []:
        text = f"[{issue.get('code')}] {issue.get('path') or '(document)'}: {issue.get('message')}"
        children.append(_warn(text) if issue.get("severity") == "warning" else _error(text))
    unsupported = result.get("unsupported_fields") or []
    if unsupported:
        children.append(html.Div([
            html.Div("Preserved but unmanaged keys", style={"fontWeight": 600,
                                                            "fontSize": "0.82rem"}),
            _muted(", ".join(unsupported[:20]), "0.76rem"),
            _muted("These stay in the raw YAML verbatim; the form does not manage them."),
        ], style={"marginTop": "8px"}))
    return html.Div(children)


def review_panel(result: Optional[Mapping[str, Any]]) -> Any:
    """Diff, hashes and identity — what the operator approves before launching."""
    if not result or not result.get("valid"):
        return _muted("Validate a configuration to review it.")
    diff = result.get("diff") or {}
    changed = diff.get("changed_fields") or []

    rows = [html.Div([
        html.Strong("Resolved identity"),
        _muted(f"variant_id: {result.get('variant_id')}"),
        _muted(f"implementation_id: {result.get('implementation_id')}"),
        _muted(f"training_path_id: {result.get('training_path_id')}"),
        _muted(f"structural_config_hash: {result.get('structural_config_hash')}"),
        _muted(f"operational_config_hash: {result.get('operational_config_hash')}"),
    ], style={"marginBottom": "10px"})]

    if changed:
        rows.append(html.Div([
            html.Strong(f"Changed from preset ({len(changed)})"),
            html.Div([
                html.Div([
                    html.Code(c["path"]),
                    html.Span(f"  {c.get('before')} → {c.get('after')}",
                              style={"fontSize": "0.8rem"}),
                    html.Span(f"  [{c.get('classification')}]", style={
                        "fontSize": "0.7rem",
                        "color": "#4c1d95" if c.get("classification") == "structural"
                                 else "#64748b"}),
                ], style={"fontSize": "0.82rem"}) for c in changed
            ], style={"marginTop": "4px"}),
        ], style={"marginBottom": "10px"}))
    else:
        rows.append(_muted("No managed field differs from the preset."))

    if diff.get("structural_changed"):
        rows.append(_warn(
            "Structural fields changed, so this is a different experiment than the "
            "preset: it resolves to a different variant_id and its results are not "
            "directly comparable with runs from the unmodified preset."))

    eligibility = result.get("launch_eligibility") or {}
    if eligibility.get("eligible"):
        note = ("Stop and Resume will be available for this run."
                if eligibility.get("stop_resume_available")
                else "This run has no learning lifecycle, so Stop/Resume will not be offered.")
        rows.append(html.Div(note, style={"fontSize": "0.8rem", "color": "#065f46"}))
    else:
        rows.append(_error(eligibility.get("reason") or "This configuration cannot be launched."))
    return html.Div(rows)


def launch_confirmation(entry: Optional[Mapping[str, Any]],
                        result: Optional[Mapping[str, Any]],
                        model_id: Optional[str]) -> Any:
    if not (entry and result and result.get("valid")):
        return None
    spec = result.get("resolved_run_spec") or {}
    runtime = spec.get("runtime") or {}
    training = spec.get("training") or {}
    return html.Div([
        html.Strong("About to launch"),
        _muted(f"preset: {entry.get('display_name')} ({entry.get('relative_path')})"),
        _muted(f"model: {model_id} · task: {(spec.get('system') or {}).get('task_id')}"),
        _muted(f"device {runtime.get('device')} · precision {runtime.get('precision')} · "
               f"workers {runtime.get('num_workers')}"),
        _muted(f"training budget: {training.get('max_updates')} optimizer updates"),
        _muted(f"training_path_id: {result.get('training_path_id')}"),
        _muted(LAUNCH_DESCRIPTION),
        _warn("There is no GPU queue or scheduler in this build: the run starts "
              "immediately and competes for local resources with anything else running."),
    ], style={"border": "1px solid #cbd5f5", "borderRadius": "6px",
              "padding": "10px", "background": "#f8fafc", "marginBottom": "10px"})


def launch_status(action: Optional[Mapping[str, Any]]) -> Any:
    if not action:
        return None
    if action.get("_error"):
        return _error(f"{action.get('reason_code') or 'error'}: {action.get('message') or ''}")
    rows = [html.Div([html.Strong("LAUNCH_RUN "),
                      html.Span(str(action.get("state") or ""),
                                style={"fontFamily": "monospace"})])]
    if action.get("error"):
        rows.append(_error(str(action["error"])))
    run_id = action.get("run_id")
    if run_id:
        rows.append(html.Div([
            html.Span("run: ", style={"fontSize": "0.8rem"}),
            dcc.Link(str(run_id)[:12] + "…", href=f"/runs/{run_id}", id="launched-run-link",
                     style={"color": "#1d4ed8", "fontSize": "0.8rem"}),
        ]))
        rows.append(_muted("The launch action is complete once the worker starts; "
                           "the run's own state tells you how the benchmark finished."))
    return html.Div(rows, id="launch-status-body", style={
        "border": "1px solid #cbd5f5", "borderRadius": "6px",
        "padding": "10px", "background": "#f8fafc"})


def layout(write_enabled: bool = False) -> html.Div:
    return html.Div([
        dcc.Store(id="nr-preset-entry"), dcc.Store(id="nr-schema"),
        dcc.Store(id="nr-validation"), dcc.Store(id="nr-action-id"),
        dcc.Store(id="nr-idem-key"),
        html.H2("New run", style={"marginTop": 0}),
        _muted("Start from a tracked preset, adjust a bounded set of fields, review the "
               "resolved configuration, then launch an immutable run."),
        html.Div(id="nr-steps"),
        html.Div([
            html.H4("1. Choose preset", style={"marginBottom": "4px"}),
            dcc.Dropdown(id="nr-preset", placeholder="Select a tracked preset…",
                         style={"fontSize": "0.85rem"}),
            html.Div(id="nr-preset-summary", style={"marginTop": "8px"}),
            html.Div([
                html.Label("Model", style={"fontSize": "0.82rem", "fontWeight": 600}),
                dcc.Dropdown(id="nr-model", placeholder="Model…",
                             style={"fontSize": "0.82rem"}),
                html.Label("Initialization", style={"fontSize": "0.82rem",
                                                    "fontWeight": 600, "marginTop": "6px"}),
                dcc.Dropdown(id="nr-init",
                             options=[{"label": "trained", "value": "trained"},
                                      {"label": "untrained", "value": "untrained"}],
                             value="trained", clearable=False,
                             style={"fontSize": "0.82rem"}),
            ], style={"marginTop": "10px", "maxWidth": "420px"}),
        ], style={"marginBottom": "16px"}),
        html.Div([
            html.H4("2. Configure", style={"marginBottom": "4px"}),
            dcc.Tabs(id="nr-edit-mode", value="form", children=[
                dcc.Tab(label="Form", value="form"),
                dcc.Tab(label="Raw YAML", value="yaml"),
            ]),
            html.Div(id="nr-form", style={"marginTop": "10px"}),
            html.Div(id="nr-yaml-wrap", children=[
                dcc.Textarea(id="nr-yaml", style={
                    "width": "100%", "height": "300px", "fontFamily": "monospace",
                    "fontSize": "0.78rem"}),
                _muted("Editing YAML edits the same draft the form does. Keys the schema "
                       "does not model are preserved verbatim and reported as unmanaged."),
            ], style={"display": "none"}),
        ], style={"marginBottom": "16px"}),
        html.Div([
            html.H4("3. Validate", style={"marginBottom": "4px"}),
            html.Button("Validate", id="nr-validate", n_clicks=0, style={
                "padding": "6px 12px", "borderRadius": "6px", "border": "1px solid #94a3b8",
                "background": "white", "cursor": "pointer"}),
            html.Div(id="nr-validation-panel", style={"marginTop": "8px"}),
        ], style={"marginBottom": "16px"}),
        html.Div([
            html.H4("4. Review", style={"marginBottom": "4px"}),
            html.Div(id="nr-review"),
        ], style={"marginBottom": "16px"}),
        html.Div([
            html.H4("5. Launch", style={"marginBottom": "4px"}),
            html.Div(id="nr-confirmation"),
            (html.Button("Launch run", id="nr-launch", n_clicks=0, style={
                "padding": "8px 14px", "borderRadius": "6px", "border": "1px solid #15803d",
                "background": "#22c55e", "color": "white", "fontWeight": 600,
                "cursor": "pointer"})
             if write_enabled else
             _warn("Launching requires write mode. Start the API and dashboard with "
                   "BENCH_CONTROL_ENABLE_WRITES=1 on a loopback bind.")),
            html.Div(id="nr-launch-status", style={"marginTop": "10px"}),
        ]),
    ])
