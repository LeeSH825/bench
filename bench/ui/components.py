"""Shared presentation helpers for the dashboard.

Accessibility note (acceptance U-09): run state is shown as a **text label**
with a coloured border, never colour alone. Every badge carries a `title`
attribute so the reason is available on focus and to a screen reader.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dash import html

#: Palette per run state. Colour is redundant with the text label by design.
STATE_STYLES: dict[str, str] = {
    "CREATED": "#6b7280",
    "VALIDATING": "#6b7280",
    "QUEUED": "#6b7280",
    "STARTING": "#0284c7",
    "RUNNING": "#0284c7",
    "COMPLETED": "#15803d",
    "FAILED": "#b91c1c",
    "CANCELLED": "#a16207",
    "ORPHANED": "#7c2d12",
    "STOP_REQUESTED": "#a16207",
    "CHECKPOINTING": "#0284c7",
    "INTERRUPTED": "#a16207",
    "RESUMING": "#0284c7",
}

FIDELITY_STYLES: dict[str, str] = {
    "verified": "#15803d",
    "partial": "#a16207",
    "unverified": "#6b7280",
    "not_applicable": "#6b7280",
}

INSTRUMENTATION_LABELS: dict[str, str] = {
    "step": "step-level events",
    "phase": "phase-level events only",
    "none": "no structured events",
}


def badge(text: str, colour: str, *, title: Optional[str] = None) -> html.Span:
    return html.Span(
        text,
        title=title or text,
        style={
            "display": "inline-block",
            "padding": "1px 8px",
            "borderRadius": "10px",
            "border": f"1px solid {colour}",
            "color": colour,
            "fontSize": "0.78rem",
            "fontWeight": 600,
            "whiteSpace": "nowrap",
        },
    )


def state_badge(state: str, *, title: Optional[str] = None) -> html.Span:
    return badge(state, STATE_STYLES.get(state, "#6b7280"), title=title)


def fidelity_badge(status: str, note: str = "") -> html.Span:
    """Paper-fidelity badge.

    The note is surfaced as the tooltip because "partial" is meaningless without
    knowing *what* deviates (DND-013).
    """
    label = {
        "verified": "fidelity: verified",
        "partial": "fidelity: partial",
        "unverified": "fidelity: unverified",
        "not_applicable": "fidelity: n/a",
    }.get(status, f"fidelity: {status}")
    return badge(label, FIDELITY_STYLES.get(status, "#6b7280"), title=note or label)


def instrumentation_badge(level: str, note: str = "") -> html.Span:
    colour = {"step": "#15803d", "phase": "#a16207", "none": "#b91c1c"}.get(level, "#6b7280")
    return badge(INSTRUMENTATION_LABELS.get(level, level), colour, title=note or level)


def resume_badge(supports_exact_resume: bool) -> html.Span:
    """Exact-resume certification badge.

    Always rendered, including the negative case. A researcher must be able to
    see at a glance that a checkpoint is *not* resume-certified; silence would
    be read as "probably fine" (risk R-05).
    """
    if supports_exact_resume:
        return badge("exact resume: certified", "#15803d", title="A parity test certifies this implementation.")
    return badge(
        "exact resume: not certified",
        "#b91c1c",
        title=(
            "No continuous-vs-resumed parity test has certified this implementation. "
            "Loading its weights is a warm start, not a resume."
        ),
    )


def error_panel(payload: Mapping[str, Any]) -> html.Div:
    return html.Div(
        [
            html.Strong(str(payload.get("error", "error"))),
            html.Div(str(payload.get("detail", "")), style={"marginTop": "4px"}),
        ],
        style={
            "border": "1px solid #b91c1c",
            "background": "#fef2f2",
            "color": "#7f1d1d",
            "padding": "10px 12px",
            "borderRadius": "6px",
            "margin": "8px 0",
        },
    )


def info_panel(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "border": "1px solid #cbd5e1",
            "background": "#f8fafc",
            "color": "#334155",
            "padding": "8px 12px",
            "borderRadius": "6px",
            "margin": "6px 0",
            "fontSize": "0.9rem",
        },
    )


def section(title: str, children: Sequence[Any], *, subtitle: Optional[str] = None) -> html.Section:
    header: list[Any] = [html.H3(title, style={"margin": "0 0 2px 0", "fontSize": "1.05rem"})]
    if subtitle:
        header.append(
            html.Div(subtitle, style={"color": "#64748b", "fontSize": "0.82rem", "marginBottom": "6px"})
        )
    return html.Section(
        [*header, *children],
        style={
            "border": "1px solid #e2e8f0",
            "borderRadius": "8px",
            "padding": "12px 14px",
            "marginBottom": "14px",
            "background": "#ffffff",
        },
    )


def key_values(items: Sequence[tuple[str, Any]]) -> html.Div:
    rows = []
    for key, value in items:
        rows.append(
            html.Div(
                [
                    html.Span(
                        key,
                        style={"color": "#64748b", "fontSize": "0.78rem", "display": "block"},
                    ),
                    html.Span(
                        "—" if value in (None, "") else str(value),
                        style={"fontFamily": "ui-monospace, monospace", "fontSize": "0.86rem", "wordBreak": "break-all"},
                    ),
                ],
                style={"minWidth": "160px", "flex": "1 1 200px", "marginBottom": "8px"},
            )
        )
    return html.Div(rows, style={"display": "flex", "flexWrap": "wrap", "gap": "12px"})


def format_bytes(value: Optional[float]) -> str:
    if value is None:
        return "—"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PiB"


PAGE_STYLE: dict[str, Any] = {
    "fontFamily": "system-ui, -apple-system, 'Segoe UI', sans-serif",
    "background": "#f1f5f9",
    "minHeight": "100vh",
    "padding": "16px 20px 40px 20px",
    "color": "#0f172a",
}
