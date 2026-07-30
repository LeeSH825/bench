from __future__ import annotations

"""Streamlit rendering for the Run Inspector "Help & guide" popover.

Everything rendered here is static Markdown from `viz.app.help_content` —
no run discovery, artifact load, trajectory load, or figure build happens
in this module. That keeps the popover free to open/close on every rerun
without adding to run-index-scan or NPZ-load cost (see
reports/VIZ_USER_GUIDE_AND_CONTEXT_HELP_REPORT.md, "Help rendering performance").
"""

import streamlit as st

from viz.app.help_content import GUIDE_TABS, HELP_TEXT


def render_user_guide() -> None:
    tabs = st.tabs([title for title, _ in GUIDE_TABS])
    for tab, (_title, body) in zip(tabs, GUIDE_TABS):
        with tab:
            st.markdown(body)


def render_help_popover() -> None:
    with st.popover(
        "Help & guide",
        icon=":material/help:",
        help=HELP_TEXT["help_button"],
        width="stretch",
    ):
        render_user_guide()
