from __future__ import annotations

from typing import Mapping

import streamlit as st


AXIS_MODE_OPTIONS: Mapping[str, str] = {
    "split": "3-axis split",
    "overlay": "Combined axes",
    "norm": "Norm only",
}


def render_axis_toggle(default: str = "split") -> str:
    keys = list(AXIS_MODE_OPTIONS.keys())
    default_key = default if default in AXIS_MODE_OPTIONS else keys[0]
    labels = [AXIS_MODE_OPTIONS[key] for key in keys]
    selected = st.radio(
        "Axis mode",
        labels,
        index=keys.index(default_key),
        horizontal=True,
        key="axis_mode",
    )
    return keys[labels.index(selected)]
