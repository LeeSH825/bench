"""Dash dashboard for the benchmark control plane.

This package is a **client** of the FastAPI service. It holds no run state, opens
no SQLite connection, and never imports `bench.control.registry` — every read
goes over HTTP to `/api/v1/...`. Two consequences follow, both intentional:

* the dashboard cannot accidentally mutate run state (DND-010), because the API
  it talks to has no write endpoints at all;
* replacing Dash with another frontend later is a rewrite of this package only
  (risk R-16).

Nothing here ever executes training. Dash callbacks fetch JSON and render it
(DND-001).
"""

from __future__ import annotations

DASHBOARD_VERSION = "0.1.0"

__all__ = ["DASHBOARD_VERSION"]
