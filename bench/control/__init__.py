"""Frontend-agnostic control plane for benchmark run execution and observation.

This package owns run identity, typed run specs, the run registry, the event
journal, worker process supervision, and resource telemetry. It deliberately
imports **no** UI framework: `bench.control` never depends on Dash, Streamlit,
or Flask (see design doc 05, DND-009). The FastAPI service in
`bench.control.api` is a thin read-only projection over these primitives, and
`bench.ui` is a thin Dash client over that HTTP API.

Layering, innermost first:

    identity / canonical      pure value objects and hashing
    config                    typed run configuration + resolution
    allocation                immutable run directory allocation
    registry                  SQLite state-of-record (WAL)
    events                    append-only JSONL journal + observer protocol
    telemetry                 CPU/GPU resource sampling
    process                   subprocess/process-group worker supervision
    api                       read-only HTTP projection
"""

from __future__ import annotations

#: Version of the ``bench.control`` contract surface as a whole. Bumped when
#: any cross-component wire format changes in a non-additive way.
CONTROL_PLANE_VERSION = "0.1.0"

__all__ = ["CONTROL_PLANE_VERSION"]
