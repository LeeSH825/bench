"""FastAPI application factory.

Read-only by construction: the app registers only `GET` routes. There is no
`POST /runs`, no stop/terminate/resume action, and no write path of any kind.
Launching happens from the CLI (`bench.control.cli`), which is a separate
process from the one serving this API.

Binding: the server binds `127.0.0.1` by default and refuses `0.0.0.0` unless
`BENCH_CONTROL_ALLOW_PUBLIC_BIND=1` is set. The control plane has no
authentication and exposes filesystem paths; putting it on a public interface
without a reverse proxy is exactly what design doc 03 §17 forbids.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .. import CONTROL_PLANE_VERSION
from .deps import configure
from .routers.runs import router as runs_router
from .routers.system import capabilities_router, router as system_router

#: Environment flag required to bind a non-loopback interface.
ALLOW_PUBLIC_BIND_ENV = "BENCH_CONTROL_ALLOW_PUBLIC_BIND"


def create_app(control_root_path: Optional[str] = None) -> FastAPI:
    """Build the read-only API application."""
    if control_root_path is not None:
        configure(control_root_path)

    app = FastAPI(
        title="Benchmark Control Plane (read-only)",
        version=CONTROL_PLANE_VERSION,
        description=(
            "Read-only observation API for benchmark runs. This service does not "
            "start, stop, or modify runs; the run registry and the worker processes "
            "are the authoritative owners of run state."
        ),
    )
    app.include_router(system_router)
    app.include_router(capabilities_router)
    app.include_router(runs_router)

    @app.get("/", include_in_schema=False)
    def index() -> dict[str, Any]:
        return {
            "service": "bench control plane",
            "version": CONTROL_PLANE_VERSION,
            "read_only": True,
            "docs": "/docs",
            "health": "/api/v1/system/health",
        }

    @app.exception_handler(FileNotFoundError)
    def _missing_file(_request: Any, exc: FileNotFoundError) -> JSONResponse:
        # A run directory that vanished under us is a 404, not a 500.
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    return app


def resolve_bind_host(requested: str) -> str:
    """Validate a bind address, refusing accidental public exposure."""
    host = str(requested).strip()
    if host in ("127.0.0.1", "localhost", "::1"):
        return host
    if os.environ.get(ALLOW_PUBLIC_BIND_ENV) == "1":
        return host
    raise SystemExit(
        f"refusing to bind {host!r}: the control-plane API has no authentication and "
        f"exposes local filesystem paths. Bind 127.0.0.1, or set "
        f"{ALLOW_PUBLIC_BIND_ENV}=1 if this host is already behind a trusted proxy."
    )


def main(argv: Optional[list[str]] = None) -> int:  # pragma: no cover - process entry point
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="bench.control.api.app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--control-root", default=None)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    host = resolve_bind_host(args.host)
    if args.control_root:
        os.environ["BENCH_CONTROL_ROOT"] = args.control_root
        configure(args.control_root)
    uvicorn.run(create_app(), host=host, port=int(args.port), log_level=args.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
