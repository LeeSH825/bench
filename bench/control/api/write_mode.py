"""Explicit, local-only write mode.

Write routes do not exist unless they are switched on. That is deliberate: the
control plane has no authentication, so a write surface that is merely
"disabled" by a runtime check is one bug away from being live. Here the routes
are never registered at all, so a POST to a write path in the default build is
a routing miss, not a permission decision.

Enabling writes additionally requires a loopback bind, and this cannot be
overridden by ``BENCH_CONTROL_ALLOW_PUBLIC_BIND`` — that flag exists to permit
*read-only* exposure behind a trusted proxy, and reusing it to open a write
surface would silently widen its meaning.
"""

from __future__ import annotations

import ipaddress
import os

#: Set to a truthy value to register the write routes.
ENABLE_WRITES_ENV = "BENCH_CONTROL_ENABLE_WRITES"

#: Every write POST must carry this header. It is not a security control — it
#: is a same-origin/CSRF speed bump that stops a plain HTML form or a naive
#: cross-site fetch from reaching a write route, since neither can set custom
#: headers without a CORS preflight this app never grants.
REQUEST_HEADER = "X-Bench-Control-Request"
REQUEST_HEADER_VALUE = "1"

#: Accepted truthy spellings. Anything else — including "yes", "on", "2" — is
#: false, because a write surface should fail closed on an ambiguous value.
_TRUE = frozenset({"1", "true"})
_FALSE = frozenset({"", "0", "false"})


def parse_bool(raw: object) -> bool:
    """Strict boolean parsing. Unrecognised values are False, never True."""
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return False


def writes_enabled(env: object = None) -> bool:
    """True when the operator explicitly asked for write routes."""
    source = env if env is not None else os.environ
    return parse_bool(source.get(ENABLE_WRITES_ENV))  # type: ignore[union-attr]


def is_loopback(host: str) -> bool:
    """True for loopback literals and hostnames that only mean loopback."""
    text = str(host).strip().strip("[]")
    if text.lower() in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


class WriteModeError(RuntimeError):
    """Write mode requested in a configuration that cannot be allowed."""


def assert_write_bind_allowed(host: str, *, enabled: bool | None = None) -> None:
    """Refuse to start a write-enabled server on a non-loopback interface.

    Called at startup rather than per-request so the operator learns
    immediately, instead of discovering it when a control action is refused.
    """
    if enabled is None:
        enabled = writes_enabled()
    if not enabled:
        return
    if is_loopback(host):
        return
    raise WriteModeError(
        "Write control requires a loopback bind because authentication is not "
        f"implemented. Refusing to serve write routes on {host!r}. "
        f"Unset {ENABLE_WRITES_ENV} to run read-only, or bind 127.0.0.1. "
        "BENCH_CONTROL_ALLOW_PUBLIC_BIND does not permit public writes."
    )
