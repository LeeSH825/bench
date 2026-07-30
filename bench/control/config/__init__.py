"""Typed run configuration: schema, resolution, and legacy compatibility.

The three modules here have distinct jobs and should not be merged:

``schema``
    Pure typed value objects plus field-level validation. Knows nothing about
    YAML, about the bench suite format, or about hashing policy.
``resolver``
    Turns a validated :class:`~bench.control.config.schema.RunSpecDraft` into an
    immutable :class:`~bench.control.config.schema.ResolvedRunSpec`, computing
    identity and the structural/operational hashes.
``compatibility``
    Adapts the **existing** bench suite YAML (`bench/configs/*.yaml`) onto the
    typed schema, without changing the suite format or the existing CLI.

This layering is what lets the control plane coexist with `run_suite.py`
instead of replacing it.
"""

from __future__ import annotations

from .schema import (  # noqa: F401
    CONFIG_SCHEMA_VERSION,
    ConfigValidationError,
    ResolvedRunSpec,
    RunSpecDraft,
    UnknownKeyPolicy,
    ValidationIssue,
)
from .resolver import resolve_run_spec  # noqa: F401

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ConfigValidationError",
    "ResolvedRunSpec",
    "RunSpecDraft",
    "UnknownKeyPolicy",
    "ValidationIssue",
    "resolve_run_spec",
]
