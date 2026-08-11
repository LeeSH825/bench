"""Backward-compatible version shim for older ``bench.init`` imports."""

from __future__ import annotations

from . import __version__

__all__ = ["__version__"]
