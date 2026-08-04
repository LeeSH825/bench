"""Canonical JSON serialization and content hashing.

Every persistent identifier in the control plane that is derived from content
(rather than allocated) is a SHA-256 over a canonical JSON encoding produced
here. Python's builtin ``hash()`` is **never** used for persistence: it is salted
per process (PYTHONHASHSEED) for str/bytes, so it is not stable across restarts.
See design doc 03 §2.2.

Canonical form rules
--------------------

* object keys sorted lexicographically by their Unicode code points
* no insignificant whitespace (``separators=(",", ":")``)
* UTF-8 output, non-ASCII characters emitted literally (``ensure_ascii=False``)
* floats that hold an integral value are **not** collapsed to int — ``1.0`` and
  ``1`` are different canonical inputs, because collapsing them would let a
  config change silently keep its hash
* ``NaN``/``Infinity`` are rejected: they are not valid JSON and comparing them
  across processes is unreliable
* tuples and sets are not silently accepted; callers must decide list ordering
  explicitly, because set iteration order is not a stable identity input
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

#: Prefix used on every content-derived identifier so that a stored value is
#: self-describing and a future migration to another digest stays detectable.
SHA256_PREFIX = "sha256:"


class CanonicalizationError(ValueError):
    """Raised when a value cannot be canonically encoded."""


def _check(value: Any, path: str) -> Any:
    """Recursively validate and normalize *value* into JSON-canonical types."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        # bool is a subclass of int and is handled above.
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise CanonicalizationError(
                f"non-finite float at {path!r} cannot be canonically encoded: {value!r}"
            )
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(
                    f"non-string mapping key at {path!r}: {key!r} ({type(key).__name__})"
                )
            out[key] = _check(item, f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_check(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        raise CanonicalizationError(
            f"set at {path!r} has no stable ordering; convert to a sorted list at the call site"
        )
    raise CanonicalizationError(
        f"unsupported type at {path!r}: {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    """Return the canonical JSON text for *value*.

    The result is stable across processes, interpreter restarts, and dict
    insertion order.
    """
    normalized = _check(value, "$")
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    """UTF-8 encoding of :func:`canonical_json`."""
    return canonical_json(value).encode("utf-8")


def content_hash(value: Any) -> str:
    """Return ``sha256:<hex>`` over the canonical encoding of *value*."""
    return SHA256_PREFIX + hashlib.sha256(canonical_bytes(value)).hexdigest()


def text_hash(text: str) -> str:
    """Return ``sha256:<hex>`` over UTF-8 *text* (for file/blob content)."""
    return SHA256_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(hash_value: str, length: int = 12) -> str:
    """Short display form of a ``sha256:``-prefixed hash.

    Presentation only — never use the truncated form as a key.
    """
    digest = hash_value[len(SHA256_PREFIX):] if hash_value.startswith(SHA256_PREFIX) else hash_value
    return digest[:length]
