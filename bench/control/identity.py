"""Canonical identity types for the control plane.

Two families of identifier live here, and they must not be confused:

**Allocated** identifiers (:class:`ExperimentId`, :class:`RunId`,
:class:`CheckpointId`, :class:`ArtifactId`) are time-ordered UUIDv7 values. They
are minted once, are unique even for byte-identical configuration, and are the
only things used as database keys and directory names. Two launches of the same
config get two different :class:`RunId` values — that is the whole point
(design doc 05, DND-004).

**Derived** identifiers (:class:`VariantId`) are SHA-256 content hashes over a
canonical JSON document. They answer "are these two runs comparable?", never
"which run is this?".

:class:`ModelId`, :class:`ImplementationId` and :class:`InitId` are validated
value objects, not free-form strings, so that a display label can never be
passed where an identity is expected.

Presentation strings are produced by :func:`variant_label` and are explicitly
**not** identities (design doc 05, DND-005).
"""

from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .canonical import content_hash, short_hash

#: Schema version of the variant-identity document. Bump when the set of fields
#: fed into the variant hash changes — that invalidates previously computed
#: variant_ids, so it must be a deliberate, visible act.
VARIANT_IDENTITY_SCHEMA_VERSION = 1

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

#: Recognised initialization provenance modes. `trained` means this run trained
#: the weights itself; `pretrained`/`loaded` mean weights came from a prior
#: artifact; `untrained` means randomly initialized. These mirror the values
#: `bench/runners/run_suite.py` already uses for `init_id`, so legacy runs map
#: onto the same vocabulary.
INIT_MODES = ("untrained", "trained", "pretrained", "loaded", "unknown")


class IdentityError(ValueError):
    """Raised when an identity value is malformed."""


# --------------------------------------------------------------------------- #
# UUIDv7 allocation
# --------------------------------------------------------------------------- #


def uuid7() -> str:
    """Generate a UUIDv7 (RFC 9562) as a lowercase hyphenated string.

    UUIDv7 is time-ordered: the leading 48 bits are a Unix millisecond
    timestamp, so lexicographic sort ≈ creation order, which keeps run listings
    and directory listings naturally ordered without a separate sort key.

    Implemented locally because Python 3.10 has no ``uuid.uuid7()`` (added in
    3.14) and the control plane must not take a dependency for 20 lines.
    Randomness comes from ``os.urandom``, so collisions across concurrent
    processes are not a practical concern (74 random bits per millisecond).
    """
    unix_ms = int(time.time() * 1000)
    rand = os.urandom(10)
    value = bytearray(16)
    value[0:6] = unix_ms.to_bytes(6, "big")
    value[6:16] = rand
    # version 7 in the high nibble of octet 6
    value[6] = (value[6] & 0x0F) | 0x70
    # RFC 4122 variant in the two high bits of octet 8
    value[8] = (value[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(value)))


def uuid7_timestamp_ms(value: str) -> int:
    """Extract the embedded millisecond timestamp from a UUIDv7 string."""
    parsed = uuid.UUID(value)
    if parsed.version != 7:
        raise IdentityError(f"not a UUIDv7: {value!r} (version {parsed.version})")
    return int.from_bytes(parsed.bytes[0:6], "big")


def _validate_uuid(value: str, kind: str) -> str:
    text = str(value).strip().lower()
    if not _UUID_RE.match(text):
        raise IdentityError(f"{kind} must be a hyphenated UUID string, got {value!r}")
    return text


def _validate_slug(value: str, kind: str) -> str:
    text = str(value).strip().lower()
    if not _SLUG_RE.match(text):
        raise IdentityError(
            f"{kind} must match {_SLUG_RE.pattern} (lowercase slug), got {value!r}"
        )
    return text


# --------------------------------------------------------------------------- #
# Allocated identifiers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, order=True)
class _UuidId:
    """Base for UUID-backed immutable identifiers."""

    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validate_uuid(self.value, type(self).__name__))

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @property
    def created_at_ms(self) -> int:
        """Creation time embedded in the UUIDv7 payload."""
        return uuid7_timestamp_ms(self.value)


@dataclass(frozen=True, order=True)
class ExperimentId(_UuidId):
    """Groups repeated runs into one logical experiment."""

    @staticmethod
    def new() -> "ExperimentId":
        return ExperimentId(uuid7())


@dataclass(frozen=True, order=True)
class RunId(_UuidId):
    """Identifies exactly one execution. Immutable, never derived from config."""

    @staticmethod
    def new() -> "RunId":
        return RunId(uuid7())


@dataclass(frozen=True, order=True)
class CheckpointId(_UuidId):
    """Identifies one checkpoint payload + manifest pair."""

    @staticmethod
    def new() -> "CheckpointId":
        return CheckpointId(uuid7())


@dataclass(frozen=True, order=True)
class ArtifactId(_UuidId):
    """Identifies one recorded output artifact."""

    @staticmethod
    def new() -> "ArtifactId":
        return ArtifactId(uuid7())


@dataclass(frozen=True, order=True)
class WorkerInstanceId(_UuidId):
    """Identifies one worker process instance (not a PID — see registry)."""

    @staticmethod
    def new() -> "WorkerInstanceId":
        return WorkerInstanceId(uuid7())


# --------------------------------------------------------------------------- #
# Validated value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, order=True)
class ModelId:
    """Algorithm family key, as used by ``bench.models.registry``.

    A ``ModelId`` alone does **not** determine behaviour: the bench registry maps
    20 model ids onto 10 adapter classes, so several ids share an implementation.
    Always pair it with an :class:`ImplementationId`.
    """

    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validate_slug(self.value, "ModelId"))

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True, order=True)
class ImplementationId:
    """Concrete adapter/upstream implementation, including a version suffix.

    Example: ``bench_split_adapter_v1``. Bump the version component whenever the
    numerical behaviour of the adapter changes, because certification
    (paper fidelity, exact resume) is granted per implementation version, never
    per model family.
    """

    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validate_slug(self.value, "ImplementationId"))

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True, order=True)
class InitId:
    """Initialization provenance: a mode plus optional source detail.

    ``mode`` is one of :data:`INIT_MODES`. ``source_checkpoint_hash`` and
    ``source_run_id`` record where loaded weights came from; both participate in
    variant identity so that "same model, two different pretrained checkpoints"
    are distinguishable — the exact collision the audit flagged.
    """

    mode: str
    source_checkpoint_hash: Optional[str] = None
    source_run_id: Optional[str] = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in INIT_MODES:
            raise IdentityError(f"init mode must be one of {INIT_MODES}, got {self.mode!r}")
        object.__setattr__(self, "mode", mode)
        if self.source_run_id is not None:
            object.__setattr__(
                self, "source_run_id", _validate_uuid(self.source_run_id, "InitId.source_run_id")
            )

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.mode

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source_checkpoint_hash": self.source_checkpoint_hash,
            "source_run_id": self.source_run_id,
        }


@dataclass(frozen=True, order=True)
class VariantId:
    """Content hash identifying a comparable implementation variant.

    Two runs share a ``VariantId`` iff they are the same algorithm, the same
    adapter implementation, the same architecture, the same initialization
    provenance, and the same structural configuration. Seeds, devices, and
    telemetry settings deliberately do **not** participate: those runs are
    genuinely comparable with each other.
    """

    value: str

    def __post_init__(self) -> None:
        text = str(self.value).strip()
        if not text.startswith("sha256:") or len(text) != len("sha256:") + 64:
            raise IdentityError(f"VariantId must be 'sha256:<64 hex>', got {self.value!r}")
        object.__setattr__(self, "value", text)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @property
    def short(self) -> str:
        """12-character display form. Presentation only, never a key."""
        return short_hash(self.value)


def compute_variant_id(
    *,
    model_id: ModelId,
    implementation_id: ImplementationId,
    init: InitId,
    architecture_fingerprint: Optional[str] = None,
    structural_config_hash: Optional[str] = None,
) -> VariantId:
    """Derive the canonical :class:`VariantId` (design doc 03 §2.3).

    The hashed document carries an explicit schema version so that adding a
    field later is a visible, versioned change rather than a silent
    re-identification of every historical run.
    """
    document = {
        "schema_version": VARIANT_IDENTITY_SCHEMA_VERSION,
        "model_id": model_id.value,
        "implementation_id": implementation_id.value,
        "architecture_fingerprint": architecture_fingerprint,
        "init": init.as_dict(),
        "structural_config_hash": structural_config_hash,
    }
    return VariantId(content_hash(document))


# --------------------------------------------------------------------------- #
# Presentation helpers — NOT identities
# --------------------------------------------------------------------------- #


def variant_label(
    *,
    model_id: ModelId,
    implementation_id: ImplementationId,
    init: InitId,
    display_name: Optional[str] = None,
) -> str:
    """Human-readable label for a variant.

    Purely for display. Never persist it, never use it as a dict/DB key, never
    parse it back. The canonical key is :class:`VariantId` (design doc 05,
    DND-005). This mirrors the existing Streamlit helper
    ``viz.app.components.model_toggle_picker.variant_label`` in spirit, but the
    control plane keeps label and identity strictly separate.
    """
    name = display_name or model_id.value
    return f"{name} · {implementation_id.value} · {init.mode}"


def describe_identity(
    *,
    model_id: ModelId,
    implementation_id: ImplementationId,
    init: InitId,
    variant_id: VariantId,
    display_name: Optional[str] = None,
) -> dict[str, Any]:
    """Flat identity summary for API/UI consumption.

    Includes both the canonical ids and the display label so that the UI never
    has to reconstruct either one.
    """
    return {
        "model_id": model_id.value,
        "implementation_id": implementation_id.value,
        "init_id": init.mode,
        "init_source_checkpoint_hash": init.source_checkpoint_hash,
        "init_source_run_id": init.source_run_id,
        "variant_id": variant_id.value,
        "variant_id_short": variant_id.short,
        "variant_label": variant_label(
            model_id=model_id,
            implementation_id=implementation_id,
            init=init,
            display_name=display_name,
        ),
    }


def init_from_mapping(value: Mapping[str, Any]) -> InitId:
    """Build an :class:`InitId` from a plain mapping (config/DB round-trip)."""
    return InitId(
        mode=str(value.get("mode", "unknown")),
        source_checkpoint_hash=value.get("source_checkpoint_hash"),
        source_run_id=value.get("source_run_id"),
    )
