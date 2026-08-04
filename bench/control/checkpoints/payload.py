"""Checkpoint payload serialisation and RNG capture.

Payloads are ``torch.save``/``torch.load`` packages, which means pickle, which
means they are only ever loaded from a trusted local control root after their
digest has been verified (ADR-CSR-009). Nothing here accepts a remote or
user-uploaded package.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .schema import (
    CHECKPOINT_SCHEMA_VERSION,
    SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS,
    AdapterTrainingState,
    CheckpointValidationError,
    RngState,
    TrainingCursor,
)

#: Keys of the top-level payload mapping. Named, so a reader never has to infer
#: structure positionally (§8).
PAYLOAD_KEYS = (
    "schema_version",
    "cursor",
    "model_slots",
    "optimizer_slots",
    "scheduler_slots",
    "grad_scaler",
    "best_state",
    "validation_state",
    "extra_state",
    "rng",
    "batch_plan",
    "resolved_run_spec",
)


def capture_rng(*, include_cuda: bool = False) -> RngState:
    """Snapshot every RNG stream that can move the numerical path."""
    cuda_states = None
    if include_cuda and torch.cuda.is_available():  # pragma: no cover - CPU certification
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return RngState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu_state=torch.get_rng_state().clone(),
        torch_cuda_states=cuda_states,
    )


def restore_rng(state: RngState) -> None:
    """Restore RNG streams captured by :func:`capture_rng`."""
    if state.python_state is not None:
        random.setstate(_as_python_state(state.python_state))
    if state.numpy_state is not None:
        np.random.set_state(_as_numpy_state(state.numpy_state))
    if state.torch_cpu_state is not None:
        torch.set_rng_state(_as_byte_tensor(state.torch_cpu_state))
    if state.torch_cuda_states and torch.cuda.is_available():  # pragma: no cover
        torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state.torch_cuda_states])


def _as_python_state(value: Any) -> tuple:
    # torch.save round-trips tuples as lists in some versions.
    if isinstance(value, list):
        return (value[0], tuple(value[1]), value[2])
    return value


def _as_numpy_state(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _as_byte_tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.to(dtype=torch.uint8, device="cpu")


def build_payload(
    *,
    cursor: TrainingCursor,
    adapter_state: AdapterTrainingState,
    rng: RngState,
    batch_plan: Optional[dict[str, Any]] = None,
    resolved_run_spec: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the payload mapping written to disk."""
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "cursor": cursor.as_dict(),
        "model_slots": adapter_state.model_slots,
        "optimizer_slots": adapter_state.optimizer_slots,
        "scheduler_slots": adapter_state.scheduler_slots,
        "grad_scaler": adapter_state.grad_scaler,
        "best_state": adapter_state.best_state,
        "validation_state": adapter_state.validation_state,
        "extra_state": adapter_state.extra_state,
        "rng": {
            "python_state": rng.python_state,
            "numpy_state": rng.numpy_state,
            "torch_cpu_state": rng.torch_cpu_state,
            "torch_cuda_states": rng.torch_cuda_states,
        },
        "batch_plan": batch_plan or {},
        "resolved_run_spec": resolved_run_spec or {},
    }


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    """Serialise a payload. Called through the atomic publisher, never directly."""
    torch.save(payload, path)


def read_payload(path: Path) -> dict[str, Any]:
    """Load a payload from a trusted, digest-verified local path.

    ``weights_only`` is deliberately **not** used: the payload intentionally
    carries optimizer state, RNG tuples, and numpy state objects that
    ``weights_only`` refuses. That is exactly why loading is gated on digest
    verification and a control-root path check upstream (ADR-CSR-009).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointValidationError(
            f"checkpoint payload at {path} is {type(payload).__name__}, expected a mapping"
        )
    version = int(payload.get("schema_version", -1))
    if version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
        raise CheckpointValidationError(
            f"checkpoint payload schema_version={version} is not readable by this build "
            f"(supported {list(SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS)}); silent migration "
            "is not performed"
        )
    return payload


def payload_to_state(payload: dict[str, Any]) -> tuple[TrainingCursor, AdapterTrainingState, RngState]:
    """Split a loaded payload into its typed parts."""
    cursor = TrainingCursor.from_dict(payload.get("cursor", {}))
    state = AdapterTrainingState(
        model_slots=dict(payload.get("model_slots") or {}),
        optimizer_slots=dict(payload.get("optimizer_slots") or {}),
        scheduler_slots=dict(payload.get("scheduler_slots") or {}),
        grad_scaler=payload.get("grad_scaler"),
        best_state=dict(payload.get("best_state") or {}),
        validation_state=dict(payload.get("validation_state") or {}),
        extra_state=dict(payload.get("extra_state") or {}),
    )
    raw_rng = dict(payload.get("rng") or {})
    rng = RngState(
        python_state=raw_rng.get("python_state"),
        numpy_state=raw_rng.get("numpy_state"),
        torch_cpu_state=raw_rng.get("torch_cpu_state"),
        torch_cuda_states=raw_rng.get("torch_cuda_states"),
    )
    return cursor, state, rng


def component_inventory(adapter_state: AdapterTrainingState) -> dict[str, list[str]]:
    """Named inventory of what the payload contains, for the manifest."""
    return {
        "model_slots": sorted(adapter_state.model_slots),
        "optimizer_slots": sorted(adapter_state.optimizer_slots),
        "scheduler_slots": sorted(adapter_state.scheduler_slots),
        "best_state": sorted(adapter_state.best_state),
        "validation_state": sorted(adapter_state.validation_state),
        "extra_state": sorted(adapter_state.extra_state),
    }
