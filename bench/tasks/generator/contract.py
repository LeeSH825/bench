from __future__ import annotations

"""
Generator Contract v1 (TG0)

All bench task-family generators must expose a deterministic function with:
    generate(task_cfg, split_cfg, seed, rng, device=None) -> GeneratorOutput

Inputs:
- task_cfg: resolved task configuration (YAML task + scenario overrides)
- split_cfg: split sizes (train/val/test)
- seed: integer seed used for deterministic generation
- rng: numpy Generator seeded from bench deterministic policy
- device: optional execution hint (generators should remain CPU-safe)

Outputs:
- x: float32 [N, T, x_dim] in canonical NTD layout
- y: float32 [N, T, y_dim] in canonical NTD layout
- meta: dict (append-only; upgraded by enforce_meta_v1)
- extras: optional aux arrays/values (e.g., q2_t/r2_t/SoW_t/task_key)

Determinism rule:
- same (task_cfg, split_cfg, seed) on CPU must produce identical first-k hashes
  for x/y and stable required meta keys.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, TypedDict

import numpy as np


CANONICAL_LAYOUT_V1 = "NTD"
GENERATOR_SCHEMA_VERSION_V1 = 1
GENERATOR_SCHEMA_TAG_V1 = "generator_contract_v1"
DEFAULT_TASK_FAMILY_V1 = "linear_gaussian_v0"


class DimsV1(TypedDict):
    x_dim: int
    y_dim: int
    T: int


class SsmBlockV1(TypedDict):
    true: Dict[str, Any]
    assumed: Dict[str, Any]


class MismatchBlockV1(TypedDict):
    enabled: bool
    kind: str
    params: Dict[str, Any]


class NoiseScheduleBlockV1(TypedDict):
    enabled: bool
    kind: str
    q2_t: Any
    r2_t: Any
    SoW_t: Any
    SoW_hat_t: Any


class SwitchingBlockV1(TypedDict):
    enabled: bool
    models: Any
    t_change: Any
    retrain_window: int


class MetaSchemaV1(TypedDict):
    schema_version: int
    task_family: str
    dims: DimsV1
    splits: Dict[str, Any]
    ssm: SsmBlockV1
    mismatch: MismatchBlockV1
    noise_schedule: NoiseScheduleBlockV1
    switching: SwitchingBlockV1


@dataclass(frozen=True)
class SplitCfg:
    n_train: int
    n_val: int
    n_test: int

    @property
    def n_total(self) -> int:
        return int(self.n_train + self.n_val + self.n_test)

    def as_dict(self) -> Dict[str, int]:
        return {
            "train": int(self.n_train),
            "val": int(self.n_val),
            "test": int(self.n_test),
        }


@dataclass(frozen=True)
class TaskCfg:
    task_id: str
    task_family: str
    system_type: str
    x_dim: int
    y_dim: int
    sequence_length_T: int
    observation: Dict[str, Any]
    noise: Dict[str, Any]
    control_input_u: bool
    ground_truth: Dict[str, Any]
    scenario_cfg: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratorOutput:
    x: np.ndarray
    y: np.ndarray
    meta: Dict[str, Any]
    extras: Dict[str, Any] = field(default_factory=dict)


class GeneratorFn(Protocol):
    def __call__(
        self,
        task_cfg: TaskCfg,
        split_cfg: SplitCfg,
        seed: int,
        rng: np.random.Generator,
        device: Optional[str] = None,
    ) -> GeneratorOutput:
        ...


def resolve_task_family(task_cfg: Mapping[str, Any]) -> str:
    raw = task_cfg.get("task_family")
    if raw is None:
        return DEFAULT_TASK_FAMILY_V1
    s = str(raw).strip()
    return s if s else DEFAULT_TASK_FAMILY_V1


def make_split_cfg(task_cfg: Mapping[str, Any]) -> SplitCfg:
    sizes = task_cfg.get("dataset_sizes") or {}
    return SplitCfg(
        n_train=int(sizes.get("N_train", 0)),
        n_val=int(sizes.get("N_val", 0)),
        n_test=int(sizes.get("N_test", 0)),
    )


def make_task_cfg(task_cfg: Mapping[str, Any], scenario_cfg: Optional[Mapping[str, Any]] = None) -> TaskCfg:
    return TaskCfg(
        task_id=str(task_cfg.get("task_id")),
        task_family=resolve_task_family(task_cfg),
        system_type=str(task_cfg.get("system_type", "unknown")),
        x_dim=int(task_cfg.get("x_dim", 0)),
        y_dim=int(task_cfg.get("y_dim", 0)),
        sequence_length_T=int(task_cfg.get("sequence_length_T", 0)),
        observation=dict(task_cfg.get("observation", {}) or {}),
        noise=dict(task_cfg.get("noise", {}) or {}),
        control_input_u=bool(task_cfg.get("control_input_u", False)),
        ground_truth=dict(task_cfg.get("ground_truth", {}) or {}),
        scenario_cfg=dict(scenario_cfg or {}),
        raw=dict(task_cfg),
    )


def coerce_ntd_float32_output(out: GeneratorOutput) -> GeneratorOutput:
    x = np.asarray(out.x, dtype=np.float32)
    y = np.asarray(out.y, dtype=np.float32)
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"generator output must be rank-3 NTD arrays, got x={x.shape}, y={y.shape}")
    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
        raise ValueError(f"generator output N/T mismatch, got x={x.shape}, y={y.shape}")
    meta = dict(out.meta or {})
    extras = dict(out.extras or {})
    return GeneratorOutput(x=x, y=y, meta=meta, extras=extras)
