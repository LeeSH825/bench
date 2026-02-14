from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from .data_format import DatasetArtifacts, SystemInfo

# === Step4 note: bench_generated canonical layout ===
# Data pipeline v0 stores x,y as [N,T,D] (CANONICAL_LAYOUT_V0="NTD").
# Repo-specific layouts (e.g., [B,D,T]) must be handled in adapters. (See DECISIONS D15)


class Task(ABC):
    """
    Bench task interface.

    Step 2 scope:
    - Define how a task exposes data artifacts and system model info.
    - Actual data generation is deferred to Step 4~.

    Key concepts:
    - task_id comes from suite YAML (e.g., A_linear_canonical_v0).
    - scenario_id is a runner-generated identifier for a sweep combination.
    """

    @property
    @abstractmethod
    def task_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_system_info(self, scenario_cfg: dict) -> SystemInfo:
        """
        Return SystemInfo describing what the model may use (or what is available).
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_artifacts(
        self,
        root_dir: str,
        seed: int,
        scenario_id: str,
        scenario_cfg: dict,
    ) -> DatasetArtifacts:
        """
        Produce (or locate) dataset artifacts for this task/scenario/seed.

        Step 2: stub only. Step 4~ will implement bench_generated generation.
        """
        raise NotImplementedError

    @abstractmethod
    def make_dataloaders(self, artifacts: DatasetArtifacts, cfg: dict) -> dict[str, Any]:
        """
        Return a dict of loaders, e.g. {"train": ..., "val": ..., "test": ...}.

        Step 2: signature only. Step 4~ will implement.
        """
        raise NotImplementedError

    def has_ground_truth(self) -> bool:
        """
        Default behavior: consult artifacts/meta in Step 4~.

        suite YAML includes `ground_truth.has_gt`.
        """
        return True

