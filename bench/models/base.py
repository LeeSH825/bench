from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

# NOTE (LOCKED RULES)
# - NLL is computed only if the model outputs covariance/variance.
#   If covariance is not supported, NLL must be recorded as "NA".
#   (SoT: /mnt/data/METRICS.md, /mnt/data/suite_*.yaml nll_policy)


@runtime_checkable
class SupportsToDevice(Protocol):
    def to(self, device: str) -> Any:  # torch-like
        ...


@dataclass(frozen=True)
class Prediction:
    """
    Prediction container (shape conventions are task-specific).
    For common KalmanNet-family code, typical shapes are:
      - y_seq: [B, y_dim, T]
      - x_hat: [B, x_dim, T]
    but the adapter MUST document its own expectations.

    cov:
      - Optional covariance/variance output.
      - If None, metrics layer must set NLL="NA" (locked rule).
    """
    x_hat: Any
    cov: Optional[Any] = None


class ModelAdapter(ABC):
    """
    Bench-facing adapter interface.

    Step 2 policy:
    - Only define the interface and minimal validation hooks.
    - Do NOT implement model-specific training/eval logic here.
    - third_party repos remain unmodified; adapters wrap them.

    Required methods are designed to support:
      - Frozen inference (no updates during test)
      - Budgeted adaptation (limited updates during test)
    (SoT: /mnt/data/FAIRNESS.md, /mnt/data/suite_shift.yaml runner.tracks)
    """

    @abstractmethod
    def setup(self, cfg: dict, system_info: Any) -> None:
        """
        Prepare the model according to model cfg and task-provided system_info.

        cfg: model-level config dict (from suite YAML `models[*]` + overrides)
        system_info: task-provided info (e.g., F/H or f/h, Q/R availability, dims)
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, train_loader: Any, val_loader: Any) -> None:
        """
        Train the model if applicable.

        Step 2: No training loop implementation is provided by the bench core.
        Adapters will implement this in later steps (Step 4~).
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, ckpt_path: str) -> None:
        """Load model weights from a checkpoint path."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[dict] = None,
        return_cov: bool = False,
    ) -> Prediction:
        """
        Predict x_hat given y_seq (and optional control u_seq).

        return_cov:
          - If True and the model supports covariance, return cov in Prediction.
          - If model doesn't support covariance, return cov=None.
        """
        raise NotImplementedError

    @abstractmethod
    def adapt(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[dict] = None,
        budget: Optional[Any] = None,
    ) -> None:
        """
        Online adaptation hook (Budgeted adaptation track).

        budget:
          - The bench runner passes an adaptation budget object/dict.
          - Adapter must enforce or respect it (Step 4~ actual enforcement).
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, out_dir: str) -> None:
        """Save adapter/model state to out_dir (checkpoint + metadata)."""
        raise NotImplementedError

