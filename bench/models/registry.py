"""
Model registry for bench adapters (Step 5)

Maps suite model_id -> Adapter class.
"""

from __future__ import annotations

from typing import Dict, Type

import torch

from .base import ModelAdapter  # type: ignore
from .kalmannet_tsp import KalmanNetTSPAdapter
from .adaptive_knet import AdaptiveKNetAdapter
from .maml_knet import MAMLKNetAdapter
from .split_knet import SplitKNetAdapter
from .me_split_knet import MESplitKNetV0Adapter
from .mb_kf import ModelBasedKFAdapter
from .basilisk_mrp_ekf import BasiliskMRPEKFAdapter


_REGISTRY: Dict[str, Type[ModelAdapter]] = {
    "kalmannet_tsp": KalmanNetTSPAdapter,
    "adaptive_knet": AdaptiveKNetAdapter,
    "maml_knet": MAMLKNetAdapter,
    "split_knet": SplitKNetAdapter,
    "me_split_knet_v0": MESplitKNetV0Adapter,
    "me_split_knet_v0_ds100": MESplitKNetV0Adapter,
    "me_split_knet_v0_ds025": MESplitKNetV0Adapter,
    "me_split_knet_v0_ds010": MESplitKNetV0Adapter,
    "me_split_knet_v0_small": MESplitKNetV0Adapter,
    "me_split_knet_v0_regstrong": MESplitKNetV0Adapter,
    "me_split_knet_v0_clip025": MESplitKNetV0Adapter,
    "oracle_kf": ModelBasedKFAdapter,
    "nominal_kf": ModelBasedKFAdapter,
    "oracle_shift_kf": ModelBasedKFAdapter,
    # Route-B closeout aliases for model-based KF baselines.
    "mb_kf_oracle": ModelBasedKFAdapter,
    "mb_kf_nominal": ModelBasedKFAdapter,
    "basilisk_mrp_ekf": BasiliskMRPEKFAdapter,
    # future:
    # "my_model": MyModelAdapter,
}


def get_adapter_class(model_id: str) -> Type[ModelAdapter]:
    if model_id not in _REGISTRY:
        raise KeyError(f"Unknown model_id={model_id}. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[model_id]


def get_model_adapter_class(model_id: str) -> Type[ModelAdapter]:
    # Compatibility alias used by run_suite._load_adapter().
    return get_adapter_class(model_id)


def list_model_ids():
    return sorted(_REGISTRY.keys())
