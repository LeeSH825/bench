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


_REGISTRY: Dict[str, Type[ModelAdapter]] = {
    "kalmannet_tsp": KalmanNetTSPAdapter,
    "adaptive_knet": AdaptiveKNetAdapter,
    # future:
    # "maml_knet": MAMLKNetAdapter,
    # "split_knet": SplitKNetAdapter,
    # "my_model": MyModelAdapter,
}


def get_adapter_class(model_id: str) -> Type[ModelAdapter]:
    if model_id not in _REGISTRY:
        raise KeyError(f"Unknown model_id={model_id}. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[model_id]


def list_model_ids():
    return sorted(_REGISTRY.keys())

