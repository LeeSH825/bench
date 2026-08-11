"""Isolated causal gyro--magnetometer compensation pilot.

The deployable namespace is intentionally limited to :mod:`model` and the
deployable replay entry points in :mod:`study`.  Generator truth and diagnostic
oracle records live only in :mod:`data` and the explicit oracle replay
namespace.
"""

from .model import GyroEncoder, MagEncoder, SideEstimator, SplitGainBackbone

__all__ = ["GyroEncoder", "MagEncoder", "SideEstimator", "SplitGainBackbone"]
