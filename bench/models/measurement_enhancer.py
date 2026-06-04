from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CausalConv1d(nn.Module):
    """1D convolution with left padding only, preserving causal time length."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        self.left_padding = int((kernel_size - 1) * dilation)
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            bias=bool(bias),
        )

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        if x_bct.ndim != 3:
            raise ValueError(f"expected [B,C,T], got {tuple(x_bct.shape)}")
        if self.left_padding > 0:
            x_bct = F.pad(x_bct, (self.left_padding, 0))
        return self.conv(x_bct)


class MeasurementEnhancer(nn.Module):
    """
    Small causal residual measurement enhancer.

    Input and output use the benchmark boundary layout [B,T,Dy]. The final
    layer is zero-initialized so y_enh = y_raw at initialization when used as a
    residual correction.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim if output_dim is not None else input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)

        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        layers = []
        in_ch = self.input_dim
        for _ in range(self.num_layers):
            layers.append(CausalConv1d(in_ch, self.hidden_dim, kernel_size=self.kernel_size))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_ch = self.hidden_dim
        self.net = nn.Sequential(*layers)
        self.out = CausalConv1d(self.hidden_dim, self.output_dim, kernel_size=1)
        nn.init.zeros_(self.out.conv.weight)
        nn.init.zeros_(self.out.conv.bias)

    def forward(self, y_raw_btd: torch.Tensor) -> torch.Tensor:
        if y_raw_btd.ndim != 3:
            raise ValueError(f"expected y_raw [B,T,D], got {tuple(y_raw_btd.shape)}")
        if int(y_raw_btd.shape[2]) != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {y_raw_btd.shape[2]}")
        z_bct = y_raw_btd.transpose(1, 2).contiguous()
        z_bct = self.net(z_bct)
        delta_bct = self.out(z_bct)
        return delta_bct.transpose(1, 2).contiguous()

    def enhance(self, y_raw_btd: torch.Tensor, *, delta_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        delta = self.forward(y_raw_btd)
        return y_raw_btd + float(delta_scale) * delta, delta

    def config_dict(self) -> Dict[str, Any]:
        return {
            "type": "causal_tcn",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "output_dim": self.output_dim,
            "residual_output": True,
            "final_layer_zero_init": True,
        }


def enhancement_regularization(delta_btd: torch.Tensor) -> Dict[str, torch.Tensor]:
    if delta_btd.ndim != 3:
        raise ValueError(f"expected delta [B,T,D], got {tuple(delta_btd.shape)}")
    l_delta = torch.mean(delta_btd.pow(2))
    if delta_btd.shape[1] > 1:
        l_smooth = torch.mean((delta_btd[:, 1:, :] - delta_btd[:, :-1, :]).pow(2))
    else:
        l_smooth = delta_btd.new_zeros(())
    return {"L_delta": l_delta, "L_smooth": l_smooth}


def enhancer_diagnostics(
    *,
    y_raw_btd: torch.Tensor,
    y_enh_btd: torch.Tensor,
    delta_btd: torch.Tensor,
    raw_delta_btd: Optional[torch.Tensor] = None,
    x_ref_btd: Optional[torch.Tensor] = None,
    y_clean_imu_btd: Optional[torch.Tensor] = None,
    imu_error_btd: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    with torch.no_grad():
        eps = torch.finfo(y_raw_btd.dtype).eps
        delta_norm_t = torch.linalg.norm(delta_btd, dim=2).mean(dim=0).detach().cpu().numpy().astype(np.float32)
        delta_norm_all = torch.linalg.norm(delta_btd, dim=2)
        raw_delta_norm = torch.linalg.norm(raw_delta_btd, dim=2) if raw_delta_btd is not None else delta_norm_all
        y_raw_norm = torch.linalg.norm(y_raw_btd, dim=2)
        y_enh_norm = torch.linalg.norm(y_enh_btd, dim=2)
        delta_to_raw_ratio = delta_norm_all / torch.clamp(y_raw_norm, min=float(eps))
        y_enh_to_raw_ratio = y_enh_norm / torch.clamp(y_raw_norm, min=float(eps))
        diag: Dict[str, Any] = {
            "delta_norm_t": delta_norm_t,
            "delta_norm_mean": float(delta_norm_all.mean().detach().cpu().item()),
            "delta_norm_max": float(delta_norm_all.max().detach().cpu().item()),
            "raw_delta_norm_mean": float(raw_delta_norm.mean().detach().cpu().item()),
            "raw_delta_norm_max": float(raw_delta_norm.max().detach().cpu().item()),
            "y_raw_norm_mean": float(y_raw_norm.mean().detach().cpu().item()),
            "y_enh_norm_mean": float(y_enh_norm.mean().detach().cpu().item()),
            "delta_to_raw_ratio_mean": float(delta_to_raw_ratio.mean().detach().cpu().item()),
            "delta_to_raw_ratio_max": float(delta_to_raw_ratio.max().detach().cpu().item()),
            "y_enh_to_raw_norm_ratio_mean": float(y_enh_to_raw_ratio.mean().detach().cpu().item()),
        }
        innovation_ref = y_clean_imu_btd if y_clean_imu_btd is not None else x_ref_btd
        if innovation_ref is not None and tuple(innovation_ref.shape) == tuple(y_raw_btd.shape):
            innovation_raw = torch.linalg.norm(y_raw_btd - innovation_ref, dim=2)
            innovation_enh = torch.linalg.norm(y_enh_btd - innovation_ref, dim=2)
            innovation_ratio = innovation_enh / torch.clamp(innovation_raw, min=float(eps))
            diag.update(
                {
                    "innovation_raw_norm_mean": float(innovation_raw.mean().detach().cpu().item()),
                    "innovation_enh_norm_mean": float(innovation_enh.mean().detach().cpu().item()),
                    "innovation_collapse_ratio": float(innovation_ratio.mean().detach().cpu().item()),
                }
            )
        if y_clean_imu_btd is not None:
            raw_to_clean = torch.mean((y_raw_btd - y_clean_imu_btd).pow(2))
            enh_to_clean = torch.mean((y_enh_btd - y_clean_imu_btd).pow(2))
            diag.update(
                {
                    "imu_y_raw_to_clean_mse": float(raw_to_clean.detach().cpu().item()),
                    "imu_y_enh_to_clean_mse": float(enh_to_clean.detach().cpu().item()),
                    "imu_mse_reduction": float((raw_to_clean - enh_to_clean).detach().cpu().item()),
                }
            )
        if imu_error_btd is not None:
            target = -imu_error_btd
            target_norm = torch.linalg.norm(target, dim=2)
            delta_to_error_ratio = delta_norm_all / torch.clamp(target_norm, min=float(eps))
            flat_delta = delta_btd.reshape(-1)
            flat_target = target.reshape(-1)
            denom = torch.clamp(torch.linalg.norm(flat_delta) * torch.linalg.norm(flat_target), min=float(eps))
            alignment = torch.dot(flat_delta, flat_target) / denom
            diag.update(
                {
                    "delta_to_imu_error_ratio_mean": float(delta_to_error_ratio.mean().detach().cpu().item()),
                    "delta_to_imu_error_ratio_max": float(delta_to_error_ratio.max().detach().cpu().item()),
                    "imu_correction_alignment": float(alignment.detach().cpu().item()),
                }
            )
        return diag
