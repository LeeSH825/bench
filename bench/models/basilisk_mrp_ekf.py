from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from bench.utils.diagnostics import format_array_stats, has_nonfinite, validate_exact_layout
from bench.utils.logging import get_logger

try:
    from .base import ModelAdapter  # type: ignore
except Exception:  # pragma: no cover
    class ModelAdapter:  # pragma: no cover
        pass


logger = get_logger(__name__)


def _as_torch_device(device: Union[str, torch.device, None]) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _to_tensor(x: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _seed_everything(seed: int, deterministic: bool = True) -> None:
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    torch.cuda.manual_seed_all(seed_i)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def _extract_batch_xy(batch: Any) -> Tuple[Any, Any]:
    if isinstance(batch, dict):
        if "x" not in batch or "y" not in batch:
            raise KeyError("Batch dict must contain keys 'x' and 'y'.")
        return batch["x"], batch["y"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(f"Unsupported dataloader batch type: {type(batch)}")


def _lookup_nested(d: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _coerce_meta_dict(system_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not system_info:
        return {}

    meta = system_info.get("meta")
    if isinstance(meta, dict):
        return dict(meta)

    meta_json = system_info.get("meta_json")
    if isinstance(meta_json, dict):
        return dict(meta_json)
    if isinstance(meta_json, str):
        try:
            decoded = json.loads(meta_json)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            return {}
    return {}


def _extract_scalar(
    cfg: Dict[str, Any],
    system_info: Dict[str, Any],
    meta: Dict[str, Any],
    *,
    name: str,
    candidates: List[Tuple[str, ...]],
    default: float,
) -> float:
    values: List[Any] = [system_info.get(name), cfg.get(name)]
    values.extend(_lookup_nested(meta, c) for c in candidates)
    values.append(default)
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return float(default)


def _as_inertia_matrix(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (3,):
        if np.any(arr <= 0.0):
            raise ValueError(f"inertia diagonal entries must be positive, got {arr}")
        return np.diag(arr)
    if arr.shape == (3, 3):
        if np.linalg.det(arr) == 0.0:
            raise ValueError("inertia matrix must be nonsingular")
        return arr
    raise ValueError(f"inertia must be length-3 diagonal or 3x3 matrix, got shape {arr.shape}")


def _as_vec3(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be length 3, got shape {arr.shape}")
    return arr


def _diag_from_scalar_or_matrix(
    value: Any,
    *,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    scalar_default: float,
) -> torch.Tensor:
    if value is None:
        return torch.eye(dim, device=device, dtype=dtype) * float(scalar_default)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == ():
        return torch.eye(dim, device=device, dtype=dtype) * float(arr)
    if arr.shape == (dim,):
        return torch.diag(_to_tensor(arr, device=device, dtype=dtype))
    if arr.shape == (dim, dim):
        return _to_tensor(arr, device=device, dtype=dtype)
    raise ValueError(f"expected scalar, diag length {dim}, or {dim}x{dim} matrix, got shape={arr.shape}")


def shadow_mrp(sigma: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """Map MRPs to the shadow set when norm(sigma) > 1."""
    norm2 = torch.sum(sigma * sigma, dim=-1, keepdim=True)
    shadow = -sigma / torch.clamp(norm2, min=float(eps))
    return torch.where(norm2 > 1.0, shadow, sigma)


def _skew_batch(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    x, y, zz = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack([z, -zz, y], dim=-1)
    row1 = torch.stack([zz, z, -x], dim=-1)
    row2 = torch.stack([-y, x, z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def mrp_B_matrix(sigma: torch.Tensor) -> torch.Tensor:
    """Return B(sigma) for MRP kinematics, batched over leading dims."""
    eye = torch.eye(3, device=sigma.device, dtype=sigma.dtype)
    batch_shape = sigma.shape[:-1]
    eye_b = eye.reshape((1,) * len(batch_shape) + (3, 3)).expand(batch_shape + (3, 3))
    s2 = torch.sum(sigma * sigma, dim=-1, keepdim=True).unsqueeze(-1)
    outer = sigma.unsqueeze(-1) * sigma.unsqueeze(-2)
    return (1.0 - s2) * eye_b + 2.0 * _skew_batch(sigma) + 2.0 * outer


class BasiliskMRPEKFAdapter(ModelAdapter):
    """
    Bench-native nonlinear EKF for Basilisk ADCS state x=[MRP sigma(3), omega(3)].

    This baseline is model-based and diagnostic by default. It uses the same
    rigid-body attitude equations as the Basilisk smoke task metadata, but it
    should not be called an oracle until the benchmark policy explicitly
    validates the assumed model/noise against the generator.
    """

    def __init__(self) -> None:
        self.device: torch.device = torch.device("cpu")
        self.dtype: torch.dtype = torch.float32
        self._cfg: Dict[str, Any] = {}
        self._run_ctx: Dict[str, Any] = {}
        self._meta: Dict[str, Any] = {}

        self._x_dim = 6
        self._y_dim = 6
        self._dt = 0.1
        self._fd_eps = 1.0e-5
        self._shadow_eps = 1.0e-12
        self._shadow_jacobian_guard = 1.0e-4
        self._jitter = 1.0e-9
        self._integration = "rk4"
        self._outputs_covariance = False

        self._inertia: Optional[torch.Tensor] = None
        self._inertia_inv: Optional[torch.Tensor] = None
        self._disturbance_torque: Optional[torch.Tensor] = None
        self._Q: Optional[torch.Tensor] = None
        self._R: Optional[torch.Tensor] = None
        self._P0: Optional[torch.Tensor] = None

        self._run_dir: Optional[Path] = None
        self._ckpt_dir: Optional[Path] = None
        self._artifacts_dir: Optional[Path] = None
        self._ledger_path: Optional[Path] = None
        self._saved_ckpt_path: Optional[Path] = None

        self.train_updates_used = 0
        self.train_outer_updates_used = 0
        self.train_inner_updates_used = 0
        self.adapt_updates_used = 0
        self.adapt_updates_per_step: Dict[int, int] = {}

        self.last_layout: Optional[str] = None
        self.last_class: Optional[str] = None
        self._debug_every = 0
        self._runtime_diag: Dict[str, Any] = {}
        self._emit_viz_diagnostics = False

    def setup(
        self,
        cfg: Dict[str, Any],
        system_info: Optional[Dict[str, Any]] = None,
        run_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        system_info = system_info or {}
        run_ctx = run_ctx or {}
        self._cfg = dict(cfg)
        self._run_ctx = dict(run_ctx)
        self._meta = _coerce_meta_dict(system_info)
        self.train_updates_used = 0
        self.train_outer_updates_used = 0
        self.train_inner_updates_used = 0
        self.adapt_updates_used = 0
        self.adapt_updates_per_step = {}
        self._runtime_diag = {}
        self._emit_viz_diagnostics = bool(run_ctx.get("emit_viz_artifacts", False))
        self._debug_every = int(run_ctx.get("debug_every", cfg.get("debug_every", system_info.get("debug_every", 0))) or 0)

        requested_device = cfg.get("device") or run_ctx.get("device") or system_info.get("device") or "cpu"
        self.device = _as_torch_device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            self.device = torch.device("cpu")

        seed = run_ctx.get("seed", cfg.get("seed", system_info.get("seed", 0)))
        deterministic = bool(run_ctx.get("deterministic", cfg.get("deterministic", True)))
        _seed_everything(int(seed), deterministic=deterministic)

        x_dim = int(system_info.get("x_dim", cfg.get("x_dim", self._meta.get("x_dim", 6))))
        y_dim = int(system_info.get("y_dim", cfg.get("y_dim", self._meta.get("y_dim", 6))))
        if x_dim != 6 or y_dim != 6:
            raise ValueError(f"basilisk_mrp_ekf requires x_dim=y_dim=6, got x_dim={x_dim}, y_dim={y_dim}")
        self._x_dim = x_dim
        self._y_dim = y_dim

        true_meta = _lookup_nested(self._meta, ("ssm", "true")) or {}
        noise_meta = self._meta.get("noise", {}) if isinstance(self._meta.get("noise"), dict) else {}
        self._dt = float(cfg.get("dt", true_meta.get("dt", self._meta.get("dt", 0.1))))
        self._fd_eps = float(cfg.get("fd_eps", 1.0e-5))
        self._shadow_jacobian_guard = float(cfg.get("shadow_jacobian_guard", 1.0e-4))
        self._jitter = float(cfg.get("covariance_jitter", 1.0e-9))
        self._integration = str(cfg.get("integration", "rk4")).strip().lower()
        if self._integration not in {"euler", "rk4"}:
            raise ValueError(f"unsupported integration={self._integration}; expected euler or rk4")
        self._outputs_covariance = bool(cfg.get("outputs_covariance", False))

        inertia = _as_inertia_matrix(cfg.get("inertia", true_meta.get("inertia", [10.0, 8.0, 6.0])))
        disturbance = _as_vec3(
            cfg.get("disturbance_torque", true_meta.get("disturbance_torque_B_Nm", [0.0, 0.0, 0.0])),
            name="disturbance_torque",
        )
        q2 = _extract_scalar(
            cfg,
            system_info,
            self._meta,
            name="q2",
            candidates=[("noise", "Q", "q2"), ("scenario_cfg", "noise", "Q", "q2")],
            default=1.0e-8,
        )
        r2 = _extract_scalar(
            cfg,
            system_info,
            self._meta,
            name="r2",
            candidates=[("noise", "R", "r2"), ("scenario_cfg", "noise", "R", "r2")],
            default=1.0e-4,
        )
        q_value = cfg.get("Q", _lookup_nested(self._meta, ("ssm", "assumed", "Q")))
        r_value = cfg.get("R", _lookup_nested(self._meta, ("ssm", "assumed", "R")))

        self._inertia = _to_tensor(inertia, device=self.device, dtype=self.dtype).reshape(3, 3)
        self._inertia_inv = torch.linalg.inv(self._inertia)
        self._disturbance_torque = _to_tensor(disturbance, device=self.device, dtype=self.dtype).reshape(3)
        self._Q = _diag_from_scalar_or_matrix(
            q_value,
            dim=6,
            device=self.device,
            dtype=self.dtype,
            scalar_default=float(q2),
        )
        self._R = _diag_from_scalar_or_matrix(
            r_value,
            dim=6,
            device=self.device,
            dtype=self.dtype,
            scalar_default=float(r2),
        )

        p0_value = cfg.get("P0", None)
        if p0_value is None and str(cfg.get("p0_source", "measurement_noise")).lower() == "measurement_noise":
            p0 = self._R.clone() * float(cfg.get("p0_scale", 1.0))
            p0 = p0 + torch.eye(6, device=self.device, dtype=self.dtype) * float(cfg.get("p0_floor", 1.0e-8))
        else:
            p0 = _diag_from_scalar_or_matrix(
                p0_value,
                dim=6,
                device=self.device,
                dtype=self.dtype,
                scalar_default=float(cfg.get("p0_scale", 1.0e-3)),
            )
        self._P0 = 0.5 * (p0 + p0.transpose(0, 1))

        self._run_dir = Path(str(run_ctx["run_dir"])).expanduser().resolve() if "run_dir" in run_ctx else None
        if self._run_dir is not None:
            self._ckpt_dir = self._run_dir / "checkpoints"
            self._artifacts_dir = self._run_dir / "artifacts"
            self._ledger_path = self._run_dir / "budget_ledger.json"
            self._ckpt_dir.mkdir(parents=True, exist_ok=True)
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            self._update_ledger(train_updates_used=0, adapt_updates_used=0, supports_budgeted=False)

        self.last_layout = "bench_BTD_to_batched_mrp_ekf_BTD"
        self.last_class = "bench.models.basilisk_mrp_ekf:BasiliskMRPEKFAdapter"
        logger.info(
            "setup basilisk_mrp_ekf device=%s dt=%s integration=%s fd_eps=%s q2=%s r2=%s layout=%s",
            self.device,
            self._dt,
            self._integration,
            self._fd_eps,
            q2,
            r2,
            self.last_layout,
        )

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        budget: Optional[Dict[str, Any]] = None,
        ckpt_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        budget = dict(budget or {})
        max_updates = int(budget.get("train_max_updates", 0))
        out_ckpt_dir = Path(ckpt_dir).expanduser().resolve() if ckpt_dir is not None else self._ckpt_dir
        if out_ckpt_dir is None:
            raise ValueError("ckpt_dir is required when adapter has no run_dir.")
        out_ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_res = self.save(out_ckpt_dir)
        train_state_path = out_ckpt_dir / "train_state.json"
        _write_json(
            train_state_path,
            {
                "status": "ok",
                "best_step": 0,
                "best_val_mse": None,
                "updates_used": 0,
                "max_updates": int(max_updates),
                "note": "basilisk_mrp_ekf is a model-based baseline; no training performed.",
            },
        )
        self._update_ledger(
            train_updates_used=0,
            adapt_updates_used=0,
            train_max_updates=max_updates,
            supports_budgeted=False,
        )
        return {
            "status": "ok",
            "ckpt_path": str(save_res["ckpt_path"]),
            "train_state_path": str(train_state_path),
            "updates_used": 0,
            "best_step": 0,
        }

    @torch.no_grad()
    def predict(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        return_cov: bool = False,
        **kwargs: Any,
    ) -> Any:
        y = _to_tensor(y_seq, device=self.device, dtype=self.dtype)
        if y.ndim != 3:
            raise ValueError(f"shape_mismatch: expected y [B,T,6], got {tuple(y.shape)}")
        bsz, t_len, y_dim = y.shape
        if int(y_dim) != 6:
            raise ValueError(f"shape_mismatch: expected y_dim=6, got {y_dim}")
        need_cov = bool((return_cov and self._outputs_covariance) or self._emit_viz_diagnostics)
        x_hat, cov = self._rollout_ekf(y, return_cov=need_cov)
        validate_exact_layout(x_hat, expected=(int(bsz), int(t_len), 6), axis_names=("B", "T", "D"), label="x_hat")
        if return_cov:
            return x_hat, cov
        return x_hat

    def eval(
        self,
        test_dl: Any,
        ckpt_path: Optional[Union[str, Path]] = None,
        track_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if ckpt_path is not None:
            self.load(str(ckpt_path))

        preds: List[torch.Tensor] = []
        covs: List[torch.Tensor] = []
        diag_accum: Dict[str, List[np.ndarray]] = {
            "innovation_norm_t": [],
            "kalman_gain_norm_t": [],
            "p_trace_t": [],
            "p_min_eig_t": [],
        }
        viz_diag_accum: Dict[str, List[torch.Tensor]] = {
            "innov": [],
            "S": [],
            "gain": [],
        }
        total_n = 0
        with torch.no_grad():
            for bi, batch in enumerate(test_dl):
                _x_raw, y_raw = _extract_batch_xy(batch)
                y = _to_tensor(y_raw, device=self.device, dtype=self.dtype)
                pred = self.predict(y, return_cov=bool(self._outputs_covariance or self._emit_viz_diagnostics))
                if isinstance(pred, tuple):
                    x_hat_b, cov_b = pred
                else:
                    x_hat_b, cov_b = pred, None
                validate_exact_layout(
                    x_hat_b,
                    expected=(int(y.shape[0]), int(y.shape[1]), 6),
                    axis_names=("B", "T", "D"),
                    label="x_hat",
                )
                total_n += int(y.shape[0])
                if self._debug_every > 0 and ((bi % self._debug_every) == 0 or has_nonfinite(x_hat_b)):
                    logger.debug("eval batch=%s %s", bi, format_array_stats("x_hat", x_hat_b))
                preds.append(x_hat_b.detach().cpu())
                if isinstance(cov_b, torch.Tensor):
                    covs.append(cov_b.detach().cpu())
                for key in diag_accum:
                    value = self._runtime_diag.get(key)
                    if value is not None:
                        diag_accum[key].append(np.asarray(value, dtype=np.float32))
                if self._emit_viz_diagnostics:
                    viz_diag = self._runtime_diag.get("viz_diagnostics")
                    if isinstance(viz_diag, dict):
                        for key in viz_diag_accum:
                            value = viz_diag.get(key)
                            if isinstance(value, torch.Tensor):
                                viz_diag_accum[key].append(value.detach().cpu())
                            elif value is not None:
                                viz_diag_accum[key].append(torch.as_tensor(value).detach().cpu())

        if not preds:
            raise RuntimeError("runtime_error: empty test dataloader.")
        x_hat = torch.cat(preds, dim=0).contiguous()
        validate_exact_layout(x_hat, expected=(int(total_n), int(x_hat.shape[1]), 6), axis_names=("N", "T", "D"), label="x_hat")

        cov_cat = torch.cat(covs, dim=0).contiguous() if covs else None
        viz_diag_cat: Dict[str, torch.Tensor] = {}
        if self._emit_viz_diagnostics:
            if cov_cat is not None:
                viz_diag_cat["P"] = cov_cat
            for key, values in viz_diag_accum.items():
                if values:
                    viz_diag_cat[key] = torch.cat(values, dim=0).contiguous()
        preds_path = None
        if self._artifacts_dir is not None and not self._emit_viz_diagnostics:
            preds_path = self._artifacts_dir / "preds_test.npz"
            if cov_cat is None:
                np.savez_compressed(preds_path, x_hat=x_hat.numpy())
            else:
                np.savez_compressed(preds_path, x_hat=x_hat.numpy(), cov=cov_cat.numpy())
        merged_diag: Dict[str, Any] = {}
        for key, values in diag_accum.items():
            if values:
                stacked = np.stack(values, axis=0)
                merged_diag[key] = np.mean(stacked, axis=0).astype(np.float32)
        if self._runtime_diag.get("covariance_psd_warnings") is not None:
            merged_diag["covariance_psd_warnings"] = np.asarray(
                [self._runtime_diag.get("covariance_psd_warnings", 0)], dtype=np.float32
            )
        self._runtime_diag = merged_diag

        self.adapt_updates_used = 0
        self.adapt_updates_per_step = {}
        self._update_ledger(train_updates_used=0, adapt_updates_used=0, supports_budgeted=False)
        result: Dict[str, Any] = {
            "status": "ok",
            "x_hat": x_hat,
            "cov": cov_cat,
            "preds_path": (str(preds_path) if preds_path is not None else None),
        }
        if viz_diag_cat:
            result["diagnostics"] = viz_diag_cat
        return result

    def adapt(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        budget: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.adapt_updates_used = 0
        self.adapt_updates_per_step = {}
        self._update_ledger(train_updates_used=0, adapt_updates_used=0, supports_budgeted=False)
        return {
            "status": "unsupported",
            "supports_budgeted": False,
            "adapt_updates_used": 0,
            "adapt_updates_per_step": {},
        }

    def save(self, ckpt_dir: Union[str, Path]) -> Dict[str, Any]:
        if self._inertia is None or self._Q is None or self._R is None or self._P0 is None:
            raise RuntimeError("setup() must be called before save().")
        out = Path(ckpt_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        ckpt_path = out / "model.pt"
        torch.save(
            {
                "dt": float(self._dt),
                "fd_eps": float(self._fd_eps),
                "integration": str(self._integration),
                "inertia": self._inertia.detach().cpu(),
                "disturbance_torque": self._disturbance_torque.detach().cpu() if self._disturbance_torque is not None else None,
                "Q": self._Q.detach().cpu(),
                "R": self._R.detach().cpu(),
                "P0": self._P0.detach().cpu(),
                "outputs_covariance": bool(self._outputs_covariance),
            },
            ckpt_path,
        )
        self._saved_ckpt_path = ckpt_path
        return {"ckpt_path": str(ckpt_path)}

    def load(self, ckpt_path: str) -> None:
        path = Path(ckpt_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"io_error: checkpoint not found: {path}")
        state = torch.load(str(path), map_location="cpu")
        if not isinstance(state, dict):
            raise RuntimeError(f"runtime_error: invalid basilisk_mrp_ekf checkpoint format at {path}")

        self._dt = float(state.get("dt", self._dt))
        self._fd_eps = float(state.get("fd_eps", self._fd_eps))
        self._integration = str(state.get("integration", self._integration))
        self._outputs_covariance = bool(state.get("outputs_covariance", self._outputs_covariance))

        for attr, key in [
            ("_inertia", "inertia"),
            ("_disturbance_torque", "disturbance_torque"),
            ("_Q", "Q"),
            ("_R", "R"),
            ("_P0", "P0"),
        ]:
            val = state.get(key)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device=self.device, dtype=self.dtype))
        if self._inertia is not None:
            self._inertia_inv = torch.linalg.inv(self._inertia)
        self._saved_ckpt_path = path

    def get_runtime_diagnostics(self) -> Dict[str, Any]:
        return dict(self._runtime_diag)

    def supports_viz_diagnostics(self) -> bool:
        return True

    def set_viz_diagnostics_enabled(self, enabled: bool) -> None:
        self._emit_viz_diagnostics = bool(enabled)

    def get_adapter_meta(self) -> Dict[str, Any]:
        return {
            "adapter_id": "basilisk_mrp_ekf",
            "adapter_version": "basilisk_mrp_ekf_v1",
            "runtime_device": str(self.device),
            "baseline": "basilisk_mrp_ekf",
            "baseline_status": "diagnostic_model_based_until_validated",
            "input_layout_bench": "BTD",
            "internal_layout_repo": "BTD_batched_nonlinear_ekf",
            "state": "x=[sigma_BN(3), omega_BN_B(3)]",
            "measurement": "h(x)=x, H=I_6",
            "initialization": "x0 is initialized from first measurement y0; P0 defaults to measurement R plus floor.",
            "integration": str(self._integration),
            "jacobian": "central finite difference of discrete propagation",
            "covariance_support": bool(self._outputs_covariance),
            "capabilities": {
                "train_supported": False,
                "eval_supported": True,
                "adapt_supported": False,
                "supports_budgeted": False,
            },
            "fairness": {
                "uses_ground_truth_in_eval": False,
                "adapt_updates_used": 0,
                "train_updates_used": 0,
            },
            "oracle_policy": {
                "is_oracle": False,
                "note": "Do not label as oracle until model/noise assumptions are explicitly approved.",
            },
        }

    def _continuous_dynamics(self, x: torch.Tensor) -> torch.Tensor:
        if self._inertia is None or self._inertia_inv is None or self._disturbance_torque is None:
            raise RuntimeError("setup() must initialize inertia and disturbance torque before propagation.")
        sigma = x[..., 0:3]
        omega = x[..., 3:6]
        b_mat = mrp_B_matrix(sigma)
        sigma_dot = 0.25 * torch.matmul(b_mat, omega.unsqueeze(-1)).squeeze(-1)
        iomega = torch.matmul(self._inertia, omega.unsqueeze(-1)).squeeze(-1)
        torque = self._disturbance_torque.reshape((1,) * (omega.ndim - 1) + (3,)).expand_as(omega)
        omega_dot = torch.matmul(self._inertia_inv, (torque - torch.cross(omega, iomega, dim=-1)).unsqueeze(-1)).squeeze(-1)
        return torch.cat([sigma_dot, omega_dot], dim=-1)

    def propagate_discrete(self, x: torch.Tensor, *, apply_shadow: bool = True) -> torch.Tensor:
        x_in = _to_tensor(x, device=self.device, dtype=self.dtype)
        if x_in.shape[-1] != 6:
            raise ValueError(f"shape_mismatch: expected state last dim 6, got {tuple(x_in.shape)}")
        dt = float(self._dt)
        if self._integration == "euler":
            out = x_in + dt * self._continuous_dynamics(x_in)
        else:
            k1 = self._continuous_dynamics(x_in)
            k2 = self._continuous_dynamics(x_in + 0.5 * dt * k1)
            k3 = self._continuous_dynamics(x_in + 0.5 * dt * k2)
            k4 = self._continuous_dynamics(x_in + dt * k3)
            out = x_in + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if apply_shadow:
            out = torch.cat([shadow_mrp(out[..., 0:3], eps=self._shadow_eps), out[..., 3:6]], dim=-1)
        return out

    def finite_difference_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        x0 = _to_tensor(x, device=self.device, dtype=self.dtype)
        if x0.ndim != 2 or x0.shape[1] != 6:
            raise ValueError(f"shape_mismatch: expected x [B,6], got {tuple(x0.shape)}")
        bsz, dim = x0.shape
        eps = float(self._fd_eps)
        eye = torch.eye(dim, device=self.device, dtype=self.dtype)
        jac = torch.empty((bsz, dim, dim), device=self.device, dtype=self.dtype)
        sigma_norm = torch.linalg.norm(x0[:, 0:3], dim=1)
        near_shadow = torch.any(torch.abs(sigma_norm - 1.0) <= float(self._shadow_jacobian_guard)).item()
        apply_shadow = not bool(near_shadow)
        warning_count = 1 if near_shadow else 0
        if warning_count:
            self._runtime_diag["shadow_jacobian_guard_hits"] = np.asarray([warning_count], dtype=np.float32)
        for j in range(dim):
            delta = eye[j].reshape(1, dim) * eps
            fp = self.propagate_discrete(x0 + delta, apply_shadow=apply_shadow)
            fm = self.propagate_discrete(x0 - delta, apply_shadow=apply_shadow)
            jac[:, :, j] = (fp - fm) / (2.0 * eps)
        return jac

    def _rollout_ekf(
        self,
        y_btd: torch.Tensor,
        *,
        return_cov: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._Q is None or self._R is None or self._P0 is None:
            raise RuntimeError("setup() must initialize Q/R/P0 before prediction.")
        B, T, Dy = y_btd.shape
        if int(Dy) != 6:
            raise ValueError(f"shape_mismatch: expected Dy=6, got {Dy}")

        eye_x = torch.eye(6, device=self.device, dtype=self.dtype).reshape(1, 6, 6).expand(B, -1, -1)
        eye_y = eye_x
        q_b = self._Q.reshape(1, 6, 6).expand(B, -1, -1)
        r_b = self._R.reshape(1, 6, 6).expand(B, -1, -1)
        p_post = self._P0.reshape(1, 6, 6).expand(B, -1, -1).clone()
        x_post = torch.cat([shadow_mrp(y_btd[:, 0, 0:3], eps=self._shadow_eps), y_btd[:, 0, 3:6]], dim=1)

        x_hist = torch.empty((B, T, 6), device=self.device, dtype=self.dtype)
        cov_hist = torch.empty((B, T, 6, 6), device=self.device, dtype=self.dtype) if return_cov else None
        collect_viz = bool(self._emit_viz_diagnostics)
        innov_hist = torch.empty((B, T, 6), device=self.device, dtype=self.dtype) if collect_viz else None
        s_hist = torch.empty((B, T, 6, 6), device=self.device, dtype=self.dtype) if collect_viz else None
        gain_hist = torch.empty((B, T, 6, 6), device=self.device, dtype=self.dtype) if collect_viz else None
        innovation_norm_t: List[float] = []
        gain_norm_t: List[float] = []
        p_trace_t: List[float] = []
        p_min_eig_t: List[float] = []
        psd_warnings = 0

        for t in range(T):
            if t == 0:
                x_pred = x_post
                p_pred = p_post
            else:
                f_t = self.finite_difference_jacobian(x_post)
                x_pred = self.propagate_discrete(x_post, apply_shadow=True)
                p_pred = torch.bmm(torch.bmm(f_t, p_post), f_t.transpose(1, 2)) + q_b

            y_t = y_btd[:, t, :]
            innov = y_t - x_pred
            s_mat = p_pred + r_b
            if self._jitter > 0.0:
                s_mat = s_mat + eye_y * float(self._jitter)
            try:
                k_gain = torch.linalg.solve(s_mat, p_pred.transpose(1, 2)).transpose(1, 2)
            except RuntimeError:
                k_gain = torch.bmm(p_pred, torch.linalg.pinv(s_mat))

            x_post = x_pred + torch.bmm(k_gain, innov.unsqueeze(-1)).squeeze(-1)
            x_post = torch.cat([shadow_mrp(x_post[:, 0:3], eps=self._shadow_eps), x_post[:, 3:6]], dim=1)
            i_kh = eye_x - k_gain
            p_post = torch.bmm(torch.bmm(i_kh, p_pred), i_kh.transpose(1, 2)) + torch.bmm(torch.bmm(k_gain, r_b), k_gain.transpose(1, 2))
            p_post = 0.5 * (p_post + p_post.transpose(1, 2))

            eig = torch.linalg.eigvalsh(p_post)
            min_eig = float(torch.min(eig).detach().cpu().item())
            if min_eig < -1.0e-6:
                psd_warnings += 1
            innovation_norm_t.append(float(torch.linalg.norm(innov).detach().cpu().item()))
            gain_norm_t.append(float(torch.linalg.norm(k_gain).detach().cpu().item()))
            p_trace_t.append(float(torch.mean(torch.diagonal(p_post, dim1=1, dim2=2).sum(dim=1)).detach().cpu().item()))
            p_min_eig_t.append(min_eig)

            x_hist[:, t, :] = x_post
            if cov_hist is not None:
                cov_hist[:, t, :, :] = p_post
            if collect_viz:
                if innov_hist is not None:
                    innov_hist[:, t, :] = innov
                if s_hist is not None:
                    s_hist[:, t, :, :] = s_mat
                if gain_hist is not None:
                    gain_hist[:, t, :, :] = k_gain
            if self._debug_every > 0 and ((t % self._debug_every) == 0 or has_nonfinite(x_post)):
                logger.debug(
                    "ekf t=%s %s innov_norm=%s K_norm=%s P_trace=%s P_min_eig=%s",
                    t,
                    format_array_stats("x_post", x_post),
                    innovation_norm_t[-1],
                    gain_norm_t[-1],
                    p_trace_t[-1],
                    p_min_eig_t[-1],
                )

        runtime_diag: Dict[str, Any] = {
            "innovation_norm_t": np.asarray(innovation_norm_t, dtype=np.float32),
            "kalman_gain_norm_t": np.asarray(gain_norm_t, dtype=np.float32),
            "p_trace_t": np.asarray(p_trace_t, dtype=np.float32),
            "p_min_eig_t": np.asarray(p_min_eig_t, dtype=np.float32),
            "covariance_psd_warnings": np.asarray([psd_warnings], dtype=np.float32),
        }
        if collect_viz:
            runtime_diag["viz_diagnostics"] = {
                "innov": innov_hist.detach().cpu() if innov_hist is not None else None,
                "S": s_hist.detach().cpu() if s_hist is not None else None,
                "gain": gain_hist.detach().cpu() if gain_hist is not None else None,
            }
        self._runtime_diag = runtime_diag
        return x_hist.contiguous(), cov_hist

    def _update_ledger(
        self,
        *,
        train_updates_used: int,
        adapt_updates_used: int,
        train_max_updates: Optional[int] = None,
        supports_budgeted: Optional[bool] = None,
    ) -> None:
        self.train_updates_used = int(train_updates_used)
        self.train_outer_updates_used = int(train_updates_used)
        self.train_inner_updates_used = 0
        self.adapt_updates_used = int(adapt_updates_used)
        if self._ledger_path is None:
            return

        current: Dict[str, Any] = {}
        if self._ledger_path.exists():
            try:
                loaded = json.loads(self._ledger_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    current = loaded
            except Exception:
                current = {}
        current["train_updates_used"] = int(train_updates_used)
        current["train_outer_updates_used"] = int(train_updates_used)
        current["train_inner_updates_used"] = 0
        current["adapt_updates_used"] = int(adapt_updates_used)
        current["adapt_updates_per_step"] = {}
        if train_max_updates is not None:
            current["train_max_updates"] = int(train_max_updates)
        if supports_budgeted is not None:
            current["supports_budgeted"] = bool(supports_budgeted)
        current["track_id"] = self._run_ctx.get("track_id")
        current["init_id"] = self._run_ctx.get("init_id")
        _write_json(self._ledger_path, current)
