from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from bench.utils.diagnostics import format_array_stats, has_nonfinite, validate_exact_layout
from bench.utils.logging import get_logger

logger = get_logger(__name__)


try:
    from .base import ModelAdapter  # type: ignore
except Exception:  # pragma: no cover
    class ModelAdapter:  # pragma: no cover
        pass


_BASELINE_MODES = {"oracle", "nominal", "oracle_shift"}


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


def _extract_q2_r2(
    cfg: Dict[str, Any],
    system_info: Dict[str, Any],
    meta: Dict[str, Any],
) -> Tuple[float, float]:
    q2_candidates = [
        system_info.get("q2"),
        _lookup_nested(meta, ("noise", "pre_shift", "Q", "q2")),
        _lookup_nested(meta, ("noise", "Q", "q2")),
        _lookup_nested(meta, ("scenario_cfg", "noise", "pre_shift", "Q", "q2")),
        _lookup_nested(meta, ("scenario_cfg", "noise", "Q", "q2")),
        _lookup_nested(meta, ("scenario_cfg", "pre_shift", "Q", "q2")),
        _lookup_nested(meta, ("scenario_cfg", "Q", "q2")),
        cfg.get("q2"),
        1e-3,
    ]
    r2_candidates = [
        system_info.get("r2"),
        _lookup_nested(meta, ("noise", "pre_shift", "R", "r2")),
        _lookup_nested(meta, ("noise", "R", "r2")),
        _lookup_nested(meta, ("scenario_cfg", "noise", "pre_shift", "R", "r2")),
        _lookup_nested(meta, ("scenario_cfg", "noise", "R", "r2")),
        _lookup_nested(meta, ("scenario_cfg", "pre_shift", "R", "r2")),
        _lookup_nested(meta, ("scenario_cfg", "R", "r2")),
        cfg.get("r2"),
        1e-3,
    ]

    q2 = 1e-3
    for v in q2_candidates:
        if v is None:
            continue
        try:
            q2 = float(v)
            break
        except Exception:
            continue

    r2 = 1e-3
    for v in r2_candidates:
        if v is None:
            continue
        try:
            r2 = float(v)
            break
        except Exception:
            continue

    return q2, r2


def _extract_shift_meta(
    cfg: Dict[str, Any],
    system_info: Dict[str, Any],
    meta: Dict[str, Any],
) -> Tuple[Optional[int], float]:
    t0_candidates = [
        system_info.get("t0_shift"),
        _lookup_nested(meta, ("noise", "shift", "t0")),
        _lookup_nested(meta, ("shift", "t0")),
        _lookup_nested(meta, ("scenario_cfg", "noise", "shift", "t0")),
        _lookup_nested(meta, ("scenario_cfg", "shift", "t0")),
        cfg.get("t0_shift"),
    ]
    t0_out: Optional[int] = None
    for v in t0_candidates:
        if v is None:
            continue
        try:
            t0_out = int(v)
            break
        except Exception:
            continue

    rscale_candidates = [
        _lookup_nested(meta, ("noise", "shift", "post_shift", "R_scale")),
        _lookup_nested(meta, ("scenario_cfg", "shift", "post_shift", "R_scale")),
        _lookup_nested(meta, ("scenario_cfg", "noise", "shift", "post_shift", "R_scale")),
        cfg.get("R_scale"),
        1.0,
    ]
    r_scale = 1.0
    for v in rscale_candidates:
        if v is None:
            continue
        try:
            r_scale = float(v)
            break
        except Exception:
            continue
    return t0_out, r_scale


def _resolve_x0_batch(
    state0: Any,
    *,
    batch_size: int,
    x_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if state0 is None:
        return torch.zeros(batch_size, x_dim, 1, device=device, dtype=dtype)

    x = _to_tensor(state0, device=device, dtype=dtype)
    if x.ndim == 1:
        if x.shape[0] != x_dim:
            raise ValueError(f"shape_mismatch: expected x0 dim={x_dim}, got {tuple(x.shape)}")
        return x.view(1, x_dim, 1).repeat(batch_size, 1, 1)
    if x.ndim == 2:
        if x.shape == (x_dim, 1):
            return x.view(1, x_dim, 1).repeat(batch_size, 1, 1)
        if x.shape == (batch_size, x_dim):
            return x.unsqueeze(2)
        raise ValueError(f"shape_mismatch: unexpected x0 shape={tuple(x.shape)}")
    if x.ndim == 3 and x.shape == (batch_size, x_dim, 1):
        return x
    raise ValueError(f"shape_mismatch: unexpected x0 rank/shape={tuple(x.shape)}")


class ModelBasedKFAdapter(ModelAdapter):
    """
    Bench-native linear KF baseline adapter.
    """

    def __init__(self) -> None:
        self.device: torch.device = torch.device("cpu")
        self.dtype: torch.dtype = torch.float32

        self._cfg: Dict[str, Any] = {}
        self._run_ctx: Dict[str, Any] = {}
        self._mode: str = "oracle"

        self._x_dim: Optional[int] = None
        self._y_dim: Optional[int] = None

        self._F: Optional[torch.Tensor] = None
        self._H: Optional[torch.Tensor] = None
        self._Q: Optional[torch.Tensor] = None
        self._R_nominal: Optional[torch.Tensor] = None
        self._R_shift: Optional[torch.Tensor] = None
        self._t0_shift: Optional[int] = None
        self._R_scale: float = 1.0
        self._uses_shift_meta: bool = False
        self._outputs_covariance: bool = False
        self._p0_scale: float = 1.0
        self._innovation_eps: float = 1e-8

        self._run_dir: Optional[Path] = None
        self._ckpt_dir: Optional[Path] = None
        self._artifacts_dir: Optional[Path] = None
        self._ledger_path: Optional[Path] = None
        self._train_state_path: Optional[Path] = None
        self._saved_ckpt_path: Optional[Path] = None

        self.train_updates_used: int = 0
        self.adapt_updates_used: int = 0

        self.last_layout: Optional[str] = None
        self.last_class: Optional[str] = None
        self._debug_every: int = 0
        self._runtime_diag: Dict[str, Any] = {}

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
        self.train_updates_used = 0
        self.adapt_updates_used = 0
        self._debug_every = int(run_ctx.get("debug_every", cfg.get("debug_every", 0)) or 0)
        self._runtime_diag = {}

        mode = str(cfg.get("baseline_mode", "")).strip().lower()
        if not mode:
            model_id = str(run_ctx.get("model_id", "")).strip().lower()
            if model_id in ("oracle_kf", "mb_kf_oracle"):
                mode = "oracle"
            elif model_id in ("nominal_kf", "mb_kf_nominal"):
                mode = "nominal"
            elif model_id in ("oracle_shift_kf",):
                mode = "oracle_shift"
            else:
                mode = "oracle"
        if mode not in _BASELINE_MODES:
            raise ValueError(f"runtime_error: unsupported baseline_mode={mode}; allowed={sorted(_BASELINE_MODES)}")
        self._mode = mode

        requested_device = (
            cfg.get("device", None)
            or run_ctx.get("device", None)
            or system_info.get("device", None)
            or "cpu"
        )
        self.device = _as_torch_device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            self.device = torch.device("cpu")

        seed = run_ctx.get("seed", cfg.get("seed", system_info.get("seed", 0)))
        deterministic = bool(run_ctx.get("deterministic", cfg.get("deterministic", True)))
        _seed_everything(int(seed), deterministic=deterministic)

        self._run_dir = Path(str(run_ctx["run_dir"])).expanduser().resolve() if "run_dir" in run_ctx else None
        if self._run_dir is not None:
            self._ckpt_dir = self._run_dir / "checkpoints"
            self._artifacts_dir = self._run_dir / "artifacts"
            self._ledger_path = self._run_dir / "budget_ledger.json"
            self._train_state_path = self._ckpt_dir / "train_state.json"
            self._ckpt_dir.mkdir(parents=True, exist_ok=True)
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            self._init_ledger()

        meta = _coerce_meta_dict(system_info)
        x_dim = system_info.get("x_dim", cfg.get("x_dim", meta.get("x_dim")))
        y_dim = system_info.get("y_dim", cfg.get("y_dim", meta.get("y_dim")))
        if x_dim is None or y_dim is None:
            raise ValueError("system_info must provide x_dim and y_dim for mb_kf setup.")
        self._x_dim = int(x_dim)
        self._y_dim = int(y_dim)

        F = system_info.get("F", system_info.get("A", None))
        H = system_info.get("H", system_info.get("C", None))
        if F is None:
            F = torch.eye(self._x_dim)
        if H is None:
            H = torch.eye(self._y_dim, self._x_dim)
        self._F = _to_tensor(F, device=self.device, dtype=self.dtype).reshape(self._x_dim, self._x_dim)
        self._H = _to_tensor(H, device=self.device, dtype=self.dtype).reshape(self._y_dim, self._x_dim)

        q2, r2 = _extract_q2_r2(cfg, system_info, meta)
        Q = system_info.get("Q", None)
        R = system_info.get("R", None)
        if Q is None:
            self._Q = torch.eye(self._x_dim, device=self.device, dtype=self.dtype) * float(q2)
        else:
            self._Q = _to_tensor(Q, device=self.device, dtype=self.dtype).reshape(self._x_dim, self._x_dim)
        if R is None:
            self._R_nominal = torch.eye(self._y_dim, device=self.device, dtype=self.dtype) * float(r2)
        else:
            self._R_nominal = _to_tensor(R, device=self.device, dtype=self.dtype).reshape(self._y_dim, self._y_dim)

        self._t0_shift, self._R_scale = _extract_shift_meta(cfg, system_info, meta)
        self._R_shift = (self._R_nominal * float(self._R_scale)).clone()
        self._uses_shift_meta = bool(
            self._mode in ("oracle", "oracle_shift")
            and self._t0_shift is not None
            and abs(float(self._R_scale) - 1.0) > 0.0
        )
        self._outputs_covariance = bool(cfg.get("outputs_covariance", False))
        self._p0_scale = float(cfg.get("p0_scale", 1.0))
        self._innovation_eps = float(cfg.get("innovation_eps", 1e-8))

        self.last_layout = "bench_BTD_to_repo_BTD"
        self.last_class = "bench.models.mb_kf:ModelBasedKFAdapter"
        logger.info(
            "setup mode=%s device=%s x_dim=%s y_dim=%s t0_shift=%s R_scale=%s layout=%s",
            self._mode,
            self.device,
            self._x_dim,
            self._y_dim,
            self._t0_shift,
            self._R_scale,
            self.last_layout,
        )

    def train(
        self,
        train_dl: Any,
        val_dl: Any,
        budget: Optional[Dict[str, Any]] = None,
        ckpt_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        # No-op by design (model-based baseline).
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
                "note": "mb_kf baseline has no trainable parameters; no-op train.",
            },
        )
        self._train_state_path = train_state_path
        self.train_updates_used = 0
        self.adapt_updates_used = 0
        self._update_ledger(
            train_updates_used=0,
            adapt_updates_used=0,
            train_max_updates=int(max_updates),
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
        y_batch: Any,
        state0: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        return_cov: bool = False,
        **kwargs: Any,
    ) -> Any:
        if self._F is None or self._H is None or self._Q is None or self._R_nominal is None:
            raise RuntimeError("setup() must be called before predict().")
        if self._x_dim is None or self._y_dim is None:
            raise RuntimeError("Adapter dimensions are not initialized.")

        y = _to_tensor(y_batch, device=self.device, dtype=self.dtype)
        if y.ndim != 3:
            raise ValueError(f"shape_mismatch: expected y [B,T,Dy], got {tuple(y.shape)}")
        bsz, _t, dy = y.shape
        if int(dy) != self._y_dim:
            raise ValueError(f"shape_mismatch: got y_dim={dy}, expected {self._y_dim}")
        logger.debug("predict input shape=%s mode=%s layout=%s", tuple(y.shape), self._mode, self.last_layout)

        ctx = dict(context or {})
        if state0 is None:
            state0 = ctx.get("x0", None)
        x0_batch = _resolve_x0_batch(
            state0,
            batch_size=int(bsz),
            x_dim=int(self._x_dim),
            device=self.device,
            dtype=self.dtype,
        )

        want_cov = bool(return_cov and self._outputs_covariance)
        x_hat, cov = self._rollout_kf(y_btd=y, x0_batch=x0_batch, return_cov=want_cov)
        validate_exact_layout(
            x_hat,
            expected=(int(bsz), int(_t), int(self._x_dim)),
            axis_names=("B", "T", "D"),
            label="x_hat",
        )
        logger.debug("predict output %s", format_array_stats("x_hat", x_hat))
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
        total_n = 0
        with torch.no_grad():
            for bi, batch in enumerate(test_dl):
                _x_raw, y_raw = _extract_batch_xy(batch)
                y = _to_tensor(y_raw, device=self.device, dtype=self.dtype)
                pred = self.predict(y, context=None, return_cov=self._outputs_covariance)
                if isinstance(pred, tuple):
                    x_hat_b = pred[0]
                    cov_b = pred[1]
                else:
                    x_hat_b = pred
                    cov_b = None
                if not isinstance(x_hat_b, torch.Tensor):
                    raise TypeError(f"runtime_error: predict() must return Tensor, got {type(x_hat_b)}")
                validate_exact_layout(
                    x_hat_b,
                    expected=(int(y.shape[0]), int(y.shape[1]), int(self._x_dim)),
                    axis_names=("B", "T", "D"),
                    label="x_hat",
                )
                total_n += int(y.shape[0])
                if self._debug_every > 0 and ((bi % self._debug_every) == 0 or has_nonfinite(x_hat_b)):
                    logger.debug("eval batch=%s %s", bi, format_array_stats("x_hat", x_hat_b))
                preds.append(x_hat_b.detach().cpu())
                if isinstance(cov_b, torch.Tensor):
                    covs.append(cov_b.detach().cpu())

        if not preds:
            raise RuntimeError("runtime_error: empty test dataloader.")
        x_hat = torch.cat(preds, dim=0).contiguous()
        validate_exact_layout(
            x_hat,
            expected=(int(total_n), int(x_hat.shape[1]), int(self._x_dim)),
            axis_names=("N", "T", "D"),
            label="x_hat",
        )

        cov_cat = None
        if covs:
            cov_cat = torch.cat(covs, dim=0).contiguous()

        preds_path = None
        if self._artifacts_dir is not None:
            preds_path = self._artifacts_dir / "preds_test.npz"
            if cov_cat is None:
                np.savez_compressed(preds_path, x_hat=x_hat.numpy())
            else:
                np.savez_compressed(preds_path, x_hat=x_hat.numpy(), cov=cov_cat.numpy())

        self.adapt_updates_used = 0
        self._update_ledger(
            train_updates_used=int(self.train_updates_used),
            adapt_updates_used=0,
        )
        return {
            "status": "ok",
            "x_hat": x_hat,
            "cov": cov_cat,
            "preds_path": (str(preds_path) if preds_path is not None else None),
        }

    def adapt(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        budget: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # No-op adaptation for model-based KF baselines.
        self.adapt_updates_used = 0
        self._update_ledger(
            train_updates_used=int(self.train_updates_used),
            adapt_updates_used=0,
            supports_budgeted=False,
        )
        return {
            "status": "unsupported",
            "supports_budgeted": False,
            "adapt_updates_used": 0,
            "adapt_updates_per_step": {},
        }

    def get_runtime_diagnostics(self) -> Dict[str, Any]:
        return dict(self._runtime_diag)

    def save(self, ckpt_dir: Union[str, Path]) -> Dict[str, Any]:
        if self._F is None or self._H is None or self._Q is None or self._R_nominal is None or self._R_shift is None:
            raise RuntimeError("setup() must be called before save().")
        out = Path(ckpt_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        ckpt_path = out / "model.pt"
        torch.save(
            {
                "baseline_mode": self._mode,
                "F": self._F.detach().cpu(),
                "H": self._H.detach().cpu(),
                "Q": self._Q.detach().cpu(),
                "R_nominal": self._R_nominal.detach().cpu(),
                "R_shift": self._R_shift.detach().cpu(),
                "t0_shift": self._t0_shift,
                "R_scale": float(self._R_scale),
                "uses_shift_meta": bool(self._uses_shift_meta),
                "outputs_covariance": bool(self._outputs_covariance),
                "p0_scale": float(self._p0_scale),
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
            raise RuntimeError(f"runtime_error: invalid mb_kf checkpoint format at {path}")

        mode = str(state.get("baseline_mode", self._mode)).strip().lower()
        if mode in _BASELINE_MODES:
            self._mode = mode

        def _load_mat(key: str, fallback: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if key not in state:
                return fallback
            val = state[key]
            if not isinstance(val, torch.Tensor):
                return fallback
            return val.to(device=self.device, dtype=self.dtype)

        self._F = _load_mat("F", self._F)
        self._H = _load_mat("H", self._H)
        self._Q = _load_mat("Q", self._Q)
        self._R_nominal = _load_mat("R_nominal", self._R_nominal)
        self._R_shift = _load_mat("R_shift", self._R_shift)
        try:
            self._t0_shift = int(state.get("t0_shift")) if state.get("t0_shift") is not None else self._t0_shift
        except Exception:
            pass
        try:
            self._R_scale = float(state.get("R_scale", self._R_scale))
        except Exception:
            pass
        if "uses_shift_meta" in state:
            self._uses_shift_meta = bool(state["uses_shift_meta"])
        if "outputs_covariance" in state:
            self._outputs_covariance = bool(state["outputs_covariance"])
        if "p0_scale" in state:
            try:
                self._p0_scale = float(state["p0_scale"])
            except Exception:
                pass
        self._saved_ckpt_path = path

    def get_adapter_meta(self) -> Dict[str, Any]:
        return {
            "adapter_id": "mb_kf",
            "adapter_version": "s10_kf_baseline_v1",
            "runtime_device": str(self.device),
            "baseline": "mb_kf",
            "mode": self._mode,
            "uses_shift_meta": bool(self._uses_shift_meta),
            "t0_shift": self._t0_shift,
            "R_scale": float(self._R_scale),
            "covariance_support": bool(self._outputs_covariance),
            "input_layout_bench": "BTD",
            "internal_layout_repo": "BTD_batched_linear_kf",
            "capabilities": {
                "train_supported": False,
                "eval_supported": True,
                "adapt_supported": False,
                "supports_budgeted": False,
            },
            "assumptions": {
                "A_linear_system": "F/H are linear system matrices from bench-generated NPZ/system_info.",
                "B_oracle_shift_info": "oracle_shift mode uses shift meta (t0,R_scale) from D15 meta_json.",
                "C_metrics_ownership": "official metrics are computed by bench layer only.",
            },
            "how_to_verify": {
                "A_linear_system": "bench/tasks/bench_generated.py (save_npz_split_v0 includes F/H/meta_json).",
                "B_oracle_shift_info": "bench/tasks/bench_generated.py noise_meta['shift']['post_shift']['R_scale'] and t0.",
                "C_metrics_ownership": "bench/runners/run_suite.py metrics block and adapter eval()/predict() outputs.",
            },
        }

    def _rollout_kf(
        self,
        *,
        y_btd: torch.Tensor,
        x0_batch: torch.Tensor,
        return_cov: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._F is None or self._H is None or self._Q is None or self._R_nominal is None or self._R_shift is None:
            raise RuntimeError("Adapter matrices are not initialized.")
        if self._x_dim is None or self._y_dim is None:
            raise RuntimeError("Adapter dimensions are not initialized.")

        if y_btd.ndim != 3:
            raise ValueError(f"shape_mismatch: expected y [B,T,Dy], got {tuple(y_btd.shape)}")
        B, T, Dy = y_btd.shape
        if int(Dy) != self._y_dim:
            raise ValueError(f"shape_mismatch: expected Dy={self._y_dim}, got {Dy}")
        if x0_batch.shape != (B, self._x_dim, 1):
            raise ValueError(
                f"shape_mismatch: expected x0_batch {(B, self._x_dim, 1)}, got {tuple(x0_batch.shape)}"
            )

        F_b = self._F.reshape(1, self._x_dim, self._x_dim).expand(B, -1, -1)
        H_b = self._H.reshape(1, self._y_dim, self._x_dim).expand(B, -1, -1)
        Q_b = self._Q.reshape(1, self._x_dim, self._x_dim).expand(B, -1, -1)
        I_x = torch.eye(self._x_dim, device=self.device, dtype=self.dtype).reshape(1, self._x_dim, self._x_dim).expand(B, -1, -1)
        I_y = torch.eye(self._y_dim, device=self.device, dtype=self.dtype).reshape(1, self._y_dim, self._y_dim).expand(B, -1, -1)
        P_post = torch.eye(self._x_dim, device=self.device, dtype=self.dtype).reshape(1, self._x_dim, self._x_dim).expand(B, -1, -1).clone()
        P_post.mul_(float(self._p0_scale))

        x_post = x0_batch.clone()
        x_hist = torch.empty((B, T, self._x_dim), device=self.device, dtype=self.dtype)
        cov_hist = torch.empty((B, T, self._x_dim, self._x_dim), device=self.device, dtype=self.dtype) if return_cov else None
        innov_norm_t: List[float] = []
        k_norm_t: List[float] = []

        for t in range(T):
            R_t = self._R_nominal
            if self._mode in ("oracle", "oracle_shift") and self._uses_shift_meta and self._t0_shift is not None:
                if int(t) >= int(self._t0_shift):
                    R_t = self._R_shift
            R_b = R_t.reshape(1, self._y_dim, self._y_dim).expand(B, -1, -1)

            x_pred = torch.bmm(F_b, x_post)
            P_pred = torch.bmm(torch.bmm(F_b, P_post), F_b.transpose(1, 2)) + Q_b

            y_t = y_btd[:, t, :].reshape(B, self._y_dim, 1)
            innov = y_t - torch.bmm(H_b, x_pred)

            PHt = torch.bmm(P_pred, H_b.transpose(1, 2))
            S = torch.bmm(torch.bmm(H_b, P_pred), H_b.transpose(1, 2)) + R_b
            if self._innovation_eps > 0.0:
                S = S + I_y * float(self._innovation_eps)
            try:
                K = torch.linalg.solve(S, PHt.transpose(1, 2)).transpose(1, 2)
            except RuntimeError:
                S_inv = torch.linalg.pinv(S)
                K = torch.bmm(PHt, S_inv)
            innov_norm_t.append(float(torch.linalg.norm(innov).detach().cpu().item()))
            k_norm_t.append(float(torch.linalg.norm(K).detach().cpu().item()))

            x_post = x_pred + torch.bmm(K, innov)
            I_KH = I_x - torch.bmm(K, H_b)
            # Joseph stabilized covariance update.
            P_post = (
                torch.bmm(torch.bmm(I_KH, P_pred), I_KH.transpose(1, 2))
                + torch.bmm(torch.bmm(K, R_b), K.transpose(1, 2))
            )

            x_hist[:, t, :] = x_post[:, :, 0]
            if cov_hist is not None:
                cov_hist[:, t, :, :] = P_post
            if self._debug_every > 0 and ((t % self._debug_every) == 0 or has_nonfinite(x_post)):
                logger.debug(
                    "kf t=%s %s innov_norm=%s K_norm=%s",
                    t,
                    format_array_stats("x_post", x_post),
                    innov_norm_t[-1],
                    k_norm_t[-1],
                )

        self._runtime_diag = {
            "innovation_norm_t": np.asarray(innov_norm_t, dtype=np.float32),
            "k_norm_t": np.asarray(k_norm_t, dtype=np.float32),
        }
        validate_exact_layout(
            x_hist,
            expected=(int(B), int(T), int(self._x_dim)),
            axis_names=("B", "T", "D"),
            label="x_hat",
        )
        return x_hist.contiguous(), cov_hist

    def _init_ledger(self) -> None:
        if self._ledger_path is None:
            return
        if self._ledger_path.exists():
            return
        _write_json(
            self._ledger_path,
            {
                "train_updates_used": 0,
                "adapt_updates_used": 0,
                "track_id": self._run_ctx.get("track_id"),
                "init_id": self._run_ctx.get("init_id"),
                "supports_budgeted": False,
            },
        )

    def _update_ledger(
        self,
        *,
        train_updates_used: int,
        adapt_updates_used: int,
        train_max_updates: Optional[int] = None,
        supports_budgeted: Optional[bool] = None,
    ) -> None:
        if self._ledger_path is None:
            return

        current: Dict[str, Any] = {}
        if self._ledger_path.exists():
            try:
                current = json.loads(self._ledger_path.read_text(encoding="utf-8"))
                if not isinstance(current, dict):
                    current = {}
            except Exception:
                current = {}

        current["train_updates_used"] = int(train_updates_used)
        current["adapt_updates_used"] = int(adapt_updates_used)
        if train_max_updates is not None:
            current["train_max_updates"] = int(train_max_updates)
        if supports_budgeted is not None:
            current["supports_budgeted"] = bool(supports_budgeted)
        current["track_id"] = self._run_ctx.get("track_id")
        current["init_id"] = self._run_ctx.get("init_id")
        _write_json(self._ledger_path, current)
