from __future__ import annotations

import contextlib
import importlib
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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


@dataclass
class _SplitImports:
    filtering_mod: Any


class _LinearSystemModel:
    """
    Minimal GSSModel-compatible wrapper for bench-provided linear systems.
    """

    def __init__(
        self,
        *,
        F: torch.Tensor,
        H: torch.Tensor,
        cov_q: torch.Tensor,
        cov_r: torch.Tensor,
        init_state: torch.Tensor,
    ) -> None:
        self.F = F
        self.H = H
        self.cov_q = cov_q
        self.cov_r = cov_r
        self.x_dim = int(F.shape[0])
        self.y_dim = int(H.shape[0])
        self.init_state = init_state.reshape(self.x_dim, 1)
        self.init_cov = torch.zeros((self.x_dim, self.x_dim), device=F.device, dtype=F.dtype)

    def f(self, current_state: torch.Tensor) -> torch.Tensor:
        return self.F @ current_state

    def g(self, current_state: torch.Tensor) -> torch.Tensor:
        return self.H @ current_state

    def Jacobian_f(self, _x: torch.Tensor) -> torch.Tensor:
        return self.F

    def Jacobian_g(self, _x: torch.Tensor) -> torch.Tensor:
        return self.H


def _bench_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _lookup_nested(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _coerce_meta_dict(system_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not system_info:
        return {}
    if isinstance(system_info.get("meta"), dict):
        return dict(system_info["meta"])

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
    model_cfg: Dict[str, Any],
    system_info: Dict[str, Any],
    meta: Dict[str, Any],
) -> Tuple[float, float]:
    q2_candidates = [
        system_info.get("q2"),
        _lookup_nested(meta, ("noise", "pre_shift", "Q", "q2")),
        _lookup_nested(meta, ("scenario_cfg", "noise", "pre_shift", "Q", "q2")),
        _lookup_nested(meta, ("noise", "Q", "q2")),
        _lookup_nested(meta, ("scenario_cfg", "noise", "Q", "q2")),
        model_cfg.get("q2"),
        1e-3,
    ]
    r2_candidates = [
        system_info.get("r2"),
        _lookup_nested(meta, ("noise", "pre_shift", "R", "r2")),
        _lookup_nested(meta, ("scenario_cfg", "noise", "pre_shift", "R", "r2")),
        _lookup_nested(meta, ("noise", "R", "r2")),
        _lookup_nested(meta, ("scenario_cfg", "noise", "R", "r2")),
        model_cfg.get("r2"),
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


def _extract_batch_xy(batch: Any) -> Tuple[Any, Any]:
    if isinstance(batch, dict):
        if "x" not in batch or "y" not in batch:
            raise KeyError("Batch dict must contain keys 'x' and 'y'.")
        return batch["x"], batch["y"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(f"Unsupported dataloader batch type: {type(batch)}")


def _resolve_repo_spec(model_cfg: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    repo_spec = (
        model_cfg.get("repo")
        or model_cfg.get("repo_root")
        or model_cfg.get("repo_path")
        or model_cfg.get("path")
    )

    entrypoints: Dict[str, Any] = {}
    repo_path: Optional[str] = None

    if isinstance(repo_spec, dict):
        repo_path = repo_spec.get("path") or repo_spec.get("repo_root") or repo_spec.get("repo_path")
        entrypoints = dict(repo_spec.get("entrypoints") or {})
    elif isinstance(repo_spec, (str, Path)):
        repo_path = str(repo_spec)
    elif repo_spec is not None:
        raise TypeError(f"Unsupported repo spec type: {type(repo_spec)}")

    if not repo_path:
        raise ValueError("Split-KalmanNet adapter requires model_cfg['repo'] as string or dict with 'path'.")

    bench_root = _bench_root_from_this_file()
    p = Path(repo_path).expanduser()
    repo_root = (bench_root / p).resolve() if not p.is_absolute() else p.resolve()
    return repo_root, entrypoints


def _normalize_repo_root(repo_root: Path) -> Path:
    bench_root = _bench_root_from_this_file()
    candidates = [
        repo_root,
        repo_root / "Split_KalmanNet",
        repo_root / "Split-KalmanNet",
        (bench_root / "third_party" / "Split_KalmanNet").resolve(),
        (bench_root / "third_party" / "Split-KalmanNet").resolve(),
    ]

    if "Split-KalmanNet" in str(repo_root):
        candidates.append(Path(str(repo_root).replace("Split-KalmanNet", "Split_KalmanNet")))
    if "Split_KalmanNet" in str(repo_root):
        candidates.append(Path(str(repo_root).replace("Split_KalmanNet", "Split-KalmanNet")))

    seen = set()
    unique: List[Path] = []
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    for c in unique:
        if (c / "config.ini").exists() and (c / "GSSFiltering" / "filtering.py").exists():
            return c

    raise FileNotFoundError(
        f"Could not locate Split-KalmanNet root from {repo_root}. "
        "Expected config.ini and GSSFiltering/filtering.py."
    )


@contextlib.contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


def _load_split_modules(repo_root: Path) -> _SplitImports:
    if not repo_root.exists():
        raise FileNotFoundError(f"Split-KalmanNet repo root not found: {repo_root}")

    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    with _pushd(repo_root):
        # Force reload under repo_root so config.ini-relative globals are stable.
        for name in ("GSSFiltering.dnn", "GSSFiltering.model", "GSSFiltering.filtering"):
            if name in sys.modules:
                importlib.reload(sys.modules[name])
            else:
                importlib.import_module(name)
        filtering_mod = importlib.import_module("GSSFiltering.filtering")

    return _SplitImports(filtering_mod=filtering_mod)


def _load_filter_class(imports: _SplitImports, repo_root: Path, class_path: str) -> Any:
    if "." not in class_path:
        cls = getattr(imports.filtering_mod, class_path, None)
        if cls is None:
            raise AttributeError(f"Split-KalmanNet class not found: {class_path}")
        return cls

    module_name, class_name = class_path.rsplit(".", 1)
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    with _pushd(repo_root):
        mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise AttributeError(f"Split-KalmanNet class not found: {class_path}")
    return cls


def _patch_split_net_for_device(kf_net: Any, device: torch.device) -> None:
    """
    third_party/Split_KalmanNet DNN modules allocate some tensors on CPU inside forward().
    We patch runtime methods to allocate on the active tensor device without editing third_party code.
    """
    cls_name = type(kf_net).__name__

    if cls_name == "DNN_SKalmanNet_GSS":
        def _initialize_hidden_patched(self):
            p = next(self.parameters())
            self.hn1 = self.hn1_init.detach().clone().to(device=p.device, dtype=p.dtype)
            self.hn2 = self.hn2_init.detach().clone().to(device=p.device, dtype=p.dtype)

        def _forward_patched(self, state_inno, observation_inno, diff_state, diff_obs, linearization_error, Jacobian):
            p = next(self.parameters())
            dev = state_inno.device if isinstance(state_inno, torch.Tensor) else p.device
            dtyp = p.dtype

            input1 = torch.cat((state_inno, diff_state, linearization_error, Jacobian), axis=0).reshape(-1)
            input2 = torch.cat((observation_inno, diff_obs, linearization_error, Jacobian), axis=0).reshape(-1)

            l1_out = self.l1(input1)
            gru_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim, device=dev, dtype=dtyp)
            gru_in[0, 0, :] = l1_out
            gru_out, self.hn1 = self.GRU1(gru_in, self.hn1.to(device=dev, dtype=dtyp))
            l2_out = self.l2(gru_out)
            pk = l2_out.reshape((self.x_dim, self.x_dim))

            l3_out = self.l3(input2)
            gru_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim, device=dev, dtype=dtyp)
            gru_in[0, 0, :] = l3_out
            gru_out, self.hn2 = self.GRU2(gru_in, self.hn2.to(device=dev, dtype=dtyp))
            l4_out = self.l4(gru_out)
            sk = l4_out.reshape((self.y_dim, self.y_dim))
            return (pk, sk)

        kf_net.initialize_hidden = MethodType(_initialize_hidden_patched, kf_net)
        kf_net.forward = MethodType(_forward_patched, kf_net)

    elif cls_name == "DNN_KalmanNet_GSS":
        def _initialize_hidden_patched(self):
            p = next(self.parameters())
            self.hn = self.hn_init.detach().clone().to(device=p.device, dtype=p.dtype)

        def _forward_patched(self, state_inno, observation_inno, diff_state, diff_obs):
            p = next(self.parameters())
            dev = state_inno.device if isinstance(state_inno, torch.Tensor) else p.device
            dtyp = p.dtype

            input_vec = torch.cat((state_inno, observation_inno, diff_state, diff_obs), axis=0).reshape(-1)
            l1_out = self.l1(input_vec)
            gru_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim, device=dev, dtype=dtyp)
            gru_in[0, 0, :] = l1_out
            gru_out, self.hn = self.GRU(gru_in, self.hn.to(device=dev, dtype=dtyp))
            l2_out = self.l2(gru_out)
            kg = torch.reshape(l2_out, (self.x_dim, self.y_dim))
            return kg

        kf_net.initialize_hidden = MethodType(_initialize_hidden_patched, kf_net)
        kf_net.forward = MethodType(_forward_patched, kf_net)

    # Prime hidden states on target device.
    if hasattr(kf_net, "initialize_hidden"):
        kf_net.initialize_hidden()


def _resolve_x0_batch(
    x0: Any,
    *,
    batch_size: int,
    x_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x = _to_tensor(x0, device=device, dtype=dtype)
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


class SplitKNetAdapter(ModelAdapter):
    """
    Route-B adapter for third_party/Split_KalmanNet.

    Integration mode: import-mode (model-only).
    - Bench controls train/val/test splits and budget counting.
    - third_party classes reused:
      GSSFiltering.filtering.Split_KalmanNet_Filter + GSSFiltering.dnn modules.
    """

    def __init__(self) -> None:
        self.repo_root: Optional[Path] = None
        self.entrypoints: Dict[str, Any] = {}
        self.device: torch.device = torch.device("cpu")
        self.dtype: torch.dtype = torch.float32

        self._imports: Optional[_SplitImports] = None
        self._filter_class_path: str = "GSSFiltering.filtering.Split_KalmanNet_Filter"
        self._filter_obj: Any = None
        self._system_model: Optional[_LinearSystemModel] = None

        self._x_dim: Optional[int] = None
        self._y_dim: Optional[int] = None
        self._T_setup: Optional[int] = None
        self._cfg: Dict[str, Any] = {}
        self._run_ctx: Dict[str, Any] = {}

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
        self._debug_every = int(run_ctx.get("debug_every", cfg.get("debug_every", 0)) or 0)
        self._runtime_diag = {}

        repo_raw, self.entrypoints = _resolve_repo_spec(cfg)
        self.repo_root = _normalize_repo_root(repo_raw)
        self._imports = _load_split_modules(self.repo_root)

        requested_device = cfg.get("device", None) or run_ctx.get("device", None) or system_info.get("device", None) or "cpu"
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

        input_layout = str(cfg.get("input_layout", "BTD")).upper()
        if input_layout != "BTD":
            raise ValueError(f"shape_mismatch: split_knet expects input_layout='BTD', got '{input_layout}'")
        self.last_layout = "bench_BTD_to_repo_stepwise_colvec"

        meta = _coerce_meta_dict(system_info)
        x_dim = system_info.get("x_dim", cfg.get("x_dim", meta.get("x_dim")))
        y_dim = system_info.get("y_dim", cfg.get("y_dim", meta.get("y_dim")))
        if x_dim is None or y_dim is None:
            raise ValueError("system_info must provide x_dim and y_dim for split_knet setup.")
        self._x_dim = int(x_dim)
        self._y_dim = int(y_dim)

        t_len = system_info.get("T", cfg.get("T", meta.get("T", cfg.get("sequence_length", 1))))
        self._T_setup = int(t_len)

        F = system_info.get("F", system_info.get("A", None))
        H = system_info.get("H", system_info.get("C", None))
        F_t = _to_tensor(F if F is not None else torch.eye(self._x_dim), device=self.device, dtype=self.dtype)
        H_t = _to_tensor(H if H is not None else torch.eye(self._y_dim, self._x_dim), device=self.device, dtype=self.dtype)

        q2, r2 = _extract_q2_r2(cfg, system_info, meta)
        Q = system_info.get("Q", None)
        R = system_info.get("R", None)
        Q_t = _to_tensor(Q, device=self.device, dtype=self.dtype) if Q is not None else (
            torch.eye(self._x_dim, device=self.device, dtype=self.dtype) * float(q2)
        )
        R_t = _to_tensor(R, device=self.device, dtype=self.dtype) if R is not None else (
            torch.eye(self._y_dim, device=self.device, dtype=self.dtype) * float(r2)
        )
        init_state = _to_tensor(
            system_info.get("x0", torch.zeros(self._x_dim, 1)),
            device=self.device,
            dtype=self.dtype,
        ).reshape(self._x_dim, 1)

        self._system_model = _LinearSystemModel(
            F=F_t,
            H=H_t,
            cov_q=Q_t,
            cov_r=R_t,
            init_state=init_state,
        )

        self._filter_class_path = str(cfg.get("estimator_class_path", "GSSFiltering.filtering.Split_KalmanNet_Filter"))
        filter_cls = _load_filter_class(self._imports, self.repo_root, self._filter_class_path)
        self._filter_obj = filter_cls(self._system_model)
        if not hasattr(self._filter_obj, "kf_net"):
            raise RuntimeError(
                "runtime_error: Split-KalmanNet filter object missing 'kf_net'. "
                "HOW TO VERIFY: third_party/Split_KalmanNet/GSSFiltering/filtering.py::Split_KalmanNet_Filter"
            )
        _patch_split_net_for_device(self._filter_obj.kf_net, self.device)
        self._filter_obj.kf_net.to(self.device)
        self._filter_obj.kf_net.eval()

        self.last_class = self._filter_class_path
        logger.info(
            "setup repo=%s class=%s device=%s x_dim=%s y_dim=%s T=%s layout=%s",
            self.repo_root,
            self.last_class,
            self.device,
            self._x_dim,
            self._y_dim,
            self._T_setup,
            self.last_layout,
        )

    def train(
        self,
        train_dl: Any,
        val_dl: Any,
        budget: Optional[Dict[str, Any]] = None,
        ckpt_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        if self._filter_obj is None or self._x_dim is None or self._y_dim is None:
            raise RuntimeError("setup() must be called before train().")

        budget = dict(budget or {})
        max_updates = int(budget.get("train_max_updates", 0))
        if max_updates <= 0:
            raise ValueError("train_max_updates must be > 0 for init_id=trained.")

        out_ckpt_dir = Path(ckpt_dir).expanduser().resolve() if ckpt_dir is not None else self._ckpt_dir
        if out_ckpt_dir is None:
            raise ValueError("ckpt_dir is required when adapter has no run_dir.")
        out_ckpt_dir.mkdir(parents=True, exist_ok=True)

        lr = float(self._cfg.get("lr", 1e-3))
        wd = float(self._cfg.get("weight_decay", self._cfg.get("wd", 0.0)))
        eval_interval = int(self._cfg.get("val_eval_interval_updates", max(1, min(10, max_updates))))
        patience_evals = int(self._cfg.get("patience_evals", budget.get("patience_evals", 0)))
        min_delta = float(self._cfg.get("min_delta", budget.get("min_delta", 0.0)))
        max_grad_norm = self._cfg.get("max_grad_norm", 10.0)
        max_grad_norm_f = float(max_grad_norm) if max_grad_norm is not None else None
        val_max_batches = int(self._cfg.get("val_max_batches", 0))
        train_init_from_gt = bool(self._cfg.get("train_init_from_gt", True))

        params = [p for p in self._filter_obj.kf_net.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("runtime_error: split_knet has no trainable parameters.")

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        loss_fn = torch.nn.MSELoss(reduction="mean")

        self._filter_obj.kf_net.train()
        updates_used = 0
        best_step = 0
        best_val = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in self._filter_obj.kf_net.state_dict().items()}
        no_improve_evals = 0
        last_train_loss = None
        val_history: List[Dict[str, float]] = []

        while updates_used < max_updates:
            for batch in train_dl:
                if updates_used >= max_updates:
                    break

                x_raw, y_raw = _extract_batch_xy(batch)
                x = _to_tensor(x_raw, device=self.device, dtype=self.dtype)
                y = _to_tensor(y_raw, device=self.device, dtype=self.dtype)

                if x.ndim != 3 or y.ndim != 3:
                    raise ValueError(f"shape_mismatch: expected rank-3 x,y; got x={tuple(x.shape)} y={tuple(y.shape)}")
                if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
                    raise ValueError(f"shape_mismatch: x/y batch-time mismatch x={tuple(x.shape)} y={tuple(y.shape)}")
                if x.shape[2] != self._x_dim or y.shape[2] != self._y_dim:
                    raise ValueError(
                        f"shape_mismatch: expected x_dim={self._x_dim} y_dim={self._y_dim}; "
                        f"got x={tuple(x.shape)} y={tuple(y.shape)}"
                    )

                x0_batch = x[:, 0, :] if train_init_from_gt else None

                optimizer.zero_grad(set_to_none=True)
                pred = self._forward_batch(y_btd=y, x0_batch=x0_batch)
                if x.shape[1] > 1:
                    loss = loss_fn(pred[:, 1:, :], x[:, 1:, :])
                else:
                    loss = loss_fn(pred, x)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"train_nan: non-finite training loss at update={updates_used}")

                loss.backward()
                if max_grad_norm_f is not None and max_grad_norm_f > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm_f)

                if updates_used >= max_updates:
                    raise RuntimeError(f"budget_overflow: attempted optimizer.step beyond max_updates={max_updates}")
                optimizer.step()
                updates_used += 1
                last_train_loss = float(loss.detach().item())

                should_eval = (updates_used % eval_interval == 0) or (updates_used == max_updates)
                if should_eval:
                    val_loss = self._compute_validation_loss(
                        val_dl=val_dl,
                        loss_fn=loss_fn,
                        max_batches=val_max_batches,
                        init_from_gt=train_init_from_gt,
                    )
                    val_history.append({"step": float(updates_used), "val_mse": float(val_loss)})
                    if (best_val - float(val_loss)) > min_delta:
                        best_val = float(val_loss)
                        best_step = int(updates_used)
                        best_state = {k: v.detach().cpu().clone() for k, v in self._filter_obj.kf_net.state_dict().items()}
                        no_improve_evals = 0
                    else:
                        no_improve_evals += 1
                    if patience_evals > 0 and no_improve_evals >= patience_evals:
                        logger.info(
                            "Early stopping Split-KalmanNet training at step=%s (patience_evals=%s)",
                            updates_used,
                            patience_evals,
                        )
                        updates_used = max_updates
                        break

        self._filter_obj.kf_net.load_state_dict(best_state, strict=True)
        ckpt_path = out_ckpt_dir / "model.pt"
        torch.save(
            {
                "state_dict": self._filter_obj.kf_net.state_dict(),
                "best_step": int(best_step),
                "best_val_mse": float(best_val),
                "train_updates_used": int(updates_used),
                "model_class": self.last_class,
            },
            ckpt_path,
        )
        self._saved_ckpt_path = ckpt_path

        train_state = {
            "status": "ok",
            "best_step": int(best_step),
            "best_val_mse": float(best_val),
            "last_train_loss": float(last_train_loss) if last_train_loss is not None else None,
            "updates_used": int(updates_used),
            "max_updates": int(max_updates),
            "val_history": val_history[-20:],
        }
        train_state_path = out_ckpt_dir / "train_state.json"
        _write_json(train_state_path, train_state)
        self._train_state_path = train_state_path

        self.train_updates_used = int(updates_used)
        self._update_ledger(
            train_updates_used=int(updates_used),
            adapt_updates_used=int(self.adapt_updates_used),
            train_max_updates=int(max_updates),
        )

        return {
            "status": "ok",
            "ckpt_path": str(ckpt_path),
            "train_state_path": str(train_state_path),
            "updates_used": int(updates_used),
            "best_step": int(best_step),
        }

    def eval(
        self,
        test_dl: Any,
        ckpt_path: Optional[Union[str, Path]] = None,
        track_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._filter_obj is None or self._x_dim is None or self._y_dim is None:
            raise RuntimeError("setup() must be called before eval().")

        if ckpt_path is not None:
            self.load(str(ckpt_path))
        elif self._saved_ckpt_path is not None:
            self.load(str(self._saved_ckpt_path))

        eval_init_from_gt = bool(self._cfg.get("eval_init_from_gt", False))
        preds: List[torch.Tensor] = []
        total_n = 0
        self._filter_obj.kf_net.eval()
        with torch.no_grad():
            for bi, batch in enumerate(test_dl):
                x_raw, y_raw = _extract_batch_xy(batch)
                y = _to_tensor(y_raw, device=self.device, dtype=self.dtype)
                x = _to_tensor(x_raw, device=self.device, dtype=self.dtype)
                x0_batch = x[:, 0, :] if eval_init_from_gt else None
                pred = self._forward_batch(y_btd=y, x0_batch=x0_batch)
                validate_exact_layout(
                    pred,
                    expected=(int(y.shape[0]), int(y.shape[1]), int(self._x_dim)),
                    axis_names=("B", "T", "D"),
                    label="x_hat",
                )
                total_n += int(y.shape[0])
                if self._debug_every > 0 and ((bi % self._debug_every) == 0 or has_nonfinite(pred)):
                    logger.debug("eval batch=%s %s", bi, format_array_stats("x_hat", pred))
                preds.append(pred.detach().cpu())

        if not preds:
            raise RuntimeError("runtime_error: empty test dataloader.")
        x_hat = torch.cat(preds, dim=0).contiguous()
        validate_exact_layout(
            x_hat,
            expected=(int(total_n), int(self._T_setup or x_hat.shape[1]), int(self._x_dim)),
            axis_names=("N", "T", "D"),
            label="x_hat",
        )

        preds_path = None
        if self._artifacts_dir is not None:
            preds_path = self._artifacts_dir / "preds_test.npz"
            np.savez_compressed(preds_path, x_hat=x_hat.numpy())

        self.adapt_updates_used = 0
        self._update_ledger(
            train_updates_used=int(self.train_updates_used),
            adapt_updates_used=0,
        )

        return {
            "status": "ok",
            "x_hat": x_hat,
            "cov": None,
            "preds_path": (str(preds_path) if preds_path is not None else None),
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
        if self._filter_obj is None or self._x_dim is None or self._y_dim is None:
            raise RuntimeError("setup() must be called before predict().")

        if state0 is None and "u_seq" in kwargs:
            _ = kwargs["u_seq"]

        y = _to_tensor(y_batch, device=self.device, dtype=self.dtype)
        ctx = dict(context or {})
        if state0 is None:
            state0 = ctx.get("x0", None)
        logger.debug("predict input shape=%s layout=%s", tuple(y.shape), self.last_layout)
        x_hat = self._forward_batch(y_btd=y, x0_batch=state0)
        validate_exact_layout(
            x_hat,
            expected=(int(y.shape[0]), int(y.shape[1]), int(self._x_dim)),
            axis_names=("B", "T", "D"),
            label="x_hat",
        )
        logger.debug("predict output %s", format_array_stats("x_hat", x_hat))
        if return_cov:
            return x_hat, None
        return x_hat

    def adapt(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        budget: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Split-KalmanNet benchmark adapter currently exposes frozen-safe no-op adaptation.
        self.adapt_updates_used = 0
        self._update_ledger(
            train_updates_used=int(self.train_updates_used),
            adapt_updates_used=0,
            adapt_updates_per_step={},
        )
        return {"status": "ok", "adapt_updates_used": 0, "adapt_updates_per_step": {}}

    def load(self, ckpt_path: str) -> None:
        if self._filter_obj is None:
            raise RuntimeError("setup() must be called before load().")
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._filter_obj.kf_net.load_state_dict(state, strict=True)
        self._filter_obj.kf_net.to(self.device)
        self._filter_obj.kf_net.eval()

    def save(self, out_dir: Union[str, Path]) -> Dict[str, Any]:
        if self._filter_obj is None:
            raise RuntimeError("setup() must be called before save().")
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        ckpt_path = out / "model.pt"
        torch.save({"state_dict": self._filter_obj.kf_net.state_dict()}, ckpt_path)
        self._saved_ckpt_path = ckpt_path
        return {"ckpt_path": str(ckpt_path)}

    def get_adapter_meta(self) -> Dict[str, Any]:
        return {
            "adapter_id": "split_knet",
            "adapter_version": "s8_gpu_device_v1",
            "runtime_device": str(self.device),
            "integration_mode": "import_model_only",
            "covariance_support": False,
            "input_layout_bench": "BTD",
            "internal_layout_repo": "stepwise_colvec",
            "class_path": self._filter_class_path,
            "entrypoints": {
                "filter": "GSSFiltering/filtering.py::Split_KalmanNet_Filter",
                "network": "GSSFiltering/dnn.py::DNN_SKalmanNet_GSS",
                "script_train_ref": "(SyntheticNL) main.py",
                "script_eval_ref": "GSSFiltering/tester.py::Tester",
            },
            "assumptions": {
                "A_input_layout": "repo filtering is step-wise [Dy,1] per time step; adapter converts BTD->stepwise",
                "B_cli_bias": "repo mains are config.ini/script-centric; bench uses import-mode class reuse",
                "C_adapt_support": "no principled test-time adaptation exposed for FAIRNESS budgeted track in this adapter",
            },
            "how_to_verify": {
                "A_input_layout": "third_party/Split_KalmanNet/GSSFiltering/filtering.py::Split_KalmanNet_Filter.filtering",
                "B_cli_bias": "third_party/Split_KalmanNet/(SyntheticNL) main.py and config.ini",
                "C_adapt_support": "search in third_party/Split_KalmanNet for online update/adapt routines beyond offline Trainer",
            },
        }

    def _forward_batch(
        self,
        *,
        y_btd: torch.Tensor,
        x0_batch: Optional[Any],
    ) -> torch.Tensor:
        if self._filter_obj is None or self._x_dim is None or self._y_dim is None:
            raise RuntimeError("Adapter/model is not initialized.")
        if y_btd.ndim != 3:
            raise ValueError(f"shape_mismatch: expected y [B,T,Dy], got {tuple(y_btd.shape)}")
        bsz, t_len, y_dim = y_btd.shape
        if int(y_dim) != int(self._y_dim):
            raise ValueError(f"shape_mismatch: got y_dim={y_dim}, expected {self._y_dim}")

        if x0_batch is None:
            x0 = torch.zeros(bsz, self._x_dim, 1, device=self.device, dtype=self.dtype)
        else:
            x0 = _resolve_x0_batch(
                x0_batch,
                batch_size=int(bsz),
                x_dim=int(self._x_dim),
                device=self.device,
                dtype=self.dtype,
            )

        preds: List[torch.Tensor] = []
        seq_norms: List[float] = []
        for bi in range(int(bsz)):
            y_seq = y_btd[bi]  # [T,Dy]
            x0_col = x0[bi]    # [Dx,1]
            pred_seq = self._rollout_one(y_td=y_seq, x0_col=x0_col)
            seq_norms.append(float(torch.linalg.norm(pred_seq).detach().cpu().item()))
            if self._debug_every > 0 and ((bi % self._debug_every) == 0 or has_nonfinite(pred_seq)):
                logger.debug("forward batch_item=%s %s", bi, format_array_stats("pred_seq", pred_seq))
            preds.append(pred_seq)
        self._runtime_diag = {"seq_norms": np.asarray(seq_norms, dtype=np.float32)}
        x_hat = torch.stack(preds, dim=0).contiguous()
        validate_exact_layout(
            x_hat,
            expected=(int(bsz), int(t_len), int(self._x_dim)),
            axis_names=("B", "T", "D"),
            label="x_hat",
        )
        return x_hat  # [B,T,Dx]

    def _rollout_one(self, *, y_td: torch.Tensor, x0_col: torch.Tensor) -> torch.Tensor:
        if self._filter_obj is None or self._x_dim is None or self._y_dim is None:
            raise RuntimeError("Adapter/model is not initialized.")
        if y_td.ndim != 2:
            raise ValueError(f"shape_mismatch: expected y_td [T,Dy], got {tuple(y_td.shape)}")
        if y_td.shape[1] != self._y_dim:
            raise ValueError(f"shape_mismatch: expected Dy={self._y_dim}, got {y_td.shape[1]}")

        t_len = int(y_td.shape[0])
        if t_len <= 0:
            raise ValueError("shape_mismatch: sequence length T must be > 0")

        # Reset hidden/history for each sequence rollout.
        self._filter_obj.reset(clean_history=True)
        self._filter_obj.state_post = x0_col.reshape(self._x_dim, 1).clone()
        self._filter_obj.state_history = self._filter_obj.state_post.clone()
        self._filter_obj.dnn_first = True

        for t in range(1, t_len):
            obs_t = y_td[t].reshape(self._y_dim, 1)
            self._filter_obj.filtering(obs_t)

        x_hist = self._filter_obj.state_history
        if not isinstance(x_hist, torch.Tensor):
            raise TypeError(f"runtime_error: state_history must be Tensor, got {type(x_hist)}")
        if x_hist.ndim != 2 or x_hist.shape[0] != self._x_dim:
            raise ValueError(f"shape_mismatch: unexpected state_history shape={tuple(x_hist.shape)}")
        if x_hist.shape[1] < t_len:
            raise ValueError(
                f"shape_mismatch: state_history too short. got={x_hist.shape[1]} expected_at_least={t_len}"
            )
        return x_hist[:, -t_len:].transpose(0, 1).contiguous()  # [T,Dx]

    def get_runtime_diagnostics(self) -> Dict[str, Any]:
        return dict(self._runtime_diag)

    @torch.no_grad()
    def _compute_validation_loss(
        self,
        *,
        val_dl: Any,
        loss_fn: torch.nn.Module,
        max_batches: int,
        init_from_gt: bool,
    ) -> float:
        if self._filter_obj is None:
            raise RuntimeError("Model is not initialized.")
        self._filter_obj.kf_net.eval()
        losses: List[float] = []

        for bi, batch in enumerate(val_dl):
            if max_batches > 0 and bi >= max_batches:
                break
            x_raw, y_raw = _extract_batch_xy(batch)
            x = _to_tensor(x_raw, device=self.device, dtype=self.dtype)
            y = _to_tensor(y_raw, device=self.device, dtype=self.dtype)
            x0_batch = x[:, 0, :] if init_from_gt else None
            pred = self._forward_batch(y_btd=y, x0_batch=x0_batch)
            if x.shape[1] > 1:
                loss = loss_fn(pred[:, 1:, :], x[:, 1:, :])
            else:
                loss = loss_fn(pred, x)
            if not torch.isfinite(loss):
                raise FloatingPointError("train_nan: non-finite validation loss.")
            losses.append(float(loss.item()))

        self._filter_obj.kf_net.train()
        if not losses:
            return float("inf")
        return float(sum(losses) / len(losses))

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
            },
        )

    def _update_ledger(
        self,
        *,
        train_updates_used: int,
        adapt_updates_used: int,
        train_max_updates: Optional[int] = None,
        adapt_updates_per_step: Optional[Dict[str, int]] = None,
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
        if adapt_updates_per_step is not None:
            current["adapt_updates_per_step"] = {str(k): int(v) for k, v in adapt_updates_per_step.items()}
        current["track_id"] = self._run_ctx.get("track_id")
        current["init_id"] = self._run_ctx.get("init_id")
        _write_json(self._ledger_path, current)
