from __future__ import annotations

import contextlib
import csv
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

# Control-plane observability. `active_observer()` returns a NullObserver unless a
# control-plane worker has installed one, so these calls are no-ops under the
# existing CLI and add no dependency to it. The fallback keeps the adapter
# importable if bench.control is unavailable.
try:
    from bench.control.events.observer import active_observer  # type: ignore
except Exception:  # pragma: no cover - control plane is optional

    def active_observer():  # type: ignore[misc]
        class _Null:
            def status(self, *args: Any, **kwargs: Any) -> None: ...
            def metric(self, *args: Any, **kwargs: Any) -> None: ...
            def log(self, *args: Any, **kwargs: Any) -> None: ...
            def artifact(self, *args: Any, **kwargs: Any) -> None: ...

        return _Null()


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
        self._train_diag_history: List[Dict[str, Any]] = []
        self._clip_applied_count: int = 0
        self._emit_viz_diagnostics: bool = False

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
        self._train_diag_history = []
        self._clip_applied_count = 0
        self._emit_viz_diagnostics = False

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
        h_np = H_t.detach().cpu().numpy()
        meta_assumed = meta.get("ssm", {}).get("assumed", {}) if isinstance(meta.get("ssm", {}), dict) else {}
        if not isinstance(meta_assumed, dict):
            meta_assumed = {}
        meta_h = meta_assumed.get("H")
        meta_h_arr = None
        if meta_h is not None:
            try:
                meta_h_arr = np.asarray(meta_h, dtype=np.float64)
            except Exception:
                meta_h_arr = None
        self._runtime_diag.update(
            {
                "runtime_H_name": str(
                    meta_assumed.get("measurement_model")
                    or meta_assumed.get("h_type")
                    or meta_assumed.get("measurement_h_type")
                    or ("identity_fallback" if H is None else "provided_H")
                ),
                "h_model_name": str(meta_assumed.get("h_type") or meta_assumed.get("measurement_model") or "unknown"),
                "runtime_H_shape": [int(v) for v in h_np.shape],
                "runtime_H_rank": int(np.linalg.matrix_rank(h_np)),
                "runtime_H_is_identity": bool(
                    h_np.shape[0] == h_np.shape[1] and np.allclose(h_np, np.eye(h_np.shape[0], dtype=np.float64))
                ),
                "runtime_H_matches_metadata": bool(
                    meta_h_arr is not None and meta_h_arr.shape == h_np.shape and np.allclose(meta_h_arr, h_np)
                ),
            }
        )

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

    def transform_measurements(
        self,
        y_btd: torch.Tensor,
        *,
        x_btd: Optional[torch.Tensor] = None,
        batch: Optional[Dict[str, Any]] = None,
        phase: str = "eval",
    ) -> torch.Tensor:
        """
        Optional measurement hook for wrappers.

        The base Split-KalmanNet adapter must remain a raw-measurement path, so
        the default implementation is identity. Subclasses may override this to
        provide causal measurement preprocessing before the third-party filter
        sees observations.
        """
        _ = x_btd, batch, phase
        return y_btd

    def measurement_extra_loss(
        self,
        *,
        x_btd: torch.Tensor,
        y_raw_btd: torch.Tensor,
        y_model_btd: torch.Tensor,
        batch: Optional[Dict[str, Any]] = None,
        phase: str = "train",
    ) -> torch.Tensor:
        _ = x_btd, y_raw_btd, y_model_btd, batch, phase
        return y_raw_btd.new_zeros(())

    def state_estimation_loss(
        self,
        *,
        pred_btd: torch.Tensor,
        x_btd: torch.Tensor,
        batch: Optional[Dict[str, Any]],
        phase: str,
        loss_fn: torch.nn.Module,
    ) -> torch.Tensor:
        """Compute the state loss; wrappers may override without changing the trainer."""
        _ = batch, phase
        if x_btd.shape[1] > 1:
            return loss_fn(pred_btd[:, 1:, :], x_btd[:, 1:, :])
        return loss_fn(pred_btd, x_btd)

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
        max_grad_norm = self._cfg.get("gradient_clip_norm", self._cfg.get("max_grad_norm", 10.0))
        max_grad_norm_f = float(max_grad_norm) if max_grad_norm is not None else None
        val_max_batches = int(self._cfg.get("val_max_batches", 0))
        train_init_from_gt = bool(self._cfg.get("train_init_from_gt", True))
        train_diag_every = int(self._cfg.get("train_diagnostics_every", self._debug_every or 0) or 0)

        params = [p for p in self._filter_obj.kf_net.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("runtime_error: split_knet has no trainable parameters.")

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        loss_fn = torch.nn.MSELoss(reduction="mean")

        _observer = active_observer()
        # One optimizer slot, not the paper's alternating scheme — the capability
        # declaration in bench/control/capabilities.py records this deviation.
        _observer.status(
            "PHASE_START",
            phase="train",
            message=f"Split-KalmanNet training: max_updates={max_updates} lr={lr} optimizer_slots=['main']",
        )

        self._filter_obj.kf_net.train()
        updates_used = 0
        best_step = 0
        best_val = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in self._filter_obj.kf_net.state_dict().items()}
        no_improve_evals = 0
        last_train_loss = None
        val_history: List[Dict[str, float]] = []
        self._train_diag_history = []
        self._clip_applied_count = 0

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

                y_model = self.transform_measurements(y, x_btd=x, batch=batch, phase="train")
                validate_exact_layout(
                    y_model,
                    expected=(int(y.shape[0]), int(y.shape[1]), int(self._y_dim)),
                    axis_names=("B", "T", "D"),
                    label="y_model",
                )

                optimizer.zero_grad(set_to_none=True)
                pred = self._forward_batch(y_btd=y_model, x0_batch=x0_batch)
                loss = self.state_estimation_loss(
                    pred_btd=pred,
                    x_btd=x,
                    batch=batch,
                    phase="train",
                    loss_fn=loss_fn,
                )
                loss = loss + self.measurement_extra_loss(
                    x_btd=x,
                    y_raw_btd=y,
                    y_model_btd=y_model,
                    batch=batch,
                    phase="train",
                )
                if not torch.isfinite(loss):
                    diag = self._make_train_diag_row(
                        update=updates_used,
                        phase="loss",
                        loss=loss,
                        x=x,
                        y=y,
                        y_model=y_model,
                        pred=pred,
                        params=params,
                        batch=batch,
                        grad_total_norm=None,
                        clip_total_norm=None,
                        clip_applied=False,
                    )
                    self._append_train_diag(diag)
                    self._write_train_nan_dump(
                        update=updates_used,
                        phase="loss",
                        x=x,
                        y=y,
                        y_model=y_model,
                        pred=pred,
                        loss=loss,
                        batch=batch,
                        diag=diag,
                    )
                    raise FloatingPointError(f"train_nan: non-finite training loss at update={updates_used}")

                loss.backward()
                grad_stats_pre = self._grad_stats(params)
                if not bool(grad_stats_pre["all_finite"]):
                    diag = self._make_train_diag_row(
                        update=updates_used,
                        phase="grad",
                        loss=loss,
                        x=x,
                        y=y,
                        y_model=y_model,
                        pred=pred,
                        params=params,
                        batch=batch,
                        grad_total_norm=float(grad_stats_pre["total_norm"]),
                        clip_total_norm=None,
                        clip_applied=False,
                    )
                    self._append_train_diag(diag)
                    self._write_train_nan_dump(
                        update=updates_used,
                        phase="grad",
                        x=x,
                        y=y,
                        y_model=y_model,
                        pred=pred,
                        loss=loss,
                        batch=batch,
                        diag=diag,
                    )
                    raise FloatingPointError(f"train_nan: non-finite gradient at update={updates_used}")

                clip_total_norm: Optional[float] = None
                clip_applied = False
                if max_grad_norm_f is not None and max_grad_norm_f > 0:
                    total_norm_t = torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm_f)
                    clip_total_norm = float(total_norm_t.detach().cpu().item())
                    clip_applied = bool(clip_total_norm > max_grad_norm_f)
                    if clip_applied:
                        self._clip_applied_count += 1

                if updates_used >= max_updates:
                    raise RuntimeError(f"budget_overflow: attempted optimizer.step beyond max_updates={max_updates}")
                optimizer.step()
                param_stats = self._param_stats(params)
                if not bool(param_stats["all_finite"]):
                    diag = self._make_train_diag_row(
                        update=updates_used,
                        phase="param",
                        loss=loss,
                        x=x,
                        y=y,
                        y_model=y_model,
                        pred=pred,
                        params=params,
                        batch=batch,
                        grad_total_norm=float(grad_stats_pre["total_norm"]),
                        clip_total_norm=clip_total_norm,
                        clip_applied=clip_applied,
                    )
                    self._append_train_diag(diag)
                    self._write_train_nan_dump(
                        update=updates_used,
                        phase="param",
                        x=x,
                        y=y,
                        y_model=y_model,
                        pred=pred,
                        loss=loss,
                        batch=batch,
                        diag=diag,
                    )
                    raise FloatingPointError(f"train_nan: non-finite parameter at update={updates_used}")
                updates_used += 1
                last_train_loss = float(loss.detach().item())
                _observer.metric(
                    "loss/train_total",
                    last_train_loss,
                    step=int(updates_used),
                    phase="train",
                    unit="mse",
                )
                if (
                    updates_used == 1
                    or updates_used == max_updates
                    or (train_diag_every > 0 and (updates_used % train_diag_every) == 0)
                ):
                    self._append_train_diag(
                        self._make_train_diag_row(
                            update=updates_used,
                            phase="train",
                            loss=loss,
                            x=x,
                            y=y,
                            y_model=y_model,
                            pred=pred,
                            params=params,
                            batch=batch,
                            grad_total_norm=float(grad_stats_pre["total_norm"]),
                            clip_total_norm=clip_total_norm,
                            clip_applied=clip_applied,
                        )
                    )

                should_eval = (updates_used % eval_interval == 0) or (updates_used == max_updates)
                if should_eval:
                    val_loss = self._compute_validation_loss(
                        val_dl=val_dl,
                        loss_fn=loss_fn,
                        max_batches=val_max_batches,
                        init_from_gt=train_init_from_gt,
                    )
                    val_history.append({"step": float(updates_used), "val_mse": float(val_loss)})
                    _observer.metric(
                        "loss/validation_total",
                        float(val_loss),
                        step=int(updates_used),
                        phase="validation",
                        unit="mse",
                    )
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
        _observer.status(
            "PHASE_END",
            phase="train",
            message=f"training finished: updates_used={updates_used} best_step={best_step}",
        )
        _observer.artifact(kind="checkpoint_weights", uri=str(ckpt_path))

        train_state = {
            "status": "ok",
            "best_step": int(best_step),
            "best_val_mse": float(best_val),
            "last_train_loss": float(last_train_loss) if last_train_loss is not None else None,
            "updates_used": int(updates_used),
            "max_updates": int(max_updates),
            "gradient_clip_norm": max_grad_norm_f,
            "clip_applied_count": int(self._clip_applied_count),
            "val_history": val_history[-20:],
            "train_diagnostics_tail": self._train_diag_history[-50:],
        }
        train_state_path = out_ckpt_dir / "train_state.json"
        _write_json(train_state_path, train_state)
        self._train_state_path = train_state_path
        train_diag_path = self._write_train_diagnostics_csv()

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
            "train_diagnostics_path": str(train_diag_path) if train_diag_path is not None else None,
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
        diagnostics: Optional[Dict[str, List[np.ndarray]]] = None
        if self._emit_viz_diagnostics:
            diagnostics = {
                "innov": [],
                "innov_valid": [],
                "gain": [],
                "gain_g1": [],
                "gain_g2": [],
            }
        total_n = 0
        self._filter_obj.kf_net.eval()
        with torch.no_grad():
            for bi, batch in enumerate(test_dl):
                x_raw, y_raw = _extract_batch_xy(batch)
                y = _to_tensor(y_raw, device=self.device, dtype=self.dtype)
                x = _to_tensor(x_raw, device=self.device, dtype=self.dtype)
                x0_batch = x[:, 0, :] if eval_init_from_gt else None
                y_model = self.transform_measurements(y, x_btd=x, batch=batch, phase="eval")
                validate_exact_layout(
                    y_model,
                    expected=(int(y.shape[0]), int(y.shape[1]), int(self._y_dim)),
                    axis_names=("B", "T", "D"),
                    label="y_model",
                )
                pred = self._forward_batch(y_btd=y_model, x0_batch=x0_batch)
                if self._emit_viz_diagnostics:
                    if diagnostics is None:
                        raise RuntimeError("runtime_error: Split-KalmanNet diagnostic history is unavailable")
                    batch_diag = self._runtime_diag.get("viz_diagnostics")
                    if not isinstance(batch_diag, dict):
                        raise RuntimeError("runtime_error: Split-KalmanNet diagnostics were not captured")
                    for key in diagnostics:
                        value = batch_diag.get(key)
                        if not isinstance(value, np.ndarray):
                            raise RuntimeError(f"runtime_error: Split-KalmanNet diagnostic {key!r} is missing")
                        diagnostics[key].append(value)
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

        self.adapt_updates_used = 0
        self._update_ledger(
            train_updates_used=int(self.train_updates_used),
            adapt_updates_used=0,
        )

        result: Dict[str, Any] = {
            "status": "ok",
            "x_hat": x_hat,
            "cov": None,
            "preds_path": None,
        }
        if self._emit_viz_diagnostics:
            if diagnostics is None:
                raise RuntimeError("runtime_error: Split-KalmanNet diagnostic history is unavailable")
            result["diagnostics"] = {
                key: np.concatenate(values, axis=0)
                for key, values in diagnostics.items()
            }
        return result

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
        y_model = self.transform_measurements(y, x_btd=None, phase="predict")
        validate_exact_layout(
            y_model,
            expected=(int(y.shape[0]), int(y.shape[1]), int(self._y_dim)),
            axis_names=("B", "T", "D"),
            label="y_model",
        )
        x_hat = self._forward_batch(y_btd=y_model, x0_batch=state0)
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
            "viz_diagnostics_version": "split_gain_components_v1",
            "diagnostic_semantics": {
                "gain": "learned_combined_kalman_gain",
                "gain_g1": "learned_split_factor_g1",
                "gain_g2": "learned_split_factor_g2",
                "innov": "measurement_residual_used_by_update",
                "validity_mask": "innov_valid",
                "combination": "gain_g1 @ H.T @ gain_g2",
                "observation_jacobian": "fixed_system_info_H_not_stored_per_trajectory",
            },
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

    def supports_viz_diagnostics(self) -> bool:
        return True

    def set_viz_diagnostics_enabled(self, enabled: bool) -> None:
        self._emit_viz_diagnostics = bool(enabled)

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

        self._runtime_diag.pop("viz_diagnostics", None)
        preds: List[torch.Tensor] = []
        seq_norms: List[float] = []
        diagnostic_rows: Optional[Dict[str, List[np.ndarray]]] = None
        if self._emit_viz_diagnostics:
            diagnostic_rows = {
                "innov": [],
                "innov_valid": [],
                "gain": [],
                "gain_g1": [],
                "gain_g2": [],
            }
        for bi in range(int(bsz)):
            y_seq = y_btd[bi]  # [T,Dy]
            x0_col = x0[bi]    # [Dx,1]
            pred_seq = self._rollout_one(y_td=y_seq, x0_col=x0_col)
            if self._emit_viz_diagnostics:
                if diagnostic_rows is None:
                    raise RuntimeError("runtime_error: Split-KalmanNet diagnostic rows are unavailable")
                sequence_diag = self._runtime_diag.pop("_viz_sequence_diagnostics", None)
                if not isinstance(sequence_diag, dict):
                    raise RuntimeError("runtime_error: missing per-sequence Split-KalmanNet diagnostics")
                for key in diagnostic_rows:
                    value = sequence_diag.get(key)
                    if not isinstance(value, np.ndarray):
                        raise RuntimeError(f"runtime_error: missing per-sequence diagnostic {key!r}")
                    diagnostic_rows[key].append(value)
            seq_norms.append(float(torch.linalg.norm(pred_seq).detach().cpu().item()))
            if self._debug_every > 0 and ((bi % self._debug_every) == 0 or has_nonfinite(pred_seq)):
                logger.debug("forward batch_item=%s %s", bi, format_array_stats("pred_seq", pred_seq))
            preds.append(pred_seq)
        self._runtime_diag.update({"seq_norms": np.asarray(seq_norms, dtype=np.float32)})
        if self._emit_viz_diagnostics:
            if diagnostic_rows is None:
                raise RuntimeError("runtime_error: Split-KalmanNet diagnostic rows are unavailable")
            self._runtime_diag["viz_diagnostics"] = {
                key: np.stack(values, axis=0)
                for key, values in diagnostic_rows.items()
            }
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

        sequence_diag: Optional[Dict[str, np.ndarray]] = None
        captures: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        hook_handle = None
        if self._emit_viz_diagnostics:
            sequence_diag = {
                "innov": np.full((t_len, self._y_dim), np.nan, dtype=np.float32),
                "innov_valid": np.zeros((t_len,), dtype=np.bool_),
                "gain": np.full((t_len, self._x_dim, self._y_dim), np.nan, dtype=np.float32),
                "gain_g1": np.full((t_len, self._x_dim, self._x_dim), np.nan, dtype=np.float32),
                "gain_g2": np.full((t_len, self._y_dim, self._y_dim), np.nan, dtype=np.float32),
            }

            def _capture_split_factors(_module: Any, inputs: Tuple[Any, ...], output: Any) -> None:
                if len(inputs) != 6 or not isinstance(output, (tuple, list)) or len(output) != 2:
                    raise RuntimeError("runtime_error: unexpected Split-KalmanNet factor interface")
                innovation = inputs[1]
                h_flat = inputs[5]
                gain_g1, gain_g2 = output
                tensors = (innovation, h_flat, gain_g1, gain_g2)
                if not all(isinstance(value, torch.Tensor) for value in tensors):
                    raise RuntimeError("runtime_error: Split-KalmanNet diagnostics must be tensors")
                h_matrix = h_flat.reshape(self._y_dim, self._x_dim)
                with torch.no_grad():
                    combined_gain = gain_g1 @ h_matrix.transpose(0, 1) @ gain_g2
                captures.append(
                    (
                        innovation.reshape(self._y_dim).detach().cpu().numpy().astype(np.float32, copy=True),
                        combined_gain.detach().cpu().numpy().astype(np.float32, copy=True),
                        gain_g1.detach().cpu().numpy().astype(np.float32, copy=True),
                        gain_g2.detach().cpu().numpy().astype(np.float32, copy=True),
                    )
                )

            hook_handle = self._filter_obj.kf_net.register_forward_hook(_capture_split_factors)

        try:
            for t in range(1, t_len):
                obs_t = y_td[t].reshape(self._y_dim, 1)
                captured_before = len(captures)
                self._filter_obj.filtering(obs_t)
                if sequence_diag is not None:
                    if len(captures) != captured_before + 1:
                        raise RuntimeError(f"runtime_error: expected one Split-KalmanNet capture at t={t}")
                    innov_t, gain_t, gain_g1_t, gain_g2_t = captures[-1]
                    sequence_diag["innov"][t] = innov_t
                    sequence_diag["innov_valid"][t] = True
                    sequence_diag["gain"][t] = gain_t
                    sequence_diag["gain_g1"][t] = gain_g1_t
                    sequence_diag["gain_g2"][t] = gain_g2_t
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if sequence_diag is not None:
            self._runtime_diag["_viz_sequence_diagnostics"] = sequence_diag

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
        out = dict(self._runtime_diag)
        out.update(
            {
                "train_diagnostics": list(self._train_diag_history[-50:]),
                "clip_applied_count": int(self._clip_applied_count),
            }
        )
        return out

    @staticmethod
    def _tensor_norm_stats(tensor: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            t = tensor.detach()
            finite = torch.isfinite(t)
            if not bool(finite.any()):
                return {
                    "norm_mean": float("nan"),
                    "norm_max": float("nan"),
                    "max_abs": float("nan"),
                    "finite_count": 0.0,
                    "nonfinite_count": float(t.numel()),
                }
            vals = t[finite]
            max_abs = float(torch.max(torch.abs(vals)).detach().cpu().item())
            if t.ndim >= 2:
                try:
                    norms = torch.linalg.norm(t.float(), dim=-1)
                    norm_mean = float(torch.mean(norms).detach().cpu().item())
                    norm_max = float(torch.max(norms).detach().cpu().item())
                except Exception:
                    norm = float(torch.linalg.norm(vals.float()).detach().cpu().item())
                    norm_mean = norm
                    norm_max = norm
            else:
                norm = float(torch.linalg.norm(vals.float()).detach().cpu().item())
                norm_mean = norm
                norm_max = norm
            return {
                "norm_mean": norm_mean,
                "norm_max": norm_max,
                "max_abs": max_abs,
                "finite_count": float(torch.count_nonzero(finite).detach().cpu().item()),
                "nonfinite_count": float(t.numel() - torch.count_nonzero(finite).detach().cpu().item()),
            }

    @staticmethod
    def _param_stats(params: Sequence[torch.nn.Parameter]) -> Dict[str, float | bool]:
        total_sq = 0.0
        max_abs = 0.0
        all_finite = True
        count = 0
        for p in params:
            data = p.detach()
            finite = torch.isfinite(data)
            all_finite = all_finite and bool(finite.all())
            if bool(finite.any()):
                vals = data[finite].float()
                total_sq += float(torch.sum(vals * vals).detach().cpu().item())
                max_abs = max(max_abs, float(torch.max(torch.abs(vals)).detach().cpu().item()))
                count += int(vals.numel())
        return {
            "total_norm": float(total_sq ** 0.5),
            "max_abs": float(max_abs),
            "all_finite": bool(all_finite),
            "finite_count": float(count),
        }

    @staticmethod
    def _grad_stats(params: Sequence[torch.nn.Parameter]) -> Dict[str, float | bool]:
        total_sq = 0.0
        max_abs = 0.0
        all_finite = True
        count = 0
        has_grad = False
        for p in params:
            if p.grad is None:
                continue
            has_grad = True
            grad = p.grad.detach()
            finite = torch.isfinite(grad)
            all_finite = all_finite and bool(finite.all())
            if bool(finite.any()):
                vals = grad[finite].float()
                total_sq += float(torch.sum(vals * vals).detach().cpu().item())
                max_abs = max(max_abs, float(torch.max(torch.abs(vals)).detach().cpu().item()))
                count += int(vals.numel())
        return {
            "total_norm": float(total_sq ** 0.5),
            "max_abs": float(max_abs),
            "all_finite": bool(all_finite),
            "finite_count": float(count),
            "has_grad": bool(has_grad),
        }

    def _mask_stats(self, batch: Any, *, like: torch.Tensor) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not isinstance(batch, dict):
            return out
        for key in ("ref_mask_seq", "measurement_mask_seq"):
            if key not in batch:
                continue
            try:
                value = _to_tensor(batch[key], device=like.device, dtype=like.dtype)
            except Exception:
                continue
            out[f"{key}_mean"] = float(value.detach().mean().cpu().item())
            out[f"{key}_min"] = float(value.detach().min().cpu().item())
            out[f"{key}_max"] = float(value.detach().max().cpu().item())
            out[f"{key}_finite"] = float(torch.isfinite(value).float().mean().detach().cpu().item())
        return out

    def _make_train_diag_row(
        self,
        *,
        update: int,
        phase: str,
        loss: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        y_model: torch.Tensor,
        pred: torch.Tensor,
        params: Sequence[torch.nn.Parameter],
        batch: Any,
        grad_total_norm: Optional[float],
        clip_total_norm: Optional[float],
        clip_applied: bool,
    ) -> Dict[str, Any]:
        x_stats = self._tensor_norm_stats(x)
        y_stats = self._tensor_norm_stats(y)
        y_model_stats = self._tensor_norm_stats(y_model)
        pred_stats = self._tensor_norm_stats(pred)
        residual_stats_local = self._tensor_norm_stats(pred - x)
        param_stats = self._param_stats(params)
        row: Dict[str, Any] = {
            "update": int(update),
            "phase": str(phase),
            "train_loss": float(loss.detach().cpu().item()) if loss.numel() == 1 else float("nan"),
            "loss_finite": bool(torch.isfinite(loss).all()),
            "grad_norm_total": grad_total_norm,
            "clip_total_norm": clip_total_norm,
            "clip_applied": bool(clip_applied),
            "clip_applied_count": int(self._clip_applied_count),
            "param_norm_total": float(param_stats["total_norm"]),
            "max_abs_param": float(param_stats["max_abs"]),
            "param_all_finite": bool(param_stats["all_finite"]),
            "x_norm_mean": x_stats["norm_mean"],
            "x_norm_max": x_stats["norm_max"],
            "x_max_abs": x_stats["max_abs"],
            "x_nonfinite_count": x_stats["nonfinite_count"],
            "y_norm_mean": y_stats["norm_mean"],
            "y_norm_max": y_stats["norm_max"],
            "y_max_abs": y_stats["max_abs"],
            "y_nonfinite_count": y_stats["nonfinite_count"],
            "y_model_norm_mean": y_model_stats["norm_mean"],
            "y_model_norm_max": y_model_stats["norm_max"],
            "pred_norm_mean": pred_stats["norm_mean"],
            "pred_norm_max": pred_stats["norm_max"],
            "pred_max_abs": pred_stats["max_abs"],
            "pred_nonfinite_count": pred_stats["nonfinite_count"],
            "residual_norm_mean": residual_stats_local["norm_mean"],
            "residual_norm_max": residual_stats_local["norm_max"],
            "max_abs_grad": float(self._grad_stats(params)["max_abs"]),
        }
        if y.shape[-1] >= 9:
            y_ref = y[..., 6:9]
            y_ref_stats = self._tensor_norm_stats(y_ref)
            row.update(
                {
                    "y_ref_norm_mean": y_ref_stats["norm_mean"],
                    "y_ref_norm_max": y_ref_stats["norm_max"],
                    "y_ref_max_abs": y_ref_stats["max_abs"],
                }
            )
        row.update(self._mask_stats(batch, like=y))
        return row

    def _append_train_diag(self, row: Dict[str, Any]) -> None:
        self._train_diag_history.append(dict(row))
        if self._debug_every > 0:
            logger.debug(
                "train diag update=%s phase=%s loss=%s grad_norm=%s clip=%s pred_norm_max=%s residual_norm_max=%s",
                row.get("update"),
                row.get("phase"),
                row.get("train_loss"),
                row.get("grad_norm_total"),
                row.get("clip_applied"),
                row.get("pred_norm_max"),
                row.get("residual_norm_max"),
            )

    def _write_train_diagnostics_csv(self) -> Optional[Path]:
        if self._run_dir is None or not self._train_diag_history:
            return None
        out = self._run_dir / "diagnostics" / "training_diagnostics.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        keys: List[str] = []
        for row in self._train_diag_history:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._train_diag_history)
        self._runtime_diag["train_diagnostics_path"] = str(out)
        return out

    def _write_train_nan_dump(
        self,
        *,
        update: int,
        phase: str,
        x: torch.Tensor,
        y: torch.Tensor,
        y_model: torch.Tensor,
        pred: torch.Tensor,
        loss: torch.Tensor,
        batch: Any,
        diag: Dict[str, Any],
    ) -> None:
        if self._run_dir is None:
            return
        diag_dir = self._run_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        arrays: Dict[str, Any] = {
            "x": x.detach().cpu().numpy(),
            "y": y.detach().cpu().numpy(),
            "y_model": y_model.detach().cpu().numpy(),
            "x_hat": pred.detach().cpu().numpy(),
            "x_finite": torch.isfinite(x).detach().cpu().numpy(),
            "y_finite": torch.isfinite(y).detach().cpu().numpy(),
            "y_model_finite": torch.isfinite(y_model).detach().cpu().numpy(),
            "x_hat_finite": torch.isfinite(pred).detach().cpu().numpy(),
        }
        if isinstance(batch, dict):
            for key in ("ref_mask_seq", "measurement_mask_seq", "measurement_clean_y_seq", "measurement_error_seq"):
                if key in batch:
                    try:
                        arrays[key] = _to_tensor(batch[key], device=x.device, dtype=x.dtype).detach().cpu().numpy()
                    except Exception:
                        continue
        np.savez_compressed(diag_dir / f"train_nan_update_{int(update)}.npz", **arrays)

        summary = dict(diag)
        summary.update(
            {
                "update": int(update),
                "phase": str(phase),
                "loss": float(loss.detach().cpu().item()) if loss.numel() == 1 else None,
                "loss_isfinite": bool(torch.isfinite(loss).all()),
            }
        )
        _write_json(diag_dir / f"train_nan_update_{int(update)}_summary.json", summary)
        self._write_train_diagnostics_csv()

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
            y_model = self.transform_measurements(y, x_btd=x, batch=batch, phase="val")
            validate_exact_layout(
                y_model,
                expected=(int(y.shape[0]), int(y.shape[1]), int(self._y_dim)),
                axis_names=("B", "T", "D"),
                label="y_model",
            )
            pred = self._forward_batch(y_btd=y_model, x0_batch=x0_batch)
            loss = self.state_estimation_loss(
                pred_btd=pred,
                x_btd=x,
                batch=batch,
                phase="val",
                loss_fn=loss_fn,
            )
            loss = loss + self.measurement_extra_loss(
                x_btd=x,
                y_raw_btd=y,
                y_model_btd=y_model,
                batch=batch,
                phase="val",
            )
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
