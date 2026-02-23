from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


try:
    from .base import ModelAdapter  # type: ignore
except Exception:  # pragma: no cover
    class ModelAdapter:  # pragma: no cover
        pass


def _import_module_from_file(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


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
        raise ValueError("Adaptive-KNet adapter requires model_cfg['repo'] with a path.")

    bench_root = Path(__file__).resolve().parents[2]
    repo_candidate = Path(repo_path).expanduser()
    if repo_candidate.is_absolute():
        repo_root = repo_candidate.resolve()
    else:
        repo_root = (bench_root / repo_candidate).resolve()
    return repo_root, entrypoints


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


def _call_general_settings_safely(general_settings_fn: Any) -> Any:
    old_argv = list(sys.argv)
    argv0 = old_argv[0] if old_argv else "bench"
    try:
        sys.argv = [argv0]
        return general_settings_fn()
    finally:
        sys.argv = old_argv


def _ensure_repo_on_syspath(repo_root: Path) -> None:
    if not repo_root.exists():
        raise FileNotFoundError(f"Adaptive-KNet repo root not found: {repo_root}")
    p = str(repo_root)
    if p not in sys.path:
        sys.path.insert(0, p)


def _validate_model_class(candidate: Any) -> bool:
    if not isinstance(candidate, type):
        return False
    if not issubclass(candidate, torch.nn.Module):
        return False
    required = ("NNBuild", "InitSequence")
    for name in required:
        if not hasattr(candidate, name):
            return False
    if not hasattr(candidate, "init_hidden") and not hasattr(candidate, "init_hidden_KNet"):
        return False
    return True


def _resolve_x0_batch(
    context: Dict[str, Any],
    *,
    batch_size: int,
    x_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x0 = None
    for key in ("x0", "x_init", "x_init_mean", "m1x_0"):
        if key in context:
            x0 = context[key]
            break

    if x0 is None:
        return torch.zeros(batch_size, x_dim, 1, device=device, dtype=dtype)

    x = _to_tensor(x0, device=device, dtype=dtype)
    if x.ndim == 1:
        if x.shape[0] != x_dim:
            raise ValueError(f"x0 shape mismatch: expected [{x_dim}], got {tuple(x.shape)}")
        return x.view(1, x_dim, 1).repeat(batch_size, 1, 1)
    if x.ndim == 2:
        if x.shape == (x_dim, 1):
            return x.view(1, x_dim, 1).repeat(batch_size, 1, 1)
        if x.shape == (batch_size, x_dim):
            return x.unsqueeze(2)
        raise ValueError(
            f"x0 shape mismatch: expected [{x_dim},1] or [{batch_size},{x_dim}], got {tuple(x.shape)}"
        )
    if x.ndim == 3 and x.shape == (batch_size, x_dim, 1):
        return x
    raise ValueError(
        f"x0 shape mismatch: expected rank 1/2/3 with x_dim={x_dim} and batch_size={batch_size}, got {tuple(x.shape)}"
    )


def _extract_batch_xy(batch: Any) -> Tuple[Any, Any]:
    if isinstance(batch, dict):
        if "x" not in batch or "y" not in batch:
            raise KeyError("Batch dict must contain keys 'x' and 'y'.")
        return batch["x"], batch["y"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(f"Unsupported dataloader batch type: {type(batch)}")


def _extract_batch_y(batch: Any) -> Any:
    if isinstance(batch, dict):
        if "y" in batch:
            return batch["y"]
        raise KeyError("Batch dict must contain key 'y'.")
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 2:
            return batch[1]
        if len(batch) == 1:
            return batch[0]
    return batch


@dataclass
class _AdaptiveImports:
    SystemModel: Any
    config_general_settings: Any


def _load_adaptive_knet(repo_root: Path) -> _AdaptiveImports:
    _ensure_repo_on_syspath(repo_root)

    try:
        from simulations.Linear_sysmdl import SystemModel  # type: ignore
        from simulations import config as adaptive_config  # type: ignore
    except Exception as e:
        logger.warning("Normal Adaptive-KNet import failed; trying file import fallback: %r", e)
        sysmdl_file = repo_root / "simulations" / "Linear_sysmdl.py"
        cfg_file = repo_root / "simulations" / "config.py"
        if not sysmdl_file.exists():
            raise FileNotFoundError(f"Missing file: {sysmdl_file}")
        if not cfg_file.exists():
            raise FileNotFoundError(f"Missing file: {cfg_file}")
        sysmdl_mod = _import_module_from_file("adaptive_knet_sysmdl", sysmdl_file)
        cfg_mod = _import_module_from_file("adaptive_knet_config", cfg_file)
        SystemModel = getattr(sysmdl_mod, "SystemModel")
        adaptive_config = cfg_mod

    return _AdaptiveImports(
        SystemModel=SystemModel,
        config_general_settings=adaptive_config.general_settings,
    )


def _load_model_class(repo_root: Path, class_path: str) -> Any:
    _ensure_repo_on_syspath(repo_root)
    if not isinstance(class_path, str) or "." not in class_path:
        raise ValueError(
            "Adaptive-KNet adapter requires model_cfg['estimator_class_path'] like "
            "'mnets.KNet_mnet.KalmanNetNN'."
        )
    module_name, class_name = class_path.rsplit(".", 1)
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        module_file = repo_root.joinpath(*module_name.split(".")).with_suffix(".py")
        if not module_file.exists():
            raise ImportError(f"Failed to import module '{module_name}': {e}") from e
        mod = _import_module_from_file(f"adaptive_knet_{module_name.replace('.', '_')}", module_file)
    cls = getattr(mod, class_name, None)
    if not _validate_model_class(cls):
        raise TypeError(
            f"Invalid estimator class '{class_path}'. "
            "HOW TO VERIFY: third_party/Adaptive-KNet-ICASSP24/mnets/* for KalmanNetNN classes."
        )
    return cls


class AdaptiveKNetAdapter(ModelAdapter):
    """
    Route-B import-mode adapter for third_party/Adaptive-KNet-ICASSP24.

    Fixed shape policy:
    - bench input/output: [B,T,D]
    - internal sequence staging: [B,D,T]
    - per-step model input: [B,D,1]
    """

    def __init__(self) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.repo_root: Optional[Path] = None
        self.entrypoints: Dict[str, Any] = {}
        self.device: torch.device = torch.device("cpu")
        self.dtype: torch.dtype = torch.float32

        self._imports: Optional[_AdaptiveImports] = None
        self._sys_model: Any = None
        self._args: Any = None

        self._x_dim: Optional[int] = None
        self._y_dim: Optional[int] = None
        self._T_setup: Optional[int] = None
        self._H: Optional[torch.Tensor] = None

        self._cfg: Dict[str, Any] = {}
        self._run_ctx: Dict[str, Any] = {}
        self._run_dir: Optional[Path] = None
        self._ckpt_dir: Optional[Path] = None
        self._artifacts_dir: Optional[Path] = None
        self._ledger_path: Optional[Path] = None
        self._saved_ckpt_path: Optional[Path] = None
        self._train_state_path: Optional[Path] = None

        self.train_updates_used: int = 0
        self.adapt_updates_used: int = 0
        self.adapt_updates_per_step: Dict[str, int] = {}

        self.last_layout: Optional[str] = None
        self.last_class: Optional[str] = None

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
        self.repo_root, self.entrypoints = _resolve_repo_spec(cfg)
        self._imports = _load_adaptive_knet(self.repo_root)

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
            raise ValueError(
                f"shape_mismatch: Adaptive-KNet adapter expects input_layout='BTD'. got '{input_layout}'"
            )
        self.last_layout = "bench_BTD_to_repo_BDT_stepwise"

        meta = _coerce_meta_dict(system_info)
        x_dim = system_info.get("x_dim", cfg.get("x_dim", meta.get("x_dim")))
        y_dim = system_info.get("y_dim", cfg.get("y_dim", meta.get("y_dim")))
        if x_dim is None or y_dim is None:
            raise ValueError("system_info must provide x_dim and y_dim for Adaptive-KNet setup.")
        self._x_dim = int(x_dim)
        self._y_dim = int(y_dim)

        T = system_info.get("T", cfg.get("T", meta.get("T", cfg.get("sequence_length", 1))))
        self._T_setup = int(T)

        F = system_info.get("F", system_info.get("A", None))
        H = system_info.get("H", system_info.get("C", None))
        if F is None:
            F = torch.eye(self._x_dim)
        if H is None:
            H = torch.eye(self._y_dim, self._x_dim)
        F_t = _to_tensor(F, device=self.device, dtype=self.dtype)
        H_t = _to_tensor(H, device=self.device, dtype=self.dtype)
        self._H = H_t

        q2, r2 = _extract_q2_r2(cfg, system_info, meta)
        Q = system_info.get("Q", None)
        R = system_info.get("R", None)
        if Q is None:
            Q_t = torch.eye(self._x_dim, device=self.device, dtype=self.dtype) * float(q2)
        else:
            Q_t = _to_tensor(Q, device=self.device, dtype=self.dtype)
        if R is None:
            R_t = torch.eye(self._y_dim, device=self.device, dtype=self.dtype) * float(r2)
        else:
            R_t = _to_tensor(R, device=self.device, dtype=self.dtype)

        args = _call_general_settings_safely(self._imports.config_general_settings)
        args.use_cuda = (self.device.type == "cuda")
        args.T = int(self._T_setup)
        args.T_test = int(self._T_setup)
        args.n_batch = int(cfg.get("batch_size", 32))
        args.lr = float(cfg.get("lr", 1e-4))
        args.wd = float(cfg.get("weight_decay", cfg.get("wd", 1e-3)))
        args.n_steps = int(cfg.get("train_steps_hint", 1))
        args.in_mult_KNet = int(cfg.get("in_mult_KNet", getattr(args, "in_mult_KNet", 5)))
        args.out_mult_KNet = int(cfg.get("out_mult_KNet", getattr(args, "out_mult_KNet", 40)))
        args.knet_trainable = bool(cfg.get("knet_trainable", True))
        args.use_context_mod = bool(cfg.get("use_context_mod", False))
        self._args = args

        SystemModel = self._imports.SystemModel
        self._sys_model = SystemModel(F_t, Q_t, H_t, R_t, args.T, args.T_test, float(q2), float(r2))
        m1x_0 = torch.zeros(self._x_dim, 1, device=self.device, dtype=self.dtype)
        m2x_0 = torch.zeros(self._x_dim, self._x_dim, device=self.device, dtype=self.dtype)
        self._sys_model.InitSequence(m1x_0, m2x_0)

        class_path = str(
            cfg.get("estimator_class_path")
            or cfg.get("estimator_class")
            or "mnets.KNet_mnet.KalmanNetNN"
        )
        model_cls = _load_model_class(self.repo_root, class_path)
        self.model = model_cls()
        self.model.NNBuild(self._sys_model, args)  # type: ignore[attr-defined]
        self.model.to(self.device)
        self.model.eval()

        self.last_class = class_path

    def train(
        self,
        train_dl: Any,
        val_dl: Any,
        budget: Optional[Dict[str, Any]] = None,
        ckpt_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("setup() must be called before train().")
        if self._x_dim is None or self._y_dim is None:
            raise RuntimeError("Adapter dimensions are not initialized.")

        budget = dict(budget or {})
        max_updates = int(budget.get("train_max_updates", 0))
        if max_updates <= 0:
            raise ValueError("train_max_updates must be > 0 for init_id=trained.")

        out_ckpt_dir = Path(ckpt_dir).expanduser().resolve() if ckpt_dir is not None else self._ckpt_dir
        if out_ckpt_dir is None:
            raise ValueError("ckpt_dir is required when adapter has no run_dir.")
        out_ckpt_dir.mkdir(parents=True, exist_ok=True)

        lr = float(self._cfg.get("lr", getattr(self._args, "lr", 1e-4)))
        wd = float(self._cfg.get("weight_decay", self._cfg.get("wd", getattr(self._args, "wd", 1e-3))))
        eval_interval = int(self._cfg.get("val_eval_interval_updates", max(1, min(10, max_updates))))
        patience_evals = int(self._cfg.get("patience_evals", budget.get("patience_evals", 0)))
        min_delta = float(self._cfg.get("min_delta", budget.get("min_delta", 0.0)))
        max_grad_norm = self._cfg.get("max_grad_norm", None)
        max_grad_norm_f = float(max_grad_norm) if max_grad_norm is not None else None
        val_max_batches = int(self._cfg.get("val_max_batches", 0))

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(
                "runtime_error: no trainable parameters in Adaptive-KNet model. "
                "Set model_cfg.knet_trainable=true."
            )

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        loss_fn = torch.nn.MSELoss(reduction="mean")

        self.model.train()
        updates_used = 0
        best_step = 0
        best_val = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
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
                    raise ValueError(f"shape_mismatch: expected x,y rank-3; got x={tuple(x.shape)} y={tuple(y.shape)}")
                if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
                    raise ValueError(
                        f"shape_mismatch: x/y batch-time mismatch x={tuple(x.shape)} y={tuple(y.shape)}"
                    )
                if x.shape[2] != self._x_dim or y.shape[2] != self._y_dim:
                    raise ValueError(
                        f"shape_mismatch: expected x_dim={self._x_dim} y_dim={self._y_dim}; "
                        f"got x={tuple(x.shape)} y={tuple(y.shape)}"
                    )

                optimizer.zero_grad(set_to_none=True)
                pred = self._forward_sequence(y, context=None)
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
                    )
                    val_history.append({"step": float(updates_used), "val_mse": float(val_loss)})
                    if (best_val - float(val_loss)) > min_delta:
                        best_val = float(val_loss)
                        best_step = int(updates_used)
                        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                        no_improve_evals = 0
                    else:
                        no_improve_evals += 1
                    if patience_evals > 0 and no_improve_evals >= patience_evals:
                        logger.info(
                            "Early stopping Adaptive-KNet training at step=%s (patience_evals=%s)",
                            updates_used,
                            patience_evals,
                        )
                        updates_used = max_updates
                        break

        self.model.load_state_dict(best_state, strict=True)
        ckpt_path = out_ckpt_dir / "model.pt"
        torch.save(
            {
                "state_dict": self.model.state_dict(),
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
            adapt_updates_per_step=self.adapt_updates_per_step,
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
        if self.model is None:
            raise RuntimeError("setup() must be called before eval().")
        if self._x_dim is None or self._y_dim is None:
            raise RuntimeError("Adapter dimensions are not initialized.")

        if ckpt_path is not None:
            self.load(str(ckpt_path))
        elif self._saved_ckpt_path is not None:
            self.load(str(self._saved_ckpt_path))

        self.model.eval()
        preds: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in test_dl:
                _, y_raw = _extract_batch_xy(batch)
                y = _to_tensor(y_raw, device=self.device, dtype=self.dtype)
                pred = self._forward_sequence(y, context=None)
                preds.append(pred.detach().cpu())

        if not preds:
            raise RuntimeError("runtime_error: empty test dataloader.")
        x_hat = torch.cat(preds, dim=0).contiguous()

        preds_path = None
        if self._artifacts_dir is not None:
            preds_path = self._artifacts_dir / "preds_test.npz"
            np.savez_compressed(preds_path, x_hat=x_hat.numpy())

        self._update_ledger(
            train_updates_used=int(self.train_updates_used),
            adapt_updates_used=int(self.adapt_updates_used),
            adapt_updates_per_step=self.adapt_updates_per_step,
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
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        return_cov: bool = False,
        **kwargs: Any,
    ) -> Any:
        if self.model is None:
            raise RuntimeError("setup() must be called before predict().")
        if self._x_dim is None or self._y_dim is None:
            raise RuntimeError("Adapter dimensions are not initialized.")

        state0 = kwargs.get("state0", None)
        ctx = dict(context or {})
        if state0 is not None:
            ctx["x0"] = state0

        y = _to_tensor(y_seq, device=self.device, dtype=self.dtype)
        self.model.eval()
        x_hat = self._forward_sequence(y, context=ctx)
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
        if self.model is None:
            raise RuntimeError("setup() must be called before adapt().")
        if self._x_dim is None or self._y_dim is None:
            raise RuntimeError("Adapter dimensions are not initialized.")

        budget_dict = dict(budget or {})
        ctx = dict(context or {})

        t0_raw = kwargs.get("t0", budget_dict.get("t0", ctx.get("t0")))
        t0: Optional[int] = None
        if t0_raw is not None:
            t0 = int(t0_raw)

        allowed_after_t0_only = bool(
            kwargs.get(
                "allowed_after_t0_only",
                budget_dict.get("allowed_after_t0_only", ctx.get("allowed_after_t0_only", False)),
            )
        )
        max_updates = int(budget_dict.get("max_updates", 0))
        max_updates_per_step = int(budget_dict.get("max_updates_per_step", 1))
        overflow_policy = str(
            budget_dict.get("overflow_policy", self._cfg.get("adapt_overflow_policy", "fail"))
        ).lower()
        track_id = str(ctx.get("track_id", self._run_ctx.get("track_id", ""))).lower()
        adapt_lr = float(self._cfg.get("adapt_lr", budget_dict.get("adapt_lr", 1e-5)))
        adapt_wd = float(self._cfg.get("adapt_weight_decay", budget_dict.get("adapt_weight_decay", 0.0)))
        max_grad_norm = self._cfg.get("adapt_max_grad_norm", budget_dict.get("adapt_max_grad_norm", None))
        max_grad_norm_f = float(max_grad_norm) if max_grad_norm is not None else None

        if track_id and track_id != "budgeted":
            raise RuntimeError(f"runtime_error: adapt() is only valid for track_id='budgeted', got '{track_id}'")
        if max_updates < 0:
            raise ValueError(f"budget_overflow: invalid max_updates={max_updates}")
        if max_updates > 200:
            raise ValueError(f"budget_overflow: max_updates must be <= 200, got {max_updates}")
        if max_updates_per_step <= 0:
            raise ValueError(f"budget_overflow: invalid max_updates_per_step={max_updates_per_step}")
        if max_updates_per_step > 1:
            raise ValueError(
                f"budget_overflow: max_updates_per_step must be <= 1, got {max_updates_per_step}"
            )

        y_all = self._collect_y_for_adapt(y_seq)
        if y_all.ndim != 3:
            raise ValueError(f"shape_mismatch: adapt expects [N,T,Dy], got {tuple(y_all.shape)}")
        if y_all.shape[2] != self._y_dim:
            raise ValueError(
                f"shape_mismatch: adapt y_dim mismatch. got={y_all.shape[2]} expected={self._y_dim}"
            )

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(
                "runtime_error: no trainable parameters for adaptation. "
                "Set model_cfg.knet_trainable=true."
            )
        optimizer = torch.optim.Adam(params, lr=adapt_lr, weight_decay=adapt_wd)
        loss_fn = torch.nn.MSELoss(reduction="mean")

        self.model.train()
        total_updates = 0
        T = int(y_all.shape[1])
        per_step: Dict[str, int] = {str(t): 0 for t in range(T)}
        loss_history: List[Dict[str, float]] = []

        for t in range(T):
            if allowed_after_t0_only and t0 is not None and t < int(t0):
                continue

            updates_this_step = 0
            while updates_this_step < max_updates_per_step:
                if total_updates >= max_updates:
                    if overflow_policy == "stop":
                        updates_this_step = max_updates_per_step
                        break
                    raise RuntimeError(
                        f"budget_overflow: adapt max_updates={max_updates} exhausted before t={t} "
                        f"(allowed_after_t0_only={allowed_after_t0_only}, t0={t0})"
                    )

                y_step = y_all[:, t : t + 1, :]  # [N,1,Dy]
                optimizer.zero_grad(set_to_none=True)
                x_hat_step = self._forward_sequence(y_step, context=ctx)
                y_hat_step = self._project_state_to_obs(x_hat_step)
                loss = loss_fn(y_hat_step, y_step)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"train_nan: non-finite adapt loss at t={t}")
                loss.backward()
                if max_grad_norm_f is not None and max_grad_norm_f > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm_f)
                optimizer.step()
                total_updates += 1
                updates_this_step += 1
                loss_history.append({"t": float(t), "loss": float(loss.detach().item())})

                # S3 fairness: max 1 update per time step (or configured upper bound).
                if updates_this_step >= max_updates_per_step:
                    break

            per_step[str(t)] = int(updates_this_step)

            if overflow_policy == "stop" and total_updates >= max_updates:
                break

        self.model.eval()
        self.adapt_updates_used = int(total_updates)
        self.adapt_updates_per_step = per_step
        self._update_ledger(
            train_updates_used=int(self.train_updates_used),
            adapt_updates_used=int(self.adapt_updates_used),
            adapt_updates_per_step=self.adapt_updates_per_step,
            adapt_t0=t0,
            allowed_after_t0_only=bool(allowed_after_t0_only),
            adapt_max_updates=int(max_updates),
            adapt_max_updates_per_step=int(max_updates_per_step),
        )

        return {
            "status": "ok",
            "adapt_updates_used": int(total_updates),
            "adapt_updates_per_step": per_step,
            "loss_history_tail": loss_history[-10:],
            "t0": t0,
            "allowed_after_t0_only": bool(allowed_after_t0_only),
        }

    def load(self, ckpt_path: str) -> None:
        if self.model is None:
            raise RuntimeError("setup() must be called before load().")
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def save(self, out_dir: Union[str, Path]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("setup() must be called before save().")
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        ckpt_path = out / "model.pt"
        torch.save({"state_dict": self.model.state_dict()}, ckpt_path)
        self._saved_ckpt_path = ckpt_path
        return {"ckpt_path": str(ckpt_path)}

    def get_adapter_meta(self) -> Dict[str, Any]:
        class_path = str(
            self._cfg.get("estimator_class_path")
            or self._cfg.get("estimator_class")
            or "mnets.KNet_mnet.KalmanNetNN"
        )
        return {
            "adapter_id": "adaptive_knet",
            "adapter_version": "s3_route_b_v1",
            "runtime_device": str(self.device),
            "covariance_support": False,
            "input_layout_bench": "BTD",
            "internal_layout_repo": "BDT_stepwise",
            "class_path": class_path,
            "entrypoints": {
                "config": "simulations/config.py::general_settings",
                "system_model": "simulations/Linear_sysmdl.py::SystemModel",
                "pipeline_train_ref": "pipelines/Pipeline_EKF.py::Pipeline_EKF.NNTrain",
                "pipeline_eval_ref": "pipelines/Pipeline_EKF.py::Pipeline_EKF.NNTest",
                "pipeline_adapt_ref": "noise_estimator/search.py::Pipeline_NE",
            },
            "adaptation_objective": "unsupervised_observation_reconstruction_mse",
            "adapt_updates_per_step_policy": "fail_if_exceeds_max_updates_per_step",
        }

    def _forward_sequence(self, y_btd: torch.Tensor, context: Optional[Dict[str, Any]]) -> torch.Tensor:
        if self.model is None or self._x_dim is None or self._y_dim is None:
            raise RuntimeError("Adapter/model is not initialized.")

        if y_btd.ndim != 3:
            raise ValueError(f"shape_mismatch: expected y [B,T,Dy], got {tuple(y_btd.shape)}")
        B, T, y_dim = y_btd.shape
        if int(y_dim) != int(self._y_dim):
            raise ValueError(f"shape_mismatch: got y_dim={y_dim}, expected {self._y_dim}")

        y_repo = y_btd.permute(0, 2, 1).contiguous()  # [B,Dy,T]
        if hasattr(self.model, "batch_size"):
            self.model.batch_size = int(B)  # type: ignore[attr-defined]

        if hasattr(self.model, "init_hidden"):
            self.model.init_hidden()  # type: ignore[attr-defined]
        elif hasattr(self.model, "init_hidden_KNet"):
            self.model.init_hidden_KNet()  # type: ignore[attr-defined]
        else:
            raise RuntimeError("runtime_error: Adaptive-KNet model missing init_hidden/init_hidden_KNet.")

        ctx = context or {}
        x0_batch = _resolve_x0_batch(
            ctx,
            batch_size=B,
            x_dim=self._x_dim,
            device=self.device,
            dtype=self.dtype,
        )
        if not hasattr(self.model, "InitSequence"):
            raise RuntimeError("runtime_error: Adaptive-KNet model missing InitSequence().")
        self.model.InitSequence(x0_batch, int(T))  # type: ignore[attr-defined]

        x_repo = torch.empty((B, self._x_dim, T), device=self.device, dtype=self.dtype)
        for t in range(T):
            y_t = y_repo[:, :, t].unsqueeze(2)  # [B,Dy,1]
            x_t = self._call_model_forward(y_t)
            if isinstance(x_t, (tuple, list)):
                x_t = x_t[0]
            if not isinstance(x_t, torch.Tensor):
                raise TypeError(f"runtime_error: model output must be Tensor, got {type(x_t)}")
            if x_t.ndim == 2:
                x_t = x_t.unsqueeze(2)
            if x_t.shape != (B, self._x_dim, 1):
                raise ValueError(
                    f"shape_mismatch: unexpected model output at t={t}: {tuple(x_t.shape)} "
                    f"(expected {(B, self._x_dim, 1)})"
                )
            x_repo[:, :, t] = x_t[:, :, 0]

        return x_repo.permute(0, 2, 1).contiguous()  # [B,T,Dx]

    def _call_model_forward(self, y_t: torch.Tensor) -> Any:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        try:
            return self.model(y_t)
        except TypeError:
            return self.model.forward(y=y_t)  # type: ignore[call-arg]

    def _project_state_to_obs(self, x_btd: torch.Tensor) -> torch.Tensor:
        if self.model is None or self._y_dim is None or self._x_dim is None:
            raise RuntimeError("Adapter/model is not initialized.")
        if x_btd.ndim != 3:
            raise ValueError(f"shape_mismatch: x_btd must be rank-3, got {tuple(x_btd.shape)}")

        x_repo = x_btd.permute(0, 2, 1).contiguous()  # [B,Dx,T]

        if hasattr(self.model, "h"):
            y_repo = self.model.h(x_repo)  # type: ignore[attr-defined]
        elif self._H is not None:
            B = x_repo.shape[0]
            H_b = self._H.view(1, self._y_dim, self._x_dim).expand(B, -1, -1)
            y_repo = torch.bmm(H_b, x_repo)
        else:
            raise RuntimeError("runtime_error: cannot project x->y (missing model.h and H matrix).")

        if not isinstance(y_repo, torch.Tensor):
            raise TypeError(f"runtime_error: projected observation must be Tensor, got {type(y_repo)}")
        if y_repo.ndim != 3:
            raise ValueError(f"shape_mismatch: projected observation rank must be 3, got {tuple(y_repo.shape)}")
        return y_repo.permute(0, 2, 1).contiguous()  # [B,T,Dy]

    @torch.no_grad()
    def _compute_validation_loss(
        self,
        *,
        val_dl: Any,
        loss_fn: torch.nn.Module,
        max_batches: int,
    ) -> float:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        self.model.eval()
        losses: List[float] = []
        for bi, batch in enumerate(val_dl):
            if max_batches > 0 and bi >= max_batches:
                break
            x_raw, y_raw = _extract_batch_xy(batch)
            x = _to_tensor(x_raw, device=self.device, dtype=self.dtype)
            y = _to_tensor(y_raw, device=self.device, dtype=self.dtype)
            pred = self._forward_sequence(y, context=None)
            loss = loss_fn(pred, x)
            if not torch.isfinite(loss):
                raise FloatingPointError("train_nan: non-finite validation loss.")
            losses.append(float(loss.item()))
        self.model.train()
        if not losses:
            return float("inf")
        return float(sum(losses) / len(losses))

    def _collect_y_for_adapt(self, y_seq: Any) -> torch.Tensor:
        if isinstance(y_seq, torch.Tensor) or isinstance(y_seq, np.ndarray):
            return _to_tensor(y_seq, device=self.device, dtype=self.dtype)

        if isinstance(y_seq, Iterable):
            ys: List[torch.Tensor] = []
            for batch in y_seq:
                y_raw = _extract_batch_y(batch)
                y_t = _to_tensor(y_raw, device=self.device, dtype=self.dtype)
                if y_t.ndim != 3:
                    raise ValueError(f"shape_mismatch: adapt stream batch y must be rank-3, got {tuple(y_t.shape)}")
                ys.append(y_t)
            if not ys:
                raise RuntimeError("runtime_error: empty adaptation stream.")
            return torch.cat(ys, dim=0).contiguous()

        raise TypeError(f"Unsupported adapt input type: {type(y_seq)}")

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
                "adapt_updates_per_step": {},
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
        adapt_t0: Optional[int] = None,
        allowed_after_t0_only: Optional[bool] = None,
        adapt_max_updates: Optional[int] = None,
        adapt_max_updates_per_step: Optional[int] = None,
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
        if adapt_t0 is not None:
            current["adapt_t0"] = int(adapt_t0)
        if allowed_after_t0_only is not None:
            current["allowed_after_t0_only"] = bool(allowed_after_t0_only)
        if adapt_max_updates is not None:
            current["adapt_max_updates"] = int(adapt_max_updates)
        if adapt_max_updates_per_step is not None:
            current["adapt_max_updates_per_step"] = int(adapt_max_updates_per_step)
        current["track_id"] = self._run_ctx.get("track_id")
        current["init_id"] = self._run_ctx.get("init_id")

        _write_json(self._ledger_path, current)
