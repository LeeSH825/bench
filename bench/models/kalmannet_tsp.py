from __future__ import annotations

import importlib.util
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    return torch.as_tensor(x, dtype=dtype, device=device)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def _lookup_nested(d: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _extract_q2_r2(meta: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    q_paths = (
        ("noise", "pre_shift", "Q", "q2"),
        ("scenario_cfg", "noise", "pre_shift", "Q", "q2"),
        ("noise", "Q", "q2"),
        ("scenario_cfg", "noise", "Q", "q2"),
    )
    r_paths = (
        ("noise", "pre_shift", "R", "r2"),
        ("scenario_cfg", "noise", "pre_shift", "R", "r2"),
        ("noise", "R", "r2"),
        ("scenario_cfg", "noise", "R", "r2"),
    )

    q2 = None
    for path in q_paths:
        v = _lookup_nested(meta, path)
        if v is None:
            continue
        try:
            q2 = float(v)
            break
        except Exception:
            continue

    r2 = None
    for path in r_paths:
        v = _lookup_nested(meta, path)
        if v is None:
            continue
        try:
            r2 = float(v)
            break
        except Exception:
            continue

    return q2, r2


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
        raise ValueError("KalmanNet_TSP adapter requires model_cfg['repo'] with a path.")

    bench_root = Path(__file__).resolve().parents[2]
    repo_candidate = Path(repo_path).expanduser()
    if repo_candidate.is_absolute():
        repo_root = repo_candidate.resolve()
    else:
        repo_root = (bench_root / repo_candidate).resolve()
    return repo_root, entrypoints


def _call_general_settings_safely(general_settings: Any) -> Any:
    # KalmanNet_TSP config.general_settings() parses sys.argv.
    old_argv = list(sys.argv)
    argv0 = old_argv[0] if old_argv else "bench"
    try:
        sys.argv = [argv0]
        return general_settings()
    finally:
        sys.argv = old_argv


def _resolve_x0(
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
        f"x0 shape mismatch: expected rank 1/2/3 with x_dim={x_dim} and batch_size={batch_size}, "
        f"got {tuple(x.shape)}"
    )


def _extract_batch_xy(batch: Any) -> Tuple[Any, Any]:
    if isinstance(batch, dict):
        if "x" not in batch or "y" not in batch:
            raise KeyError("Batch dict must contain keys 'x' and 'y'.")
        return batch["x"], batch["y"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(f"Unsupported dataloader batch type: {type(batch)}")


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


@dataclass
class _KNetImports:
    KalmanNetNN: Any
    SystemModel: Any
    config_general_settings: Any


def _load_kalmannet_tsp(repo_root: Path) -> _KNetImports:
    if not repo_root.exists():
        raise FileNotFoundError(f"KalmanNet_TSP repo root not found: {repo_root}")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from KNet.KalmanNet_nn import KalmanNetNN  # type: ignore
        from Simulations.Linear_sysmdl import SystemModel  # type: ignore
        from Simulations import config as kn_config  # type: ignore

        return _KNetImports(
            KalmanNetNN=KalmanNetNN,
            SystemModel=SystemModel,
            config_general_settings=kn_config.general_settings,
        )
    except Exception as e:
        logger.warning("Normal KalmanNet_TSP import failed; trying file import fallback: %r", e)

    kn_file = repo_root / "KNet" / "KalmanNet_nn.py"
    sysmdl_file = repo_root / "Simulations" / "Linear_sysmdl.py"
    cfg_file = repo_root / "Simulations" / "config.py"

    if not kn_file.exists():
        raise FileNotFoundError(f"Missing file: {kn_file}")
    if not sysmdl_file.exists():
        raise FileNotFoundError(f"Missing file: {sysmdl_file}")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing file: {cfg_file}")

    kn_mod = _import_module_from_file("kalmannet_tsp_KalmanNet_nn", kn_file)
    sys_mod = _import_module_from_file("kalmannet_tsp_Linear_sysmdl", sysmdl_file)
    cfg_mod = _import_module_from_file("kalmannet_tsp_config", cfg_file)

    return _KNetImports(
        KalmanNetNN=getattr(kn_mod, "KalmanNetNN"),
        SystemModel=getattr(sys_mod, "SystemModel"),
        config_general_settings=getattr(cfg_mod, "general_settings"),
    )


class KalmanNetTSPAdapter(ModelAdapter):
    """
    Route-B import-mode adapter for third_party/KalmanNet_TSP.

    Data layout contract inside this adapter:
    - bench input/output: [B,T,D]
    - KalmanNet_TSP forward expects per-step [B,D,1]
    - internal sequence staging uses [B,D,T]
    """

    def __init__(self) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.repo_root: Optional[Path] = None
        self.entrypoints: Dict[str, Any] = {}
        self.device: torch.device = torch.device("cpu")
        self.dtype: torch.dtype = torch.float32

        self._imports: Optional[_KNetImports] = None
        self._sys_model: Any = None

        self._x_dim: Optional[int] = None
        self._y_dim: Optional[int] = None
        self._T_setup: Optional[int] = None

        self._F: Optional[torch.Tensor] = None
        self._H: Optional[torch.Tensor] = None
        self._Q: Optional[torch.Tensor] = None
        self._R: Optional[torch.Tensor] = None

        self._args: Any = None
        self._cfg: Dict[str, Any] = {}
        self._run_ctx: Dict[str, Any] = {}
        self._run_dir: Optional[Path] = None
        self._ckpt_dir: Optional[Path] = None
        self._artifacts_dir: Optional[Path] = None
        self._ledger_path: Optional[Path] = None
        self._train_state_path: Optional[Path] = None
        self._saved_ckpt_path: Optional[Path] = None

        self.last_layout: Optional[str] = None
        self.last_class: Optional[str] = None

        self.train_updates_used: int = 0
        self.adapt_updates_used: int = 0

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
        self._imports = _load_kalmannet_tsp(self.repo_root)

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

        meta = _coerce_meta_dict(system_info)
        x_dim = system_info.get("x_dim", cfg.get("x_dim", meta.get("x_dim")))
        y_dim = system_info.get("y_dim", cfg.get("y_dim", meta.get("y_dim")))
        if x_dim is None or y_dim is None:
            raise ValueError("system_info must provide x_dim and y_dim for KalmanNet_TSP setup.")
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
        self._F = _to_tensor(F, device=self.device, dtype=self.dtype)
        self._H = _to_tensor(H, device=self.device, dtype=self.dtype)

        Q = system_info.get("Q", None)
        R = system_info.get("R", None)
        q2_meta, r2_meta = _extract_q2_r2(meta)
        q2 = system_info.get("q2", cfg.get("q2", q2_meta if q2_meta is not None else 1e-3))
        r2 = system_info.get("r2", cfg.get("r2", r2_meta if r2_meta is not None else 1e-3))

        if Q is None:
            self._Q = torch.eye(self._x_dim, device=self.device, dtype=self.dtype) * float(q2)
        else:
            self._Q = _to_tensor(Q, device=self.device, dtype=self.dtype)

        if R is None:
            self._R = torch.eye(self._y_dim, device=self.device, dtype=self.dtype) * float(r2)
        else:
            self._R = _to_tensor(R, device=self.device, dtype=self.dtype)

        args = _call_general_settings_safely(self._imports.config_general_settings)
        args.use_cuda = (self.device.type == "cuda")
        args.T = int(self._T_setup)
        args.T_test = int(self._T_setup)
        args.n_batch = int(cfg.get("batch_size", 32))
        args.lr = float(cfg.get("lr", 1e-4))
        args.wd = float(cfg.get("weight_decay", cfg.get("wd", 1e-3)))
        args.n_steps = int(cfg.get("train_steps_hint", 1))
        if "in_mult_KNet" in cfg:
            args.in_mult_KNet = int(cfg["in_mult_KNet"])
        if "out_mult_KNet" in cfg:
            args.out_mult_KNet = int(cfg["out_mult_KNet"])
        self._args = args

        SystemModel = self._imports.SystemModel
        self._sys_model = SystemModel(self._F, self._Q, self._H, self._R, args.T, args.T_test)

        m1x_0 = torch.zeros(self._x_dim, 1, device=self.device, dtype=self.dtype)
        m2x_0 = torch.zeros(self._x_dim, self._x_dim, device=self.device, dtype=self.dtype)
        self._sys_model.InitSequence(m1x_0, m2x_0)

        KalmanNetNN = self._imports.KalmanNetNN
        self.model = KalmanNetNN()
        self.model.NNBuild(self._sys_model, args)
        self.model.to(self.device)
        self.model.eval()

        self.last_class = type(self.model).__name__
        self.last_layout = "bench_BTD_to_repo_BDT_stepwise"

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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm_f)

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
                            "Early stopping KalmanNet_TSP training at step=%s (patience_evals=%s)",
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

        # frozen track: no updates in eval/adapt
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
        # Backward compatible aliases used by older callers.
        if state0 is None and "u_seq" in kwargs:
            _ = kwargs["u_seq"]

        if self.model is None:
            raise RuntimeError("setup() must be called before predict().")
        self.model.eval()

        ctx = dict(context or {})
        if state0 is not None:
            ctx["x0"] = state0

        y = _to_tensor(y_batch, device=self.device, dtype=self.dtype)
        x_hat = self._forward_sequence(y, context=ctx)
        if return_cov:
            return x_hat, None
        return x_hat

    def load(self, ckpt_path: str) -> None:
        if self.model is None:
            raise RuntimeError("setup() must be called before load().")
        try:
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def save(self, ckpt_dir: Union[str, Path]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("setup() must be called before save().")
        out = Path(ckpt_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        ckpt_path = out / "model.pt"
        torch.save({"state_dict": self.model.state_dict()}, ckpt_path)
        self._saved_ckpt_path = ckpt_path
        return {"ckpt_path": str(ckpt_path)}

    def adapt(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        budget: Optional[Any] = None,
    ) -> None:
        # S2 scope: frozen track only.
        self.adapt_updates_used = 0
        self._update_ledger(
            train_updates_used=int(self.train_updates_used),
            adapt_updates_used=0,
        )
        return None

    def get_adapter_meta(self) -> Dict[str, Any]:
        return {
            "adapter_id": "kalmannet_tsp",
            "adapter_version": "s2_route_b_v1",
            "runtime_device": str(self.device),
            "covariance_support": False,
            "input_layout_bench": "BTD",
            "internal_layout_repo": "BDT_stepwise",
            "entrypoints": {
                "config": "Simulations/config.py::general_settings",
                "system_model": "Simulations/Linear_sysmdl.py::SystemModel",
                "model": "KNet/KalmanNet_nn.py::KalmanNetNN",
                "pipeline_candidate": "Pipelines/Pipeline_EKF.py::Pipeline_EKF",
            },
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
        if not hasattr(self.model, "init_hidden_KNet"):
            raise RuntimeError("runtime_error: KalmanNet_TSP model missing init_hidden_KNet().")
        self.model.init_hidden_KNet()  # type: ignore[attr-defined]

        ctx = context or {}
        x0 = None
        for key in ("x0", "x_init", "x_init_mean", "m1x_0"):
            if key in ctx:
                x0 = ctx[key]
                break
        if x0 is None:
            m1_0_batch = torch.zeros(B, self._x_dim, 1, device=self.device, dtype=self.dtype)
        else:
            m1_0_batch = _resolve_x0(
                x0,
                batch_size=B,
                x_dim=self._x_dim,
                device=self.device,
                dtype=self.dtype,
            )

        if not hasattr(self.model, "InitSequence"):
            raise RuntimeError("runtime_error: KalmanNet_TSP model missing InitSequence().")
        self.model.InitSequence(m1_0_batch, int(T))  # type: ignore[attr-defined]

        x_repo = torch.empty((B, self._x_dim, T), device=self.device, dtype=self.dtype)
        for t in range(T):
            y_t = y_repo[:, :, t].unsqueeze(2)  # [B,Dy,1]
            x_t = self.model(y_t)
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
        current["track_id"] = self._run_ctx.get("track_id")
        current["init_id"] = self._run_ctx.get("init_id")
        _write_json(self._ledger_path, current)
