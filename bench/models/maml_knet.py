from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Type, Union

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


def _bench_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


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
        raise ValueError("MAML-KalmanNet adapter requires model_cfg['repo'] as string or dict with 'path'.")

    bench_root = _bench_root_from_this_file()
    p = Path(repo_path).expanduser()
    repo_root = (bench_root / p).resolve() if not p.is_absolute() else p.resolve()
    return repo_root, entrypoints


def _normalize_repo_root(repo_root: Path) -> Path:
    """
    Accept several layout variants:
      - .../MAML-KalmanNet                 (code root)
      - .../MAML_KalmanNet                 (wrapper root, code nested in MAML-KalmanNet/)
      - .../MAML_KalmanNet/MAML-KalmanNet  (code root)
    """
    candidates: List[Path] = []
    candidates.append(repo_root)
    candidates.append(repo_root / "MAML-KalmanNet")
    candidates.append(repo_root / "MAML_KalmanNet")

    if "MAML-KalmanNet" in str(repo_root):
        candidates.append(Path(str(repo_root).replace("MAML-KalmanNet", "MAML_KalmanNet")))
    if "MAML_KalmanNet" in str(repo_root):
        candidates.append(Path(str(repo_root).replace("MAML_KalmanNet", "MAML-KalmanNet")))

    bench_root = _bench_root_from_this_file()
    candidates.append((bench_root / "third_party" / "MAML_KalmanNet").resolve())
    candidates.append((bench_root / "third_party" / "MAML_KalmanNet" / "MAML-KalmanNet").resolve())
    candidates.append((bench_root / "third_party" / "MAML-KalmanNet").resolve())

    seen = set()
    unique: List[Path] = []
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    for c in unique:
        if (c / "filter.py").exists() and (c / "state_dict_learner.py").exists():
            return c

    raise FileNotFoundError(
        f"Could not locate MAML-KalmanNet code root from {repo_root}. "
        "Expected filter.py + state_dict_learner.py."
    )


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
            f"x0 2D shape mismatch: expected [{x_dim},1] or [{batch_size},{x_dim}], got {tuple(x.shape)}"
        )
    if x.ndim == 3:
        if x.shape == (batch_size, x_dim, 1):
            return x
        raise ValueError(
            f"x0 3D shape mismatch: expected [{batch_size},{x_dim},1], got {tuple(x.shape)}"
        )
    raise ValueError(f"Unsupported x0 rank={x.ndim}")


def _build_y_layout_candidates(
    y: torch.Tensor,
    *,
    expected_y_dim: int,
    expected_T: Optional[int],
) -> List[Tuple[str, torch.Tensor, int, int]]:
    if y.ndim != 3:
        return []

    candidates: List[Tuple[str, torch.Tensor, int, int]] = []
    b0, a1, a2 = y.shape

    # Benchmark canonical [B,T,y] -> repo [B,y,T]
    if a2 == expected_y_dim:
        candidates.append(("BTD", y.permute(0, 2, 1).contiguous(), b0, a1))

    # Already [B,y,T]
    if a1 == expected_y_dim:
        candidates.append(("BYT", y.contiguous(), b0, a2))

    # [T,B,y] -> [B,y,T]
    if a2 == expected_y_dim and (expected_T is None or b0 == expected_T):
        candidates.append(("TBY", y.permute(1, 2, 0).contiguous(), a1, b0))

    if not candidates:
        candidates.append(("BTD_fallback", y.permute(0, 2, 1).contiguous(), b0, a1))

    seen = set()
    unique: List[Tuple[str, torch.Tensor, int, int]] = []
    for name, y_byt, batch_size, seq_len in candidates:
        key = (tuple(y_byt.shape), batch_size, seq_len)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, y_byt, batch_size, seq_len))
    return unique


class _LinearSystemBridge:
    def __init__(
        self,
        F: torch.Tensor,
        H: torch.Tensor,
        *,
        x_dim: int,
        y_dim: int,
        device: torch.device,
        q_qry: int,
    ) -> None:
        self.F = F
        self.H = H
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.device = device
        self.init_state_filter = torch.zeros(self.x_dim, 1, device=self.device)
        self.ekf_cov = torch.zeros((1, self.x_dim, self.y_dim), device=self.device).repeat(max(1, int(q_qry)), 1, 1)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        batched_F = self.F.view(1, self.x_dim, self.x_dim).expand(x.shape[0], -1, -1)
        return torch.bmm(batched_F, x)

    def g(self, x: torch.Tensor) -> torch.Tensor:
        batched_H = self.H.view(1, self.y_dim, self.x_dim).expand(x.shape[0], -1, -1)
        return torch.bmm(batched_H, x)

    def Jacobian_f(self, x: torch.Tensor, is_seq: bool = True) -> torch.Tensor:
        if is_seq:
            return self.F.view(1, self.x_dim, self.x_dim).repeat(x.shape[0], 1, 1)
        return self.F

    def Jacobian_g(self, x: torch.Tensor, is_seq: bool = True) -> torch.Tensor:
        if is_seq:
            return self.H.view(1, self.y_dim, self.x_dim).repeat(x.shape[0], 1, 1)
        return self.H


class MAMLKNetAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.model: Any = None
        self.repo_root: Optional[Path] = None
        self.entrypoints: Dict[str, Any] = {}
        self.device: torch.device = torch.device("cpu")
        self.dtype: torch.dtype = torch.float32

        self._args: Any = None
        self._bridge: Any = None
        self._filter: Any = None
        self._filter_cls: Optional[Type[Any]] = None
        self._cfg: Dict[str, Any] = {}
        self._run_ctx: Dict[str, Any] = {}
        self._run_dir: Optional[Path] = None
        self._ckpt_dir: Optional[Path] = None
        self._artifacts_dir: Optional[Path] = None
        self._ledger_path: Optional[Path] = None
        self._train_state_path: Optional[Path] = None
        self._saved_ckpt_path: Optional[Path] = None

        self._x_dim: Optional[int] = None
        self._y_dim: Optional[int] = None
        self._T_setup: Optional[int] = None

        self.train_updates_used: int = 0
        self.train_outer_updates_used: int = 0
        self.train_inner_updates_used: int = 0
        self.adapt_updates_used: int = 0

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
        repo_root_raw, self.entrypoints = _resolve_repo_spec(cfg)
        self.repo_root = _normalize_repo_root(repo_root_raw)
        self.train_updates_used = 0
        self.train_outer_updates_used = 0
        self.train_inner_updates_used = 0
        self.adapt_updates_used = 0

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

        self._ckpt_dir = None
        self._artifacts_dir = None
        self._ledger_path = None
        self._train_state_path = None
        self._run_dir = Path(str(run_ctx["run_dir"])).expanduser().resolve() if "run_dir" in run_ctx else None
        if self._run_dir is not None:
            self._ckpt_dir = self._run_dir / "checkpoints"
            self._artifacts_dir = self._run_dir / "artifacts"
            self._ledger_path = self._run_dir / "budget_ledger.json"
            self._train_state_path = self._ckpt_dir / "train_state.json"
            self._ckpt_dir.mkdir(parents=True, exist_ok=True)
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            self._init_ledger()

        x_dim = system_info.get("x_dim", cfg.get("x_dim"))
        y_dim = system_info.get("y_dim", cfg.get("y_dim"))
        if x_dim is None or y_dim is None:
            raise ValueError("system_info must provide x_dim and y_dim for MAML-KalmanNet setup.")
        self._x_dim = int(x_dim)
        self._y_dim = int(y_dim)
        T = system_info.get("T", cfg.get("T", cfg.get("sequence_length", 1)))
        self._T_setup = int(T)

        F = system_info.get("F", None)
        H = system_info.get("H", None)
        if F is None:
            F = torch.eye(self._x_dim)
        if H is None:
            H = torch.eye(self._y_dim, self._x_dim)
        F_t = _to_tensor(F, device=self.device, dtype=self.dtype)
        H_t = _to_tensor(H, device=self.device, dtype=self.dtype)

        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        try:
            filter_mod = importlib.import_module("filter")
        except Exception:
            filter_mod = _import_module_from_file("maml_knet_filter", self.repo_root / "filter.py")
        FilterCls = getattr(filter_mod, "Filter")
        self._filter_cls = FilterCls

        batch_size = int(cfg.get("batch_size", 8) or 8)
        query_size = int(cfg.get("query_size", cfg.get("q_qry", max(2, batch_size))) or max(2, batch_size))
        self._args = SimpleNamespace(
            update_lr=float(cfg.get("update_lr", 5e-2)),
            meta_lr=float(cfg.get("meta_lr", cfg.get("outer_lr", cfg.get("lr", 1e-3)))),
            update_step=int(cfg.get("update_step", 1)),
            update_step_test=int(cfg.get("update_step_test", 1)),
            batch_size=batch_size,
            q_qry=query_size,
            use_cuda=(self.device.type == "cuda"),
        )

        is_linear_net = bool(cfg.get("is_linear_net", True))

        self._bridge = _LinearSystemBridge(
            F_t,
            H_t,
            x_dim=self._x_dim,
            y_dim=self._y_dim,
            device=self.device,
            q_qry=self._args.q_qry,
        )
        self.model = self._bridge
        self._filter = FilterCls(self._args, self._bridge, is_linear_net=is_linear_net)

        ckpt = self._resolve_default_checkpoint(cfg)
        if ckpt is not None:
            self._safe_load_checkpoint(ckpt)
            self._saved_ckpt_path = ckpt

        self.last_class = "filter.Filter + state_dict_learner.Learner"
        self.last_layout = "bench_BTD_to_repo_BYT_stepwise_filter"

    def _resolve_default_checkpoint(self, cfg: Dict[str, Any]) -> Optional[Path]:
        candidates: List[Any] = []
        for key in ("ckpt_path", "checkpoint", "checkpoint_path", "weights", "weights_path", "basenet_path"):
            if key in cfg:
                candidates.append(cfg.get(key))

        if self.repo_root is None:
            return None

        candidates.extend(
            [
                self.repo_root / "MAML_model" / "linear" / "basenet.pt",
                self.repo_root / "MAML_model" / "nonlinear" / "basenet.pt",
                self.repo_root / "MAML_model" / "UZH" / "basenet.pt",
            ]
        )

        bench_root = _bench_root_from_this_file()
        for c in candidates:
            if c is None:
                continue
            p = Path(c).expanduser()
            if not p.is_absolute():
                p = (bench_root / p).resolve()
            if p.exists():
                return p
        return None

    def _safe_load_checkpoint(self, ckpt_path: Path) -> None:
        if self._filter is None:
            raise RuntimeError("setup() must be called before checkpoint loading.")

        try:
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        if not isinstance(state, dict):
            logger.warning("Checkpoint %s is not a state_dict dict; skipping load.", ckpt_path)
            return

        model_state = self._filter.train_net.state_dict()
        compatible: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k not in model_state:
                continue
            if not isinstance(v, torch.Tensor):
                continue
            if tuple(v.shape) != tuple(model_state[k].shape):
                continue
            compatible[k] = v

        if not compatible:
            logger.warning("No compatible weights found in checkpoint %s; using initialized weights.", ckpt_path)
            return

        merged = dict(model_state)
        merged.update(compatible)
        self._filter.train_net.load_state_dict(merged, strict=False)
        logger.info("Loaded %d compatible tensors from checkpoint %s", len(compatible), ckpt_path)

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        budget: Optional[Dict[str, Any]] = None,
        ckpt_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        if self._filter is None or self._x_dim is None or self._y_dim is None:
            raise RuntimeError("setup() must be called before train().")

        budget = dict(budget or {})
        max_updates = int(budget.get("train_max_updates", 0))
        if max_updates <= 0:
            raise ValueError("train_max_updates must be > 0 for init_id=trained.")

        out_ckpt_dir = Path(ckpt_dir).expanduser().resolve() if ckpt_dir is not None else self._ckpt_dir
        if out_ckpt_dir is None:
            raise ValueError("ckpt_dir is required when adapter has no run_dir.")
        out_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # MAML-style training knobs (pinned by adapter config; no runtime discovery).
        meta_task_num = int(self._cfg.get("meta_task_num", self._cfg.get("task_num", 4)))
        support_size = int(self._cfg.get("support_size", self._cfg.get("k_spt", 4)))
        query_size = int(self._cfg.get("query_size", self._cfg.get("q_qry", 4)))
        inner_steps = int(self._cfg.get("inner_steps", self._cfg.get("update_step", 1)))
        inner_lr = float(self._cfg.get("inner_lr", self._cfg.get("update_lr", 5e-2)))
        outer_lr = float(self._cfg.get("outer_lr", self._cfg.get("meta_lr", self._cfg.get("lr", 1e-3))))
        weight_decay = float(self._cfg.get("weight_decay", self._cfg.get("wd", 0.0)))
        max_grad_norm = self._cfg.get("max_grad_norm", 1.0)
        max_grad_norm_f = float(max_grad_norm) if max_grad_norm is not None else None
        val_eval_interval = int(self._cfg.get("val_eval_interval_updates", max(1, min(10, max_updates))))
        patience_evals = int(self._cfg.get("patience_evals", budget.get("patience_evals", 0)))
        min_delta = float(self._cfg.get("min_delta", budget.get("min_delta", 0.0)))
        val_max_batches = int(self._cfg.get("val_max_batches", 0))

        if meta_task_num <= 0:
            raise ValueError("meta_task_num must be > 0.")
        if support_size <= 0 or query_size <= 0:
            raise ValueError("support_size and query_size must both be > 0.")
        if inner_steps <= 0:
            raise ValueError("inner_steps must be > 0.")

        x_train_btd, y_train_btd = self._materialize_xy(train_loader)
        n_train = int(x_train_btd.shape[0])
        if n_train <= 0:
            raise RuntimeError("runtime_error: empty train split for maml_knet.")
        if n_train < 2:
            raise RuntimeError("runtime_error: train split too small for MAML-style support/query sampling.")

        x_train_bxt = x_train_btd.permute(0, 2, 1).contiguous()
        y_train_byt = y_train_btd.permute(0, 2, 1).contiguous()

        self._args.update_lr = float(inner_lr)
        self._args.meta_lr = float(outer_lr)
        self._args.update_step = int(inner_steps)

        base_net = self._filter.train_net
        base_params = [p for p in base_net.parameters() if p.requires_grad]
        if not base_params:
            raise RuntimeError("runtime_error: maml_knet has no trainable parameters.")

        outer_optim = torch.optim.Adam(base_params, lr=outer_lr, weight_decay=weight_decay)
        base_net.train()
        seed = int(self._run_ctx.get("seed", self._cfg.get("seed", 0)))
        sample_gen = torch.Generator(device="cpu")
        sample_gen.manual_seed(seed + 1337)

        updates_used = 0
        best_step = 0
        best_val = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in base_net.state_dict().items()}
        no_improve_evals = 0
        last_train_loss: Optional[float] = None
        val_history: List[Dict[str, float]] = []
        inner_steps_total = 0
        meta_tasks_seen = 0
        stop_early = False

        while updates_used < max_updates and not stop_early:
            grad_accum = [torch.zeros_like(p) for p in base_params]
            query_losses: List[float] = []
            tasks_used = 0

            for _ in range(meta_task_num):
                support_idx, query_idx = self._sample_episode_indices(
                    n_total=n_train,
                    support_size=support_size,
                    query_size=query_size,
                    generator=sample_gen,
                )
                support_state = x_train_bxt[support_idx]
                support_obs = y_train_byt[support_idx]
                query_state = x_train_bxt[query_idx]
                query_obs = y_train_byt[query_idx]

                task_model = self._clone_task_model(base_net)
                self._prepare_task_model(task_model, support_batch=int(support_state.shape[0]), query_batch=int(query_state.shape[0]))
                self._filter.batch_size = int(support_state.shape[0])
                self._filter.args.q_qry = int(query_state.shape[0])
                self._filter.data_idx = 0

                inner_optim = torch.optim.SGD(
                    [p for p in task_model.parameters() if p.requires_grad],
                    lr=inner_lr,
                )

                task_ok = True
                for _k in range(inner_steps):
                    support_loss = self._filter.compute_x_post(
                        support_state,
                        support_obs,
                        task_net=task_model,
                        use_initial_state=True,
                    )
                    if not torch.isfinite(support_loss):
                        task_ok = False
                        break
                    inner_optim.zero_grad(set_to_none=True)
                    support_loss.backward()
                    if max_grad_norm_f is not None and max_grad_norm_f > 0:
                        torch.nn.utils.clip_grad_norm_(task_model.parameters(), max_norm=max_grad_norm_f)
                    inner_optim.step()
                    inner_steps_total += 1

                if not task_ok:
                    continue

                query_loss = self._filter.compute_x_post_qry(
                    query_state,
                    query_obs,
                    task_net=task_model,
                    use_initial_state=True,
                )
                if not torch.isfinite(query_loss):
                    continue

                task_params = [p for p in task_model.parameters() if p.requires_grad]
                if len(task_params) != len(base_params):
                    raise RuntimeError("runtime_error: task/base parameter mismatch in MAML training loop.")

                for p in task_params:
                    if p.grad is not None:
                        p.grad.zero_()
                query_loss.backward()

                for gi, p_task in enumerate(task_params):
                    if p_task.grad is not None:
                        grad_accum[gi].add_(p_task.grad.detach())

                query_losses.append(float(query_loss.detach().item()))
                tasks_used += 1
                meta_tasks_seen += 1

            if tasks_used <= 0:
                raise FloatingPointError("train_nan: all sampled meta-tasks produced non-finite losses.")

            outer_optim.zero_grad(set_to_none=True)
            for gi, p_base in enumerate(base_params):
                p_base.grad = grad_accum[gi] / float(tasks_used)
            if max_grad_norm_f is not None and max_grad_norm_f > 0:
                torch.nn.utils.clip_grad_norm_(base_params, max_norm=max_grad_norm_f)

            if updates_used >= max_updates:
                raise RuntimeError(f"budget_overflow: attempted optimizer.step beyond max_updates={max_updates}")
            outer_optim.step()
            updates_used += 1
            last_train_loss = float(sum(query_losses) / max(1, len(query_losses)))

            should_eval = (updates_used % val_eval_interval == 0) or (updates_used == max_updates)
            if should_eval:
                val_loss = self._compute_validation_loss(
                    val_dl=val_loader,
                    max_batches=val_max_batches,
                )
                val_history.append({"step": float(updates_used), "val_mse": float(val_loss)})
                if (best_val - float(val_loss)) > min_delta:
                    best_val = float(val_loss)
                    best_step = int(updates_used)
                    best_state = {k: v.detach().cpu().clone() for k, v in base_net.state_dict().items()}
                    no_improve_evals = 0
                else:
                    no_improve_evals += 1
                if patience_evals > 0 and no_improve_evals >= patience_evals:
                    logger.info(
                        "Early stopping MAML-KalmanNet training at step=%s (patience_evals=%s)",
                        updates_used,
                        patience_evals,
                    )
                    stop_early = True

        base_net.load_state_dict(best_state, strict=True)
        ckpt_path = out_ckpt_dir / "model.pt"
        torch.save(
            {
                "state_dict": base_net.state_dict(),
                "best_step": int(best_step),
                "best_val_mse": float(best_val),
                "train_updates_used": int(updates_used),
                "train_outer_updates_used": int(updates_used),
                "train_inner_updates_used": int(inner_steps_total),
                "inner_steps_total": int(inner_steps_total),
                "meta_tasks_seen": int(meta_tasks_seen),
                "class_path": "filter.Filter + state_dict_learner.Learner",
            },
            ckpt_path,
        )
        self._saved_ckpt_path = ckpt_path

        train_state_path = out_ckpt_dir / "train_state.json"
        train_state = {
            "status": "ok",
            "best_step": int(best_step),
            "best_val_mse": float(best_val),
            "last_train_loss": float(last_train_loss) if last_train_loss is not None else None,
            "updates_used": int(updates_used),
            "train_outer_updates_used": int(updates_used),
            "train_inner_updates_used": int(inner_steps_total),
            "max_updates": int(max_updates),
            "val_history": val_history[-20:],
            "meta_train": {
                "meta_task_num": int(meta_task_num),
                "support_size": int(support_size),
                "query_size": int(query_size),
                "inner_steps": int(inner_steps),
                "inner_lr": float(inner_lr),
                "outer_lr": float(outer_lr),
                "inner_steps_total": int(inner_steps_total),
                "meta_tasks_seen": int(meta_tasks_seen),
            },
        }
        _write_json(train_state_path, train_state)
        self._train_state_path = train_state_path

        self.train_outer_updates_used = int(updates_used)
        self.train_inner_updates_used = int(inner_steps_total)
        self.train_updates_used = int(updates_used)
        self.adapt_updates_used = 0
        self._update_ledger(
            train_outer_updates_used=int(updates_used),
            train_inner_updates_used=int(inner_steps_total),
            adapt_updates_used=0,
            train_max_updates=int(max_updates),
            meta_tasks_seen=int(meta_tasks_seen),
        )

        return {
            "status": "ok",
            "ckpt_path": str(ckpt_path),
            "train_state_path": str(train_state_path),
            "updates_used": int(updates_used),
            "best_step": int(best_step),
        }

    def _materialize_xy(self, loader: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._x_dim is None or self._y_dim is None:
            raise RuntimeError("Adapter dimensions are not initialized.")

        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for batch in loader:
            x_raw, y_raw = _extract_batch_xy(batch)
            x = _to_tensor(x_raw, device=self.device, dtype=self.dtype)
            y = _to_tensor(y_raw, device=self.device, dtype=self.dtype)
            if x.ndim != 3 or y.ndim != 3:
                raise ValueError(f"shape_mismatch: expected x,y rank-3; got x={tuple(x.shape)} y={tuple(y.shape)}")
            if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
                raise ValueError(f"shape_mismatch: x/y batch-time mismatch x={tuple(x.shape)} y={tuple(y.shape)}")
            if x.shape[2] != self._x_dim or y.shape[2] != self._y_dim:
                raise ValueError(
                    f"shape_mismatch: expected x_dim={self._x_dim} y_dim={self._y_dim}; got x={tuple(x.shape)} y={tuple(y.shape)}"
                )
            xs.append(x)
            ys.append(y)

        if not xs:
            raise RuntimeError("runtime_error: empty dataloader.")
        return torch.cat(xs, dim=0).contiguous(), torch.cat(ys, dim=0).contiguous()

    def _sample_episode_indices(
        self,
        *,
        n_total: int,
        support_size: int,
        query_size: int,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_pick = int(support_size + query_size)
        if n_total >= n_pick:
            idx = torch.randperm(n_total, generator=generator)[:n_pick]
        else:
            idx = torch.randint(low=0, high=n_total, size=(n_pick,), generator=generator)
        return idx[:support_size], idx[support_size:]

    def _clone_task_model(self, base_net: Any) -> Any:
        if self._x_dim is None or self._y_dim is None:
            raise RuntimeError("Adapter dimensions are not initialized.")
        cls = type(base_net)
        is_linear_net = bool(self._cfg.get("is_linear_net", True))
        try:
            task_model = cls(self._x_dim, self._y_dim, self._args, is_linear_net).to(self.device)
        except TypeError:
            task_model = cls(self._x_dim, self._y_dim, self._args).to(self.device)
        task_model.load_state_dict(base_net.state_dict(), strict=True)
        task_model.train()
        return task_model

    def _prepare_task_model(self, task_model: Any, *, support_batch: int, query_batch: int) -> None:
        hidden_dim = int(task_model.gru_hidden_dim)
        gru_layers = int(task_model.gru_n_layer)
        p = next(task_model.parameters())
        task_model.hn_train_init = torch.zeros(
            gru_layers, support_batch, hidden_dim, device=p.device, dtype=p.dtype
        )
        task_model.hn_qry_init = torch.zeros(
            gru_layers, query_batch, hidden_dim, device=p.device, dtype=p.dtype
        )

    def eval(
        self,
        test_loader: Any,
        ckpt_path: Optional[Union[str, Path]] = None,
        track_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._filter is None:
            raise RuntimeError("setup() must be called before eval().")

        if ckpt_path is not None:
            self.load(str(ckpt_path))
            self._saved_ckpt_path = Path(str(ckpt_path)).expanduser().resolve()
        elif self._saved_ckpt_path is not None and self._saved_ckpt_path.exists():
            self.load(str(self._saved_ckpt_path))

        preds: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in test_loader:
                _x_raw, y_raw = _extract_batch_xy(batch)
                y = _to_tensor(y_raw, device=self.device, dtype=self.dtype)
                pred = self.predict(y, context=None, return_cov=False)
                if not isinstance(pred, torch.Tensor):
                    raise TypeError(f"runtime_error: predict() must return Tensor for eval, got {type(pred)}")
                preds.append(pred.detach().cpu())

        if not preds:
            raise RuntimeError("runtime_error: empty test dataloader.")
        x_hat = torch.cat(preds, dim=0).contiguous()

        preds_path = None
        if self._artifacts_dir is not None:
            preds_path = self._artifacts_dir / "preds_test.npz"
            np.savez_compressed(preds_path, x_hat=x_hat.numpy())

        self.adapt_updates_used = 0
        self._update_ledger(
            train_outer_updates_used=int(self.train_outer_updates_used),
            train_inner_updates_used=int(self.train_inner_updates_used),
            adapt_updates_used=0,
        )

        return {
            "status": "ok",
            "x_hat": x_hat,
            "cov": None,
            "preds_path": (str(preds_path) if preds_path is not None else None),
        }

    def load(self, ckpt_path: str) -> None:
        p = Path(ckpt_path).expanduser().resolve()
        self._safe_load_checkpoint(p)
        self._saved_ckpt_path = p

    @torch.no_grad()
    def predict(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        return_cov: bool = False,
    ) -> Any:
        if self._filter is None or self._x_dim is None or self._y_dim is None:
            raise RuntimeError("setup() must be called before predict().")

        y = _to_tensor(y_seq, device=self.device, dtype=self.dtype)
        if y.ndim != 3:
            raise ValueError(f"Expected y_seq with rank 3, got {tuple(y.shape)}")

        ctx = context or {}
        expected_T = ctx.get("T", self._T_setup)
        expected_T = int(expected_T) if expected_T is not None else None

        candidates = _build_y_layout_candidates(y, expected_y_dim=self._y_dim, expected_T=expected_T)
        errors: List[str] = []

        for layout_name, y_byt, batch_size, seq_len in candidates:
            try:
                x_hat = self._predict_stepwise(y_byt, batch_size=batch_size, seq_len=seq_len, context=ctx)
                self.last_layout = f"{layout_name}->BYT(stepwise_filter)"
                if return_cov:
                    return x_hat, None
                return x_hat
            except Exception as e:
                errors.append(f"{layout_name}: {type(e).__name__}: {e}")

        tried = ", ".join([name for name, _, _, _ in candidates])
        details = " | ".join(errors[:3]) if errors else "no layout attempts recorded"
        raise RuntimeError(
            "MAML-KalmanNet forward failed for all tried layouts. "
            f"tried=[{tried}] errors=[{details}] "
            "HOW TO VERIFY: third_party/MAML_KalmanNet/MAML-KalmanNet/filter.py filtering() and state_dict_learner.py."
        )

    def _prepare_runtime_batch(self, batch_size: int) -> None:
        if self._filter is None:
            raise RuntimeError("Filter is not initialized.")
        train_net = self._filter.train_net
        hidden_dim = int(train_net.gru_hidden_dim)
        gru_layers = int(train_net.gru_n_layer)
        if train_net.hn_train_init.shape[1] != batch_size:
            train_net.hn_train_init = torch.zeros(gru_layers, batch_size, hidden_dim, device=self.device)
        if train_net.hn_qry_init.shape[1] != batch_size:
            train_net.hn_qry_init = torch.zeros(gru_layers, batch_size, hidden_dim, device=self.device)
        self._filter.batch_size = int(batch_size)
        self._filter.args.q_qry = int(batch_size)

    def _predict_stepwise(
        self,
        y_byt: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        context: Dict[str, Any],
    ) -> torch.Tensor:
        if self._filter is None or self._x_dim is None or self._y_dim is None:
            raise RuntimeError("Adapter/model is not initialized.")

        if y_byt.ndim != 3:
            raise ValueError(f"Expected y_byt [B,y,T], got {tuple(y_byt.shape)}")
        if y_byt.shape[0] != batch_size:
            raise ValueError(f"batch_size mismatch: tensor has {y_byt.shape[0]}, expected {batch_size}")
        if y_byt.shape[1] != self._y_dim:
            raise ValueError(f"y_dim mismatch: tensor has {y_byt.shape[1]}, expected {self._y_dim}")
        if y_byt.shape[2] != seq_len:
            raise ValueError(f"seq_len mismatch: tensor has {y_byt.shape[2]}, expected {seq_len}")

        self._prepare_runtime_batch(batch_size)
        self._filter.reset_net(is_train=True)
        self._filter.train_net.initialize_hidden(is_train=True)

        x0_batch = _resolve_x0_batch(
            context,
            batch_size=batch_size,
            x_dim=self._x_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self._filter.state_post = x0_batch.clone()
        self._filter.state_history = x0_batch.clone()
        self._filter.y_predict_history = torch.zeros((batch_size, self._y_dim, 1), device=self.device, dtype=self.dtype)

        for t in range(seq_len):
            y_t = y_byt[:, :, t].unsqueeze(2)  # [B,y,1]
            self._filter.filtering(y_t, self._filter.train_net)

        x_hist = self._filter.state_history  # [B,x,T+1] or [B,x,T]
        if x_hist.ndim != 3:
            raise RuntimeError(f"Unexpected state_history rank: {tuple(x_hist.shape)}")
        if x_hist.shape[2] == seq_len + 1:
            x_repo = x_hist[:, :, 1:]
        elif x_hist.shape[2] == seq_len:
            x_repo = x_hist
        else:
            raise RuntimeError(
                f"Unexpected state_history length: {x_hist.shape[2]} (expected {seq_len} or {seq_len + 1})"
            )

        return x_repo.permute(0, 2, 1).contiguous()

    @torch.no_grad()
    def _compute_validation_loss(
        self,
        *,
        val_dl: Any,
        max_batches: int,
    ) -> float:
        if self._filter is None:
            raise RuntimeError("Model is not initialized.")
        self._filter.train_net.eval()
        losses: List[float] = []
        loss_fn = torch.nn.MSELoss(reduction="mean")

        for bi, batch in enumerate(val_dl):
            if max_batches > 0 and bi >= max_batches:
                break
            x_raw, y_raw = _extract_batch_xy(batch)
            x = _to_tensor(x_raw, device=self.device, dtype=self.dtype)
            y = _to_tensor(y_raw, device=self.device, dtype=self.dtype)
            pred = self.predict(y, context=None, return_cov=False)
            if not isinstance(pred, torch.Tensor):
                raise TypeError(f"runtime_error: predict() must return Tensor during validation, got {type(pred)}")
            loss = loss_fn(pred, x)
            if not torch.isfinite(loss):
                raise FloatingPointError("train_nan: non-finite validation loss.")
            losses.append(float(loss.item()))

        self._filter.train_net.train()
        if not losses:
            return float("inf")
        return float(sum(losses) / len(losses))

    def adapt(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        budget: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Route-B policy for MAML-KNet v0 shift semantics:
        # budgeted per-timestep adaptation is not mapped yet; keep explicit no-op.
        self.adapt_updates_used = 0
        self._update_ledger(
            train_outer_updates_used=int(self.train_outer_updates_used),
            train_inner_updates_used=int(self.train_inner_updates_used),
            adapt_updates_used=0,
            supports_budgeted=False,
        )
        return {
            "status": "unsupported",
            "supports_budgeted": False,
            "adapt_updates_used": 0,
            "adapt_updates_per_step": {},
        }

    def save(self, out_dir: str) -> Dict[str, Any]:
        if self._filter is None:
            raise RuntimeError("setup() must be called before save().")
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        ckpt = out / "model.pt"
        torch.save({"state_dict": self._filter.train_net.state_dict()}, ckpt)
        self._saved_ckpt_path = ckpt
        return {"ckpt_path": str(ckpt)}

    def get_adapter_meta(self) -> Dict[str, Any]:
        return {
            "adapter_id": "maml_knet",
            "adapter_version": "s9_route_b_train_v1",
            "runtime_device": str(self.device),
            "covariance_support": False,
            "input_layout_bench": "BTD",
            "internal_layout_repo": "BYT_stepwise_filter",
            "class_path": "filter.Filter + state_dict_learner.Learner",
            "integration_mode": "import_model_only",
            "meta_train": {
                "meta_task_num": int(self._cfg.get("meta_task_num", self._cfg.get("task_num", 4))),
                "support_size": int(self._cfg.get("support_size", self._cfg.get("k_spt", 4))),
                "query_size": int(self._cfg.get("query_size", self._cfg.get("q_qry", 4))),
                "inner_steps": int(self._cfg.get("inner_steps", self._cfg.get("update_step", 1))),
                "inner_lr": float(self._cfg.get("inner_lr", self._cfg.get("update_lr", 5e-2))),
                "outer_lr": float(self._cfg.get("outer_lr", self._cfg.get("meta_lr", self._cfg.get("lr", 1e-3)))),
                "first_order": True,
                "budget_semantics": "train_max_updates counts outer optimizer.step() only",
            },
            "entrypoints": {
                "filter": "filter.py::Filter",
                "learner": "state_dict_learner.py::Learner",
                "meta_ref": "meta.py::Meta",
                "cli_ref": "main_linear.py::main",
            },
            "capabilities": {
                "train_supported": True,
                "eval_supported": True,
                "adapt_supported": False,
                "supports_budgeted": False,
            },
            "assumptions": {
                "A_input_layout": "Filter routines consume state/obs as [B,D,T]; adapter converts bench BTD internally.",
                "B_task_semantics": "v0 shift t0 per-step adaptation semantics are not mapped to MAML episode adaptation in this adapter.",
                "C_metrics_ownership": "third_party metrics are ignored; bench computes official metrics from adapter x_hat only.",
            },
            "how_to_verify": {
                "A_input_layout": "third_party/MAML_KalmanNet/MAML-KalmanNet/filter.py::compute_x_post/compute_x_post_qry",
                "B_task_semantics": "third_party/MAML_KalmanNet/MAML-KalmanNet/meta.py::forward",
                "C_metrics_ownership": "bench/runners/run_suite.py metrics block and this adapter eval()/predict().",
            },
        }

    def _init_ledger(self) -> None:
        if self._ledger_path is None:
            return
        if self._ledger_path.exists():
            return
        _write_json(
            self._ledger_path,
            {
                "train_updates_used": 0,
                "train_outer_updates_used": 0,
                "train_inner_updates_used": 0,
                "adapt_updates_used": 0,
                "track_id": self._run_ctx.get("track_id"),
                "init_id": self._run_ctx.get("init_id"),
                "supports_budgeted": False,
            },
        )

    def _update_ledger(
        self,
        *,
        train_outer_updates_used: int,
        train_inner_updates_used: int,
        adapt_updates_used: int,
        train_max_updates: Optional[int] = None,
        meta_tasks_seen: Optional[int] = None,
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

        current["train_outer_updates_used"] = int(train_outer_updates_used)
        current["train_inner_updates_used"] = int(train_inner_updates_used)
        # Backward-compatible alias for legacy consumers.
        current["train_updates_used"] = int(train_outer_updates_used)
        current["adapt_updates_used"] = int(adapt_updates_used)
        if train_max_updates is not None:
            current["train_max_updates"] = int(train_max_updates)
        if meta_tasks_seen is not None:
            current["meta_tasks_seen"] = int(meta_tasks_seen)
        if supports_budgeted is not None:
            current["supports_budgeted"] = bool(supports_budgeted)
        current["track_id"] = self._run_ctx.get("track_id")
        current["init_id"] = self._run_ctx.get("init_id")
        _write_json(self._ledger_path, current)
