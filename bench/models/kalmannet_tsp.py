# bench/models/kalmannet_tsp.py
from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


# --- Optional base-class import (keeps this file drop-in safe) ---
try:
    from .base import ModelAdapter  # type: ignore
except Exception:  # pragma: no cover
    class ModelAdapter:  # minimal fallback
        pass


def _import_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


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


def _resolve_repo_spec(model_cfg: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    """
    Accepts:
      repo: "third_party/KalmanNet_TSP"
      repo: {path: "third_party/KalmanNet_TSP", entrypoints: {...}}
    """
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
    elif repo_spec is None:
        repo_path = None
    else:
        raise TypeError(f"Unsupported repo spec type: {type(repo_spec)} (value={repo_spec!r})")

    if not repo_path:
        raise ValueError(
            "KalmanNet_TSP adapter requires model_cfg['repo'] (str or {path:..., entrypoints:{...}})."
        )

    # Interpret relative paths as relative to bench repo root (…/bench)
    bench_root = Path(__file__).resolve().parents[2]
    p = Path(repo_path).expanduser()
    repo_root = (bench_root / p).resolve() if not p.is_absolute() else p.resolve()

    return repo_root, entrypoints


@dataclass
class _KNetImports:
    KalmanNetNN: Any
    SystemModel: Any
    config_general_settings: Any  # function


def _load_kalmannet_tsp(repo_root: Path) -> _KNetImports:
    """
    Import strategy:
      1) add repo_root to sys.path and try normal imports (preferred)
      2) fall back to direct file import (robust if packages lack __init__.py)
    """
    if not repo_root.exists():
        raise FileNotFoundError(f"KalmanNet_TSP repo_root not found: {repo_root}")

    # Try normal import path first
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
    except Exception as e1:
        logger.warning("Normal import failed, falling back to file import: %r", e1)

    # File import fallback
    kn_file = repo_root / "KNet" / "KalmanNet_nn.py"
    sysmdl_file = repo_root / "Simulations" / "Linear_sysmdl.py"
    cfg_file = repo_root / "Simulations" / "config.py"

    if not kn_file.exists():
        raise FileNotFoundError(f"Missing: {kn_file}")
    if not sysmdl_file.exists():
        raise FileNotFoundError(f"Missing: {sysmdl_file}")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing: {cfg_file}")

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
    Import-mode adapter for KalmanNet_TSP.

    Key points (fixes your failures):
      - model_cfg['repo'] may be dict → we extract repo['path']
      - KalmanNet_TSP forward expects per-step y:[B,n,1]; we loop over T
      - 'device/f/h' missing → always build SystemModel and call NNBuild()
    """

    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.repo_root: Optional[Path] = None
        self.entrypoints: Dict[str, Any] = {}
        self.device: torch.device = torch.device("cpu")
        self.dtype: torch.dtype = torch.float32

        self._imports: Optional[_KNetImports] = None
        self._sys_model: Any = None

        self._m: Optional[int] = None  # x_dim
        self._n: Optional[int] = None  # y_dim
        self._F: Optional[torch.Tensor] = None
        self._H: Optional[torch.Tensor] = None
        self._Q: Optional[torch.Tensor] = None
        self._R: Optional[torch.Tensor] = None

    # ----------------------------
    # Setup
    # ----------------------------
    def setup(self, model_cfg: Dict[str, Any], system_info: Optional[Dict[str, Any]] = None) -> None:
        self.repo_root, self.entrypoints = _resolve_repo_spec(model_cfg)

        # bench runners pass --device separately; prefer model_cfg first, then system_info, then cpu
        cfg_device = model_cfg.get("device", None)
        sys_device = (system_info or {}).get("device", None)
        self.device = _as_torch_device(cfg_device or sys_device or "cpu")

        # be tolerant: if user asked cuda but cuda isn't available, don't hard-crash here
        if self.device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            self.device = torch.device("cpu")

        # imports from third_party repo
        self._imports = _load_kalmannet_tsp(self.repo_root)

        # extract system matrices/dims
        self._configure_system(model_cfg=model_cfg, system_info=system_info)

        # build args from repo defaults, then override minimal fields
        args = self._imports.config_general_settings()
        args.use_cuda = (self.device.type == "cuda")
        # args.n_batch is used as initial batch_size in repo; we overwrite model.batch_size at runtime anyway
        args.n_batch = int(model_cfg.get("eval_batch_size", model_cfg.get("batch_size", 1)) or 1)

        # allow optional overrides (keeps adapter controllable without editing third_party)
        if "in_mult_KNet" in model_cfg:
            args.in_mult_KNet = int(model_cfg["in_mult_KNet"])
        if "out_mult_KNet" in model_cfg:
            args.out_mult_KNet = int(model_cfg["out_mult_KNet"])

        # set T/T_test if available
        T_guess = int((system_info or {}).get("T", model_cfg.get("sequence_length", 0)) or 0)
        if T_guess > 0:
            args.T = T_guess
            args.T_test = T_guess

        # build SystemModel and KNet
        SystemModel = self._imports.SystemModel
        self._sys_model = SystemModel(self._F, self._Q, self._H, self._R, args.T, args.T_test)  # type: ignore[arg-type]

        # init x0/cov (repo expects these fields to exist)
        m = int(self._m or self._F.shape[0])
        m1x_0 = torch.zeros(m, 1, device=self.device, dtype=self.dtype)
        m2x_0 = torch.zeros(m, m, device=self.device, dtype=self.dtype)
        self._sys_model.InitSequence(m1x_0, m2x_0)

        KalmanNetNN = self._imports.KalmanNetNN
        self.model = KalmanNetNN()
        # NNBuild sets internal f/h and device handling in repo
        self.model.NNBuild(self._sys_model, args)  # type: ignore[call-arg]
        self.model.eval()

        logger.info(
            "KalmanNet_TSP adapter setup OK. device=%s repo=%s",
            str(self.device),
            str(self.repo_root),
        )

    def _configure_system(self, model_cfg: Dict[str, Any], system_info: Optional[Dict[str, Any]]) -> None:
        sysi = system_info or {}

        # dims
        x_dim = sysi.get("x_dim", sysi.get("m", model_cfg.get("x_dim")))
        y_dim = sysi.get("y_dim", sysi.get("n", model_cfg.get("y_dim")))
        if x_dim is None or y_dim is None:
            # last-resort defaults (suite_shift is 5/5; but we prefer explicit)
            x_dim = x_dim or 5
            y_dim = y_dim or 5

        self._m = int(x_dim)
        self._n = int(y_dim)

        # matrices: accept multiple key aliases
        F = sysi.get("F", sysi.get("A", sysi.get("Phi", None)))
        H = sysi.get("H", sysi.get("C", sysi.get("Obs", None)))

        if F is None:
            # fallback: identity
            F = torch.eye(self._m)
        if H is None:
            # fallback: identity (square case)
            H = torch.eye(self._n, self._m)

        self._F = _to_tensor(F, device=self.device, dtype=self.dtype)
        self._H = _to_tensor(H, device=self.device, dtype=self.dtype)

        # Q/R: accept either full matrix or scalar scales
        Q = sysi.get("Q", None)
        R = sysi.get("R", None)

        if Q is None:
            q2 = sysi.get("q2", model_cfg.get("q2", 1e-3))
            self._Q = torch.eye(self._m, device=self.device, dtype=self.dtype) * float(q2)
        else:
            self._Q = _to_tensor(Q, device=self.device, dtype=self.dtype)

        if R is None:
            r2 = sysi.get("r2", model_cfg.get("r2", 1e-3))
            self._R = torch.eye(self._n, device=self.device, dtype=self.dtype) * float(r2)
        else:
            self._R = _to_tensor(R, device=self.device, dtype=self.dtype)

    # ----------------------------
    # Predict (forward-only)
    # ----------------------------
    @torch.no_grad()
    def predict(
        self,
        y_seq: torch.Tensor,
        u_seq: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
        return_cov: bool = False,
    ) -> Any:
        if self.model is None or self._imports is None:
            raise RuntimeError("Model is not initialized. Call setup() first.")

        # Bench canonical layout: [B, T, y_dim]
        if not isinstance(y_seq, torch.Tensor):
            y_seq = torch.as_tensor(y_seq)

        if y_seq.dim() != 3:
            raise ValueError(f"Expected y_seq with shape [B,T,n], got {tuple(y_seq.shape)}")

        B, T, n = y_seq.shape
        if self._n is not None and int(n) != int(self._n):
            raise RuntimeError(f"y_dim mismatch: got {n}, expected {self._n}")

        # repo internal: [B, n, T]
        y_repo = y_seq.to(device=self.device, dtype=self.dtype).permute(0, 2, 1).contiguous()

        # init per-batch state
        # model.batch_size is used inside repo reshape paths
        try:
            setattr(self.model, "batch_size", int(B))
        except Exception:
            pass

        # init hidden state (repo defines this; if missing, let it fail loudly)
        if hasattr(self.model, "init_hidden_KNet"):
            self.model.init_hidden_KNet()

        # initial x0: accept from context if provided, else zeros
        m = int(self._m or 0)
        x0 = None
        ctx = context or {}
        for k in ("x0", "x_init", "x_init_mean", "m1x_0"):
            if k in ctx:
                x0 = ctx[k]
                break

        if x0 is None:
            x0_t = torch.zeros(B, m, 1, device=self.device, dtype=self.dtype)
        else:
            x0_t = _to_tensor(x0, device=self.device, dtype=self.dtype)
            if x0_t.dim() == 2:  # [B,m] or [m,1]
                if x0_t.shape[0] == m and x0_t.shape[1] == 1:
                    x0_t = x0_t.unsqueeze(0).repeat(B, 1, 1)  # [B,m,1]
                else:
                    x0_t = x0_t.view(B, m, 1)
            elif x0_t.dim() == 1:
                x0_t = x0_t.view(1, m, 1).repeat(B, 1, 1)
            elif x0_t.dim() == 3:
                pass
            else:
                raise ValueError(f"Unsupported x0 shape: {tuple(x0_t.shape)}")

        if hasattr(self.model, "InitSequence"):
            self.model.InitSequence(x0_t, T)

        # run per-time-step (repo expects y:[B,n,1])
        x_repo = torch.empty((B, m, T), device=self.device, dtype=self.dtype)
        for t in range(T):
            y_t = y_repo[:, :, t].unsqueeze(2)  # [B,n,1]
            x_t = self.model(y_t)  # expected [B,m,1]
            if isinstance(x_t, (tuple, list)):
                x_t = x_t[0]
            if x_t.dim() == 2:
                x_t = x_t.unsqueeze(2)
            x_repo[:, :, t] = x_t.squeeze(2)

        # back to bench canonical: [B, T, m]
        x_hat = x_repo.permute(0, 2, 1).contiguous()

        if return_cov:
            # KalmanNet_TSP repo doesn't expose covariance here; NA by policy
            return x_hat, None
        return x_hat

    # ----------------------------
    # Train / Adapt (not in Step3)
    # ----------------------------
    def train(self, train_loader=None, val_loader=None) -> None:
        raise NotImplementedError("train() is not implemented yet for KalmanNet_TSP (forward-only adapter).")

