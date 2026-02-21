from __future__ import annotations

import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

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
    candidates = (
        ("noise", "pre_shift", "Q", "q2"),
        ("scenario_cfg", "noise", "pre_shift", "Q", "q2"),
    )
    q2 = None
    for path in candidates:
        v = _lookup_nested(meta, path)
        if v is not None:
            try:
                q2 = float(v)
                break
            except Exception:
                pass

    candidates_r = (
        ("noise", "pre_shift", "R", "r2"),
        ("scenario_cfg", "noise", "pre_shift", "R", "r2"),
    )
    r2 = None
    for path in candidates_r:
        v = _lookup_nested(meta, path)
        if v is not None:
            try:
                r2 = float(v)
                break
            except Exception:
                pass

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
        raise ValueError(
            "KalmanNet_TSP adapter requires model_cfg['repo'] as string or dict with 'path'."
        )

    bench_root = Path(__file__).resolve().parents[2]
    repo_candidate = Path(repo_path).expanduser()
    repo_root = (bench_root / repo_candidate).resolve() if not repo_candidate.is_absolute() else repo_candidate.resolve()
    return repo_root, entrypoints


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
        logger.warning("Normal KalmanNet_TSP import failed, trying file import fallback: %r", e)

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


def _call_general_settings_safely(general_settings: Any) -> Any:
    # third_party/Simulations/config.py uses argparse.parse_args();
    # isolate benchmark CLI flags so this call is deterministic.
    old_argv = list(sys.argv)
    argv0 = old_argv[0] if old_argv else "bench"
    try:
        sys.argv = [argv0]
        return general_settings()
    finally:
        sys.argv = old_argv


def _resolve_x0(x0: Any, *, batch_size: int, x_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = _to_tensor(x0, device=device, dtype=dtype)
    if x.ndim == 1:
        if x.shape[0] != x_dim:
            raise ValueError(f"x0 1D shape mismatch: expected [{x_dim}], got {tuple(x.shape)}")
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
    raise ValueError(f"Unsupported x0 rank {x.ndim}; expected 1D/2D/3D.")


class KalmanNetTSPAdapter(ModelAdapter):
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

        self.last_layout: Optional[str] = None
        self.last_class: Optional[str] = None

    def setup(self, model_cfg: Dict[str, Any], system_info: Optional[Dict[str, Any]] = None) -> None:
        system_info = system_info or {}
        self.repo_root, self.entrypoints = _resolve_repo_spec(model_cfg)
        self._imports = _load_kalmannet_tsp(self.repo_root)

        requested_device = model_cfg.get("device", None) or system_info.get("device", None) or "cpu"
        self.device = _as_torch_device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            self.device = torch.device("cpu")

        meta = _coerce_meta_dict(system_info)

        x_dim = system_info.get("x_dim", model_cfg.get("x_dim", meta.get("x_dim")))
        y_dim = system_info.get("y_dim", model_cfg.get("y_dim", meta.get("y_dim")))
        if x_dim is None or y_dim is None:
            raise ValueError("system_info must provide x_dim and y_dim for KalmanNet_TSP setup.")
        self._x_dim = int(x_dim)
        self._y_dim = int(y_dim)

        T = system_info.get("T", model_cfg.get("T", meta.get("T", model_cfg.get("sequence_length", 1))))
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
        q2 = system_info.get("q2", model_cfg.get("q2", q2_meta if q2_meta is not None else 1e-3))
        r2 = system_info.get("r2", model_cfg.get("r2", r2_meta if r2_meta is not None else 1e-3))

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
        args.n_batch = int(model_cfg.get("eval_batch_size", model_cfg.get("batch_size", 1)) or 1)

        if "in_mult_KNet" in model_cfg:
            args.in_mult_KNet = int(model_cfg["in_mult_KNet"])
        if "out_mult_KNet" in model_cfg:
            args.out_mult_KNet = int(model_cfg["out_mult_KNet"])

        SystemModel = self._imports.SystemModel
        self._sys_model = SystemModel(self._F, self._Q, self._H, self._R, args.T, args.T_test)

        m1x_0 = torch.zeros(self._x_dim, 1, device=self.device, dtype=self.dtype)
        m2x_0 = torch.zeros(self._x_dim, self._x_dim, device=self.device, dtype=self.dtype)
        self._sys_model.InitSequence(m1x_0, m2x_0)

        KalmanNetNN = self._imports.KalmanNetNN
        self.model = KalmanNetNN()
        self.model.NNBuild(self._sys_model, args)
        self.model.eval()

        self.last_class = type(self.model).__name__
        self.last_layout = "stepwise_BTD_to_BnT"

    def train(self, train_loader: Any, val_loader: Any) -> None:
        raise NotImplementedError("KalmanNet_TSP adapter MVP supports inference only.")

    def load(self, ckpt_path: str) -> None:
        if self.model is None:
            raise RuntimeError("setup() must be called before load().")
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)

    @torch.no_grad()
    def predict(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        return_cov: bool = False,
    ) -> Any:
        if self.model is None:
            raise RuntimeError("setup() must be called before predict().")
        if self._x_dim is None or self._y_dim is None:
            raise RuntimeError("Adapter dimensions are not initialized.")

        y = _to_tensor(y_seq, device=self.device, dtype=self.dtype)
        if y.ndim != 3:
            raise ValueError(f"Expected y_seq [B,T,y_dim], got shape {tuple(y.shape)}")

        B, T, y_dim = y.shape
        if int(y_dim) != int(self._y_dim):
            raise RuntimeError(f"y_dim mismatch: got {y_dim}, expected {self._y_dim}")

        y_repo = y.permute(0, 2, 1).contiguous()

        self.model.batch_size = int(B)  # type: ignore[attr-defined]
        if not hasattr(self.model, "init_hidden_KNet"):
            raise RuntimeError("KalmanNet_TSP model missing init_hidden_KNet().")
        self.model.init_hidden_KNet()  # type: ignore[attr-defined]

        x0 = None
        ctx = context or {}
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
            raise RuntimeError("KalmanNet_TSP model missing InitSequence().")
        self.model.InitSequence(m1_0_batch, int(T))  # type: ignore[attr-defined]

        x_repo = torch.empty((B, self._x_dim, T), device=self.device, dtype=self.dtype)
        for t in range(T):
            y_t = y_repo[:, :, t].unsqueeze(2)  # [B,n,1]
            x_t = self.model(y_t)
            if isinstance(x_t, (tuple, list)):
                x_t = x_t[0]
            if not isinstance(x_t, torch.Tensor):
                raise TypeError(f"Model output must be Tensor, got {type(x_t)}")
            if x_t.ndim == 2:
                x_t = x_t.unsqueeze(2)
            if x_t.shape != (B, self._x_dim, 1):
                raise RuntimeError(
                    f"Unexpected model output shape at t={t}: {tuple(x_t.shape)} "
                    f"(expected {(B, self._x_dim, 1)})"
                )
            x_repo[:, :, t] = x_t[:, :, 0]

        x_hat = x_repo.permute(0, 2, 1).contiguous()  # [B,T,x_dim]
        if return_cov:
            return x_hat, None
        return x_hat

    def adapt(
        self,
        y_seq: Any,
        u_seq: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        budget: Optional[Any] = None,
    ) -> None:
        # Frozen-track MVP: no online adaptation.
        return None

    def save(self, out_dir: str) -> None:
        if self.model is None:
            raise RuntimeError("setup() must be called before save().")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "model_state.pt")

