"""
Adaptive-KNet-ICASSP24 adapter (import-mode) — Step 5 MVP

Goal:
- Convert bench_generated y_seq [B,T,y] (NTD) to repo-expected layouts and run a single forward pass.
- third_party code is NOT modified.

Reality check:
Adaptive-KNet repo structures vary. This adapter uses a heuristic importer:
- Prefer modules under `filters/` and classes containing "KNet" (case-insensitive)
- Instantiate with common patterns (no-arg, ctor(sys_model), Build(sys_model))
- Try forward with multiple input layouts: [B,y,T] then [B,T,y] then [T,B,y]

ASSUMPTION + HOW TO VERIFY:
- Repo uses a torch.nn.Module-like class for the estimator.
- Verify in third_party/Adaptive-KNet-ICASSP24/filters and pipelines where forward/inference happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import importlib
import inspect
import logging
import pkgutil
import sys

import numpy as np
import torch

try:
    from .base import ModelAdapter  # type: ignore
except Exception:  # pragma: no cover
    ModelAdapter = object


_LOG = logging.getLogger(__name__)


def _get_logger() -> logging.Logger:
    try:
        from ..utils.logging import get_logger  # type: ignore
        return get_logger(__name__)
    except Exception:
        return _LOG


def _bench_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_repo_on_syspath(repo_root: Path) -> None:
    if not repo_root.exists():
        raise FileNotFoundError(f"Adaptive-KNet repo root not found: {repo_root}")
    p = str(repo_root)
    if p not in sys.path:
        sys.path.insert(0, p)


def _to_tensor(a: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a.to(device=device, dtype=dtype)
    return torch.as_tensor(a, dtype=dtype, device=device)


def _bench_ntd_to_repo_byt(y_ntd: torch.Tensor) -> torch.Tensor:
    return y_ntd.transpose(1, 2).contiguous()


def _infer_out_to_bench_ntd(x_hat: torch.Tensor, T: int) -> torch.Tensor:
    if x_hat.ndim != 3:
        raise ValueError(f"Expected 3D output, got {tuple(x_hat.shape)}")

    B, A, C = x_hat.shape
    if A == T:  # [B,T,x]
        return x_hat
    if C == T:  # [B,x,T]
        return x_hat.transpose(1, 2)
    if B == T:  # [T,B,x]
        return x_hat.permute(1, 0, 2)

    _get_logger().warning("Unknown output layout; returning as-is. shape=%s T=%d", tuple(x_hat.shape), T)
    return x_hat


@dataclass
class _SystemInfo:
    F: Optional[np.ndarray]
    H: Optional[np.ndarray]
    T: int
    x_dim: int
    y_dim: int
    meta: Dict[str, Any]


class AdaptiveKNetAdapter(ModelAdapter):
    def __init__(self) -> None:
        self._log = _get_logger()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.repo_root: Optional[Path] = None

        self.sys_model: Any = None
        self.model: Optional[torch.nn.Module] = None

    def setup(self, cfg: Dict[str, Any], system_info: Dict[str, Any]) -> None:
        requested = (cfg.get("device") or cfg.get("runner_device") or "cuda").lower()
        if requested.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        bench_root = _bench_root_from_this_file()
        rel = (cfg.get("repo") or {}).get("path") or "third_party/Adaptive-KNet-ICASSP24"
        self.repo_root = (bench_root / rel).resolve()
        _ensure_repo_on_syspath(self.repo_root)

        meta = system_info.get("meta") or {}
        T = int(system_info.get("T"))
        x_dim = int(system_info.get("x_dim"))
        y_dim = int(system_info.get("y_dim"))
        F = system_info.get("F", None)
        H = system_info.get("H", None)

        si = _SystemInfo(F=F, H=H, T=T, x_dim=x_dim, y_dim=y_dim, meta=meta)
        self.sys_model = self._try_build_system_model(si)  # best-effort; may be None

        # heuristic import of model core
        ModelCls = self._find_model_class(repo_root=self.repo_root)
        self.model = self._try_instantiate_model(ModelCls, self.sys_model)
        self.model.to(self.device)
        self.model.eval()

        self._log.info("Adaptive-KNet adapter setup OK. device=%s repo=%s model=%s",
                       self.device, self.repo_root, ModelCls.__name__)

    def load(self, ckpt_path: str) -> None:
        if self.model is None:
            raise RuntimeError("setup() must be called before load().")
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self._log.info("Loaded checkpoint: %s", ckpt_path)

    @torch.no_grad()
    def predict(
        self,
        y_seq: torch.Tensor,
        u_seq: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
        return_cov: bool = False,
    ) -> Any:
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call setup() first.")

        y_ntd = _to_tensor(y_seq, device=self.device, dtype=self.dtype)

        # Try multiple layouts (repo uncertain)
        candidates = [
            ("BYT", _bench_ntd_to_repo_byt(y_ntd)),  # [B,y,T]
            ("BTY", y_ntd.contiguous()),             # [B,T,y]
            ("TBY", y_ntd.permute(1, 0, 2).contiguous()),  # [T,B,y]
        ]

        last_err = None
        out = None
        for name, y_in in candidates:
            try:
                out = self._try_forward(self.model, y_in)
                self._log.info("Adaptive-KNet forward OK with input layout=%s shape=%s", name, tuple(y_in.shape))
                break
            except Exception as e:
                last_err = e
                continue
        if out is None:
            raise RuntimeError(
                "Adaptive-KNet forward failed for all tried layouts. "
                "HOW TO VERIFY: third_party/Adaptive-KNet-ICASSP24/filters or pipelines for expected input dims."
            ) from last_err

        T = int((context or {}).get("T") or y_ntd.shape[1])
        x_hat = _infer_out_to_bench_ntd(out, T=T)

        if return_cov:
            return x_hat, None
        return x_hat

    def adapt(
        self,
        y_seq: torch.Tensor,
        u_seq: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
        budget: Optional[Dict[str, Any]] = None,
    ) -> Any:
        raise NotImplementedError("Step6 implements budgeted adaptation via repo-specific API (if supported).")

    def save(self, out_dir: str) -> None:
        if self.model is None:
            raise RuntimeError("setup() must be called before save().")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "model_state.pt")

    # -----------------------------
    # internals
    # -----------------------------

    def _try_build_system_model(self, si: _SystemInfo) -> Any:
        """
        Best-effort SystemModel builder.
        If the repo doesn't expose a SystemModel, returns None.
        """
        # Try common KalmanNet-style SystemModel name/location
        for mod_name in ["simulations.linear_sysmdl", "simulations.Linear_sysmdl", "Simulations.Linear_sysmdl"]:
            try:
                m = importlib.import_module(mod_name)
                SystemModel = getattr(m, "SystemModel")
                break
            except Exception:
                SystemModel = None
        if SystemModel is None:
            return None

        F = si.F if si.F is not None else np.eye(si.x_dim, dtype=np.float32)
        H = si.H if si.H is not None else np.eye(si.y_dim, si.x_dim, dtype=np.float32)

        # q2/r2 are optional; use suite defaults if absent
        try:
            q2 = float(si.meta["noise"]["pre_shift"]["Q"]["q2"])
        except Exception:
            q2 = 1.0e-3
        try:
            r2 = float(si.meta["noise"]["pre_shift"]["R"]["r2"])
        except Exception:
            r2 = 1.0e-3

        Q = q2 * np.eye(si.x_dim, dtype=np.float32)
        R = r2 * np.eye(si.y_dim, dtype=np.float32)

        tF = _to_tensor(F, self.device)
        tH = _to_tensor(H, self.device)
        tQ = _to_tensor(Q, self.device)
        tR = _to_tensor(R, self.device)

        # Attempt a few ctor variants
        for args in [(tF, tQ, tH, tR, si.T, si.T), (tF, tQ, tH, tR, si.T)]:
            try:
                return SystemModel(*args)
            except Exception:
                pass
        try:
            return SystemModel(F=tF, Q=tQ, H=tH, R=tR, T=si.T, T_test=si.T)
        except Exception:
            return None

    def _find_model_class(self, repo_root: Path) -> Type[torch.nn.Module]:
        """
        Heuristic:
        - scan subpackages `filters`, `mnets`, `hnets`
        - choose first class that subclasses nn.Module and contains "knet" in name
        """
        preferred_pkgs = ["filters", "mnets", "hnets", "models"]
        for pkg in preferred_pkgs:
            pkg_path = repo_root / pkg
            if not pkg_path.exists() or not pkg_path.is_dir():
                continue

            for m in pkgutil.iter_modules([str(pkg_path)]):
                mod_name = f"{pkg}.{m.name}"
                try:
                    mod = importlib.import_module(mod_name)
                except Exception:
                    continue

                for _, obj in vars(mod).items():
                    if not isinstance(obj, type):
                        continue
                    if not issubclass(obj, torch.nn.Module):
                        continue
                    nm = obj.__name__.lower()
                    if "knet" in nm or ("kalman" in nm and "net" in nm):
                        self._log.info("Selected Adaptive-KNet model class candidate: %s.%s", mod_name, obj.__name__)
                        return obj

        raise ImportError(
            "Could not find a torch.nn.Module class containing 'KNet' in Adaptive-KNet repo. "
            "HOW TO VERIFY: third_party/Adaptive-KNet-ICASSP24/filters for the estimator class."
        )

    def _try_instantiate_model(self, ModelCls: Type[torch.nn.Module], sys_model: Any) -> torch.nn.Module:
        # no-arg
        try:
            m = ModelCls()
            if hasattr(m, "Build") and callable(getattr(m, "Build")) and sys_model is not None:
                try:
                    m.Build(sys_model)
                except Exception:
                    pass
            return m
        except Exception:
            pass

        # ctor(sys_model)
        try:
            if sys_model is not None:
                return ModelCls(sys_model)
        except Exception:
            pass

        # ctor(x_dim,y_dim) best-effort (rare)
        try:
            return ModelCls()
        except Exception as e:
            raise RuntimeError(
                "Failed to instantiate Adaptive-KNet model class. "
                "HOW TO VERIFY: constructor signature of the selected class."
            ) from e

    def _try_forward(self, model: torch.nn.Module, y_in: torch.Tensor) -> torch.Tensor:
        try:
            out = model(y_in)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not isinstance(out, torch.Tensor):
                raise TypeError(f"Model output is not a Tensor: {type(out)}")
            return out
        except Exception as e1:
            # Try forward with named arg if signature supports it
            try:
                sig = inspect.signature(model.forward)
                if "y" in sig.parameters:
                    out = model.forward(y=y_in)
                else:
                    out = model.forward(y_in)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if not isinstance(out, torch.Tensor):
                    raise TypeError(f"Model output is not a Tensor: {type(out)}")
                return out
            except Exception as e2:
                raise RuntimeError(f"forward failed. primary={repr(e1)} secondary={repr(e2)}")

