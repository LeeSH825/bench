from __future__ import annotations

import os
import random
import hashlib
from typing import Any

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set global seeds for reproducibility.

    Locked rules:
    - suite YAML: runner.deterministic=true (default)
    - FAIRNESS: 동일 seed / deterministic 옵션 원칙
      (SoT: /mnt/data/FAIRNESS.md, /mnt/data/suite_*.yaml)

    Behavior (best effort):
    - Python random, PYTHONHASHSEED
    - numpy random (if available)
    - torch CPU/CUDA seeds (if available)
    - cudnn deterministic + benchmark off
    - torch deterministic algorithms (if supported)

    Note:
    - Complete determinism is not guaranteed for all ops/hardware.
      If repo/model cannot be fully deterministic, record it and report variance across seeds.
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # numpy (optional)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    # torch (optional)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # torch >= 1.8
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except Exception:
        pass


# Backward-compatible alias (if Step 2 used set_global_seed name)
def set_global_seed(seed: int, deterministic: bool = True) -> None:
    set_seed(seed, deterministic=deterministic)

def stable_int_seed_v0(*parts: Any) -> int:
    """
    Stable 32-bit seed from arbitrary parts (strings/ints).
    """
    s = "|".join(str(p) for p in parts)
    h = hashlib.sha1(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)

def numpy_rng_v0(seed: int) -> np.random.Generator:
    """
    Numpy Generator wrapper (deterministic given seed).
    """
    return np.random.default_rng(int(seed))
