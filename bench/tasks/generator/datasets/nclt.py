from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ...data_format import CANONICAL_LAYOUT_V0
from ..contract import GeneratorOutput, SplitCfg, TaskCfg, coerce_ntd_float32_output, make_split_cfg, make_task_cfg
from .common import (
    INTERNAL_SPLIT_PAYLOADS_KEY,
    DatasetMissingError,
    dataset_root_is_available,
    deep_merge,
    deterministic_windows,
    find_first_existing,
    load_xy_npz,
    resolve_dataset_root,
)
from ....utils.seeding import numpy_rng_v0, stable_int_seed_v0


NCLT_ENV_VAR = "NCLT_ROOT"
NCLT_DATASET_NAME = "nclt"

_NCLT_EXPECTED_LAYOUT = (
    "$NCLT_ROOT/prepared/2012-01-22.npz   # keys: x,y (rank-2 or rank-3)",
    "$NCLT_ROOT/prepared/2012-04-29.npz   # keys: x,y (rank-2 or rank-3)",
    "$NCLT_ROOT/sessions/<session>/xy.npz # alternative layout",
)

_NCLT_PREP_HINT = (
    "Download NCLT raw data from the official source and convert sessions to NPZ with x/y arrays. "
    "This benchmark keeps raw data external by design."
)


_NCLT_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    # Split-KalmanNet (TVT 2023, p.5)
    "split_kalmannet_2023": {
        "paper": "Split-KalmanNet (TVT 2023)",
        "sampling_hz": 1.0,
        "splits": {
            "train": {"session": "2012-01-22", "L": 80, "T": 50, "stride": 1},
            "val": {"session": "2012-01-22", "L": 5, "T": 200, "stride": 1},
            "test": {"session": "2012-04-29", "L": 1, "T": 2000, "stride": 1},
        },
    },
    # KalmanNet (TSP 2022, p.14)
    "kalmannet_2022": {
        "paper": "KalmanNet (TSP 2022)",
        "sampling_hz": None,
        "splits": {
            "train": {"session": "unspecified_train_pool", "L": 23, "T": 200, "stride": 1},
            "val": {"session": "unspecified_val_pool", "L": 2, "T": 200, "stride": 1},
            "test": {"session": "unspecified_test_pool", "L": 1, "T": 277, "stride": 1},
        },
    },
}

_NCLT_PROTOCOL_ALIASES = {
    "split_kalmannet_2023": "split_kalmannet_2023",
    "split_knet_2023": "split_kalmannet_2023",
    "splitkalmannet_2023": "split_kalmannet_2023",
    "kalmannet_2022": "kalmannet_2022",
    "knet_2022": "kalmannet_2022",
}


def has_nclt_dataset_root() -> bool:
    return dataset_root_is_available(NCLT_ENV_VAR)


def resolve_nclt_root() -> Path:
    return resolve_dataset_root(
        dataset=NCLT_DATASET_NAME,
        env_var=NCLT_ENV_VAR,
        expected_layout_lines=_NCLT_EXPECTED_LAYOUT,
        prep_hint=_NCLT_PREP_HINT,
    )


def nclt_protocol_spec() -> Dict[str, Any]:
    return json.loads(json.dumps(_NCLT_PROTOCOLS))


def _dataset_cfg(task_cfg: TaskCfg) -> Dict[str, Any]:
    raw = task_cfg.raw.get("dataset", {}) if isinstance(task_cfg.raw, Mapping) else {}
    sc = task_cfg.scenario_cfg.get("dataset", {}) if isinstance(task_cfg.scenario_cfg, Mapping) else {}
    raw_map = dict(raw) if isinstance(raw, Mapping) else {}
    sc_map = dict(sc) if isinstance(sc, Mapping) else {}
    return deep_merge(raw_map, sc_map)


def _resolve_protocol_id(dataset_cfg: Mapping[str, Any]) -> str:
    raw = dataset_cfg.get("protocol", "split_kalmannet_2023")
    key = str(raw).strip().lower()
    canonical = _NCLT_PROTOCOL_ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            f"Unsupported NCLT protocol '{raw}'. Supported protocols: {sorted(_NCLT_PROTOCOLS.keys())}"
        )
    return canonical


def _session_candidates(root: Path, session: str) -> Tuple[Path, ...]:
    sid = str(session)
    return (
        root / "prepared" / f"{sid}.npz",
        root / "sessions" / f"{sid}.npz",
        root / "sessions" / sid / "xy.npz",
        root / f"{sid}.npz",
    )


def _load_session(root: Path, session: str) -> Tuple[np.ndarray, np.ndarray, Path]:
    candidate = find_first_existing(_session_candidates(root, session))
    if candidate is None:
        expected = "\n".join(f"  - {p}" for p in _session_candidates(root, session))
        raise DatasetMissingError(
            dataset=NCLT_DATASET_NAME,
            env_var=NCLT_ENV_VAR,
            message=(
                f"Could not find session '{session}' under {root}.\n"
                f"Tried:\n{expected}\n"
                "Please convert raw logs into one of these NPZ paths with x/y arrays."
            ),
        )
    x_raw, y_raw, _extras = load_xy_npz(npz_path=candidate, dataset=NCLT_DATASET_NAME, env_var=NCLT_ENV_VAR)
    return x_raw, y_raw, candidate


def _resolve_split_spec(
    *,
    protocol: Mapping[str, Any],
    split_name: str,
    dataset_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    base = dict(protocol.get("splits", {}).get(split_name, {}) or {})
    overrides = dataset_cfg.get("split_overrides", {})
    if isinstance(overrides, Mapping):
        ov = overrides.get(split_name, {})
        if isinstance(ov, Mapping):
            base = deep_merge(base, dict(ov))
    stride_global = dataset_cfg.get("window_stride")
    if stride_global is not None and "stride" not in base:
        base["stride"] = int(stride_global)
    return {
        "session": str(base.get("session", "")),
        "L": int(base.get("L", 0)),
        "T": int(base.get("T", 0)),
        "stride": int(base.get("stride", 1)),
    }


def build_cache(
    task_cfg: TaskCfg,
    split_cfg: SplitCfg,
    seed: int,
    rng: Optional[np.random.Generator],
    device: Optional[str] = None,
    *,
    suite_name: str = "",
    scenario_id: str = "",
    task_family: str = "nclt_v0",
) -> GeneratorOutput:
    _ = (split_cfg, seed, rng, device)  # deterministic slicing has no stochastic branch in scaffold.
    root = resolve_nclt_root()
    dataset_cfg = _dataset_cfg(task_cfg)
    protocol_id = _resolve_protocol_id(dataset_cfg)
    protocol = dict(_NCLT_PROTOCOLS[protocol_id])

    session_cache: Dict[str, Tuple[np.ndarray, np.ndarray, Path]] = {}
    session_cursor: Dict[str, int] = {}
    split_payloads: Dict[str, Dict[str, Any]] = {}
    split_meta: Dict[str, Any] = {}

    for split_name in ("train", "val", "test"):
        spec = _resolve_split_spec(protocol=protocol, split_name=split_name, dataset_cfg=dataset_cfg)
        session = str(spec["session"])
        l_count = int(spec["L"])
        t_len = int(spec["T"])
        stride = int(spec["stride"])
        if l_count < 0 or t_len <= 0 or stride <= 0:
            raise ValueError(
                f"Invalid NCLT split spec for {split_name}: L={l_count}, T={t_len}, stride={stride}"
            )
        if session not in session_cache:
            session_cache[session] = _load_session(root, session)
            session_cursor.setdefault(session, 0)
        x_raw, y_raw, src_path = session_cache[session]
        start_window = int(session_cursor.get(session, 0))
        x_split, y_split, starts = deterministic_windows(
            x_raw=x_raw,
            y_raw=y_raw,
            n_windows=l_count,
            window_t=t_len,
            stride=stride,
            start_window=start_window,
            dataset=NCLT_DATASET_NAME,
            env_var=NCLT_ENV_VAR,
            context=f"NCLT split={split_name} session={session}",
        )
        session_cursor[session] = start_window + l_count
        split_payloads[split_name] = {
            "x": x_split,
            "y": y_split,
            "extras": {
                "window_start_seq": starts.astype(np.float32, copy=False),
            },
        }
        split_meta[split_name] = {
            "session": str(session),
            "source_npz": str(src_path),
            "N": int(l_count),
            "L": int(l_count),
            "T": int(t_len),
            "stride": int(stride),
            "sampling_hz": protocol.get("sampling_hz"),
            "paper_protocol": str(protocol_id),
        }

    proxy_split = "train"
    if int(split_payloads["train"]["x"].shape[0]) == 0:
        for s_name in ("val", "test"):
            if int(split_payloads[s_name]["x"].shape[0]) > 0:
                proxy_split = s_name
                break
    x_proxy = np.asarray(split_payloads[proxy_split]["x"], dtype=np.float32)
    y_proxy = np.asarray(split_payloads[proxy_split]["y"], dtype=np.float32)
    if x_proxy.ndim != 3 or y_proxy.ndim != 3:
        raise RuntimeError(f"NCLT proxy split must be rank-3 NTD, got x={x_proxy.shape}, y={y_proxy.shape}")

    ssm_common = {
        "type": "real_dataset",
        "dataset": "nclt",
        "params": {
            "protocol": str(protocol_id),
            "paper": str(protocol.get("paper", "")),
            "sampling_hz": protocol.get("sampling_hz"),
            "x0_kind": "from_dataset",
        },
    }
    meta_common: Dict[str, Any] = {
        "format_version": "0.1",
        "canonical_layout": CANONICAL_LAYOUT_V0,
        "schema_version": 1,
        "task_family": str(task_family),
        "suite_name": str(suite_name),
        "task_id": str(task_cfg.task_id),
        "scenario_id": str(scenario_id),
        "seed": int(seed),
        "dataset": {
            "name": "NCLT",
            "id": NCLT_DATASET_NAME,
            "root_env": NCLT_ENV_VAR,
            "root_path": str(root),
            "version": "external",
            "protocol": str(protocol_id),
            "paper": str(protocol.get("paper", "")),
            "external_data": True,
            "note": "Raw NCLT data is external. Bench caches converted NPZ artifacts only.",
        },
        "ssm": {
            "true": dict(ssm_common),
            "assumed": dict(ssm_common),
        },
        "mismatch": {
            "enabled": False,
            "kind": "none",
            "params": {},
        },
        "noise_schedule": {
            "enabled": False,
            "kind": "stationary",
            "params": {},
            "q2_t": "not_applicable_real_dataset",
            "r2_t": "not_applicable_real_dataset",
            "SoW_t": "not_applicable_real_dataset",
            "SoW_hat_t": None,
        },
        "switching": {
            "enabled": False,
            "models": [],
            "t_change": None,
            "retrain_window": 0,
        },
        "splits": split_meta,
    }
    out = GeneratorOutput(
        x=x_proxy,
        y=y_proxy,
        meta=meta_common,
        extras={INTERNAL_SPLIT_PAYLOADS_KEY: split_payloads},
    )
    return coerce_ntd_float32_output(out)


def generate_nclt_v0(
    *,
    suite_name: str,
    task_cfg_dict: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    seed: int,
    scenario_id: str,
    task_family: str = "nclt_v0",
) -> Tuple[GeneratorOutput, Optional[np.ndarray], Optional[np.ndarray]]:
    task_cfg = make_task_cfg(task_cfg_dict, scenario_cfg=scenario_cfg)
    split_cfg = make_split_cfg(task_cfg_dict)
    data_seed = stable_int_seed_v0("data", suite_name, task_cfg.task_id, scenario_id, int(seed))
    rng = numpy_rng_v0(data_seed)
    out = build_cache(
        task_cfg=task_cfg,
        split_cfg=split_cfg,
        seed=int(seed),
        rng=rng,
        device=None,
        suite_name=suite_name,
        scenario_id=scenario_id,
        task_family=task_family,
    )
    return out, None, None
