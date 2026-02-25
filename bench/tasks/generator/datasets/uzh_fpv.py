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


UZH_FPV_ENV_VAR = "UZH_FPV_ROOT"
UZH_FPV_DATASET_NAME = "uzh_fpv"

_UZH_EXPECTED_LAYOUT = (
    "$UZH_FPV_ROOT/prepared/6th_indoor_forward_facing.npz # keys: x,y",
    "$UZH_FPV_ROOT/sequences/6th_indoor_forward_facing.npz # alternative",
    "$UZH_FPV_ROOT/sequences/6th_indoor_forward_facing/xy.npz # alternative",
)

_UZH_PREP_HINT = (
    "Download/prepare UZH-FPV raw data and convert the target trajectory into NPZ with x/y arrays. "
    "The benchmark does not vendor raw datasets."
)


_UZH_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    # MAML-KalmanNet (TSP 2025, p.14)
    "maml_kalmannet_2025": {
        "paper": "MAML-KalmanNet (TSP 2025)",
        "session": "6th_indoor_forward_facing",
        "sampling_hz": 100.0,
        "total_steps": 3020,
        "splits": {
            "train": {"L": 25, "T": 80, "stride": 80},
            "val": {"L": 0, "T": 80, "stride": 80},
            "test": {"L": 1, "T": 1020},
        },
    }
}

_UZH_PROTOCOL_ALIASES = {
    "maml_kalmannet_2025": "maml_kalmannet_2025",
    "maml_knet_2025": "maml_kalmannet_2025",
    "uzh_fpv_2025": "maml_kalmannet_2025",
}


def has_uzh_fpv_dataset_root() -> bool:
    return dataset_root_is_available(UZH_FPV_ENV_VAR)


def resolve_uzh_fpv_root() -> Path:
    return resolve_dataset_root(
        dataset=UZH_FPV_DATASET_NAME,
        env_var=UZH_FPV_ENV_VAR,
        expected_layout_lines=_UZH_EXPECTED_LAYOUT,
        prep_hint=_UZH_PREP_HINT,
    )


def uzh_fpv_protocol_spec() -> Dict[str, Any]:
    return json.loads(json.dumps(_UZH_PROTOCOLS))


def _dataset_cfg(task_cfg: TaskCfg) -> Dict[str, Any]:
    raw = task_cfg.raw.get("dataset", {}) if isinstance(task_cfg.raw, Mapping) else {}
    sc = task_cfg.scenario_cfg.get("dataset", {}) if isinstance(task_cfg.scenario_cfg, Mapping) else {}
    raw_map = dict(raw) if isinstance(raw, Mapping) else {}
    sc_map = dict(sc) if isinstance(sc, Mapping) else {}
    return deep_merge(raw_map, sc_map)


def _resolve_protocol_id(dataset_cfg: Mapping[str, Any]) -> str:
    raw = dataset_cfg.get("protocol", "maml_kalmannet_2025")
    key = str(raw).strip().lower()
    canonical = _UZH_PROTOCOL_ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            f"Unsupported UZH-FPV protocol '{raw}'. Supported protocols: {sorted(_UZH_PROTOCOLS.keys())}"
        )
    return canonical


def _session_candidates(root: Path, session: str) -> Tuple[Path, ...]:
    sid = str(session)
    alias = sid.replace("-", "_").replace(" ", "_")
    return (
        root / "prepared" / f"{sid}.npz",
        root / "prepared" / f"{alias}.npz",
        root / "sequences" / f"{sid}.npz",
        root / "sequences" / f"{alias}.npz",
        root / "sequences" / sid / "xy.npz",
        root / "sequences" / alias / "xy.npz",
        root / f"{sid}.npz",
        root / f"{alias}.npz",
    )


def _load_session(root: Path, session: str) -> Tuple[np.ndarray, np.ndarray, Path]:
    candidate = find_first_existing(_session_candidates(root, session))
    if candidate is None:
        expected = "\n".join(f"  - {p}" for p in _session_candidates(root, session))
        raise DatasetMissingError(
            dataset=UZH_FPV_DATASET_NAME,
            env_var=UZH_FPV_ENV_VAR,
            message=(
                f"Could not find UZH-FPV session '{session}' under {root}.\n"
                f"Tried:\n{expected}\n"
                "Please convert/prep the sequence into one of these NPZ paths with x/y arrays."
            ),
        )
    x_raw, y_raw, _extras = load_xy_npz(npz_path=candidate, dataset=UZH_FPV_DATASET_NAME, env_var=UZH_FPV_ENV_VAR)
    return x_raw, y_raw, candidate


def _extract_uzh_splits(
    *,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    protocol: Mapping[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    splits = dict(protocol.get("splits", {}) or {})
    train_spec = dict(splits.get("train", {}) or {})
    val_spec = dict(splits.get("val", {}) or {})
    test_spec = dict(splits.get("test", {}) or {})

    l_train = int(train_spec.get("L", 25))
    t_train = int(train_spec.get("T", 80))
    stride_train = int(train_spec.get("stride", 80))
    l_val = int(val_spec.get("L", 0))
    t_val = int(val_spec.get("T", t_train))
    l_test = int(test_spec.get("L", 1))
    t_test = int(test_spec.get("T", 1020))
    if l_train < 0 or l_val < 0 or l_test < 0 or t_train <= 0 or t_val <= 0 or t_test <= 0:
        raise ValueError(
            "Invalid UZH split spec: "
            f"train(L={l_train},T={t_train},stride={stride_train}) "
            f"val(L={l_val},T={t_val}) test(L={l_test},T={t_test})"
        )

    x = np.asarray(x_raw, dtype=np.float64)
    y = np.asarray(y_raw, dtype=np.float64)
    if x.ndim != y.ndim:
        raise DatasetMissingError(
            dataset=UZH_FPV_DATASET_NAME,
            env_var=UZH_FPV_ENV_VAR,
            message=f"UZH-FPV x/y rank mismatch x.shape={x.shape}, y.shape={y.shape}",
        )

    payloads: Dict[str, Dict[str, Any]] = {}
    split_meta: Dict[str, Dict[str, Any]] = {}

    x_train, y_train, starts_train = deterministic_windows(
        x_raw=x,
        y_raw=y,
        n_windows=l_train,
        window_t=t_train,
        stride=max(1, stride_train),
        start_window=0,
        dataset=UZH_FPV_DATASET_NAME,
        env_var=UZH_FPV_ENV_VAR,
        context="UZH-FPV train split",
    )
    payloads["train"] = {
        "x": x_train,
        "y": y_train,
        "extras": {"window_start_seq": starts_train.astype(np.float32, copy=False)},
    }

    x_val, y_val, starts_val = deterministic_windows(
        x_raw=x,
        y_raw=y,
        n_windows=l_val,
        window_t=t_val,
        stride=max(1, stride_train),
        start_window=l_train,
        dataset=UZH_FPV_DATASET_NAME,
        env_var=UZH_FPV_ENV_VAR,
        context="UZH-FPV val split",
    )
    payloads["val"] = {
        "x": x_val,
        "y": y_val,
        "extras": {"window_start_seq": starts_val.astype(np.float32, copy=False)},
    }

    if x.ndim == 2:
        total_steps = int(x.shape[0])
        protocol_total_steps = int(protocol.get("total_steps", total_steps))
        if total_steps < t_test:
            raise DatasetMissingError(
                dataset=UZH_FPV_DATASET_NAME,
                env_var=UZH_FPV_ENV_VAR,
                message=f"UZH-FPV total steps ({total_steps}) shorter than test T={t_test}",
            )
        if total_steps >= protocol_total_steps:
            test_start = protocol_total_steps - t_test
        else:
            test_start = total_steps - t_test
        test_start = max(0, int(test_start))
        if l_test != 1:
            raise ValueError("UZH-FPV protocol currently supports test L=1 only")
        x_test = np.asarray(x[test_start : test_start + t_test, :], dtype=np.float32)[None, :, :]
        y_test = np.asarray(y[test_start : test_start + t_test, :], dtype=np.float32)[None, :, :]
        starts_test = np.asarray([test_start], dtype=np.int64)
    elif x.ndim == 3:
        if l_test != 1:
            raise ValueError("UZH-FPV pre-windowed path currently supports test L=1 only")
        n_seq = int(x.shape[0])
        test_index = max(0, min(n_seq - 1, l_train + l_val))
        if int(x.shape[1]) < t_test or int(y.shape[1]) < t_test:
            raise DatasetMissingError(
                dataset=UZH_FPV_DATASET_NAME,
                env_var=UZH_FPV_ENV_VAR,
                message=f"UZH-FPV pre-windowed sequence length too short for test T={t_test}",
            )
        x_test = np.asarray(x[test_index : test_index + 1, :t_test, :], dtype=np.float32)
        y_test = np.asarray(y[test_index : test_index + 1, :t_test, :], dtype=np.float32)
        starts_test = np.asarray([test_index], dtype=np.int64)
    else:
        raise DatasetMissingError(
            dataset=UZH_FPV_DATASET_NAME,
            env_var=UZH_FPV_ENV_VAR,
            message=f"Unsupported UZH-FPV array rank x.shape={x.shape}, y.shape={y.shape}",
        )

    payloads["test"] = {
        "x": x_test,
        "y": y_test,
        "extras": {"window_start_seq": starts_test.astype(np.float32, copy=False)},
    }

    split_meta["train"] = {
        "session": str(protocol.get("session", "")),
        "N": int(l_train),
        "L": int(l_train),
        "T": int(t_train),
        "stride": int(stride_train),
        "paper_ratio": "2/3",
    }
    split_meta["val"] = {
        "session": str(protocol.get("session", "")),
        "N": int(l_val),
        "L": int(l_val),
        "T": int(t_val),
        "stride": int(stride_train),
        "paper_ratio": "not_specified",
    }
    split_meta["test"] = {
        "session": str(protocol.get("session", "")),
        "N": int(l_test),
        "L": int(l_test),
        "T": int(t_test),
        "stride": None,
        "paper_ratio": "remaining_1/3",
    }
    return payloads, split_meta


def build_cache(
    task_cfg: TaskCfg,
    split_cfg: SplitCfg,
    seed: int,
    rng: Optional[np.random.Generator],
    device: Optional[str] = None,
    *,
    suite_name: str = "",
    scenario_id: str = "",
    task_family: str = "uzh_fpv_v0",
) -> GeneratorOutput:
    _ = (split_cfg, seed, rng, device)  # deterministic slicing only.
    root = resolve_uzh_fpv_root()
    dataset_cfg = _dataset_cfg(task_cfg)
    protocol_id = _resolve_protocol_id(dataset_cfg)
    protocol = dict(_UZH_PROTOCOLS[protocol_id])
    session = str(dataset_cfg.get("session", protocol.get("session", "6th_indoor_forward_facing")))
    x_raw, y_raw, src_path = _load_session(root, session)

    split_payloads, split_meta = _extract_uzh_splits(x_raw=x_raw, y_raw=y_raw, protocol=protocol)
    split_meta["train"]["source_npz"] = str(src_path)
    split_meta["val"]["source_npz"] = str(src_path)
    split_meta["test"]["source_npz"] = str(src_path)
    split_meta["train"]["paper_protocol"] = str(protocol_id)
    split_meta["val"]["paper_protocol"] = str(protocol_id)
    split_meta["test"]["paper_protocol"] = str(protocol_id)
    split_meta["train"]["sampling_hz"] = protocol.get("sampling_hz")
    split_meta["val"]["sampling_hz"] = protocol.get("sampling_hz")
    split_meta["test"]["sampling_hz"] = protocol.get("sampling_hz")

    proxy_split = "train"
    if int(split_payloads["train"]["x"].shape[0]) == 0:
        proxy_split = "test"
    x_proxy = np.asarray(split_payloads[proxy_split]["x"], dtype=np.float32)
    y_proxy = np.asarray(split_payloads[proxy_split]["y"], dtype=np.float32)
    if x_proxy.ndim != 3 or y_proxy.ndim != 3:
        raise RuntimeError(f"UZH-FPV proxy split must be rank-3 NTD, got x={x_proxy.shape}, y={y_proxy.shape}")

    ssm_common = {
        "type": "real_dataset",
        "dataset": "uzh_fpv",
        "params": {
            "protocol": str(protocol_id),
            "paper": str(protocol.get("paper", "")),
            "sampling_hz": protocol.get("sampling_hz"),
            "dt": 0.01,
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
            "name": "UZH-FPV",
            "id": UZH_FPV_DATASET_NAME,
            "root_env": UZH_FPV_ENV_VAR,
            "root_path": str(root),
            "version": "external",
            "protocol": str(protocol_id),
            "paper": str(protocol.get("paper", "")),
            "external_data": True,
            "note": "Raw UZH-FPV data is external. Bench caches converted NPZ artifacts only.",
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


def generate_uzh_fpv_v0(
    *,
    suite_name: str,
    task_cfg_dict: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    seed: int,
    scenario_id: str,
    task_family: str = "uzh_fpv_v0",
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
