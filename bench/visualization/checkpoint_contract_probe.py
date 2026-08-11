from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .replay_checkpoint_contract import (
    REPLAY_CHECKPOINT_CONTRACT_FILENAME,
    load_replay_checkpoint_contract,
    summarize_replay_checkpoint_contract,
)


CHECKPOINT_CONTRACT_PROBE_FILENAME = "checkpoint_contract_probe.json"
_STATE_DICT_KEYS = (
    "state_dict",
    "model_state_dict",
    "model",
    "weights",
    "network",
)
_CONFIG_TOKENS = ("config", "cfg", "args", "hparam", "hyper")
_NORMALIZER_TOKENS = ("normaliz", "mean", "std", "scale", "scaler")
_STEP_TOKENS = ("epoch", "step", "update", "iteration")


def _load_torch_checkpoint(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ValueError(
            "PyTorch is required to inspect checkpoint files but is not "
            "available in this Python environment"
        ) from exc

    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError(
            f"failed to load checkpoint {path} with torch.load: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _mapping_keys(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    return sorted(str(key) for key in value.keys())


def _looks_like_state_dict(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    tensor_like = 0
    for item in value.values():
        if hasattr(item, "shape") and hasattr(item, "dtype"):
            tensor_like += 1
    return tensor_like > 0 and tensor_like == len(value)


def _state_dict_candidates(checkpoint: Any) -> list[str]:
    if not isinstance(checkpoint, Mapping):
        return []
    candidates = [
        key
        for key in _STATE_DICT_KEYS
        if key in checkpoint and isinstance(checkpoint[key], Mapping)
    ]
    if _looks_like_state_dict(checkpoint):
        candidates.append("<root>")
    return candidates


def _keys_matching(keys: list[str], tokens: tuple[str, ...]) -> list[str]:
    return [
        key
        for key in keys
        if any(token in key.lower() for token in tokens)
    ]


def _infer_model_family(
    model_id: str | None,
    checkpoint: Any,
) -> str | None:
    candidates = [str(model_id or "")]
    if isinstance(checkpoint, Mapping):
        for key in ("model_id", "model_class", "adapter_id"):
            if key in checkpoint:
                candidates.append(str(checkpoint[key]))
    joined = " ".join(candidates).lower()
    if "mock_checkpoint_adapter" in joined:
        return "phase6d_mock"
    if "kalmannet" in joined or "kalman_net" in joined:
        return "kalmannet"
    if "adaptive" in joined and "knet" in joined:
        return "adaptive_knet"
    if "split" in joined and ("knet" in joined or "kalman" in joined):
        return "split_knet"
    if "maml" in joined and ("knet" in joined or "kalman" in joined):
        return "maml_knet"
    return None


def _support_summary(
    *,
    model_id: str | None,
    checkpoint: Any,
) -> tuple[bool, str, list[str]]:
    warnings: list[str] = []
    effective_model_id = str(model_id or "").strip()
    if not effective_model_id and isinstance(checkpoint, Mapping):
        effective_model_id = str(checkpoint.get("model_id", "")).strip()

    if effective_model_id == "mock_checkpoint_adapter":
        if not isinstance(checkpoint, Mapping):
            return False, "Mock checkpoint must be a mapping.", warnings
        missing = [
            key for key in ("gain", "bias") if key not in checkpoint
        ]
        if missing:
            return (
                False,
                f"Mock checkpoint is missing required keys: {missing}.",
                warnings,
            )
        return (
            True,
            "The explicit test-only mock checkpoint contract is supported.",
            warnings,
        )

    if effective_model_id == "kalmannet_tsp":
        return (
            False,
            "KalmanNet_TSP weights alone do not define replay setup. "
            "Phase 6D still requires a compatible x_dim/y_dim architecture, "
            "model hyperparameters, F/H system matrices, initialization "
            "policy, and any preprocessing contract.",
            warnings,
        )

    if effective_model_id:
        return (
            False,
            f"No explicit Phase 6D replay adapter is registered for "
            f"model_id={effective_model_id!r}.",
            warnings,
        )

    warnings.append(
        "No model_id was supplied or found in the checkpoint; support "
        "cannot be selected safely."
    )
    return False, "Checkpoint model family is not explicit.", warnings


def probe_checkpoint_contract(
    checkpoint: str | Path,
    *,
    model_id: str | None = None,
    model_config: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_input = Path(checkpoint).expanduser().resolve()
    if not checkpoint_input.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_input}")

    config_path: Path | None = None
    replay_contract: dict[str, Any] | None = None
    replay_contract_path: Path | None = None
    if checkpoint_input.is_dir():
        replay_contract = load_replay_checkpoint_contract(checkpoint_input)
        replay_contract_path = (
            checkpoint_input / REPLAY_CHECKPOINT_CONTRACT_FILENAME
        )
        checkpoint_path = Path(
            replay_contract["_resolved_paths"]["checkpoint_path"]
        )
    elif checkpoint_input.is_file():
        checkpoint_path = checkpoint_input
        sibling_contract = (
            checkpoint_path.parent / REPLAY_CHECKPOINT_CONTRACT_FILENAME
        )
        if (
            checkpoint_path.name == "checkpoint.pt"
            and sibling_contract.exists()
        ):
            replay_contract = load_replay_checkpoint_contract(
                sibling_contract
            )
            replay_contract_path = sibling_contract
    else:
        raise ValueError(
            f"checkpoint must be a file or package directory: "
            f"{checkpoint_input}"
        )

    if model_config is not None:
        config_path = Path(model_config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"model config not found: {config_path}")
        is_contract = config_path.is_dir() or (
            config_path.is_file()
            and config_path.name == REPLAY_CHECKPOINT_CONTRACT_FILENAME
        )
        if is_contract:
            replay_contract = load_replay_checkpoint_contract(config_path)
            replay_contract_path = (
                config_path / REPLAY_CHECKPOINT_CONTRACT_FILENAME
                if config_path.is_dir()
                else config_path
            )
        elif not config_path.is_file():
            raise ValueError(
                f"model config must be a file or package directory: "
                f"{config_path}"
            )

    if replay_contract is not None:
        contract_checkpoint = Path(
            replay_contract["_resolved_paths"]["checkpoint_path"]
        ).resolve()
        if checkpoint_input.is_file() and checkpoint_path != contract_checkpoint:
            raise ValueError(
                "checkpoint input does not match replay contract "
                f"checkpoint_path: input={checkpoint_path}, "
                f"contract={contract_checkpoint}"
            )
        checkpoint_path = contract_checkpoint

    effective_model_id = str(model_id or "").strip()
    if replay_contract is not None:
        contract_model_id = str(replay_contract["model_id"])
        if effective_model_id and contract_model_id != effective_model_id:
            raise ValueError(
                f"replay contract model_id={contract_model_id!r} does not "
                f"match requested model_id={effective_model_id!r}"
            )
        effective_model_id = contract_model_id

    resolved_run_dir: Path | None = None
    if run_dir is not None:
        resolved_run_dir = Path(run_dir).expanduser().resolve()

    loaded = _load_torch_checkpoint(checkpoint_path)
    top_level_keys = _mapping_keys(loaded)
    supported, support_reason, warnings = _support_summary(
        model_id=effective_model_id or model_id,
        checkpoint=loaded,
    )
    if replay_contract is not None:
        from .checkpoint_replay_adapters import (
            get_real_checkpoint_replay_model_ids,
            get_test_checkpoint_replay_model_ids,
        )

        registered = set(get_real_checkpoint_replay_model_ids())
        registered.update(get_test_checkpoint_replay_model_ids())
        if effective_model_id in registered:
            supported = True
            support_reason = (
                "A valid replay checkpoint contract exists and an explicit "
                f"adapter is registered for model_id={effective_model_id!r}."
            )
        else:
            supported = False
            support_reason = (
                "A valid replay contract exists, but no real adapter is "
                f"registered for model_id={effective_model_id!r}."
            )
    if (
        replay_contract is None
        and config_path is None
        and effective_model_id == "kalmannet_tsp"
    ):
        warnings.append(
            "No model_config was supplied; checkpoint weights do not "
            "contain the KalmanNet setup contract."
        )

    optimizer_keys = [
        key for key in top_level_keys if "optimizer" in key.lower()
    ]
    return {
        "schema_version": "checkpoint_contract_probe_v1",
        "checkpoint_input": str(checkpoint_input),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": True,
        "checkpoint_size_bytes": int(checkpoint_path.stat().st_size),
        "model_id": effective_model_id or model_id,
        "model_config_path": (
            None if config_path is None else str(config_path)
        ),
        "run_dir": (
            None if resolved_run_dir is None else str(resolved_run_dir)
        ),
        "top_level_type": type(loaded).__name__,
        "top_level_keys": top_level_keys,
        "state_dict_key_candidates": _state_dict_candidates(loaded),
        "config_key_candidates": _keys_matching(
            top_level_keys,
            _CONFIG_TOKENS,
        ),
        "normalizer_key_candidates": _keys_matching(
            top_level_keys,
            _NORMALIZER_TOKENS,
        ),
        "optimizer_key_present": bool(optimizer_keys),
        "epoch_or_step_keys": _keys_matching(
            top_level_keys,
            _STEP_TOKENS,
        ),
        "inferred_model_family": _infer_model_family(
            effective_model_id or model_id,
            loaded,
        ),
        "replay_contract_present": replay_contract is not None,
        "replay_contract_valid": replay_contract is not None,
        "replay_contract_path": (
            None
            if replay_contract_path is None
            else str(replay_contract_path)
        ),
        "replay_contract_summary": (
            None
            if replay_contract is None
            else summarize_replay_checkpoint_contract(replay_contract)
        ),
        "expected_state_dim": (
            None
            if replay_contract is None
            else int(replay_contract["state_dim"])
        ),
        "expected_measurement_dim": (
            None
            if replay_contract is None
            else int(replay_contract["measurement_dim"])
        ),
        "observed_state": (
            None
            if replay_contract is None
            else list(replay_contract["observed_state"])
        ),
        "package_supported_for_phase6d": bool(
            replay_contract is not None and supported
        ),
        "supported_for_phase6d": bool(supported),
        "support_reason": support_reason,
        "warnings": warnings,
        "errors": [],
    }


def save_checkpoint_contract_probe(
    probe: dict[str, Any],
    out_dir: str | Path,
) -> Path:
    output_dir = Path(out_dir).expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / CHECKPOINT_CONTRACT_PROBE_FILENAME
        temporary = output_path.with_name(f".{output_path.name}.tmp")
        temporary.write_text(
            json.dumps(probe, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    except OSError as exc:
        raise RuntimeError(
            f"failed to write checkpoint contract probe under {output_dir}"
        ) from exc
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a checkpoint without instantiating its model."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--model-config")
    parser.add_argument("--run-dir")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    probe = probe_checkpoint_contract(
        args.checkpoint,
        model_id=args.model_id,
        model_config=args.model_config,
        run_dir=args.run_dir,
    )
    path = save_checkpoint_contract_probe(probe, args.out_dir)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
