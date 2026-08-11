from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ADCSStateSchema:
    attitude_indices: tuple[int, int, int]
    attitude_type: str
    angular_rate_indices: tuple[int, int, int]
    angular_rate_type: str
    gyro_bias_indices: Optional[tuple[int, int, int]] = None
    gyro_bias_type: Optional[str] = None
    attitude_convention: str = "MRP sigma_BN"
    time_unit: str = "s"
    schema_source: str = "explicit"


def validate_indices(
    name: str,
    indices: Sequence[int],
    *,
    x_dim: int,
    length: int = 3,
) -> tuple[int, int, int]:
    if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
        raise ValueError(f"{name} must be a sequence of {length} integer indices")
    if len(indices) != int(length):
        raise ValueError(
            f"{name} must contain exactly {length} indices, got {len(indices)}"
        )

    parsed: list[int] = []
    for position, value in enumerate(indices):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"{name}[{position}] must be an integer, got {value!r}"
            )
        idx = int(value)
        if idx < 0 or idx >= int(x_dim):
            raise ValueError(
                f"{name}[{position}]={idx} is out of range for x_dim={x_dim}"
            )
        parsed.append(idx)

    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} contains duplicate indices: {parsed}")
    return parsed[0], parsed[1], parsed[2]


def _require_block(
    state_schema: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    block = state_schema.get(name)
    if not isinstance(block, Mapping):
        raise ValueError(f"state_schema.{name} is required and must be a mapping")
    return block


def _parse_type(block: Mapping[str, Any], field: str, *, default: Optional[str] = None) -> str:
    raw = block.get("type", default)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field}.type is required and must be a non-empty string")
    return raw.strip().lower()


def parse_adcs_state_schema(
    meta: Mapping[str, Any],
    *,
    x_dim: int,
) -> ADCSStateSchema:
    if not isinstance(meta, Mapping):
        raise ValueError("prediction metadata must be a mapping")
    if int(x_dim) <= 0:
        raise ValueError(f"x_dim must be positive, got {x_dim}")

    raw_schema = meta.get("state_schema")
    if raw_schema is None:
        if int(x_dim) < 6:
            raise ValueError(
                "state_schema is missing and fallback ADCS layout requires x_dim >= 6, "
                f"got x_dim={x_dim}"
            )
        return ADCSStateSchema(
            attitude_indices=(0, 1, 2),
            attitude_type="mrp",
            angular_rate_indices=(3, 4, 5),
            angular_rate_type="rad_s",
            gyro_bias_indices=((6, 7, 8) if int(x_dim) >= 9 else None),
            gyro_bias_type=("rad_s" if int(x_dim) >= 9 else None),
            attitude_convention=str(meta.get("attitude_convention", "MRP sigma_BN")),
            time_unit=str(meta.get("time_unit", "s")),
            schema_source="fallback_default_adcs",
        )
    if not isinstance(raw_schema, Mapping):
        raise ValueError("state_schema must be a mapping")

    attitude = _require_block(raw_schema, "attitude")
    angular_rate = _require_block(raw_schema, "angular_rate")
    attitude_type = _parse_type(attitude, "state_schema.attitude")
    if attitude_type != "mrp":
        raise ValueError(
            "state_schema.attitude.type must be 'mrp' in Phase 2, "
            f"got {attitude_type!r}"
        )
    angular_rate_type = _parse_type(
        angular_rate,
        "state_schema.angular_rate",
        default="rad_s",
    )
    attitude_indices = validate_indices(
        "state_schema.attitude.indices",
        attitude.get("indices"),
        x_dim=x_dim,
    )
    angular_rate_indices = validate_indices(
        "state_schema.angular_rate.indices",
        angular_rate.get("indices"),
        x_dim=x_dim,
    )

    gyro_bias_indices: Optional[tuple[int, int, int]] = None
    gyro_bias_type: Optional[str] = None
    gyro_bias = raw_schema.get("gyro_bias")
    if gyro_bias is not None:
        if not isinstance(gyro_bias, Mapping):
            raise ValueError("state_schema.gyro_bias must be a mapping when provided")
        gyro_bias_indices = validate_indices(
            "state_schema.gyro_bias.indices",
            gyro_bias.get("indices"),
            x_dim=x_dim,
        )
        gyro_bias_type = _parse_type(
            gyro_bias,
            "state_schema.gyro_bias",
            default="rad_s",
        )

    return ADCSStateSchema(
        attitude_indices=attitude_indices,
        attitude_type=attitude_type,
        angular_rate_indices=angular_rate_indices,
        angular_rate_type=angular_rate_type,
        gyro_bias_indices=gyro_bias_indices,
        gyro_bias_type=gyro_bias_type,
        attitude_convention=str(meta.get("attitude_convention", "MRP sigma_BN")),
        time_unit=str(meta.get("time_unit", "s")),
        schema_source="explicit",
    )


def adcs_state_schema_to_dict(schema: ADCSStateSchema) -> dict[str, Any]:
    state_schema: dict[str, Any] = {
        "attitude": {
            "type": schema.attitude_type,
            "name": "sigma_BN",
            "indices": list(schema.attitude_indices),
        },
        "angular_rate": {
            "type": schema.angular_rate_type,
            "name": "omega_BN_B",
            "indices": list(schema.angular_rate_indices),
        },
    }
    if schema.gyro_bias_indices is not None:
        state_schema["gyro_bias"] = {
            "type": schema.gyro_bias_type or "rad_s",
            "name": "gyro_bias",
            "indices": list(schema.gyro_bias_indices),
            "optional": True,
        }
    return state_schema
