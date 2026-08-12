from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


VIZARD_CONVENTION_SCHEMA_VERSION = "vizard_convention_v1"
SUPPORTED_VIZARD_CONVENTION_IDS = (
    "direct",
    "attitude_inverse",
    "omega_negated",
    "attitude_inverse_omega_negated",
)

_CONVENTION_SPECS: dict[str, dict[str, Any]] = {
    "direct": {
        "attitude_mrp_mapping": "sigma_BN_direct",
        "omega_mapping": "omega_BN_B_direct",
        "sigma_sign": 1.0,
        "omega_sign": 1.0,
    },
    "attitude_inverse": {
        "attitude_mrp_mapping": "sigma_BN_inverse",
        "omega_mapping": "omega_BN_B_direct",
        "sigma_sign": -1.0,
        "omega_sign": 1.0,
    },
    "omega_negated": {
        "attitude_mrp_mapping": "sigma_BN_direct",
        "omega_mapping": "omega_BN_B_negated",
        "sigma_sign": 1.0,
        "omega_sign": -1.0,
    },
    "attitude_inverse_omega_negated": {
        "attitude_mrp_mapping": "sigma_BN_inverse",
        "omega_mapping": "omega_BN_B_negated",
        "sigma_sign": -1.0,
        "omega_sign": -1.0,
    },
}

_SIGMA_COLUMNS = ("sigma_BN_1", "sigma_BN_2", "sigma_BN_3")
_OMEGA_COLUMNS = (
    "omega_BN_B_x_rad_s",
    "omega_BN_B_y_rad_s",
    "omega_BN_B_z_rad_s",
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in convention file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"convention file must contain a JSON object: {path}")
    return payload


def build_vizard_convention(
    convention_id: str,
    *,
    manual_confirmation_status: str = "pending",
    source_run_dir: str | Path | None = None,
    confirmed_by: str | None = None,
    confirmed_at_utc: str | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    convention_key = str(convention_id)
    if convention_key not in SUPPORTED_VIZARD_CONVENTION_IDS:
        raise ValueError(
            f"unsupported vizard convention_id={convention_key!r}; "
            f"expected one of {SUPPORTED_VIZARD_CONVENTION_IDS}"
        )
    convention = {
        "schema_version": VIZARD_CONVENTION_SCHEMA_VERSION,
        "convention_id": convention_key,
        "attitude_mrp_mapping": _CONVENTION_SPECS[convention_key][
            "attitude_mrp_mapping"
        ],
        "omega_mapping": _CONVENTION_SPECS[convention_key]["omega_mapping"],
        "requires_manual_vizard_confirmation": True,
        "manual_confirmation_status": str(manual_confirmation_status),
        "source_run_dir": (
            None if source_run_dir is None else str(Path(source_run_dir).resolve())
        ),
        "confirmed_by": confirmed_by,
        "confirmed_at_utc": confirmed_at_utc,
        "notes": list(notes or []),
    }
    if convention["manual_confirmation_status"] == "confirmed":
        convention["confirmed_by"] = confirmed_by or "manual_vizard_inspection"
        convention["confirmed_at_utc"] = confirmed_at_utc or _now_utc()
    validate_vizard_convention(convention)
    return convention


def validate_vizard_convention(
    convention: Mapping[str, Any],
    *,
    require_confirmed: bool = False,
) -> dict[str, Any]:
    if not isinstance(convention, Mapping):
        raise ValueError("vizard convention must be a mapping")
    schema_version = convention.get("schema_version")
    if schema_version != VIZARD_CONVENTION_SCHEMA_VERSION:
        raise ValueError(
            "invalid vizard convention schema_version="
            f"{schema_version!r}"
        )
    convention_id = str(convention.get("convention_id", "")).strip()
    if convention_id not in SUPPORTED_VIZARD_CONVENTION_IDS:
        raise ValueError(
            f"unsupported vizard convention_id={convention_id!r}; "
            f"expected one of {SUPPORTED_VIZARD_CONVENTION_IDS}"
        )
    spec = _CONVENTION_SPECS[convention_id]
    attitude_mapping = str(convention.get("attitude_mrp_mapping", "")).strip()
    omega_mapping = str(convention.get("omega_mapping", "")).strip()
    if attitude_mapping != spec["attitude_mrp_mapping"]:
        raise ValueError(
            f"invalid attitude_mrp_mapping for convention_id={convention_id!r}: "
            f"{attitude_mapping!r}"
        )
    if omega_mapping != spec["omega_mapping"]:
        raise ValueError(
            f"invalid omega_mapping for convention_id={convention_id!r}: "
            f"{omega_mapping!r}"
        )
    if not bool(convention.get("requires_manual_vizard_confirmation", False)):
        raise ValueError("vizard convention must require manual confirmation")
    confirmation_status = str(
        convention.get("manual_confirmation_status", "")
    ).strip()
    if confirmation_status not in {"pending", "confirmed"}:
        raise ValueError(
            "manual_confirmation_status must be 'pending' or 'confirmed'"
        )
    if require_confirmed and confirmation_status != "confirmed":
        raise ValueError(
            f"vizard convention {convention_id!r} is not confirmed"
        )

    normalized_notes = convention.get("notes", [])
    if normalized_notes is None:
        normalized_notes = []
    if not isinstance(normalized_notes, list) or not all(
        isinstance(item, str) for item in normalized_notes
    ):
        raise ValueError("vizard convention notes must be a list of strings")

    source_run_dir = convention.get("source_run_dir")
    if source_run_dir is not None and not isinstance(source_run_dir, str):
        raise ValueError("source_run_dir must be a string or null")
    confirmed_by = convention.get("confirmed_by")
    if confirmed_by is not None and not isinstance(confirmed_by, str):
        raise ValueError("confirmed_by must be a string or null")
    confirmed_at_utc = convention.get("confirmed_at_utc")
    if confirmed_at_utc is not None and not isinstance(confirmed_at_utc, str):
        raise ValueError("confirmed_at_utc must be a string or null")

    return {
        "schema_version": VIZARD_CONVENTION_SCHEMA_VERSION,
        "convention_id": convention_id,
        "attitude_mrp_mapping": attitude_mapping,
        "omega_mapping": omega_mapping,
        "requires_manual_vizard_confirmation": True,
        "manual_confirmation_status": confirmation_status,
        "source_run_dir": source_run_dir,
        "confirmed_by": confirmed_by,
        "confirmed_at_utc": confirmed_at_utc,
        "notes": list(normalized_notes),
    }


def load_vizard_convention(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return validate_vizard_convention(source)
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"vizard convention file not found: {path}")
    if path.is_dir():
        path = path / "vizard_convention_locked.json"
        if not path.exists():
            raise FileNotFoundError(f"vizard convention file not found: {path}")
    return validate_vizard_convention(_read_json(path))


def save_vizard_convention(
    convention: Mapping[str, Any],
    path: str | Path,
) -> Path:
    normalized = validate_vizard_convention(convention)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def convention_signs(convention: Mapping[str, Any]) -> tuple[float, float]:
    normalized = validate_vizard_convention(convention)
    spec = _CONVENTION_SPECS[normalized["convention_id"]]
    return float(spec["sigma_sign"]), float(spec["omega_sign"])


def apply_vizard_convention_to_frame(
    frame: pd.DataFrame,
    convention: Mapping[str, Any],
) -> pd.DataFrame:
    normalized = validate_vizard_convention(convention)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame")
    missing = [
        column
        for column in (*_SIGMA_COLUMNS, *_OMEGA_COLUMNS)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            "Vizard spacecraft state frame is missing convention columns: "
            f"{missing}"
        )
    spec = _CONVENTION_SPECS[normalized["convention_id"]]
    transformed = frame.copy()
    sigma_sign = float(spec["sigma_sign"])
    omega_sign = float(spec["omega_sign"])
    transformed.loc[:, _SIGMA_COLUMNS] = (
        transformed.loc[:, _SIGMA_COLUMNS].to_numpy(dtype=float) * sigma_sign
    )
    transformed.loc[:, _OMEGA_COLUMNS] = (
        transformed.loc[:, _OMEGA_COLUMNS].to_numpy(dtype=float) * omega_sign
    )
    return transformed


def convention_summary(convention: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_vizard_convention(convention)
    spec = _CONVENTION_SPECS[normalized["convention_id"]]
    return {
        "schema_version": normalized["schema_version"],
        "convention_id": normalized["convention_id"],
        "attitude_mrp_mapping": normalized["attitude_mrp_mapping"],
        "omega_mapping": normalized["omega_mapping"],
        "manual_confirmation_status": normalized["manual_confirmation_status"],
        "requires_manual_vizard_confirmation": True,
        "attitude_sign": float(spec["sigma_sign"]),
        "omega_sign": float(spec["omega_sign"]),
        "confirmed_by": normalized.get("confirmed_by"),
        "confirmed_at_utc": normalized.get("confirmed_at_utc"),
        "source_run_dir": normalized.get("source_run_dir"),
        "notes": list(normalized.get("notes", [])),
    }
