"""Preset catalog, schema descriptor and validation — all read-only.

These are registered unconditionally: browsing presets and previewing a
resolved spec computes nothing durable and allocates no run, so they are safe
in the default read-only build. Only the launch route is gated on write mode.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field

from ...config.descriptor import descriptor_document
from ...config.gui_service import parse_submitted_yaml, validate_config
from ...config.presets import (
    PresetCatalog,
    PresetError,
    PresetNotFound,
    PresetUnsafe,
)
from ...registry.sqlite import SqliteRegistry
from ..deps import get_registry

router = APIRouter(prefix="/api/v1/config", tags=["config"])

#: Bound the request body independently of the YAML parser's own limits.
MAX_BODY_YAML_CHARS = 512 * 1024


class ValidateRequest(BaseModel):
    """A validation request carries *data*, never a filesystem path."""

    preset_id: Optional[str] = Field(None, description="Tracked preset to start from")
    yaml_text: Optional[str] = Field(None, max_length=MAX_BODY_YAML_CHARS)
    task_id: Optional[str] = None
    model_id: Optional[str] = None
    init_id: Optional[str] = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    include_diff: bool = True


@router.get("/presets")
def list_presets() -> dict[str, Any]:
    """Tracked presets only. This is an allowlist, not a directory listing."""
    catalog = PresetCatalog()
    entries = catalog.list()
    return {
        "schema_version": 1,
        "count": len(entries),
        "preset_root": "bench/configs",
        "presets": [e.as_dict() for e in entries],
        "note": (
            "Only files tracked by Git under the preset root are listed. "
            "Untracked or out-of-root files are not addressable."
        ),
    }


@router.get("/presets/{preset_id}")
def get_preset(preset_id: str) -> dict[str, Any]:
    catalog = PresetCatalog()
    try:
        entry, text = catalog.get(preset_id)
    except PresetNotFound as exc:
        raise HTTPException(status_code=404, detail={
            "reason_code": "UNKNOWN_PRESET", "message": str(exc)}) from exc
    except PresetUnsafe as exc:
        raise HTTPException(status_code=400, detail={
            "reason_code": "UNSAFE_PRESET_PATH", "message": str(exc)}) from exc
    return {"schema_version": 1, **entry.as_dict(), "yaml_text": text}


@router.get("/schema")
def get_schema() -> dict[str, Any]:
    """Descriptor derived from the same dataclasses the resolver uses."""
    return descriptor_document()


@router.post("/validate")
def validate(
    payload: ValidateRequest = Body(...),
    registry: SqliteRegistry = Depends(get_registry),
) -> dict[str, Any]:
    """Validate and preview. Allocates nothing, writes nothing.

    Deliberately available without write mode: a preview has no side effect,
    and refusing it would only push users toward launching blind.
    """
    catalog = PresetCatalog()
    baseline_document = None

    if payload.preset_id:
        try:
            _entry, preset_text = catalog.get(payload.preset_id)
        except PresetNotFound as exc:
            raise HTTPException(status_code=404, detail={
                "reason_code": "UNKNOWN_PRESET", "message": str(exc)}) from exc
        except PresetError as exc:
            raise HTTPException(status_code=400, detail={
                "reason_code": "PRESET_ERROR", "message": str(exc)}) from exc
        try:
            baseline_document = parse_submitted_yaml(preset_text)
        except PresetError as exc:
            raise HTTPException(status_code=422, detail={
                "reason_code": "PRESET_UNPARSEABLE", "message": str(exc)}) from exc
        source_document = baseline_document

    if payload.yaml_text is not None:
        try:
            source_document = parse_submitted_yaml(payload.yaml_text)
        except PresetError as exc:
            # A YAML syntax error is a client error with line/column, not a 500.
            return {
                "schema_version": 1, "valid": False,
                "issues": [{"path": "", "code": "YAML_PARSE_ERROR",
                            "message": str(exc), "severity": "error"}],
                "unsupported_fields": [], "resolved_run_spec": None,
                "canonical_yaml": None, "structural_config_hash": None,
                "operational_config_hash": None, "variant_id": None,
                "training_path_id": None, "implementation_id": None,
                "launch_eligibility": {"eligible": False,
                                       "reason_code": "INVALID_CONFIG",
                                       "reason": "The submitted YAML could not be parsed."},
                "diff": {},
            }

    if payload.preset_id is None and payload.yaml_text is None:
        raise HTTPException(status_code=400, detail={
            "reason_code": "NOTHING_TO_VALIDATE",
            "message": "supply preset_id, yaml_text, or both"})

    result = validate_config(
        suite_document=source_document,
        task_id=payload.task_id, model_id=payload.model_id,
        overrides=payload.overrides, init_id=payload.init_id,
        baseline_document=baseline_document if payload.include_diff else None,
        registry=registry,
    )
    return result.as_dict()
