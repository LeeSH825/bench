from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .phase6c_replay_visualization import (
    PHASE6C_SUMMARY_FILENAME,
    run_phase6c_replay_visualization,
)
from .vizard_basilisk_wrapper import (
    BASILISK_INPUT_FILENAME,
    BASILISK_INPUT_MANIFEST_FILENAME,
    MANUAL_CHECK_FILENAME,
    build_basilisk_vizard_offline_input,
)
from .vizard_convention import (
    SUPPORTED_VIZARD_CONVENTION_IDS,
    apply_vizard_convention_to_frame,
    build_vizard_convention,
    convention_summary,
    load_vizard_convention,
)
from .vizard_export import (
    VIZARD_MANIFEST_FILENAME,
    VIZARD_STATES_FILENAME,
    export_vizard_offline,
)
from .vizard_native_bridge import (
    NATIVE_BRIDGE_LOG_FILENAME,
    NATIVE_BRIDGE_MANIFEST_FILENAME,
    NATIVE_OUTPUT_MANIFEST_FILENAME,
    NATIVE_PLAYBACK_FILENAME,
    run_vizard_native_bridge,
)
from .vizard_phase5c_review import REVIEW_README_FILENAME


PHASE7_MANIFEST_FILENAME = "phase7_convention_manifest.json"
PHASE7_README_FILENAME = "README_phase7_vizard_check.md"
PHASE7_TEMPLATE_FILENAME = "vizard_convention_report_template.md"
PHASE7_REPORT_FILENAME = "vizard_convention_report.md"
PHASE7_LOCKED_FILENAME = "vizard_convention_locked.json"
CANDIDATES_DIRNAME = "candidates"
_SOURCE_TIMESERIES_CSV = "adcs_timeseries.csv"
_SOURCE_TIMESERIES_META = "adcs_timeseries_meta.json"
_SOURCE_PLOTS_DIR = "plots"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return payload


def _copy_file(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _copy_tree(source_dir: Path, destination_dir: Path) -> list[str]:
    copied: list[str] = []
    if not source_dir.is_dir():
        return copied
    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_dir)
        target = destination_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target.as_posix())
    return copied


def _source_phase6c_outputs(
    run_dir: Path,
    *,
    trajectory_id: int,
    position_source: str,
) -> tuple[Path, Path, Path]:
    artifacts = run_dir / "artifacts"
    timeseries_csv = artifacts / _SOURCE_TIMESERIES_CSV
    vizard_csv = artifacts / "vizard" / VIZARD_STATES_FILENAME
    summary_path = artifacts / PHASE6C_SUMMARY_FILENAME
    plots_dir = artifacts / _SOURCE_PLOTS_DIR
    if (
        timeseries_csv.is_file()
        and vizard_csv.is_file()
        and summary_path.is_file()
        and plots_dir.is_dir()
    ):
        return timeseries_csv, vizard_csv, summary_path

    summary_path = run_phase6c_replay_visualization(
        run_dir,
        trajectory_id=int(trajectory_id),
        position_source=position_source,
        require_native_success=False,
        include_review_bundle=False,
        create_zip=False,
    )
    return (
        artifacts / _SOURCE_TIMESERIES_CSV,
        artifacts / "vizard" / VIZARD_STATES_FILENAME,
        summary_path,
    )


def _stage_candidate(
    *,
    source_run_dir: Path,
    source_summary_path: Path,
    source_timeseries_csv: Path,
    source_timeseries_meta: Path,
    source_plots_dir: Path,
    candidate_dir: Path,
    convention: Mapping[str, Any],
    trajectory_id: int,
    position_source: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=".phase7_candidate_") as tmp:
        work_root = Path(tmp).resolve()
        work_artifacts = work_root / "artifacts"
        work_artifacts.mkdir(parents=True, exist_ok=True)
        _copy_file(source_timeseries_csv, work_artifacts / _SOURCE_TIMESERIES_CSV)
        _copy_file(source_timeseries_meta, work_artifacts / _SOURCE_TIMESERIES_META)

        export_csv, export_manifest = export_vizard_offline(
            work_root,
            trajectory_id=int(trajectory_id),
            position_source=position_source,
            out_dir=work_artifacts / "vizard",
            vizard_convention=convention,
        )
        basilisk_csv, basilisk_manifest, basilisk_readme = (
            build_basilisk_vizard_offline_input(
                work_root,
                input_csv=export_csv,
                out_dir=work_artifacts / "vizard" / "basilisk",
            )
        )

        native_dir = work_artifacts / "vizard" / "basilisk" / "native"
        native_dir.mkdir(parents=True, exist_ok=True)
        native_manifest_path: Path
        native_log_path: Path
        try:
            native_manifest_path, native_log_path = run_vizard_native_bridge(
                work_root,
                input_csv=basilisk_csv,
                out_dir=native_dir,
                mode="attempt-native",
                require_basilisk=False,
                require_native_success=False,
            )
        except Exception as exc:
            native_manifest_path = native_dir / NATIVE_BRIDGE_MANIFEST_FILENAME
            native_log_path = native_dir / NATIVE_BRIDGE_LOG_FILENAME
            _write_json_atomic(
                native_manifest_path,
                {
                    "schema_version": "vizard_native_bridge_v1",
                    "mode": "attempt-native",
                    "native_conversion_attempted": False,
                    "native_conversion_status": "orchestrator_native_bridge_failed",
                    "native_conversion_error": f"{type(exc).__name__}: {exc}",
                    "official_metrics_affected": False,
                },
            )
            native_log_path.write_text(
                f"native bridge failed in Phase 7 orchestrator: {type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
        native_playback = native_dir / NATIVE_PLAYBACK_FILENAME
        candidate_frame = pd.read_csv(export_csv)

        candidate_dir.mkdir(parents=True, exist_ok=True)
        _copy_file(export_csv, candidate_dir / VIZARD_STATES_FILENAME)
        _copy_file(
            export_manifest,
            candidate_dir / VIZARD_MANIFEST_FILENAME,
        )
        _copy_file(basilisk_csv, candidate_dir / BASILISK_INPUT_FILENAME)
        _copy_file(
            basilisk_manifest,
            candidate_dir / BASILISK_INPUT_MANIFEST_FILENAME,
        )
        _copy_file(basilisk_readme, candidate_dir / MANUAL_CHECK_FILENAME)
        _copy_tree(native_dir, candidate_dir / "native")

        candidate_manifest_path = candidate_dir / "candidate_manifest.json"
        export_manifest_obj = _read_json(export_manifest)
        native_manifest_obj = _read_json(native_manifest_path)
        candidate_manifest = {
            "schema_version": "phase7_convention_candidate_v1",
            "convention_id": convention["convention_id"],
            "attitude_mrp_mapping": convention["attitude_mrp_mapping"],
            "omega_mapping": convention["omega_mapping"],
            "sigma_transform": convention["attitude_mrp_mapping"],
            "omega_transform": convention["omega_mapping"],
            "source_run_dir": str(source_run_dir),
            "source_phase6c_summary": str(source_summary_path),
            "source_timeseries_csv": str(source_timeseries_csv),
            "source_timeseries_meta": str(source_timeseries_meta),
            "source_plots_dir": str(source_plots_dir),
            "output_dir": str(candidate_dir),
            "trajectory_id": int(trajectory_id),
            "position_source": position_source,
            "num_timestamps": int(export_manifest_obj.get("num_timestamps", 0)),
            "time_start_s": float(candidate_frame["time_s"].min()),
            "time_end_s": float(candidate_frame["time_s"].max()),
            "duration_s": float(
                candidate_frame["time_s"].max() - candidate_frame["time_s"].min()
            ),
            "spacecraft_count": int(
                candidate_frame["sc_name"]
                .drop_duplicates()
                .size
            ),
            "spacecraft_names": (
                candidate_frame["sc_name"]
                .drop_duplicates()
                .tolist()
            ),
            "native_conversion_status": native_manifest_obj.get(
                "native_conversion_status",
                "unknown",
            ),
            "playback_path": (
                str(candidate_dir / "native" / NATIVE_PLAYBACK_FILENAME)
                if (candidate_dir / "native" / NATIVE_PLAYBACK_FILENAME).is_file()
                else None
            ),
            "vizard_export_manifest": str(candidate_dir / VIZARD_MANIFEST_FILENAME),
            "dataFileToViz_input_manifest": str(
                candidate_dir / BASILISK_INPUT_MANIFEST_FILENAME
            ),
            "native_bridge_manifest": str(
                candidate_dir / "native" / NATIVE_BRIDGE_MANIFEST_FILENAME
            ),
            "native_bridge_log": str(
                candidate_dir / "native" / NATIVE_BRIDGE_LOG_FILENAME
            ),
            "official_metrics_affected": False,
            "notes": (
                "Phase 7 candidate convention package generated from a Phase 6G "
                "real replay output for manual Vizard inspection."
            ),
        }
        _write_json_atomic(candidate_manifest_path, candidate_manifest)
        return candidate_manifest


def _phase7_readme(
    *,
    pred_run_dir: Path,
    output_dir: Path,
    plots_dir: Path,
) -> str:
    candidates = "\n".join(
        [
            f"- `{output_dir / CANDIDATES_DIRNAME / name / 'native' / NATIVE_PLAYBACK_FILENAME}`"
            for name in SUPPORTED_VIZARD_CONVENTION_IDS
        ]
    )
    return f"""# Phase 7 Vizard Convention Package

This package prepares four explicit candidate sign/frame transforms from
`{pred_run_dir}` for manual Vizard review. It does not change official benchmark
metrics and does not decide the convention automatically.

## What To Open

Inspect these candidate playback files in Vizard:

{candidates}

## Compare Against Plots

Use the source replay plots from:

- `{plots_dir / 'rpy_true_vs_hat.png'}`
- `{plots_dir / 'rpy_error.png'}`
- `{plots_dir / 'omega_true_vs_hat.png'}`
- `{plots_dir / 'omega_error_norm.png'}`
- `{plots_dir / 'mrp_error_norm.png'}`

RPY uses Euler 3-2-1 and can be ambiguous near +/-90 degree pitch. Always check
`mrp_error_norm.png` together with the Euler plots.

## Expected Spacecraft

- `SC_true`
- `SC_estimated`

The candidate transforms are:

1. `direct`
2. `attitude_inverse`
3. `omega_negated`
4. `attitude_inverse_omega_negated`

The user must decide which candidate matches the physical sign/frame motion in
Vizard. Do not treat any candidate as confirmed until `vizard_convention_locked.json`
is written after manual inspection.
"""


def _phase7_template(
    *,
    phase7_dir: Path,
    source_run_dir: Path,
    candidate_ids: list[str],
) -> str:
    candidate_lines = "\n".join(
        f"- [ ] `{candidate_id}`" for candidate_id in candidate_ids
    )
    return f"""# Vizard Convention Review Template

Use this after opening the candidate playbacks in Vizard.

## Environment

- Source replay run: `{source_run_dir}`
- Phase 7 directory: `{phase7_dir}`
- Reviewer:
- Review date:
- Basilisk version:
- Vizard version:

## Candidate Check

{candidate_lines}

## Decision

- [ ] direct
- [ ] attitude_inverse
- [ ] omega_negated
- [ ] attitude_inverse_omega_negated

## Notes

- Which candidate matches expected attitude motion?
- Which candidate matches expected omega direction?
- Does `SC_estimated` remain close to `SC_true` when the plots show small errors?
- Does any candidate appear mirrored or inverted?

## Final Convention

- Selected convention_id:
- Confirmed by:
- Confirmed at UTC:
- Notes:
"""


def _phase7_locked_report(
    *,
    phase7_dir: Path,
    locked: Mapping[str, Any],
) -> str:
    return f"""# Vizard Convention Locked Report

This file records the convention selected after manual Vizard inspection.

## Locked Convention

- Convention ID: `{locked.get("convention_id")}`
- Attitude mapping: `{locked.get("attitude_mrp_mapping")}`
- Omega mapping: `{locked.get("omega_mapping")}`
- Manual confirmation status: `{locked.get("manual_confirmation_status")}`
- Confirmed by: `{locked.get("confirmed_by")}`
- Confirmed at UTC: `{locked.get("confirmed_at_utc")}`
- Source Phase 7 directory: `{phase7_dir}`

## Notes

{chr(10).join(f"- {note}" for note in locked.get("notes", [])) or "-"}
"""


def _clear_previous(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    shutil.rmtree(output_dir)


def build_phase7_vizard_convention_package(
    pred_run_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    trajectory_id: int = 0,
    position_source: str = "dummy_circular_orbit",
    overwrite: bool = False,
) -> tuple[Path, Path]:
    run_dir = Path(pred_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"prediction run directory not found: {run_dir}")

    output_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else run_dir / "artifacts" / "vizard" / "phase7_convention"
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"phase7 output directory already exists: {output_dir}"
        )
    if output_dir.exists() and overwrite:
        _clear_previous(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_timeseries_csv, _, source_summary_path = _source_phase6c_outputs(
        run_dir,
        trajectory_id=int(trajectory_id),
        position_source=position_source,
    )
    artifacts_dir = run_dir / "artifacts"
    source_timeseries_meta = artifacts_dir / _SOURCE_TIMESERIES_META
    if not source_timeseries_meta.is_file():
        raise FileNotFoundError(
            f"source ADCS timeseries metadata not found: {source_timeseries_meta}"
        )
    source_plots_dir = artifacts_dir / _SOURCE_PLOTS_DIR
    if not source_plots_dir.is_dir():
        raise FileNotFoundError(f"source plots directory not found: {source_plots_dir}")

    plots_dir = output_dir / _SOURCE_PLOTS_DIR
    copied_plots = _copy_tree(source_plots_dir, plots_dir)
    summary_copy = output_dir / PHASE6C_SUMMARY_FILENAME
    _copy_file(source_summary_path, summary_copy)

    candidates_root = output_dir / CANDIDATES_DIRNAME
    candidate_records: list[dict[str, Any]] = []
    for convention_id in SUPPORTED_VIZARD_CONVENTION_IDS:
        candidate_dir = candidates_root / convention_id
        convention = build_vizard_convention(
            convention_id,
            source_run_dir=run_dir,
            notes=[
                "Candidate transform generated for manual Vizard inspection.",
            ],
        )
        record = _stage_candidate(
            source_run_dir=run_dir,
            source_summary_path=source_summary_path,
            source_timeseries_csv=source_timeseries_csv,
            source_timeseries_meta=source_timeseries_meta,
            source_plots_dir=source_plots_dir,
            candidate_dir=candidate_dir,
            convention=convention,
            trajectory_id=int(trajectory_id),
            position_source=position_source,
        )
        candidate_records.append(record)

    manifest_path = output_dir / PHASE7_MANIFEST_FILENAME
    readme_path = output_dir / PHASE7_README_FILENAME
    template_path = output_dir / PHASE7_TEMPLATE_FILENAME
    manifest = {
        "schema_version": "phase7_vizard_convention_v1",
        "input_pred_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "source_phase6c_summary": str(summary_copy),
        "trajectory_id": int(trajectory_id),
        "position_source": position_source,
        "candidate_ids": list(SUPPORTED_VIZARD_CONVENTION_IDS),
        "candidates": [
            {
                "convention_id": record["convention_id"],
                "candidate_dir": str(output_dir / CANDIDATES_DIRNAME / record["convention_id"]),
                "candidate_manifest": str(
                    output_dir
                    / CANDIDATES_DIRNAME
                    / record["convention_id"]
                    / "candidate_manifest.json"
                ),
                "native_conversion_status": record["native_conversion_status"],
                "playback_path": record["playback_path"],
            }
            for record in candidate_records
        ],
        "plots_dir": str(plots_dir),
        "plots_copied": copied_plots,
        "manual_confirmation_status": "pending",
        "official_metrics_affected": False,
        "notes": (
            "Phase 7 Vizard convention package generated from a Phase 6G real "
            "replay output. Manual Vizard inspection is required before a "
            "convention can be locked."
        ),
    }
    _write_json_atomic(manifest_path, manifest)
    readme_path.write_text(
        _phase7_readme(
            pred_run_dir=run_dir,
            output_dir=output_dir,
            plots_dir=plots_dir,
        ),
        encoding="utf-8",
    )
    template_path.write_text(
        _phase7_template(
            phase7_dir=output_dir,
            source_run_dir=run_dir,
            candidate_ids=list(SUPPORTED_VIZARD_CONVENTION_IDS),
        ),
        encoding="utf-8",
    )
    return manifest_path, readme_path


def lock_vizard_convention(
    phase7_dir: str | Path,
    convention_id: str,
    *,
    confirmed_by: str = "manual_vizard_inspection",
    notes: str | list[str] | None = None,
) -> tuple[Path, Path]:
    phase7_path = Path(phase7_dir).expanduser().resolve()
    if not phase7_path.is_dir():
        raise FileNotFoundError(f"phase7 directory not found: {phase7_path}")
    if convention_id not in SUPPORTED_VIZARD_CONVENTION_IDS:
        raise ValueError(
            f"unsupported convention_id={convention_id!r}; "
            f"expected one of {SUPPORTED_VIZARD_CONVENTION_IDS}"
        )
    if isinstance(notes, str):
        note_list = [notes] if notes.strip() else []
    else:
        note_list = list(notes or [])
    locked = build_vizard_convention(
        convention_id,
        manual_confirmation_status="confirmed",
        source_run_dir=phase7_path,
        confirmed_by=confirmed_by,
        notes=note_list,
    )
    locked_path = phase7_path / PHASE7_LOCKED_FILENAME
    report_path = phase7_path / PHASE7_REPORT_FILENAME
    _write_json_atomic(locked_path, locked)
    report_path.write_text(
        _phase7_locked_report(phase7_dir=phase7_path, locked=locked),
        encoding="utf-8",
    )
    return locked_path, report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or lock Vizard convention verification artifacts."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pred-run-dir", default=None)
    mode.add_argument("--lock-convention", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--trajectory-id", type=int, default=0)
    parser.add_argument(
        "--position-source",
        choices=("fixed_origin", "dummy_circular_orbit"),
        default="dummy_circular_orbit",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--phase7-dir", default=None)
    parser.add_argument("--confirmed-by", default="manual_vizard_inspection")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args(argv)

    if args.pred_run_dir is not None:
        manifest_path, readme_path = build_phase7_vizard_convention_package(
            args.pred_run_dir,
            out_dir=args.out_dir,
            trajectory_id=int(args.trajectory_id),
            position_source=args.position_source,
            overwrite=bool(args.overwrite),
        )
        print(f"wrote {manifest_path}")
        print(f"wrote {readme_path}")
        return 0

    if args.phase7_dir is None:
        parser.error("--phase7-dir is required with --lock-convention")
    locked_path, report_path = lock_vizard_convention(
        args.phase7_dir,
        str(args.lock_convention),
        confirmed_by=str(args.confirmed_by),
        notes=args.notes,
    )
    print(f"wrote {locked_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
