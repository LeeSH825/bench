from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .vizard_frame_checks import (
    FRAME_CHECK_README_FILENAME,
    POSITIVE_YAW_FILENAME,
    TRUE_ESTIMATED_OFFSET_FILENAME,
    ZERO_ATTITUDE_FILENAME,
)
from .vizard_native_bridge import (
    NATIVE_BRIDGE_MANIFEST_FILENAME,
    NATIVE_PLAYBACK_FILENAME,
    run_vizard_native_bridge,
)


REVIEW_MANIFEST_FILENAME = "phase5c_review_manifest.json"
REVIEW_README_FILENAME = "README_phase5c_review.md"
VERIFICATION_REPORT_FILENAME = "vizard_frame_verification_report.md"
REVIEW_ZIP_FILENAME = "phase5c_review_bundle.zip"
FRAME_CHECK_NATIVE_MANIFEST_FILENAME = "frame_check_native_manifest.json"

_FRAME_CHECK_PLAYBACK_NAMES = {
    ZERO_ATTITUDE_FILENAME: "zero_attitude_vizard_playback.bin",
    POSITIVE_YAW_FILENAME: "small_positive_yaw_vizard_playback.bin",
    TRUE_ESTIMATED_OFFSET_FILENAME: (
        "true_vs_estimated_offset_vizard_playback.bin"
    ),
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def convert_frame_check_fixtures_to_native(
    frame_check_dir: str | Path,
    *,
    require_native_success: bool = False,
) -> Path:
    """Convert Phase 5B frame fixtures through the guarded native bridge."""
    source_dir = Path(frame_check_dir).expanduser().resolve()
    native_dir = source_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for source_name, playback_name in _FRAME_CHECK_PLAYBACK_NAMES.items():
        source_path = source_dir / source_name
        result: dict[str, Any] = {
            "input_csv": str(source_path),
            "output_playback": None,
            "native_conversion_status": "missing_input",
            "native_conversion_error": None,
        }
        if not source_path.exists():
            results.append(result)
            continue

        try:
            with tempfile.TemporaryDirectory(
                prefix=".frame_native_",
                dir=native_dir,
            ) as tmp:
                bridge_dir = Path(tmp) / "bridge"
                bridge_manifest_path, _ = run_vizard_native_bridge(
                    input_csv=source_path,
                    out_dir=bridge_dir,
                    mode="attempt-native",
                    require_native_success=False,
                )
                bridge_manifest = _read_json(bridge_manifest_path)
                status = str(
                    bridge_manifest.get(
                        "native_conversion_status",
                        "attempted_failed",
                    )
                )
                result["native_conversion_status"] = status
                result["native_conversion_error"] = bridge_manifest.get(
                    "native_conversion_error"
                )
                playback_source = bridge_dir / NATIVE_PLAYBACK_FILENAME
                if status == "attempted_success" and playback_source.exists():
                    playback_path = native_dir / playback_name
                    shutil.copy2(playback_source, playback_path)
                    result["output_playback"] = str(playback_path)
        except Exception as exc:
            result["native_conversion_status"] = "attempted_failed"
            result["native_conversion_error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)

    successful = [
        item
        for item in results
        if item["native_conversion_status"] == "attempted_success"
    ]
    manifest_path = native_dir / FRAME_CHECK_NATIVE_MANIFEST_FILENAME
    manifest = {
        "schema_version": "frame_check_native_manifest_v1",
        "input_frame_check_dir": str(source_dir),
        "output_dir": str(native_dir),
        "fixtures": results,
        "num_fixtures": len(results),
        "num_successful": len(successful),
        "all_successful": len(successful) == len(results),
        "official_metrics_affected": False,
        "notes": (
            "Frame-convention fixtures converted through the guarded Phase 5B "
            "native bridge. Vizard was not launched."
        ),
    }
    _write_json(manifest_path, manifest)

    if require_native_success and not manifest["all_successful"]:
        raise RuntimeError(
            "one or more frame-check native conversions failed; "
            f"see {manifest_path}"
        )
    return manifest_path


def _frame_native_outputs_complete(frame_check_dir: Path) -> bool:
    native_dir = frame_check_dir / "native"
    manifest = _read_json(native_dir / FRAME_CHECK_NATIVE_MANIFEST_FILENAME)
    if manifest.get("schema_version") != "frame_check_native_manifest_v1":
        return False
    if not manifest.get("all_successful", False):
        return False
    return all(
        (native_dir / playback_name).is_file()
        for playback_name in _FRAME_CHECK_PLAYBACK_NAMES.values()
    )


def _review_readme(
    *,
    run_dir: Path,
    native_playback_present: bool,
    plots_present: bool,
    frame_checks_present: bool,
) -> str:
    playback = (
        "`native/vizard_playback.bin`"
        if native_playback_present
        else "No main native playback file was included."
    )
    return f"""# Phase 5C Vizard Review Package

This self-contained package collects downstream artifacts from `{run_dir}` for
one human visualization review. It does not alter or recompute official
benchmark metrics and is not a publication-grade benchmark result.

## Start Here

- Main playback: {playback}
- Manual checklist: `{VERIFICATION_REPORT_FILENAME}`
- Native bridge details: `native/{NATIVE_BRIDGE_MANIFEST_FILENAME}`
- Frame-check instructions: `frame_check/{FRAME_CHECK_README_FILENAME}`

## Plot Review

Plots included: `{str(plots_present).lower()}`.

Inspect these first when present:

1. `plots/rpy_true_vs_hat.png`
2. `plots/rpy_error.png`
3. `plots/omega_true_vs_hat.png`
4. `plots/omega_error_norm.png`
5. `plots/mrp_error_norm.png`

RPY uses Euler 3-2-1 roll/pitch/yaw and can be ambiguous near +/-90 degree
pitch. Always inspect `mrp_error_norm.png` together with the Euler plots.

## Spacecraft Mapping

- `SC_true`: trajectory ground truth.
- `SC_estimated`: estimator output for the same timestamps.

Different orientations are expected when their MRPs differ. Position may be
fixed or a synthetic circular orbit and is visualization-only unless a source
manifest explicitly states otherwise.

Frame-check fixtures included: `{str(frame_checks_present).lower()}`. Native
fixture playback files, when conversion succeeded, are under
`frame_check/native/`.

This workflow does not launch Vizard, perform live streaming, or run online
filter inference.
"""


def _verification_report(
    *,
    run_dir: Path,
    basilisk_version: Optional[str],
    playback_path: Optional[str],
) -> str:
    return f"""# Vizard Frame Verification Report

Do not pre-mark this report. Complete it during a manual Vizard session.

## Environment

- Basilisk version: `{basilisk_version or "unknown"}`
- Run directory: `{run_dir}`
- Playback file: `{playback_path or "not generated"}`
- Vizard version:
- Reviewer:
- Review date:

## Main Playback Check

- [ ] Vizard opens the playback file.
- [ ] `SC_true` is visible.
- [ ] `SC_estimated` is visible.
- [ ] Spacecraft motion matches the fixed/dummy-orbit source expectation.
- [ ] True and estimated orientations differ when their MRPs differ.

## Frame Convention Check

### Zero Attitude

- Fixture: `frame_check/native/zero_attitude_vizard_playback.bin`
- [ ] Body and inertial axes align for `sigma_BN = [0, 0, 0]`.
- Observation:

### Small Positive Yaw

- Fixture: `frame_check/native/small_positive_yaw_vizard_playback.bin`
- [ ] The displayed rotation direction matches the expected positive z rotation.
- [ ] No unexpected roll or pitch is visible.
- Observation:

### True Versus Estimated Offset

- Fixture: `frame_check/native/true_vs_estimated_offset_vizard_playback.bin`
- [ ] `SC_true` remains at identity attitude.
- [ ] `SC_estimated` shows the known positive yaw offset.
- [ ] Spacecraft names are not swapped.
- Observation:

## Decision

- `sigma_BN` direct usage: [ ] PASS  [ ] FAIL  [ ] NEEDS_INVERSION
- `omega_BN_B` mapping: [ ] PASS  [ ] FAIL  [ ] UNCLEAR
- True/estimated mapping: [ ] PASS  [ ] FAIL

## Notes


## Final Decision

- [ ] Keep current convention.
- [ ] Invert the MRP convention in export.
- [ ] Further investigation is needed.
"""


def _copy_if_present(
    source: Path,
    destination: Path,
    included: list[str],
) -> bool:
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    included.append(destination.as_posix())
    return True


def _clear_previous_package(output_dir: Path) -> None:
    previous = _read_json(output_dir / REVIEW_MANIFEST_FILENAME)
    managed = previous.get("included_artifacts", [])
    if isinstance(managed, list):
        for value in managed:
            if not isinstance(value, str):
                continue
            candidate = (output_dir / value).resolve()
            try:
                candidate.relative_to(output_dir.resolve())
            except ValueError:
                continue
            if candidate.is_file():
                candidate.unlink()
    for filename in (
        REVIEW_MANIFEST_FILENAME,
        REVIEW_README_FILENAME,
        VERIFICATION_REPORT_FILENAME,
        REVIEW_ZIP_FILENAME,
    ):
        path = output_dir / filename
        if path.is_file():
            path.unlink()


def build_phase5c_review_package(
    run_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    include_plots: bool = True,
    include_native_playback: bool = True,
    include_frame_checks: bool = True,
    create_zip: bool = True,
) -> tuple[Path, Path]:
    input_run_dir = Path(run_dir).expanduser().resolve()
    if not input_run_dir.is_dir():
        raise ValueError(f"run_dir is not a directory: {input_run_dir}")
    artifacts_dir = input_run_dir / "artifacts"
    output_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else artifacts_dir / "vizard" / "phase5c_review"
    )

    frame_check_dir = artifacts_dir / "vizard" / "basilisk" / "frame_check"
    frame_native_error: Optional[str] = None
    if (
        include_frame_checks
        and frame_check_dir.is_dir()
        and not _frame_native_outputs_complete(frame_check_dir)
    ):
        try:
            convert_frame_check_fixtures_to_native(frame_check_dir)
        except Exception as exc:
            frame_native_error = f"{type(exc).__name__}: {exc}"

    copy_specs = [
        (artifacts_dir / "preds_test.npz", Path("preds_test.npz")),
        (
            artifacts_dir / "preds_test_meta.json",
            Path("preds_test_meta.json"),
        ),
        (
            artifacts_dir / "adcs_timeseries.csv",
            Path("adcs_timeseries.csv"),
        ),
        (
            artifacts_dir / "adcs_timeseries_meta.json",
            Path("adcs_timeseries_meta.json"),
        ),
        (
            artifacts_dir / "phase5c_demo_summary.json",
            Path("phase5c_demo_summary.json"),
        ),
        (
            artifacts_dir / "phase5c_demo_config.json",
            Path("phase5c_demo_config.json"),
        ),
        (
            artifacts_dir / "toy_training_trace.json",
            Path("toy_training_trace.json"),
        ),
        (
            artifacts_dir / "vizard" / "vizard_spacecraft_states.csv",
            Path("vizard_spacecraft_states.csv"),
        ),
        (
            artifacts_dir / "vizard" / "vizard_export_manifest.json",
            Path("vizard_export_manifest.json"),
        ),
        (
            artifacts_dir / "vizard" / "basilisk" / "dataFileToViz_input.csv",
            Path("dataFileToViz_input.csv"),
        ),
        (
            artifacts_dir
            / "vizard"
            / "basilisk"
            / "dataFileToViz_input_manifest.json",
            Path("dataFileToViz_input_manifest.json"),
        ),
        (
            artifacts_dir
            / "vizard"
            / "basilisk"
            / "README_manual_vizard_check.md",
            Path("README_manual_vizard_check.md"),
        ),
        (
            artifacts_dir
            / "vizard"
            / "basilisk"
            / "native"
            / "basilisk_api_probe.json",
            Path("native/basilisk_api_probe.json"),
        ),
        (
            artifacts_dir
            / "vizard"
            / "basilisk"
            / "native"
            / NATIVE_BRIDGE_MANIFEST_FILENAME,
            Path(f"native/{NATIVE_BRIDGE_MANIFEST_FILENAME}"),
        ),
        (
            artifacts_dir
            / "vizard"
            / "basilisk"
            / "native"
            / "native_bridge_log.txt",
            Path("native/native_bridge_log.txt"),
        ),
        (
            artifacts_dir
            / "vizard"
            / "basilisk"
            / "native"
            / "native_conversion_output_manifest.json",
            Path("native/native_conversion_output_manifest.json"),
        ),
    ]
    if include_native_playback:
        copy_specs.append(
            (
                artifacts_dir
                / "vizard"
                / "basilisk"
                / "native"
                / NATIVE_PLAYBACK_FILENAME,
                Path(f"native/{NATIVE_PLAYBACK_FILENAME}"),
            )
        )

    expected_optional = [str(source) for source, _ in copy_specs]
    meaningful_sources = [source for source, _ in copy_specs if source.is_file()]
    plot_sources: list[Path] = []
    if include_plots:
        plots_dir = artifacts_dir / "plots"
        if plots_dir.is_dir():
            plot_sources = sorted(
                path
                for path in plots_dir.iterdir()
                if path.is_file()
                and (path.suffix.lower() == ".png" or path.suffix == ".json")
            )
        expected_optional.append(str(plots_dir / "*.png"))

    frame_sources: list[Path] = []
    if include_frame_checks and frame_check_dir.is_dir():
        frame_sources = sorted(
            path
            for path in frame_check_dir.rglob("*")
            if path.is_file()
        )
        expected_optional.append(str(frame_check_dir / "*"))

    if not meaningful_sources and not plot_sources and not frame_sources:
        raise ValueError(
            f"no meaningful Phase 1-5B artifacts found under {artifacts_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_package(output_dir)
    included_relative: list[str] = []
    for source, relative in copy_specs:
        destination = output_dir / relative
        if _copy_if_present(source, destination, included_relative):
            included_relative[-1] = relative.as_posix()
    for source in plot_sources:
        relative = Path("plots") / source.name
        _copy_if_present(source, output_dir / relative, included_relative)
        included_relative[-1] = relative.as_posix()
    for source in frame_sources:
        relative = Path("frame_check") / source.relative_to(frame_check_dir)
        _copy_if_present(source, output_dir / relative, included_relative)
        included_relative[-1] = relative.as_posix()

    native_playback_present = (
        output_dir / "native" / NATIVE_PLAYBACK_FILENAME
    ).is_file()
    plots_present = any(path.startswith("plots/") for path in included_relative)
    frame_checks_present = any(
        path.startswith("frame_check/") for path in included_relative
    )
    probe = _read_json(
        artifacts_dir
        / "vizard"
        / "basilisk"
        / "native"
        / "basilisk_api_probe.json"
    )
    readme_path = output_dir / REVIEW_README_FILENAME
    report_path = output_dir / VERIFICATION_REPORT_FILENAME
    manifest_path = output_dir / REVIEW_MANIFEST_FILENAME
    zip_path = output_dir / REVIEW_ZIP_FILENAME

    readme_path.write_text(
        _review_readme(
            run_dir=input_run_dir,
            native_playback_present=native_playback_present,
            plots_present=plots_present,
            frame_checks_present=frame_checks_present,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        _verification_report(
            run_dir=input_run_dir,
            basilisk_version=probe.get("basilisk_version"),
            playback_path=(
                f"native/{NATIVE_PLAYBACK_FILENAME}"
                if native_playback_present
                else None
            ),
        ),
        encoding="utf-8",
    )

    included_sources = {
        str(source.resolve())
        for source, _ in copy_specs
        if source.is_file()
    }
    included_sources.update(str(path.resolve()) for path in plot_sources)
    included_sources.update(str(path.resolve()) for path in frame_sources)
    missing_optional = sorted(
        path
        for path in expected_optional
        if "*" in path or str(Path(path).resolve()) not in included_sources
    )
    if plot_sources:
        missing_optional = [
            path for path in missing_optional if not path.endswith("*.png")
        ]
    if frame_sources:
        missing_optional = [
            path for path in missing_optional if not path.endswith("frame_check/*")
        ]
    if frame_native_error:
        missing_optional.append(
            f"frame-check native conversion failed: {frame_native_error}"
        )

    manifest = {
        "schema_version": "phase5c_review_v1",
        "input_run_dir": str(input_run_dir),
        "output_dir": str(output_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "included_artifacts": sorted(included_relative),
        "missing_optional_artifacts": missing_optional,
        "native_playback_present": native_playback_present,
        "plots_present": plots_present,
        "frame_checks_present": frame_checks_present,
        "review_readme": str(readme_path),
        "verification_report": str(report_path),
        "review_bundle_zip": str(zip_path) if create_zip else None,
        "official_metrics_affected": False,
        "notes": (
            "Human-checkable Vizard review package. Manual frame verification "
            "is required before finalizing MRP sign and frame conventions."
        ),
    }
    _write_json(manifest_path, manifest)

    if create_zip:
        tmp_zip = zip_path.with_name(f".{zip_path.name}.tmp")
        try:
            with zipfile.ZipFile(
                tmp_zip,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
                archive_relatives = sorted(
                    set(
                        included_relative
                        + [
                            REVIEW_MANIFEST_FILENAME,
                            REVIEW_README_FILENAME,
                            VERIFICATION_REPORT_FILENAME,
                        ]
                    )
                )
                for relative in archive_relatives:
                    path = output_dir / relative
                    if path.is_file():
                        archive.write(path, relative)
            tmp_zip.replace(zip_path)
        except Exception as exc:
            if tmp_zip.exists():
                tmp_zip.unlink()
            raise RuntimeError(
                f"failed to create Phase 5C review zip: {exc}"
            ) from exc
    elif zip_path.exists():
        zip_path.unlink()

    return manifest_path, readme_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a human-checkable Phase 5C Vizard review package."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-native-playback", action="store_true")
    parser.add_argument("--no-frame-checks", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args(argv)

    manifest_path, readme_path = build_phase5c_review_package(
        args.run_dir,
        out_dir=args.out_dir,
        include_plots=not args.no_plots,
        include_native_playback=not args.no_native_playback,
        include_frame_checks=not args.no_frame_checks,
        create_zip=not args.no_zip,
    )
    print(f"wrote {manifest_path}")
    print(f"wrote {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
