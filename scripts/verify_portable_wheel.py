#!/usr/bin/env python3
"""Verify that a wheel contains the tracked portable benchmark surface."""

from __future__ import annotations

import argparse
import configparser
import subprocess
import sys
import zipfile
from pathlib import Path


REQUIRED_MEMBERS = (
    "bench/__init__.py",
    "bench/tasks/smoke_data.py",
    "bench/tasks/replay_generated_data.py",
    "bench/runners/run_suite.py",
    "bench/models/registry.py",
    "bench/models/g1_snn_split_knet.py",
    "bench/models/spike_ra_knet.py",
    "bench/models/spike_split_knet.py",
    "bench/side_gyro_mag_comp_v1/__init__.py",
    "bench/side_gyro_mag_comp_v1/data.py",
    "bench/side_gyro_mag_comp_v1/model.py",
    "bench/side_gyro_mag_comp_v1/study.py",
    "bench/side_gyro_mag_comp_pilot/__init__.py",
    "bench/side_gyro_mag_comp_pilot/data.py",
    "bench/side_gyro_mag_comp_pilot/model.py",
    "bench/side_gyro_mag_comp_pilot/oracle_decomposition.py",
    "bench/side_gyro_mag_comp_pilot/runner.py",
    "bench/side_gyro_mag_comp_pilot/study.py",
    "bench/reports/make_report.py",
    "bench/control/cli.py",
    "bench/control/api/app.py",
    "bench/control/api/routers/actions.py",
    "bench/control/api/routers/config.py",
    "bench/control/checkpoints/resume_coordinator.py",
    "bench/control/config/gui_service.py",
    "bench/ui/dash_app.py",
    "bench/ui/pages/new_run.py",
    "bench/configs/suite_kf_baseline_smoke.yaml",
    "bench/configs/suite_phase6a_replay_short.yaml",
    "bench/configs/suite_phase6a_replay_long.yaml",
    "bench/configs/suite_phase6f_kalmannet_adcs_tiny.yaml",
    "bench/configs/suite_basilisk_spike_ra_stage2_event.yaml",
    "bench/configs/suite_basilisk_spike_split_smoke.yaml",
    "bench/configs/side_gyro_mag_comp_v1.yaml",
    "bench/configs/side_gyro_mag_comp_pilot.yaml",
    "bench/visualization/phase6b_checkpoint_replay.py",
    "bench/visualization/phase6g_kalmannet_export.py",
    "bench/visualization/phase7_vizard_convention.py",
    "bench/visualization/vizard_native_bridge.py",
    "viz/app/main.py",
)

EXPECTED_CONSOLE_SCRIPTS = {
    "bench-smoke-data": "bench.tasks.smoke_data:main",
    "bench-run-suite": "bench.runners.run_suite:main",
    "bench-make-report": "bench.reports.make_report:main",
    "bench-control": "bench.control.cli:main",
    "bench-control-api": "bench.control.api.app:main",
    "bench-dashboard": "bench.ui.dash_app:main",
}

FORBIDDEN_PREFIXES = ("runs/", "reports/", "bench_data_cache/", "experiments/")


def _entry_points(archive: zipfile.ZipFile) -> dict[str, str]:
    candidates = [name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")]
    if len(candidates) != 1:
        return {}
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(archive.read(candidates[0]).decode("utf-8"))
    if not parser.has_section("console_scripts"):
        return {}
    return {name: value.strip() for name, value in parser.items("console_scripts")}


def verify(wheel_path: Path) -> list[str]:
    """Return human-readable contract violations for ``wheel_path``."""
    failures: list[str] = []
    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        entries = _entry_points(archive)

    for member in REQUIRED_MEMBERS:
        if member not in names:
            failures.append(f"missing wheel member: {member}")

    for name in sorted(names):
        if name.startswith(FORBIDDEN_PREFIXES):
            failures.append(f"generated/research payload packaged: {name}")
        if "/__pycache__/" in f"/{name}" or name.endswith((".pyc", ".pyo")):
            failures.append(f"Python cache packaged: {name}")

    if not any(name.endswith(".dist-info/licenses/LICENSE") for name in names):
        failures.append("MIT LICENSE is absent from wheel metadata")

    for command, target in EXPECTED_CONSOLE_SCRIPTS.items():
        actual = entries.get(command)
        if actual != target:
            failures.append(f"console script {command!r}: expected {target!r}, found {actual!r}")
    return failures


def untracked_required_sources(source_root: Path) -> list[str]:
    """Return required sources that are not in Git, preventing dirty-only PASS."""
    if not (source_root / ".git").exists():
        return []
    required_sources = (*REQUIRED_MEMBERS, "LICENSE")
    result = subprocess.run(
        ["git", "-C", str(source_root), "ls-files", "--error-unmatch", "--", *required_sources],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        return []
    tracked = set(
        subprocess.run(
            ["git", "-C", str(source_root), "ls-files", "--", *required_sources],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    return [member for member in required_sources if member not in tracked]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="Wheel produced by pip/uv build")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Git worktree whose tracked sources must provide every required member",
    )
    args = parser.parse_args(argv)

    wheel_path = args.wheel.expanduser().resolve()
    if not wheel_path.is_file():
        parser.error(f"wheel not found: {wheel_path}")

    failures = verify(wheel_path)
    failures.extend(
        f"required source is not tracked: {member}"
        for member in untracked_required_sources(args.source_root.expanduser().resolve())
    )
    if failures:
        print("portable wheel check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"portable wheel check passed: {wheel_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
