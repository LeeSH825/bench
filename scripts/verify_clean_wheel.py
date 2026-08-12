#!/usr/bin/env python3
"""Build HEAD from a Git archive, install its wheel, and smoke every public CLI."""

from __future__ import annotations

import os
import site
import subprocess
import sys
import tarfile
import tempfile
import venv
from pathlib import Path

from verify_portable_wheel import verify


CONSOLE_SCRIPTS = (
    "bench-smoke-data",
    "bench-run-suite",
    "bench-make-report",
    "bench-control",
    "bench-control-api",
    "bench-dashboard",
)


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="bench-clean-wheel-") as raw_tmp:
        tmp = Path(raw_tmp)
        archive_path = tmp / "source.tar"
        export_root = tmp / "source"
        wheelhouse = tmp / "wheelhouse"
        environment = tmp / "venv"
        export_root.mkdir()
        wheelhouse.mkdir()

        run(
            ["git", "archive", "--format=tar", f"--output={archive_path}", "HEAD"],
            cwd=repo_root,
        )
        with tarfile.open(archive_path) as archive:
            archive.extractall(export_root)

        run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheelhouse), str(export_root)],
            cwd=tmp,
        )
        wheels = list(wheelhouse.glob("bench-*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"expected one bench wheel, found: {wheels}")
        failures = verify(wheels[0])
        if failures:
            raise SystemExit("clean wheel contract failed:\n  - " + "\n  - ".join(failures))

        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
        bin_dir = environment / ("Scripts" if os.name == "nt" else "bin")
        python = bin_dir / ("python.exe" if os.name == "nt" else "python")
        run(
            [str(python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheels[0])],
            cwd=tmp,
        )

        # A child venv does not inherit packages installed in the invoking
        # venv, even with system_site_packages=True. Make only the caller's
        # dependency directories visible; the assertions below still require
        # bench and viz themselves to resolve from this clean child venv.
        caller_sites = [
            str(Path(path).resolve())
            for path in site.getsitepackages()
            if Path(path).is_dir()
        ]
        child_site = Path(
            subprocess.check_output(
                [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
                cwd=tmp,
                text=True,
            ).strip()
        )
        (child_site / "bench-caller-dependencies.pth").write_text(
            "".join(f"{path}\n" for path in caller_sites),
            encoding="utf-8",
        )

        clean_env = os.environ.copy()
        clean_env.pop("PYTHONPATH", None)
        clean_env["PYTHONNOUSERSITE"] = "1"
        import_check = (
            "from pathlib import Path; "
            "import bench, bench.control.cli, bench.control.checkpoints.resume_coordinator, "
            "bench.control.api.routers.actions, bench.control.config.gui_service, "
            "bench.runners.run_suite, bench.tasks.smoke_data, "
            "bench.tasks.replay_generated_data, "
            "bench.models.g1_snn_split_knet, bench.models.spike_ra_knet, "
            "bench.models.spike_split_knet, "
            "bench.side_gyro_mag_comp_v1.data, "
            "bench.side_gyro_mag_comp_v1.model, "
            "bench.side_gyro_mag_comp_v1.study, "
            "bench.side_gyro_mag_comp_pilot.data, "
            "bench.side_gyro_mag_comp_pilot.model, "
            "bench.side_gyro_mag_comp_pilot.oracle_decomposition, "
            "bench.side_gyro_mag_comp_pilot.runner, "
            "bench.side_gyro_mag_comp_pilot.study, "
            "bench.visualization.phase6b_checkpoint_replay, "
            "bench.visualization.phase6g_kalmannet_export, "
            "bench.visualization.phase7_vizard_convention, "
            "bench.visualization.vizard_native_bridge, viz; "
            f"assert Path(bench.__file__).resolve().is_relative_to(Path({str(environment)!r})); "
            f"assert Path(viz.__file__).resolve().is_relative_to(Path({str(environment)!r})); "
            "from bench.models.registry import list_model_ids; "
            "assert {'g1_snn_split_knet', 'spike_ra_knet', 'spike_split_knet'} "
            ".issubset(set(list_model_ids())); "
            "import yaml; "
            "config_root = Path(bench.__file__).resolve().parent / 'configs'; "
            "spike_configs = sorted(config_root.glob('suite_basilisk_spike_*.yaml')); "
            "assert len(spike_configs) == 12; "
            "assert all(isinstance(yaml.safe_load(path.read_text(encoding='utf-8')), dict) "
            "for path in spike_configs); "
            "side_configs = [config_root / 'side_gyro_mag_comp_v1.yaml', "
            "config_root / 'side_gyro_mag_comp_pilot.yaml']; "
            "assert all(isinstance(yaml.safe_load(path.read_text(encoding='utf-8')), dict) "
            "for path in side_configs)"
        )
        run([str(python), "-c", import_check], cwd=tmp, env=clean_env)
        for command in CONSOLE_SCRIPTS:
            run([str(bin_dir / command), "--help"], cwd=tmp, env=clean_env)

        print(f"clean wheel and CLI checks passed: {wheels[0].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
