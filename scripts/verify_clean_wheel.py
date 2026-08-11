#!/usr/bin/env python3
"""Build HEAD from a Git archive, install its wheel, and smoke every public CLI."""

from __future__ import annotations

import os
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

        clean_env = os.environ.copy()
        clean_env.pop("PYTHONPATH", None)
        clean_env["PYTHONNOUSERSITE"] = "1"
        import_check = (
            "from pathlib import Path; "
            "import bench, bench.control.cli, bench.control.checkpoints.resume_coordinator, "
            "bench.control.api.routers.actions, bench.control.config.gui_service, "
            "bench.runners.run_suite, bench.tasks.smoke_data, viz; "
            f"assert not Path(bench.__file__).resolve().is_relative_to(Path({str(repo_root)!r}))"
        )
        run([str(python), "-c", import_check], cwd=tmp, env=clean_env)
        for command in CONSOLE_SCRIPTS:
            run([str(bin_dir / command), "--help"], cwd=tmp, env=clean_env)

        print(f"clean wheel and CLI checks passed: {wheels[0].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
