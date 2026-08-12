"""Run the bounded ADCS replay/Vizard regression manifest used by CI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ADCS_REPLAY_TEST_PATHS = (
    "bench/tests/test_adcs_event_metrics.py",
    "bench/tests/test_adcs_plots.py",
    "bench/tests/test_adcs_timeseries.py",
    "bench/tests/test_basilisk_api_probe.py",
    "bench/tests/test_checkpoint_contract_probe.py",
    "bench/tests/test_checkpoint_replay_adapters.py",
    "bench/tests/test_phase5c_demo.py",
    "bench/tests/test_phase6a_replay_input.py",
    "bench/tests/test_phase6b_checkpoint_replay.py",
    "bench/tests/test_phase6c_replay_visualization.py",
    "bench/tests/test_phase6d_checkpoint_replay.py",
    "bench/tests/test_phase6e_adapter_contract.py",
    "bench/tests/test_phase6e_checkpoint_package.py",
    "bench/tests/test_phase6f_kalmannet_export.py",
    "bench/tests/test_phase6f_kalmannet_replay_adapter.py",
    "bench/tests/test_phase6g_kalmannet_replay_path.py",
    "bench/tests/test_phase6g_kalmannet_tiny_train_export.py",
    "bench/tests/test_phase7_vizard_convention.py",
    "bench/tests/test_pred_artifact_schema.py",
    "bench/tests/test_replay_checkpoint_contract.py",
    "bench/tests/test_replay_generated_training_bridge.py",
    "bench/tests/test_replay_suite_scenario.py",
    "bench/tests/test_vizard_basilisk_wrapper.py",
    "bench/tests/test_vizard_export.py",
    "bench/tests/test_vizard_native_bridge.py",
    "bench/tests/test_vizard_phase5c_review.py",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def validate_manifest(repository_root: Path) -> None:
    if len(ADCS_REPLAY_TEST_PATHS) != 26:
        raise RuntimeError(
            f"ADCS replay CI manifest must contain 26 paths, found "
            f"{len(ADCS_REPLAY_TEST_PATHS)}"
        )
    if len(set(ADCS_REPLAY_TEST_PATHS)) != len(ADCS_REPLAY_TEST_PATHS):
        raise RuntimeError("ADCS replay CI manifest contains duplicate paths")
    forbidden = ("spike", "phase2")
    for relative_path in ADCS_REPLAY_TEST_PATHS:
        lowered = relative_path.lower()
        if any(token in lowered for token in forbidden):
            raise RuntimeError(
                f"ADCS replay CI manifest crosses a forbidden scope: {relative_path}"
            )
        if not (repository_root / relative_path).is_file():
            raise FileNotFoundError(
                f"ADCS replay CI test path is missing: {relative_path}"
            )


def main() -> int:
    repository_root = _repository_root()
    validate_manifest(repository_root)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-s",
        "-p",
        "no:cacheprovider",
        *ADCS_REPLAY_TEST_PATHS,
    ]
    return subprocess.call(command, cwd=repository_root)


if __name__ == "__main__":
    raise SystemExit(main())
