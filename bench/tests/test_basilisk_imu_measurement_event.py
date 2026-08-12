from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from bench.tasks.bench_generated import prepare_bench_generated_v0
from bench.tasks.data_format import LoadedSplitV0, load_npz_split_v0
from bench.tasks.generator.datasets.common import DatasetMissingError


@dataclass
class BasiliskIMUMeasurementEventResult:
    ok: bool
    skipped: bool
    note: str
    npz_path: Path
    event_gyro_std_mean: float = float("nan")
    non_event_gyro_std_mean: float = float("nan")
    event_gyro_error_std_mean: float = float("nan")
    non_event_gyro_error_std_mean: float = float("nan")


def _task_cfg(*, task_id: str, event_enabled: bool) -> Dict[str, Any]:
    task: Dict[str, Any] = {
        "task_id": task_id,
        "task_name": "Basilisk IMU Measurement Event Smoke",
        "task_family": "basilisk_imu_adcs_v0",
        "system_type": "nonlinear",
        "x_dim": 6,
        "y_dim": 6,
        "sequence_length_T": 40 if event_enabled else 12,
        "dataset_sizes": {
            "N_train": 4 if event_enabled else 1,
            "N_val": 2 if event_enabled else 1,
            "N_test": 2 if event_enabled else 1,
        },
        "simulation": {
            "dt": 0.1,
            "inertia": [10.0, 8.0, 6.0],
            "disturbance_torque": [0.0, 0.0, 0.0],
            "sigma0_std": 0.05,
            "sigma0_max_norm": 0.25,
            "omega0_std": 0.01,
        },
        "noise": {"Q": {"type": "scaled_identity", "q2": 1.0e-8}},
        "observation": {
            "type": "basilisk_imu_sensor",
            "h_type": "imu_sensor_packet",
            "measurement_mode": "gyro_delta_angle",
        },
        "imu": {
            "measurement_mode": "gyro_delta_angle",
            "gyro_noise_std": 2.0e-4,
            "accel_noise_std": 0.0,
            "gyro_bias_std": 0.0,
            "accel_bias_std": 0.0,
            "sensor_pos_B": [0.0, 0.0, 0.0],
            "body_to_platform_euler321_rad": [0.0, 0.0, 0.0],
        },
        "control_input_u": False,
        "ground_truth": {"has_gt": True},
        "sweep": {},
    }
    if event_enabled:
        task["event_disturbance"] = {
            "enabled": True,
            "event_start_frac": 0.45,
            "event_duration_frac": 0.20,
            "gyro_noise_scale_event": 100.0,
            "gyro_bias_jump_std": 0.02,
            "event_type": "measurement_gyro_bias_jump",
        }
    return task


def _load_splits(artifact: Any) -> List[Tuple[str, LoadedSplitV0]]:
    return [
        ("train", load_npz_split_v0(artifact.train.path)),
        ("val", load_npz_split_v0(artifact.val.path)),
        ("test", load_npz_split_v0(artifact.test.path)),
    ]


def _failed(note: str, npz_path: Path = Path("")) -> BasiliskIMUMeasurementEventResult:
    return BasiliskIMUMeasurementEventResult(
        ok=False,
        skipped=False,
        note=note,
        npz_path=npz_path,
    )


def run_basilisk_imu_measurement_event_tests() -> BasiliskIMUMeasurementEventResult:
    suite_name = "basilisk_imu_measurement_event_smoke"
    with tempfile.TemporaryDirectory(prefix="basilisk_imu_measurement_event_") as tmp:
        cache_root = Path(tmp).resolve()
        try:
            disabled_artifacts = prepare_bench_generated_v0(
                suite_name=suite_name,
                task_cfg=_task_cfg(
                    task_id="Basilisk_IMU_measurement_event_disabled_smoke_v0",
                    event_enabled=False,
                ),
                seed=0,
                cache_root=cache_root,
                scenario_overrides={},
            )
        except DatasetMissingError as exc:
            return BasiliskIMUMeasurementEventResult(
                ok=True,
                skipped=True,
                note=f"Basilisk unavailable; measurement-event integration skipped: {exc}",
                npz_path=Path(""),
            )
        except Exception as exc:
            return _failed(f"non-event Basilisk IMU generation failed: {type(exc).__name__}: {exc}")

        if not disabled_artifacts:
            return _failed("non-event prepare_bench_generated_v0 returned no artifacts")
        disabled_train = load_npz_split_v0(disabled_artifacts[0].train.path)
        if "measurement_event" in disabled_train.meta:
            return _failed("disabled event config unexpectedly added meta.measurement_event", disabled_artifacts[0].train.path)
        disabled_event_keys = sorted(key for key in disabled_train.extras if key.startswith("event_"))
        if disabled_event_keys:
            return _failed(
                f"disabled event config unexpectedly added extras: {disabled_event_keys}",
                disabled_artifacts[0].train.path,
            )

        try:
            enabled_artifacts = prepare_bench_generated_v0(
                suite_name=suite_name,
                task_cfg=_task_cfg(
                    task_id="Basilisk_IMU_measurement_event_enabled_smoke_v0",
                    event_enabled=True,
                ),
                seed=0,
                cache_root=cache_root,
                scenario_overrides={},
            )
        except Exception as exc:
            return _failed(f"event Basilisk IMU generation failed: {type(exc).__name__}: {exc}")
        if not enabled_artifacts:
            return _failed("event prepare_bench_generated_v0 returned no artifacts")

        artifact = enabled_artifacts[0]
        npz_path = artifact.train.path
        expected_n = {"train": 4, "val": 2, "test": 2}
        expected_t = 40
        expected_start = 18
        expected_duration = 8
        expected_end = 26
        required_extras = {
            "event_flag_seq",
            "event_bias_component_seq",
            "event_noise_component_seq",
            "event_start_seq",
            "event_end_seq",
            "event_duration_seq",
        }

        y_parts: List[np.ndarray] = []
        error_parts: List[np.ndarray] = []
        flag_parts: List[np.ndarray] = []
        for split_name, split in _load_splits(artifact):
            missing = sorted(required_extras.difference(split.extras))
            if missing:
                return _failed(f"{split_name} split missing event extras: {missing}", npz_path)

            n_split = expected_n[split_name]
            event_flag = np.asarray(split.extras["event_flag_seq"], dtype=np.float64)
            event_bias = np.asarray(split.extras["event_bias_component_seq"], dtype=np.float64)
            event_noise = np.asarray(split.extras["event_noise_component_seq"], dtype=np.float64)
            if event_flag.shape != (n_split, expected_t, 1):
                return _failed(f"{split_name} event_flag_seq shape={event_flag.shape}", npz_path)
            if event_bias.shape != (n_split, expected_t, 6):
                return _failed(f"{split_name} event_bias_component_seq shape={event_bias.shape}", npz_path)
            if event_noise.shape != (n_split, expected_t, 6):
                return _failed(f"{split_name} event_noise_component_seq shape={event_noise.shape}", npz_path)

            for key, expected_value in (
                ("event_start_seq", expected_start),
                ("event_end_seq", expected_end),
                ("event_duration_seq", expected_duration),
            ):
                values = np.asarray(split.extras[key], dtype=np.int64)
                if values.shape != (n_split,) or not np.all(values == expected_value):
                    return _failed(f"{split_name} {key} mismatch: shape={values.shape} values={values}", npz_path)

            event_mask = event_flag[:, :, 0] > 0.5
            if int(np.sum(event_mask)) != n_split * expected_duration:
                return _failed(f"{split_name} event_flag_seq active sample count mismatch", npz_path)
            if np.any(event_mask[:, :expected_start]) or np.any(event_mask[:, expected_end:]):
                return _failed(f"{split_name} event_flag_seq is active outside configured window", npz_path)
            if not np.all(event_mask[:, expected_start:expected_end]):
                return _failed(f"{split_name} event_flag_seq is inactive inside configured window", npz_path)

            if np.any(event_bias[~event_mask] != 0.0) or np.any(event_noise[~event_mask] != 0.0):
                return _failed(f"{split_name} event components are nonzero outside event window", npz_path)
            if np.any(event_bias[:, :, 3:] != 0.0) or np.any(event_noise[:, :, 3:] != 0.0):
                return _failed(f"{split_name} event components modified non-gyro measurement columns", npz_path)
            if not np.any(event_bias[event_mask] != 0.0) or not np.any(event_noise[event_mask] != 0.0):
                return _failed(f"{split_name} event bias/noise components are unexpectedly all zero", npz_path)
            for i in range(n_split):
                active_bias = event_bias[i, expected_start:expected_end, 0:3]
                if not np.allclose(active_bias, active_bias[0:1], rtol=0.0, atol=0.0):
                    return _failed(f"{split_name} trajectory {i} bias jump is not constant in event window", npz_path)

            measurement_event = split.meta.get("measurement_event")
            if not isinstance(measurement_event, dict) or not bool(measurement_event.get("enabled", False)):
                return _failed(f"{split_name} meta.measurement_event missing or disabled", npz_path)
            if str(measurement_event.get("event_type")) != "measurement_gyro_bias_jump":
                return _failed(f"{split_name} meta.measurement_event.event_type mismatch", npz_path)
            storage = measurement_event.get("storage", {})
            expected_storage = {
                "event_flag": "event_flag_seq",
                "event_bias_component": "event_bias_component_seq",
                "event_noise_component": "event_noise_component_seq",
                "event_start": "event_start_seq",
                "event_end": "event_end_seq",
                "event_duration": "event_duration_seq",
            }
            if not isinstance(storage, dict) or any(storage.get(k) != v for k, v in expected_storage.items()):
                return _failed(f"{split_name} meta.measurement_event.storage mismatch", npz_path)
            if (
                int(measurement_event.get("event_start", -1)) != expected_start
                or int(measurement_event.get("event_end", -1)) != expected_end
                or int(measurement_event.get("event_duration", -1)) != expected_duration
            ):
                return _failed(f"{split_name} meta.measurement_event window mismatch", npz_path)
            if (
                not np.isclose(float(measurement_event.get("event_start_frac", -1.0)), 0.45)
                or not np.isclose(float(measurement_event.get("event_duration_frac", -1.0)), 0.20)
                or not np.isclose(float(measurement_event.get("gyro_noise_scale_event", -1.0)), 100.0)
                or not np.isclose(float(measurement_event.get("gyro_bias_jump_std", -1.0)), 0.02)
                or str(measurement_event.get("units")) != "rad/s"
                or bool(measurement_event.get("truth_dynamics_modified", True))
            ):
                return _failed(f"{split_name} meta.measurement_event semantics mismatch", npz_path)

            imu_gyro = np.asarray(split.extras["imu_gyro_seq"], dtype=np.float64)
            if not np.allclose(split.y[:, :, 0:3], imu_gyro, rtol=1.0e-6, atol=1.0e-7):
                return _failed(f"{split_name} imu_gyro_seq does not match event-modified y gyro", npz_path)

            y_parts.append(np.asarray(split.y[:, :, 0:3], dtype=np.float64))
            error_parts.append(np.asarray(split.extras["imu_error_seq"][:, :, 0:3], dtype=np.float64))
            flag_parts.append(event_mask)

        y_gyro = np.concatenate(y_parts, axis=0)
        gyro_error = np.concatenate(error_parts, axis=0)
        event_mask = np.concatenate(flag_parts, axis=0)
        event_gyro_std_mean = float(np.mean(np.std(y_gyro[event_mask], axis=0)))
        non_event_gyro_std_mean = float(np.mean(np.std(y_gyro[~event_mask], axis=0)))
        event_gyro_error_std_mean = float(np.mean(np.std(gyro_error[event_mask], axis=0)))
        non_event_gyro_error_std_mean = float(np.mean(np.std(gyro_error[~event_mask], axis=0)))

        if not event_gyro_std_mean > non_event_gyro_std_mean:
            return _failed(
                "event gyro measurement std did not exceed non-event std: "
                f"event={event_gyro_std_mean:.6e} non_event={non_event_gyro_std_mean:.6e}",
                npz_path,
            )
        if not event_gyro_error_std_mean > non_event_gyro_error_std_mean:
            return _failed(
                "event gyro residual std did not exceed non-event std: "
                f"event={event_gyro_error_std_mean:.6e} non_event={non_event_gyro_error_std_mean:.6e}",
                npz_path,
            )

    return BasiliskIMUMeasurementEventResult(
        ok=True,
        skipped=False,
        note=(
            "Basilisk IMU measurement-event checks passed "
            f"(gyro std event={event_gyro_std_mean:.6e}, non-event={non_event_gyro_std_mean:.6e}; "
            f"gyro residual std event={event_gyro_error_std_mean:.6e}, "
            f"non-event={non_event_gyro_error_std_mean:.6e})"
        ),
        npz_path=npz_path,
        event_gyro_std_mean=event_gyro_std_mean,
        non_event_gyro_std_mean=non_event_gyro_std_mean,
        event_gyro_error_std_mean=event_gyro_error_std_mean,
        non_event_gyro_error_std_mean=non_event_gyro_error_std_mean,
    )


if __name__ == "__main__":
    result = run_basilisk_imu_measurement_event_tests()
    status = "SKIP" if result.skipped else ("PASS" if result.ok else "FAIL")
    print(f"[{status}] {result.note}")
    raise SystemExit(0 if result.ok else 1)
