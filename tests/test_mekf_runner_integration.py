from __future__ import annotations

import copy
import inspect
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.estimators.mekf import MEKFState
from bench.models.mekf import DatasetIdentity, MEKFEventReplayBridge
from bench.models.registry import (
    get_adapter_class,
    get_typed_event_bridge_class,
    list_model_ids,
    list_typed_event_bridge_ids,
)
from bench.runners import run_suite as runner
from bench.tasks.bench_generated import prepare_mekf_unit_st_v1
from bench.tasks.generator.mekf_events import (
    MEKFEventTable,
    compute_semantic_hashes,
    replay_trajectory,
)


SMOKE_YAML = Path("bench/configs/suite_phase1a_unit_st_smoke.yaml")
MODEL = {"model_id": "mekf_event_replay_v1", "registry_kind": "typed_event_bridge"}


def _task(producer_id: str, *, task_id: str | None = None) -> dict:
    common = {
        "num_trajectories": 3,
        "duration_s": 0.2,
        "gyro_rate_hz": 20,
        "star_tracker_rate_hz": 5,
        "initial_attitude_max_rad": 0.15,
        "angular_rate_max_rad_s": 0.1,
        "gyro_bias_max_rad_s": 0.005,
        "gyro_noise_std_rad_s": 5.0e-4,
        "randomize_star_tracker_sign": True,
        "train_fraction": 0.6,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
    }
    if producer_id == "synthetic-unit-st-v1":
        common.update(
            star_tracker_noise_std_rad=1.0e-3,
            star_tracker_R_diagonal_rad2=[1.0e-6, 1.0e-6, 1.0e-6],
        )
        default_task_id = "cp4_synthetic"
    else:
        common.update(
            star_tracker_R_rad2=[
                [1.0e-6, 0.0, 0.0],
                [0.0, 1.0e-6, 0.0],
                [0.0, 0.0, 1.0e-6],
            ],
            star_tracker_noise_scale=1.0,
            representative_mass_kg=10.0,
            representative_spherical_inertia_kg_m2=7.0,
        )
        default_task_id = "cp4_basilisk"
    return {
        "task_id": task_id or default_task_id,
        "task_family": "mekf_unit_st_v1",
        "metadata": {"validation_tier": "representative_tier_0_smoke"},
        "typed_event_dataset": {
            "producer_id": producer_id,
            "cache_namespace": "p1a-cp4-tests-v1",
            "generator_config": common,
        },
        "mekf_replay": {
            "initial_time_s": 0.0,
            "initial_state": {
                "q_NB": [1.0, 0.0, 0.0, 0.0],
                "b_g_rad_s": [0.0, 0.0, 0.0],
                "P_diagonal": [1.0e-3, 1.0e-3, 1.0e-3, 1.0e-4, 1.0e-4, 1.0e-4],
            },
            "process_noise": {
                "Q_c_diagonal": [1.0e-10] * 3 + [1.0e-12] * 3,
            },
            "evaluation_split": "test",
            "metric_confidence_level": 0.95,
        },
        "sweep": {},
    }


def _suite(tmp_path: Path, name: str = "cp4_integration_test") -> dict:
    return {
        "suite": {"name": name},
        "runner": {
            "device": "cpu",
            "precision": "fp32",
            "deterministic": True,
            "tracks": [{"track_id": "frozen", "adaptation_enabled": False}],
        },
        "reporting": {
            "output_dir_template": str(
                tmp_path
                / "runs/{suite.name}/{task_id}/{model_id}/{track_id}/seed_{seed}/scenario_{scenario_id}"
            )
        },
    }


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: dict,
    seed: int,
    *,
    suite_name: str = "cp4_integration_test",
    model: dict | None = None,
    init_id: str = "untrained",
    track_id: str = "frozen",
) -> dict:
    monkeypatch.setenv("BENCH_DATA_CACHE", str(tmp_path / "cache"))
    return runner.run_one(
        suite=_suite(tmp_path, suite_name),
        task=task,
        model=model or MODEL,
        scenario_settings={},
        seed=seed,
        track_id=track_id,
        device_str="cpu",
        precision="fp32",
        init_id=init_id,
    )


def _runner_npz(run_result: dict) -> tuple[dict[str, np.ndarray], dict]:
    artifact_dir = Path(run_result["run_dir"]) / "artifacts/mekf_replay"
    manifest_bytes = (artifact_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes.decode("ascii"))
    assert manifest_bytes == json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    filenames = [item["filename"] for item in manifest["trajectory_files"]]
    assert len(filenames) == 1
    with np.load(artifact_dir / filenames[0], allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    return arrays, manifest


def _prepared_bridge(tmp_path: Path, task: dict, seed: int):
    prepared = prepare_mekf_unit_st_v1(
        suite_name="cp4_integration_test",
        task_cfg=task,
        seed=seed,
        cache_root=tmp_path / "cache",
    )
    identity = DatasetIdentity.from_verified(prepared.manifest, prepared.semantic_hashes)
    initial_state, initial_time_s, Q_c, _resolved = runner._p1a_mekf_filter_configuration(task)
    trajectory_id = int(prepared.trajectory_split.test_ids[0])
    bridge = MEKFEventReplayBridge(expected_dataset_identity=identity)
    artifact = bridge.replay_events(
        prepared.dataset.events,
        trajectory_id,
        initial_state,
        initial_time_s,
        Q_c,
        identity,
    )
    return prepared, identity, initial_state, initial_time_s, Q_c, artifact


def test_cp4_01_smoke_yaml_contract_and_fixed_ids() -> None:
    suite = runner.load_suite_yaml(SMOKE_YAML)
    assert suite["suite"]["name"] == "phase1a_unit_st_smoke"
    assert suite["seeds"] == [6101, 6102, 6103]
    assert {task["task_family"] for task in suite["tasks"]} == {"mekf_unit_st_v1"}
    assert {task["typed_event_dataset"]["producer_id"] for task in suite["tasks"]} == {
        "synthetic-unit-st-v1",
        "basilisk-unit-st-v1",
    }
    assert [model["model_id"] for model in suite["models"]] == ["mekf_event_replay_v1"]
    assert all(task["metadata"]["flight_grade"] is False for task in suite["tasks"])


def test_cp4_02_dispatch_and_separate_registry_are_append_only(tmp_path: Path) -> None:
    assert "mekf_event_replay_v1" not in list_model_ids()
    assert list_typed_event_bridge_ids() == ["mekf_event_replay_v1"]
    assert get_typed_event_bridge_class("mekf_event_replay_v1") is MEKFEventReplayBridge
    with pytest.raises(KeyError):
        get_adapter_class("mekf_event_replay_v1")
    prepared = prepare_mekf_unit_st_v1(
        suite_name="cp4_registry",
        task_cfg=_task("synthetic-unit-st-v1"),
        seed=6200,
        cache_root=tmp_path,
    )
    assert prepared.producer_id == "synthetic-unit-st-v1"


@pytest.mark.parametrize("seed", (6201, 6202, 6203))
def test_cp4_03_fresh_and_cache_hit_synthetic_runner_property(
    seed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task("synthetic-unit-st-v1")
    fresh = _run(tmp_path, monkeypatch, task, seed, suite_name=f"cp4_synth_{seed}")
    fresh_arrays, fresh_manifest = _runner_npz(fresh)
    hit = _run(tmp_path, monkeypatch, task, seed, suite_name=f"cp4_synth_{seed}")
    hit_arrays, hit_manifest = _runner_npz(hit)
    assert fresh["status"] == hit["status"] == "ok"
    assert fresh["cache_state"] == "fresh_generation"
    assert hit["cache_state"] == "verified_cache_hit"
    assert fresh_manifest["dataset_identity"] == hit_manifest["dataset_identity"]
    for name in fresh_arrays:
        assert np.array_equal(fresh_arrays[name], hit_arrays[name])


@pytest.mark.parametrize("seed", (6301, 6302, 6303))
def test_cp4_04_fresh_and_cache_hit_basilisk_runner_property(
    seed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task("basilisk-unit-st-v1")
    fresh = _run(tmp_path, monkeypatch, task, seed, suite_name=f"cp4_basilisk_{seed}")
    fresh_arrays, fresh_manifest = _runner_npz(fresh)
    hit = _run(tmp_path, monkeypatch, task, seed, suite_name=f"cp4_basilisk_{seed}")
    hit_arrays, hit_manifest = _runner_npz(hit)
    assert fresh["status"] == hit["status"] == "ok"
    assert fresh["cache_state"] == "fresh_generation"
    assert hit["cache_state"] == "verified_cache_hit"
    assert fresh_manifest["dataset_identity"] == hit_manifest["dataset_identity"]
    for name in fresh_arrays:
        assert np.array_equal(fresh_arrays[name], hit_arrays[name])


def test_cp4_05_verified_cache_is_exact_three_file_sidecar(tmp_path: Path) -> None:
    task = _task("synthetic-unit-st-v1")
    fresh = prepare_mekf_unit_st_v1(
        suite_name="cp4_cache", task_cfg=task, seed=6401, cache_root=tmp_path
    )
    hit = prepare_mekf_unit_st_v1(
        suite_name="cp4_cache", task_cfg=task, seed=6401, cache_root=tmp_path
    )
    assert fresh.cache_state == "fresh_generation"
    assert hit.cache_state == "verified_cache_hit"
    assert {path.name for path in hit.dataset_dir.iterdir()} == {
        "manifest.json",
        "truth.npz",
        "events.npz",
    }
    assert fresh.semantic_hashes == hit.semantic_hashes
    assert fresh.dataset_config_hash == hit.dataset_config_hash


def test_cp4_06_stale_source_fingerprint_cache_is_rejected(tmp_path: Path) -> None:
    task = _task("synthetic-unit-st-v1")
    prepared = prepare_mekf_unit_st_v1(
        suite_name="cp4_stale", task_cfg=task, seed=6402, cache_root=tmp_path
    )
    manifest = copy.deepcopy(prepared.manifest)
    manifest.pop("semantic_hashes", None)
    manifest["source_fingerprints"]["bench/estimators/mekf.py"] = "0" * 64
    hashes = compute_semantic_hashes(prepared.dataset, manifest)
    manifest["semantic_hashes"] = hashes.as_dict()
    (prepared.dataset_dir / "manifest.json").write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    )
    with pytest.raises(ValueError, match="source fingerprint mismatch"):
        prepare_mekf_unit_st_v1(
            suite_name="cp4_stale", task_cfg=task, seed=6402, cache_root=tmp_path
        )


def test_cp4_07_direct_bridge_runner_q_b_P_r_S_and_identity_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task("synthetic-unit-st-v1")
    result = _run(tmp_path, monkeypatch, task, 6501)
    runner_arrays, runner_manifest = _runner_npz(result)
    prepared, identity, state, initial_time_s, Q_c, bridge_artifact = _prepared_bridge(
        tmp_path, task, 6501
    )
    direct = replay_trajectory(
        prepared.dataset.events,
        bridge_artifact.trajectory_id,
        state,
        initial_time_s,
        Q_c,
    )
    comparisons = {
        "q_hat_NB": direct.q_NB_history,
        "b_hat_rad_s": direct.b_g_history,
        "P": direct.P_history,
        "st_residual": direct.star_tracker_residual,
        "st_S": direct.star_tracker_S,
    }
    for name, direct_value in comparisons.items():
        assert np.array_equal(direct_value, getattr(bridge_artifact, name))
        assert np.array_equal(direct_value, runner_arrays[name])
    assert runner_manifest["dataset_identity"] == identity.as_dict()


def test_cp4_08_nonsemantic_model_and_task_metadata_preserve_realization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task("synthetic-unit-st-v1")
    first = _run(tmp_path, monkeypatch, task, 6502, model={**MODEL, "display_name": "first"})
    arrays_first, manifest_first = _runner_npz(first)
    changed_task = copy.deepcopy(task)
    changed_task["metadata"]["review_note"] = "nonsemantic"
    second = _run(
        tmp_path,
        monkeypatch,
        changed_task,
        6502,
        model={**MODEL, "display_name": "second", "training_note": "ignored"},
    )
    arrays_second, manifest_second = _runner_npz(second)
    assert manifest_first["dataset_config_hash"] == manifest_second["dataset_config_hash"]
    assert manifest_first["dataset_identity"] == manifest_second["dataset_identity"]
    for name in arrays_first:
        assert np.array_equal(arrays_first[name], arrays_second[name])
    dataset_manifest = json.dumps(manifest_second["dataset_identity"], sort_keys=True)
    assert "display_name" not in dataset_manifest and "training_note" not in dataset_manifest


def test_cp4_09_truth_never_crosses_estimator_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[set[str], bool]] = []
    original = MEKFEventReplayBridge.replay_events

    def guarded(self, *args, **kwargs):
        calls.append((set(inspect.signature(original).parameters), hasattr(args[0], "truth")))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MEKFEventReplayBridge, "replay_events", guarded)
    result = _run(tmp_path, monkeypatch, _task("synthetic-unit-st-v1"), 6503)
    assert result["status"] == "ok" and calls
    assert all(not has_truth for _parameters, has_truth in calls)
    parameters = calls[0][0]
    assert not parameters & {"truth", "oracle", "label", "future"}
    source = inspect.getsource(runner._run_p1a_mekf_event_replay)
    assert source.index("bridge.replay_events(") < source.index("truth=prepared.dataset.truth")


def test_cp4_10_truth_join_is_exact_and_never_interpolates(tmp_path: Path) -> None:
    task = _task("synthetic-unit-st-v1")
    prepared, _identity, _state, _time, _Q, artifact = _prepared_bridge(tmp_path, task, 6504)
    q_true, b_true = runner._p1a_exact_truth_join(prepared.dataset.truth, artifact)
    assert q_true.shape == artifact.q_hat_NB.shape
    assert b_true.shape == artifact.b_hat_rad_s.shape
    bad = SimpleNamespace(
        trajectory_id=artifact.trajectory_id,
        timestamp_s=np.array(artifact.timestamp_s, copy=True),
    )
    bad.timestamp_s[0] = np.nextafter(bad.timestamp_s[0], np.inf)
    with pytest.raises(ValueError, match="timestamp mismatch"):
        runner._p1a_exact_truth_join(prepared.dataset.truth, bad)


def test_cp4_11_lossless_artifact_round_trip_and_no_truth_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(tmp_path, monkeypatch, _task("synthetic-unit-st-v1"), 6505)
    arrays, _manifest = _runner_npz(result)
    assert set(arrays) == {
        "event_index",
        "event_order",
        "timestamp_s",
        "sensor_code",
        "q_hat_NB",
        "b_hat_rad_s",
        "P",
        "st_event_index",
        "st_event_order",
        "st_timestamp_s",
        "st_residual",
        "st_S",
    }
    assert not any("true" in name or "truth" in name for name in arrays)
    assert all(not value.dtype.hasobject for value in arrays.values())


def test_cp4_12_star_tracker_evidence_is_compact_and_exactly_aligned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(tmp_path, monkeypatch, _task("synthetic-unit-st-v1"), 6506)
    arrays, manifest = _runner_npz(result)
    star_rows = np.flatnonzero(arrays["sensor_code"] == np.int16(2))
    assert 0 < arrays["st_residual"].shape[0] < arrays["q_hat_NB"].shape[0]
    assert np.array_equal(arrays["st_event_index"], arrays["event_index"][star_rows])
    assert np.array_equal(arrays["st_event_order"], arrays["event_order"][star_rows])
    assert np.array_equal(arrays["st_timestamp_s"], arrays["timestamp_s"][star_rows])
    assert manifest["star_tracker_update_count"] == arrays["st_residual"].shape[0]


def test_cp4_13_star_tracker_q_and_negative_q_replay_are_identical(tmp_path: Path) -> None:
    task = _task("synthetic-unit-st-v1")
    prepared, identity, state, initial_time_s, Q_c, positive = _prepared_bridge(
        tmp_path, task, 6507
    )
    kwargs = {
        field.name: np.array(getattr(prepared.dataset.events, field.name), copy=True)
        for field in fields(prepared.dataset.events)
    }
    kwargs["star_tracker_q_NB"] *= -1.0
    negative_events = MEKFEventTable(**kwargs)
    negative = MEKFEventReplayBridge().replay_events(
        negative_events,
        positive.trajectory_id,
        state,
        initial_time_s,
        Q_c,
        identity,
    )
    for name in ("q_hat_NB", "b_hat_rad_s", "P", "st_residual", "st_S"):
        assert np.array_equal(getattr(positive, name), getattr(negative, name))


def test_cp4_14_exact_pair_never_calls_dense_float32_runner_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy dense path was entered")

    for name in (
        "_load_split_npz",
        "_make_loader",
        "_predict_batches",
        "_load_adapter",
        "_try_call_train",
        "_try_call_eval",
    ):
        monkeypatch.setattr(runner, name, forbidden)
    result = _run(tmp_path, monkeypatch, _task("synthetic-unit-st-v1"), 6508)
    assert result["status"] == "ok"
    source = inspect.getsource(runner._run_p1a_mekf_event_replay).lower()
    for forbidden_token in ("float32", "y_seq", "zero-filled", "_seqdataset"):
        assert forbidden_token not in source


def test_cp4_15_training_and_adaptation_are_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(
        tmp_path,
        monkeypatch,
        _task("synthetic-unit-st-v1"),
        6509,
        init_id="trained",
    )
    assert result["status"] == "failed"
    assert "disables training and adaptation" in result["error"]
    assert not (Path(result["run_dir"]) / "artifacts/mekf_replay").exists()


def test_cp4_16_replay_failure_leaves_no_partial_valid_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_replay(*_args, **_kwargs):
        raise RuntimeError("injected replay failure")

    monkeypatch.setattr(MEKFEventReplayBridge, "replay_events", fail_replay)
    result = _run(tmp_path, monkeypatch, _task("synthetic-unit-st-v1"), 6510)
    artifact_parent = Path(result["run_dir"]) / "artifacts"
    assert result["status"] == "failed"
    assert not (artifact_parent / "mekf_replay").exists()
    assert not list(artifact_parent.glob(".mekf_replay.partial.*"))
    assert (Path(result["run_dir"]) / "failure.json").is_file()


def test_cp4_17_partial_pair_fails_before_legacy_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "_load_split_npz",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dense load")),
    )
    result = _run(
        tmp_path,
        monkeypatch,
        _task("synthetic-unit-st-v1"),
        6511,
        model={"model_id": "oracle_kf"},
    )
    assert result["status"] == "failed"
    assert "must be selected as an exact pair" in result["error"]


def test_cp4_18_manifest_provenance_metrics_and_spd_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(tmp_path, monkeypatch, _task("synthetic-unit-st-v1"), 6512)
    _arrays, manifest = _runner_npz(result)
    metrics = json.loads((Path(result["run_dir"]) / "metrics.json").read_text())
    assert manifest["artifact_contract_version"] == "p1a-cp4-mekf-replay-artifact-v1"
    assert manifest["task_family"] == "mekf_unit_st_v1"
    assert manifest["model_id"] == "mekf_event_replay_v1"
    assert manifest["truth_in_trajectory_npz"] is False
    assert set(manifest["dataset_identity"]) == {
        "schema_version",
        "generator_id",
        "convention_id",
        "truth_hash",
        "sensor_payload_hash",
        "event_order_hash",
        "manifest_hash",
        "dataset_hash",
    }
    canonical = metrics["canonical_mekf"]
    assert canonical["metric_contract"] == "p1a-canonical-mekf-metrics-v1"
    assert canonical["nis"]["count"] == manifest["star_tracker_update_count"]
    assert canonical["nees"]["count"] == manifest["processed_event_count"]
    assert canonical["spd"]["P"]["all_cholesky_succeeded"] is True
    assert canonical["spd"]["S"]["all_cholesky_succeeded"] is True
    assert canonical["spd"]["P"]["minimum_eigenvalue"] > 0.0
    assert canonical["spd"]["S"]["minimum_eigenvalue"] > 0.0
