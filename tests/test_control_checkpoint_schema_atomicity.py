"""Checkpoint v1 atomicity, corruption detection, and reconciliation.

The property under test throughout: a checkpoint is either fully published and
catalogued, or it is not a checkpoint. Every fault point below must degrade to
something the reconciler can adjudicate — never to a catalog row that promises
state which was never made durable (DND-CSR-010).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
import torch

from bench.control.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    AdapterTrainingState,
    BatchPlan,
    CheckpointError,
    CheckpointKind,
    CheckpointService,
    CheckpointValidationError,
    RngState,
    TrainingCursor,
    ValidationStatus,
    capture_rng,
    check_manifest_compatibility,
    reconcile_run_checkpoints,
    validate_package,
)
from bench.control.checkpoints.atomic import TEMP_SUFFIX, sha256_file
from bench.control.checkpoints.schema import MANIFEST_FILENAME, PAYLOAD_FILENAME
from bench.control.registry.schema import ExperimentRecord, RunRecord, RunState
from bench.control.registry.sqlite import SqliteRegistry, utc_now


def _state() -> AdapterTrainingState:
    return AdapterTrainingState(
        model_slots={"model": {"w": torch.ones(4, 4)}},
        optimizer_slots={"main": {"state": {}, "param_groups": [{"lr": 1e-3}]}},
        best_state={"weights": {"w": torch.ones(4, 4)}},
    )


def _service(tmp_path: Path, registry=None, events=None) -> CheckpointService:
    return CheckpointService(
        tmp_path / "run", registry=registry, event_writer=events, control_root=tmp_path
    )


def _save(service: CheckpointService, **overrides):
    kwargs = dict(
        run_id="run-1",
        kind=CheckpointKind.PERIODIC,
        cursor=TrainingCursor(global_update=5, batch_plan_position=5),
        adapter_state=_state(),
        rng=capture_rng(),
        identity={"model_id": "kalmannet_tsp", "implementation_id": "impl_v1"},
        structural_config_hash="sha256:cfg",
        dataset_fingerprint="sha256:data",
        batch_plan=BatchPlan(dataset_length=6, batch_size=2, seed=1),
    )
    kwargs.update(overrides)
    return service.save(**kwargs)


def _registry(tmp_path: Path) -> tuple[SqliteRegistry, str]:
    registry = SqliteRegistry(tmp_path / "registry.sqlite3")
    experiment = ExperimentRecord(experiment_id=str(uuid.uuid4()), name="t", created_at=utc_now())
    registry.upsert_experiment(experiment)
    registry.create_run(
        RunRecord(
            run_id="run-1",
            experiment_id=experiment.experiment_id,
            state=RunState.RUNNING,
            state_version=0,
            created_at=utc_now(),
            updated_at=utc_now(),
            model_id="kalmannet_tsp",
            run_dir=str(tmp_path / "run"),
        )
    )
    return registry, "run-1"


# -- happy path --------------------------------------------------------------


def test_published_checkpoint_validates_and_catalogs(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    service = _service(tmp_path, registry=registry)
    saved = _save(service)

    report = validate_package(saved.directory)
    assert report.valid, report.errors
    assert report.payload_sha256 == saved.payload_sha256

    manifest = json.loads((saved.directory / MANIFEST_FILENAME).read_text())
    # No training-path provenance was supplied, so this is a v1 package and
    # keeps exactly its v1 meaning. The version is derived from the proof
    # available, never imposed (continuation gate B0).
    assert manifest["schema_version"] == 1
    assert manifest.get("training_path_id") is None
    assert manifest["kind"] == "periodic"
    assert manifest["resume_boundary"] == "optimizer_update"
    assert manifest["component_inventory"]["optimizer_slots"] == ["main"]

    row = registry.get_checkpoint(saved.checkpoint_id)
    assert row["validation_status"] == "VALID"
    assert row["payload_sha256"] == saved.payload_sha256
    assert registry.get_run("run-1").latest_checkpoint_id == saved.checkpoint_id


def test_supplying_a_training_path_publishes_a_v2_package(tmp_path: Path) -> None:
    """v2 exists precisely when the package can prove which loop produced it."""
    registry, _ = _registry(tmp_path)
    service = _service(tmp_path, registry=registry)
    saved = _save(
        service,
        training_path_id="control_resumable_v1",
        training_path_contract_version=1,
    )
    manifest = json.loads((saved.directory / MANIFEST_FILENAME).read_text())
    assert manifest["schema_version"] == CHECKPOINT_SCHEMA_VERSION == 2
    assert manifest["training_path_id"] == "control_resumable_v1"

    row = registry.get_checkpoint(saved.checkpoint_id)
    assert row["checkpoint_schema_version"] == 2
    assert row["training_path_id"] == "control_resumable_v1"
    assert validate_package(saved.directory).valid


def test_v1_package_is_valid_but_not_write_control_eligible(tmp_path: Path) -> None:
    """The distinction the whole gate rests on: valid != launch-eligible."""
    from bench.control.checkpoints import evaluate_resume_eligibility

    saved = _save(_service(tmp_path), kind=CheckpointKind.INTERRUPT)
    report = validate_package(saved.directory)
    assert report.valid, "a v1 package must remain a valid artifact"

    eligibility = evaluate_resume_eligibility(manifest=report.manifest)
    assert not eligibility.eligible
    assert "CHECKPOINT_TRAINING_PATH_UNPROVEN" in eligibility.reason_codes
    assert "CHECKPOINT_SCHEMA_NOT_WRITE_CONTROL_CERTIFIED" in eligibility.reason_codes


def test_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    saved = _save(_service(tmp_path))
    path = saved.directory / MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = 99
    path.write_text(json.dumps(manifest))
    report = validate_package(saved.directory)
    assert not report.valid
    assert any("not readable" in e for e in report.errors)


def test_no_temp_files_survive_a_successful_publish(tmp_path: Path) -> None:
    saved = _save(_service(tmp_path))
    assert not list(saved.directory.glob(f"*{TEMP_SUFFIX}"))
    assert (saved.directory / PAYLOAD_FILENAME).exists()


def test_checkpoints_are_immutable(tmp_path: Path) -> None:
    service = _service(tmp_path)
    saved = _save(service)
    with pytest.raises(CheckpointError, match="immutable"):
        _save(service, checkpoint_id=saved.checkpoint_id)


def test_best_checkpoint_updates_the_run_pointer(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    service = _service(tmp_path, registry=registry)
    saved = _save(service, kind=CheckpointKind.BEST)
    assert registry.get_run("run-1").best_checkpoint_id == saved.checkpoint_id


# -- corruption --------------------------------------------------------------


def test_payload_corruption_is_detected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    saved = _save(service)

    payload = saved.directory / PAYLOAD_FILENAME
    data = bytearray(payload.read_bytes())
    data[len(data) // 2] ^= 0xFF
    payload.write_bytes(bytes(data))

    report = validate_package(saved.directory)
    assert not report.valid
    assert any("digest mismatch" in e for e in report.errors)
    with pytest.raises(CheckpointValidationError):
        service.load(saved.checkpoint_id)


def test_truncated_payload_is_detected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    saved = _save(service)
    payload = saved.directory / PAYLOAD_FILENAME
    payload.write_bytes(payload.read_bytes()[: -64])

    report = validate_package(saved.directory)
    assert not report.valid
    with pytest.raises(CheckpointValidationError):
        service.load(saved.checkpoint_id)


def test_missing_payload_is_detected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    saved = _save(service)
    (saved.directory / PAYLOAD_FILENAME).unlink()
    report = validate_package(saved.directory)
    assert not report.valid
    assert any("missing payload" in e for e in report.errors)


def test_missing_manifest_is_not_a_checkpoint(tmp_path: Path) -> None:
    service = _service(tmp_path)
    saved = _save(service)
    (saved.directory / MANIFEST_FILENAME).unlink()
    assert not validate_package(saved.directory).valid
    assert service.list_on_disk() == []


def test_future_schema_version_is_refused_not_migrated(tmp_path: Path) -> None:
    service = _service(tmp_path)
    saved = _save(service)
    manifest_path = saved.directory / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = CHECKPOINT_SCHEMA_VERSION + 1
    manifest_path.write_text(json.dumps(manifest))

    report = validate_package(saved.directory)
    assert not report.valid
    assert any("schema_version" in e for e in report.errors)


# -- fault injection ---------------------------------------------------------


@pytest.mark.parametrize(
    "fault_point",
    ["after_payload_write", "before_payload_rename", "before_manifest_write"],
)
def test_crash_before_publication_leaves_no_checkpoint(tmp_path: Path, fault_point: str) -> None:
    """A crash before both renames must leave nothing the catalog can see."""
    registry, _ = _registry(tmp_path)
    service = _service(tmp_path, registry=registry)

    class _Crash(RuntimeError):
        pass

    def hook(name: str, _path: Path) -> None:
        if name == fault_point:
            raise _Crash(name)

    with pytest.raises(_Crash):
        _save(service, fault_hook=hook)

    assert service.list_on_disk() == []
    assert registry.list_checkpoints("run-1") == []
    assert registry.get_run("run-1").latest_checkpoint_id is None


def test_crash_after_manifest_before_registry_is_adopted_by_reconciler(tmp_path: Path) -> None:
    """Files complete, row missing: the reconciler adopts after verifying bytes."""
    registry, _ = _registry(tmp_path)
    service = _service(tmp_path, registry=registry)

    class _Crash(RuntimeError):
        pass

    def hook(name: str, _path: Path) -> None:
        if name == "after_manifest_rename":
            raise _Crash(name)

    with pytest.raises(_Crash):
        _save(service, fault_hook=hook)

    # The package is on disk and complete, but uncatalogued.
    on_disk = service.list_on_disk()
    assert len(on_disk) == 1
    assert registry.list_checkpoints("run-1") == []

    report = reconcile_run_checkpoints(
        run_id="run-1", run_dir=tmp_path / "run", registry=registry
    )
    assert report.catalogued == on_disk
    row = registry.get_checkpoint(on_disk[0])
    assert row["validation_status"] == "VALID"


def test_row_without_package_is_marked_invalid(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    service = _service(tmp_path, registry=registry)
    saved = _save(service)

    import shutil

    shutil.rmtree(saved.directory)

    report = reconcile_run_checkpoints(
        run_id="run-1", run_dir=tmp_path / "run", registry=registry
    )
    assert saved.checkpoint_id in report.invalidated
    assert registry.get_checkpoint(saved.checkpoint_id)["validation_status"] == "INVALID"


def test_corrupt_catalogued_package_is_marked_invalid(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    service = _service(tmp_path, registry=registry)
    saved = _save(service)

    payload = saved.directory / PAYLOAD_FILENAME
    payload.write_bytes(payload.read_bytes()[:-32])

    report = reconcile_run_checkpoints(
        run_id="run-1", run_dir=tmp_path / "run", registry=registry
    )
    assert saved.checkpoint_id in report.invalidated
    assert registry.get_checkpoint(saved.checkpoint_id)["validation_status"] == "INVALID"


def test_leftover_temp_file_is_reported_not_catalogued(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    service = _service(tmp_path, registry=registry)
    saved = _save(service)
    (saved.directory / (PAYLOAD_FILENAME + TEMP_SUFFIX)).write_bytes(b"partial")

    report = reconcile_run_checkpoints(
        run_id="run-1", run_dir=tmp_path / "run", registry=registry
    )
    assert report.temp_leftovers
    assert saved.checkpoint_id not in report.quarantined
    # Reported, and deliberately still on disk: it may belong to a live writer.
    assert (saved.directory / (PAYLOAD_FILENAME + TEMP_SUFFIX)).exists()


# -- compatibility -----------------------------------------------------------


def test_incompatible_dataset_is_refused(tmp_path: Path) -> None:
    service = _service(tmp_path)
    saved = _save(service)
    report = check_manifest_compatibility(
        saved.manifest,
        expected={"model_id": "kalmannet_tsp", "dataset_fingerprint": "sha256:other"},
    )
    assert not report.compatible
    assert any("dataset_fingerprint" in m for m in report.mismatches)


def test_incompatible_model_is_refused(tmp_path: Path) -> None:
    service = _service(tmp_path)
    saved = _save(service)
    with pytest.raises(Exception):
        service.load(saved.checkpoint_id, expected={"model_id": "split_knet"})


def test_payload_without_optimizer_slots_is_not_exact_resume(tmp_path: Path) -> None:
    service = _service(tmp_path)
    state = _state()
    state.optimizer_slots = {}
    saved = _save(service, adapter_state=state)
    report = validate_package(saved.directory)
    assert not report.valid
    assert any("optimizer" in e for e in report.errors)


def test_loading_outside_the_control_root_is_refused(tmp_path: Path) -> None:
    """Payloads are pickle; the path is part of the trust boundary."""
    service = _save(_service(tmp_path))
    stray = CheckpointService(tmp_path / "run", control_root=tmp_path / "elsewhere")
    with pytest.raises(CheckpointError, match="control root"):
        stray.load(service.checkpoint_id)


# -- batch plan --------------------------------------------------------------


def _take(iterator, count):
    return [next(iterator) for _ in range(count)]


def test_batch_plan_is_deterministic_and_seekable() -> None:
    plan = BatchPlan(dataset_length=10, batch_size=3, seed=7)
    assert plan.plan_id == BatchPlan(dataset_length=10, batch_size=3, seed=7).plan_id
    assert plan.plan_id != BatchPlan(dataset_length=10, batch_size=3, seed=8).plan_id

    sequential = [item[1].tolist() for item in _take(plan.iter_from(0), 12)]
    seeked = [item[1].tolist() for item in _take(plan.iter_from(5), 7)]
    assert seeked == sequential[5:12]

    positions = [item[0] for item in _take(plan.iter_from(5), 3)]
    assert positions == [5, 6, 7]

    _, _, direct = plan.batch_at(9)
    assert direct.tolist() == sequential[9]
