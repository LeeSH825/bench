"""POST /api/v1/runs/launch: gating, idempotency, conflicts, durability.

The launch route is the only surface that can create work from a browser, so
every test here is about *not* creating more work than asked: never a second
run for a retried request, and never a run at all for a request that was
refused.

Worker spawning is stubbed. What is exercised for real is the registry, the
run allocation, the provenance snapshots and the action lifecycle — the parts
that must survive a crash. Real process launch is covered by the E2E gate.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bench.control.api.write_mode import ENABLE_WRITES_ENV, REQUEST_HEADER
from bench.control.config.presets import PresetCatalog
from bench.control.launch_coordinator import (
    ACTION_LAUNCH_RUN,
    LaunchCoordinator,
)
from bench.control.process.manager import WorkerManager
from bench.control.registry.schema import RunState, WorkerRecord
from bench.control.registry.sqlite import SqliteRegistry, utc_now

LAUNCH_PATH = "/api/v1/runs/launch"
PRESET_FILE = "suite_train_smoke.yaml"
MODEL = "kalmannet_tsp"


class StubManager(WorkerManager):
    """Real allocation, stubbed spawn.

    ``prepare_run`` is inherited untouched: run directories, the resolved spec
    file and the registry row must be the genuine article. Only the subprocess
    is replaced, so a test can assert "exactly one worker was started" without
    training anything.
    """

    def __init__(self, *args, fail_launch: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.launched: list[str] = []
        self.prepared: list[str] = []
        self.fail_launch = fail_launch

    def prepare_run(self, spec, **kwargs):
        location = super().prepare_run(spec, **kwargs)
        self.prepared.append(spec.run_id.value)
        return location

    def launch(self, spec, **kwargs):
        run_id = spec.run_id.value
        if self.fail_launch:
            raise RuntimeError("simulated spawn failure")
        self.launched.append(run_id)
        for state in (RunState.VALIDATING, RunState.QUEUED, RunState.STARTING):
            self.registry.transition(run_id, to_state=state, actor="stub", reason="stub")
        worker_id = str(uuid.uuid4())
        self.registry.register_worker(WorkerRecord(
            worker_instance_id=worker_id, run_id=run_id, host="test", pid=-1,
            process_group_id=-1, process_start_time=0.0, worker_token="t",
            started_at=utc_now(), last_heartbeat_at=utc_now(), state="RUNNING"))

        class _Result:
            worker_instance_id = worker_id

        return _Result()


@pytest.fixture(scope="module")
def preset():
    catalog = PresetCatalog()
    entry = next(e for e in catalog.list() if e.relative_path.endswith(PRESET_FILE))
    return entry


def _client(tmp_path, monkeypatch, *, enabled: bool = True, fail_launch: bool = False):
    monkeypatch.setenv("BENCH_CONTROL_ROOT", str(tmp_path))
    if enabled:
        monkeypatch.setenv(ENABLE_WRITES_ENV, "1")
    else:
        monkeypatch.delenv(ENABLE_WRITES_ENV, raising=False)

    from bench.control.api import deps
    from bench.control.api.app import create_app

    deps.configure(str(tmp_path))
    registry = SqliteRegistry(Path(tmp_path) / "registry.sqlite3")
    manager = StubManager(registry, control_root_path=str(tmp_path),
                          fail_launch=fail_launch)
    app = create_app(str(tmp_path))
    app.dependency_overrides[deps.get_manager] = lambda: manager
    app.dependency_overrides[deps.get_registry] = lambda: registry
    return TestClient(app), registry, manager


def _headers(key: str) -> dict[str, str]:
    return {REQUEST_HEADER: "1", "Idempotency-Key": key}


def _body(preset, **overrides):
    payload = {"preset_id": preset.preset_id, "preset_digest": preset.content_digest,
               "model_id": MODEL, "init_id": "trained"}
    payload.update(overrides)
    return payload


def _counts(registry) -> tuple[int, int]:
    runs = registry.list_runs()
    actions = registry.connection.execute(
        "SELECT COUNT(*) FROM run_actions").fetchone()[0]
    return len(runs), int(actions)


# -- gating ------------------------------------------------------------------


def test_launch_route_absent_without_write_mode(tmp_path, monkeypatch, preset) -> None:
    client, registry, manager = _client(tmp_path, monkeypatch, enabled=False)
    spec = client.get("/openapi.json").json()
    assert LAUNCH_PATH not in spec["paths"]
    response = client.post(LAUNCH_PATH, json=_body(preset), headers=_headers("k"))
    # 405, not 404: the read-only build still has GET /runs/{run_id}, which the
    # path matches. Either way it is a routing refusal — no handler ran.
    assert response.status_code in (404, 405), response.text
    assert _counts(registry) == (0, 0)
    assert manager.prepared == []


def test_missing_control_header_is_refused(tmp_path, monkeypatch, preset) -> None:
    client, registry, manager = _client(tmp_path, monkeypatch)
    response = client.post(LAUNCH_PATH, json=_body(preset),
                           headers={"Idempotency-Key": "k"})
    assert response.status_code == 400
    assert _counts(registry) == (0, 0)
    assert manager.prepared == []


def test_missing_idempotency_key_is_refused(tmp_path, monkeypatch, preset) -> None:
    client, registry, manager = _client(tmp_path, monkeypatch)
    response = client.post(LAUNCH_PATH, json=_body(preset),
                           headers={REQUEST_HEADER: "1"})
    assert response.status_code == 400
    assert _counts(registry) == (0, 0)
    assert manager.prepared == []


# -- refusals allocate nothing ----------------------------------------------


def test_unknown_preset_is_404_with_no_side_effects(tmp_path, monkeypatch) -> None:
    client, registry, manager = _client(tmp_path, monkeypatch)
    response = client.post(LAUNCH_PATH, headers=_headers("k"), json={
        "preset_id": "no-such-preset.0000", "preset_digest": "x", "model_id": MODEL})
    assert response.status_code == 404
    assert response.json()["detail"]["reason_code"] == "UNKNOWN_PRESET"
    assert _counts(registry) == (0, 0)
    assert manager.prepared == []


def test_invalid_config_is_422_with_no_side_effects(tmp_path, monkeypatch, preset) -> None:
    client, registry, manager = _client(tmp_path, monkeypatch)
    response = client.post(LAUNCH_PATH, headers=_headers("k"),
                           json=_body(preset, model_id="not_a_model"))
    assert response.status_code == 422
    assert "UNKNOWN_MODEL" in response.json()["detail"]["reason_codes"]
    assert _counts(registry) == (0, 0)
    assert manager.prepared == [] and manager.launched == []


def test_stale_preset_digest_is_409_with_no_side_effects(tmp_path, monkeypatch, preset) -> None:
    """The preset file moved between preview and launch."""
    client, registry, manager = _client(tmp_path, monkeypatch)
    response = client.post(LAUNCH_PATH, headers=_headers("k"),
                           json=_body(preset, preset_digest="sha256:stale"))
    assert response.status_code == 409
    assert _counts(registry) == (0, 0)
    assert manager.prepared == []


def test_preview_hash_mismatch_is_409_with_no_side_effects(tmp_path, monkeypatch, preset) -> None:
    """The operator approved a preview whose resolved identity no longer holds."""
    client, registry, manager = _client(tmp_path, monkeypatch)
    response = client.post(LAUNCH_PATH, headers=_headers("k"), json=_body(
        preset, expected_structural_config_hash="sha256:not-what-resolved"))
    assert response.status_code == 409
    assert _counts(registry) == (0, 0)
    assert manager.prepared == []


def test_matching_preview_hash_is_accepted(tmp_path, monkeypatch, preset) -> None:
    client, registry, manager = _client(tmp_path, monkeypatch)
    preview = client.post("/api/v1/config/validate", json={
        "preset_id": preset.preset_id, "model_id": MODEL, "init_id": "trained"}).json()
    assert preview["valid"], preview["issues"]
    response = client.post(LAUNCH_PATH, headers=_headers("k"), json=_body(
        preset,
        expected_structural_config_hash=preview["structural_config_hash"],
        expected_operational_config_hash=preview["operational_config_hash"]))
    assert response.status_code == 202, response.text
    assert len(manager.launched) == 1


# -- idempotency -------------------------------------------------------------


def test_same_key_repeated_yields_one_run_and_one_worker(tmp_path, monkeypatch, preset) -> None:
    client, registry, manager = _client(tmp_path, monkeypatch)
    codes, run_ids, action_ids = [], set(), set()
    for _ in range(5):
        response = client.post(LAUNCH_PATH, json=_body(preset), headers=_headers("same"))
        codes.append(response.status_code)
        body = response.json()
        run_ids.add(body["run_id"])
        action_ids.add(body["action_id"])
    assert codes == [202, 200, 200, 200, 200], codes
    assert len(run_ids) == 1 and len(action_ids) == 1
    assert _counts(registry) == (1, 1)
    assert manager.prepared == manager.launched
    assert len(manager.launched) == 1


def test_same_key_with_a_different_draft_is_409(tmp_path, monkeypatch, preset) -> None:
    """Replaying a key against different config is a client bug, not a retry."""
    client, registry, manager = _client(tmp_path, monkeypatch)
    first = client.post(LAUNCH_PATH, json=_body(preset), headers=_headers("dup"))
    assert first.status_code == 202
    second = client.post(LAUNCH_PATH, headers=_headers("dup"),
                         json=_body(preset, overrides={"training.max_updates": 42}))
    assert second.status_code == 409
    assert _counts(registry) == (1, 1)
    assert len(manager.launched) == 1


def test_same_config_under_different_keys_yields_distinct_runs(tmp_path, monkeypatch, preset) -> None:
    """Identical config is not a duplicate request — the key decides."""
    client, registry, manager = _client(tmp_path, monkeypatch)
    run_ids = set()
    for key in ("a", "b", "c"):
        response = client.post(LAUNCH_PATH, json=_body(preset), headers=_headers(key))
        assert response.status_code == 202, response.text
        run_ids.add(response.json()["run_id"])
    assert len(run_ids) == 3
    assert _counts(registry) == (3, 3)
    assert len(manager.launched) == 3


# -- restart boundaries ------------------------------------------------------


def test_restart_between_action_and_allocation_launches_once(tmp_path, monkeypatch, preset) -> None:
    """Crash after the action row exists: recovery must adopt, not duplicate."""
    _client_unused, registry, manager = _client(tmp_path, monkeypatch)
    coordinator = LaunchCoordinator(registry=registry, manager=manager,
                                    control_root=tmp_path)
    outcome = coordinator.request_launch(
        preset_id=preset.preset_id, preset_digest=preset.content_digest,
        idempotency_key="crash-1", model_id=MODEL, init_id="trained", launch=False)
    assert outcome.run_id is None
    assert _counts(registry) == (0, 1)

    # "Restart": a brand new coordinator sees only durable state.
    recovered = LaunchCoordinator(registry=registry, manager=manager,
                                  control_root=tmp_path)
    outcomes = recovered.reconcile_open_actions()
    assert len(outcomes) == 1 and outcomes[0].run_id
    assert _counts(registry) == (1, 1)
    assert len(manager.launched) == 1

    # Reconciling again is a no-op: the action is terminal.
    assert recovered.reconcile_open_actions() == []
    assert len(manager.launched) == 1


def test_restart_between_allocation_and_launch_adopts_the_run(tmp_path, monkeypatch, preset) -> None:
    """Crash after the run exists: settle again must reuse the same run."""
    _client_unused, registry, manager = _client(tmp_path, monkeypatch)
    coordinator = LaunchCoordinator(registry=registry, manager=manager,
                                    control_root=tmp_path)
    action = coordinator.request_launch(
        preset_id=preset.preset_id, preset_digest=preset.content_digest,
        idempotency_key="crash-2", model_id=MODEL, init_id="trained", launch=False)
    params = json.loads(registry.get_action(action.action_id)["parameters_json"])
    run_id = coordinator._allocate(action.action_id, params, None, None)
    assert registry.get_action(action.action_id)["result_child_run_id"] == run_id
    assert manager.launched == []

    settled = LaunchCoordinator(registry=registry, manager=manager,
                                control_root=tmp_path).settle(action.action_id)
    assert settled.run_id == run_id
    assert _counts(registry) == (1, 1)
    assert manager.launched == [run_id]


def test_spawn_failure_fails_the_action_and_does_not_leave_a_live_run(
        tmp_path, monkeypatch, preset) -> None:
    client, registry, manager = _client(tmp_path, monkeypatch, fail_launch=True)
    response = client.post(LAUNCH_PATH, json=_body(preset), headers=_headers("boom"))
    assert response.status_code == 202
    body = response.json()
    assert body["state"] == "FAILED"
    action = registry.get_action(body["action_id"])
    assert action["status"] == "FAILED" and action["failure_reason"]
    record = registry.get_run(body["run_id"])
    assert record.state == RunState.CANCELLED
    assert record.exit_code == 52


def test_coordinator_failure_maps_to_503(tmp_path, monkeypatch, preset) -> None:
    client, registry, _manager = _client(tmp_path, monkeypatch)
    import bench.control.launch_coordinator as module

    def _explode(*_args, **_kwargs):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(module.LaunchCoordinator, "request_launch", _explode)
    response = client.post(LAUNCH_PATH, json=_body(preset), headers=_headers("k"))
    assert response.status_code == 503
    assert response.json()["detail"]["reason_code"] == "MANAGER_UNAVAILABLE"
    assert _counts(registry) == (0, 0)


# -- provenance --------------------------------------------------------------


def test_launch_writes_provenance_without_the_idempotency_key(tmp_path, monkeypatch, preset) -> None:
    client, registry, _manager = _client(tmp_path, monkeypatch)
    secret_key = "operator-secret-key-1234"
    response = client.post(LAUNCH_PATH, json=_body(preset), headers=_headers(secret_key))
    assert response.status_code == 202, response.text
    run_dir = Path(registry.get_run(response.json()["run_id"]).run_dir)

    expected = ("original_preset.yaml", "submitted_draft.yaml",
                "config_validation.json", "launch_request.json",
                "resolved_run_spec.json")
    for name in expected:
        assert (run_dir / name).is_file(), f"missing provenance file {name}"

    request = json.loads((run_dir / "launch_request.json").read_text())
    assert request["launch_source"] == "gui"
    assert request["preset_digest"] == preset.content_digest
    assert request["structural_config_hash"] and request["variant_id"]

    # An idempotency key is a client credential, not run provenance.
    for path in run_dir.rglob("*"):
        if path.is_file():
            assert secret_key not in path.read_text(encoding="utf-8", errors="ignore"), path


def test_launch_action_is_pollable_and_typed(tmp_path, monkeypatch, preset) -> None:
    client, registry, _manager = _client(tmp_path, monkeypatch)
    body = client.post(LAUNCH_PATH, json=_body(preset), headers=_headers("k")).json()
    assert body["action_type"] == ACTION_LAUNCH_RUN
    assert body["run_url"] == f"/api/v1/runs/{body['run_id']}"
    polled = client.get(f"/api/v1/actions/{body['action_id']}")
    assert polled.status_code == 200
    assert polled.json()["state"] == "COMPLETED"
    assert client.get(body["run_url"]).status_code == 200


def test_polling_a_launch_action_still_names_its_run(tmp_path, monkeypatch, preset) -> None:
    """Regression: the launch action's run lives in result_child_run_id.

    Returning the raw (NULL) ``run_id`` made the UI lose the link to the run it
    had just started as soon as the POST response was replaced by a poll.
    """
    client, _registry, _manager = _client(tmp_path, monkeypatch)
    body = client.post(LAUNCH_PATH, json=_body(preset), headers=_headers("k")).json()
    polled = client.get(f"/api/v1/actions/{body['action_id']}").json()
    assert polled["run_id"] == body["run_id"] is not None
    assert polled["child_run_id"] == body["run_id"]
