"""Write-control API: gating, security, idempotency, and error mapping.

Write routes are *registered* only in write mode, so in the default build a
POST to a write path is a routing miss rather than a permission decision.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from bench.control.api.write_mode import (
    ENABLE_WRITES_ENV,
    REQUEST_HEADER,
    WriteModeError,
    assert_write_bind_allowed,
    is_loopback,
    parse_bool,
    writes_enabled,
)
from bench.control.registry.schema import ExperimentRecord, RunRecord, RunState
from bench.control.registry.sqlite import SqliteRegistry, utc_now

HDRS = {REQUEST_HEADER: "1", "Idempotency-Key": "k-1"}


def _seed(root):
    registry = SqliteRegistry(root / "registry.sqlite3")
    exp = ExperimentRecord(experiment_id=str(uuid.uuid4()), name="t", created_at=utc_now())
    registry.upsert_experiment(exp)
    run_dir = root / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    registry.create_run(RunRecord(
        run_id="r1", experiment_id=exp.experiment_id, state=RunState.CREATED,
        state_version=0, created_at=utc_now(), updated_at=utc_now(),
        model_id="kalmannet_tsp", implementation_id="bench_kalmannet_tsp_adapter_v1",
        run_dir=str(run_dir), training_path_id="control_resumable_v1"))
    for st in (RunState.VALIDATING, RunState.QUEUED, RunState.STARTING, RunState.RUNNING):
        registry.transition("r1", to_state=st, actor="test", reason="seed")
    registry.create_run(RunRecord(
        run_id="legacy", experiment_id=exp.experiment_id, state=RunState.CREATED,
        state_version=0, created_at=utc_now(), updated_at=utc_now(),
        model_id="kalmannet_tsp", run_dir=str(root / "runs" / "legacy"),
        training_path_id="legacy_train_v1"))
    for st in (RunState.VALIDATING, RunState.QUEUED, RunState.STARTING, RunState.RUNNING):
        registry.transition("legacy", to_state=st, actor="test", reason="seed")
    registry.close()
    return registry


def _client(tmp_path, monkeypatch, enabled: bool, live_worker: bool = False):
    monkeypatch.setenv("BENCH_CONTROL_ROOT", str(tmp_path))
    if enabled:
        monkeypatch.setenv(ENABLE_WRITES_ENV, "1")
    else:
        monkeypatch.delenv(ENABLE_WRITES_ENV, raising=False)
    _seed(tmp_path)
    if live_worker:
        # Register this test process as the worker so liveness checks pass;
        # the PID is real and owned by the test.
        import os
        from bench.control.registry.schema import WorkerRecord
        reg = SqliteRegistry(tmp_path / "registry.sqlite3")
        reg.register_worker(WorkerRecord(
            worker_instance_id=str(uuid.uuid4()), run_id="r1", host="test",
            pid=os.getpid(), process_group_id=os.getpgrp(),
            process_start_time=0.0, worker_token="t", started_at=utc_now(),
            last_heartbeat_at=utc_now(), state="RUNNING"))
        reg.close()
    from bench.control.api.app import create_app
    return TestClient(create_app(str(tmp_path)))


# -- boolean and loopback parsing -------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("TRUE", True),
    ("0", False), ("false", False), ("", False), (None, False),
    ("yes", False), ("on", False), ("2", False), ("maybe", False),
])
def test_write_flag_parsing_fails_closed(raw, expected) -> None:
    """An ambiguous value must never enable a write surface."""
    assert parse_bool(raw) is expected


@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", True), ("localhost", True), ("::1", True), ("[::1]", True),
    ("0.0.0.0", False), ("::", False), ("192.168.1.10", False), ("example.com", False),
])
def test_loopback_detection(host, expected) -> None:
    assert is_loopback(host) is expected


def test_public_bind_with_writes_is_refused(monkeypatch) -> None:
    monkeypatch.setenv(ENABLE_WRITES_ENV, "1")
    with pytest.raises(WriteModeError, match="loopback"):
        assert_write_bind_allowed("0.0.0.0")
    # ...and the read-only public-bind escape hatch must not unlock writes.
    monkeypatch.setenv("BENCH_CONTROL_ALLOW_PUBLIC_BIND", "1")
    with pytest.raises(WriteModeError):
        assert_write_bind_allowed("0.0.0.0")


def test_public_bind_without_writes_is_allowed(monkeypatch) -> None:
    monkeypatch.delenv(ENABLE_WRITES_ENV, raising=False)
    assert_write_bind_allowed("0.0.0.0")  # read-only exposure is a separate decision


# -- default read-only mode --------------------------------------------------


#: Routes that change durable state. None of these may exist without write mode.
MUTATING_PATH_SUFFIXES = ("/actions/stop", "/actions/resume", "/runs/launch")


def test_default_mode_registers_no_mutating_routes(tmp_path, monkeypatch) -> None:
    """The invariant is "no state-changing route", not "no POST verb".

    POST /api/v1/config/validate exists in read-only mode by design: it parses
    and resolves a config and returns hashes, allocating nothing and writing
    nothing. Refusing it would only push operators toward launching without a
    preview. Every route that *does* change state stays absent.
    """
    client = _client(tmp_path, monkeypatch, enabled=False)
    spec = client.app.openapi()

    mutating = [p for p in spec["paths"]
                if any(p.endswith(sfx) for sfx in MUTATING_PATH_SUFFIXES)]
    assert mutating == [], f"read-only build exposes mutating routes: {mutating}"

    posts = sorted(p for p, ops in spec["paths"].items() if "post" in ops)
    assert posts == ["/api/v1/config/validate"], (
        f"unexpected POST routes in read-only mode: {posts}")

    assert client.post("/api/v1/runs/r1/actions/stop", headers=HDRS,
                       json={}).status_code in (404, 405)
    assert client.post("/api/v1/runs/launch", headers=HDRS,
                       json={}).status_code in (404, 405)


def test_validate_in_read_only_mode_has_no_side_effects(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, enabled=False)
    registry = SqliteRegistry(tmp_path / "registry.sqlite3")
    before = len(registry.list_runs(limit=100))
    r = client.post("/api/v1/config/validate", json={"yaml_text": "not: a suite"})
    assert r.status_code in (200, 400, 422)
    assert len(registry.list_runs(limit=100)) == before


def test_default_mode_reports_write_capabilities_false(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, enabled=False)
    cp = client.get("/api/v1/capabilities").json()["control_plane"]
    assert cp["write_control_enabled"] is False
    assert cp["graceful_stop_api"] is False and cp["resume_api"] is False
    assert cp["dash_stop_control"] is False and cp["dash_resume_control"] is False
    assert cp["write_mode_loopback_only"] is True
    assert cp["authentication"] is False


# -- write mode --------------------------------------------------------------


def test_write_mode_registers_the_routes(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, enabled=True)
    spec = client.app.openapi()
    assert "/api/v1/runs/launch" in spec["paths"]
    for path in ("/api/v1/runs/{run_id}/actions/stop",
                 "/api/v1/checkpoints/{checkpoint_id}/actions/resume",
                 "/api/v1/actions/{action_id}"):
        assert path in spec["paths"]
    cp = client.get("/api/v1/capabilities").json()["control_plane"]
    assert cp["write_control_enabled"] is True and cp["graceful_stop_api"] is True


def test_missing_control_header_is_rejected(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, enabled=True)
    r = client.post("/api/v1/runs/r1/actions/stop",
                    headers={"Idempotency-Key": "k"}, json={})
    assert r.status_code == 400
    assert r.json()["detail"]["reason_code"] == "MISSING_CONTROL_HEADER"


def test_missing_idempotency_key_is_rejected(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, enabled=True)
    r = client.post("/api/v1/runs/r1/actions/stop", headers={REQUEST_HEADER: "1"}, json={})
    assert r.status_code == 400
    assert r.json()["detail"]["reason_code"] == "MISSING_IDEMPOTENCY_KEY"


def test_unknown_run_is_404(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, enabled=True)
    r = client.post("/api/v1/runs/nope/actions/stop", headers=HDRS, json={})
    assert r.status_code == 404


def test_legacy_training_path_is_422(tmp_path, monkeypatch) -> None:
    """An uncertified envelope is unprocessable, not merely conflicting."""
    client = _client(tmp_path, monkeypatch, enabled=True)
    r = client.post("/api/v1/runs/legacy/actions/stop", headers=HDRS, json={})
    assert r.status_code == 422
    assert r.json()["detail"]["reason_code"] == "TRAINING_PATH_NOT_RESUMABLE"


def test_stale_state_version_is_409_with_no_side_effect(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, enabled=True, live_worker=True)
    r = client.post("/api/v1/runs/r1/actions/stop", headers=HDRS,
                    json={"expected_state_version": 999})
    assert r.status_code == 409
    assert r.json()["detail"]["reason_code"] == "STALE_STATE_VERSION"
    registry = SqliteRegistry(tmp_path / "registry.sqlite3")
    assert registry.open_action("r1", action="stop") is None


def test_stop_without_a_live_worker_is_refused(tmp_path, monkeypatch) -> None:
    """UI gating and enforcement must agree (ADR-WC-020).

    The seeded run is RUNNING with no worker row, which is exactly the state
    where a stop could be recorded and never acted on.
    """
    client = _client(tmp_path, monkeypatch, enabled=True)
    r = client.post("/api/v1/runs/r1/actions/stop", headers=HDRS, json={})
    assert r.status_code == 409
    assert r.json()["detail"]["reason_code"] == "NO_LIVE_WORKER"
    registry = SqliteRegistry(tmp_path / "registry.sqlite3")
    assert registry.open_action("r1", action="stop") is None


def _accept_stop(client):
    """Bypass the liveness gate for idempotency tests by seeding a live worker."""
    return client.post("/api/v1/runs/r1/actions/stop", headers=HDRS, json={})


def test_stop_is_accepted_and_idempotent(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, enabled=True, live_worker=True)
    first = client.post("/api/v1/runs/r1/actions/stop", headers=HDRS, json={})
    assert first.status_code == 202
    body = first.json()
    assert body["action_type"] == "STOP_GRACEFUL"
    assert body["state"] == "REQUESTED"
    assert body["idempotency_reused"] is False
    assert body["status_url"].endswith(body["action_id"])

    for _ in range(4):
        again = client.post("/api/v1/runs/r1/actions/stop", headers=HDRS, json={})
        assert again.status_code in (200, 202)
        assert again.json()["action_id"] == body["action_id"]
        assert again.json()["idempotency_reused"] is True

    registry = SqliteRegistry(tmp_path / "registry.sqlite3")
    assert len([a for a in registry.list_actions("r1") if a["action"] == "stop"]) == 1


def test_same_key_different_target_is_409(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, enabled=True, live_worker=True)
    assert client.post("/api/v1/runs/r1/actions/stop", headers=HDRS, json={}).status_code == 202
    r = client.post("/api/v1/runs/legacy/actions/stop", headers=HDRS, json={})
    assert r.status_code == 409
    assert r.json()["detail"]["reason_code"] == "IDEMPOTENCY_KEY_REUSED"


def test_action_resource_round_trip_hides_the_key(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, enabled=True, live_worker=True)
    action_id = client.post("/api/v1/runs/r1/actions/stop", headers=HDRS,
                            json={}).json()["action_id"]
    r = client.get(f"/api/v1/actions/{action_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["action_id"] == action_id
    assert body["terminal"] is False
    assert "idempotency_key" not in body
    assert "k-1" not in r.text
    assert client.get("/api/v1/actions/nope").status_code == 404


def test_unknown_checkpoint_resume_is_404(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, enabled=True)
    r = client.post("/api/v1/checkpoints/nope/actions/resume", headers=HDRS, json={})
    assert r.status_code == 404


def test_eligibility_read_model_is_served(tmp_path, monkeypatch) -> None:
    """The UI renders this; it must not recompute the conditions."""
    client = _client(tmp_path, monkeypatch, enabled=True)
    block = client.get("/api/v1/runs/r1").json()["action_eligibility"]
    assert block["write_control_enabled"] is True
    # No live worker for this seeded run, so a stop is correctly refused and
    # the reason says so. The live-worker case is covered by the real E2E.
    assert block["stop_action"]["eligible"] is False
    assert block["stop_action"]["reason_code"] == "NO_LIVE_WORKER"
    assert block["resume_action"]["eligible"] is False
    assert block["resume_action"]["reason_code"]
    assert block["resume_action"]["reason"]

    legacy = client.get("/api/v1/runs/legacy").json()["action_eligibility"]
    assert legacy["stop_action"]["eligible"] is False
    assert legacy["stop_action"]["reason_code"] == "TRAINING_PATH_NOT_RESUMABLE"
    assert "legacy_train_v1" in legacy["stop_action"]["reason"]
