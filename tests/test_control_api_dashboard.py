"""API and Dash dashboard acceptance tests (design doc 06, A-01 … A-06, U-01 … U-10).

Browser scope, stated honestly: this host has no Chrome/Chromium/Firefox and no
webdriver, so `dash[testing]`/Selenium/Playwright cannot run here. The dashboard
is therefore exercised **server-side** — page routes are fetched over HTTP and
callbacks are dispatched through Dash's own `_dash-update-component` endpoint,
which is the same code path a browser triggers. This verifies routing, layout
construction, data binding, and callback output; it does **not** verify
client-side JavaScript or visual rendering. See known_limitations.md.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from typing import Any

from bench.control.allocation import allocate_run_directory, write_run_spec
from bench.control.api import deps
from bench.control.api.app import create_app, resolve_bind_host
from bench.control.config.resolver import resolve_run_spec
from bench.control.config.schema import (
    DatasetSection,
    ExperimentSection,
    RunSpecDraft,
    RuntimeSection,
    SystemSection,
    TrainingSection,
)
from bench.control.events.writer import EventWriter
from bench.control.identity import ExperimentId, ImplementationId, ModelId
from bench.control.registry.schema import ExperimentRecord, RunRecord, RunState
from bench.control.registry.sqlite import SqliteRegistry

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None  # type: ignore


def make_spec(model_id: str = "split_knet", seed: int = 0):
    from bench.control.capabilities import implementation_id_for

    draft = RunSpecDraft(
        experiment=ExperimentSection(experiment_id=ExperimentId.new().value, name="api-test"),
        model_id=ModelId(model_id),
        implementation_id=ImplementationId(implementation_id_for(model_id)),
        system=SystemSection(task_id="task_a", scenario_id="scen_a", state_dim=2, observation_dim=2),
        dataset=DatasetSection(dataset_id="ds"),
        training=TrainingSection(enabled=True, max_updates=10, batch_size=2, validation_interval_updates=2),
        runtime=RuntimeSection(device="cpu", seed=seed),
    )
    return resolve_run_spec(draft)


class ControlFixture(unittest.TestCase):
    """Builds a small control root with a completed run and a failed run."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry = SqliteRegistry(self.root / "registry.sqlite3")
        self.addCleanup(self._cleanup)

        self.completed_run_id = self._make_run(RunState.COMPLETED, model_id="split_knet", with_events=True)
        self.failed_run_id = self._make_run(RunState.FAILED, model_id="kalmannet_tsp", with_events=False)

        deps.configure(self.root)
        self.addCleanup(lambda: deps.configure(None))

    def _cleanup(self) -> None:
        try:
            self.registry.close()
        except Exception:
            pass
        self._tmp.cleanup()

    def _make_run(self, final_state: RunState, *, model_id: str, with_events: bool) -> str:
        spec = make_spec(model_id=model_id)
        experiment_id = ExperimentId(spec.draft.experiment.experiment_id)
        self.registry.upsert_experiment(ExperimentRecord(experiment_id=experiment_id.value, name="api-test"))
        location = allocate_run_directory(
            run_id=spec.run_id, experiment_id=experiment_id, control_root=self.root
        )
        write_run_spec(location, spec)
        record = self.registry.create_run(
            RunRecord(
                run_id=spec.run_id.value,
                experiment_id=experiment_id.value,
                state=RunState.CREATED,
                state_version=0,
                created_at="",
                updated_at="",
                model_id=spec.model_id.value,
                implementation_id=spec.implementation_id.value,
                init_id=spec.init_id.mode,
                variant_id=spec.variant_id.value,
                task_id=spec.draft.system.task_id,
                scenario_id=spec.draft.system.scenario_id,
                seed=spec.draft.runtime.seed,
                device="cpu",
                run_dir=str(location.root),
                structural_config_hash=spec.structural_config_hash,
                operational_config_hash=spec.operational_config_hash,
                resolved_spec_hash=spec.resolved_spec_hash,
            )
        )
        if with_events:
            with EventWriter(location.events_path, spec.run_id.value) as writer:
                writer.status("RUNNING", phase="train")
                for step in range(1, 11):
                    writer.metric("loss/train_total", 1.0 / step, step=step, phase="train")
                    if step % 5 == 0:
                        writer.metric("loss/validation_total", 1.2 / step, step=step, phase="validation")
                writer.metric("metric/test_mse", 0.05, step=10, phase="test")
                writer.resource({"process_tree_cpu_percent": 42.0, "process_tree_rss_bytes": 1024**3, "gpu": None})
                writer.status("COMPLETED", phase="report")
            location.stdout_path.write_text("hello from the worker\n", encoding="utf-8")
            location.stderr_path.write_text("", encoding="utf-8")
        else:
            location.stdout_path.write_text("starting\n", encoding="utf-8")
            location.stderr_path.write_text("Traceback (most recent call last):\nBoom\n", encoding="utf-8")
            (location.artifacts_dir / "traceback.txt").write_text("Traceback\nBoom\n", encoding="utf-8")
            location.failure_path.write_text(json.dumps({"message": "Boom"}), encoding="utf-8")

        for state in (RunState.VALIDATING, RunState.QUEUED, RunState.STARTING, RunState.RUNNING):
            record = self.registry.transition(record.run_id, to_state=state)
        self.registry.transition(
            record.run_id,
            to_state=final_state,
            fields={
                "exit_code": 0 if final_state is RunState.COMPLETED else 40,
                "terminal_reason": "completed" if final_state is RunState.COMPLETED else "execution_failure",
                "error_summary": None if final_state is RunState.COMPLETED else "ExecutionError: Boom",
                "global_step": 10,
            },
        )
        return spec.run_id.value


@unittest.skipIf(TestClient is None, "fastapi TestClient unavailable")
class ApiTests(ControlFixture):
    def setUp(self) -> None:
        super().setUp()
        self.client = TestClient(create_app())

    # -- system -------------------------------------------------------------

    def test_health_reports_each_component(self) -> None:
        """A-06: degraded subsystems must be distinguishable."""
        payload = self.client.get("/api/v1/system/health").json()
        self.assertIn(payload["status"], ("ok", "degraded", "error"))
        for component in ("registry", "worker_manager", "telemetry"):
            self.assertIn(component, payload["components"])
            self.assertIn("status", payload["components"][component])
        self.assertTrue(payload["read_only"])

    def test_gpus_endpoint_is_null_safe(self) -> None:
        payload = self.client.get("/api/v1/system/gpus").json()
        self.assertIn("available", payload)
        self.assertIsInstance(payload["devices"], list)
        if not payload["available"]:
            self.assertTrue(payload["note"], "absence must be explained, not implied by an empty list")

    def test_capabilities_declares_absent_features_as_false(self) -> None:
        payload = self.client.get("/api/v1/capabilities").json()
        control = payload["control_plane"]
        for absent in ("graceful_stop", "force_terminate", "exact_resume", "launch_from_ui", "authentication"):
            self.assertFalse(control[absent], f"{absent} must be declared absent in this build")
        self.assertTrue(control["read_only_dashboard"])
        self.assertTrue(payload["models"])
        self.assertFalse(any(model["supports_exact_resume"] for model in payload["models"]))

    def test_schema_versions_are_exposed(self) -> None:
        versions = self.client.get("/api/v1/capabilities").json()["schema_versions"]
        for key in ("control_plane", "config", "registry", "event", "api"):
            self.assertIn(key, versions)

    def test_state_machine_marks_schema_only_states(self) -> None:
        payload = self.client.get("/api/v1/system/state-machine").json()
        self.assertEqual(
            set(payload["schema_only_states"]),
            {"STOP_REQUESTED", "CHECKPOINTING", "INTERRUPTED", "RESUMING"},
        )
        self.assertIn("RUNNING", payload["active_states_this_build"])

    # -- runs ---------------------------------------------------------------

    def test_list_runs(self) -> None:
        payload = self.client.get("/api/v1/runs").json()
        self.assertEqual(payload["total"], 2)
        ids = {run["run_id"] for run in payload["runs"]}
        self.assertEqual(ids, {self.completed_run_id, self.failed_run_id})
        for run in payload["runs"]:
            self.assertIn("identity", run)
            self.assertIn("paper_fidelity_status", run["identity"])

    def test_list_runs_filters(self) -> None:
        completed = self.client.get("/api/v1/runs", params={"state": "COMPLETED"}).json()
        self.assertEqual([run["run_id"] for run in completed["runs"]], [self.completed_run_id])
        by_model = self.client.get("/api/v1/runs", params={"model_id": "kalmannet_tsp"}).json()
        self.assertEqual([run["run_id"] for run in by_model["runs"]], [self.failed_run_id])

    def test_list_runs_rejects_unknown_state(self) -> None:
        self.assertEqual(self.client.get("/api/v1/runs", params={"state": "NOPE"}).status_code, 400)

    def test_list_runs_is_bounded(self) -> None:
        """Performance guard: an unbounded list query is how dashboards die."""
        self.assertEqual(self.client.get("/api/v1/runs", params={"limit": 99999}).status_code, 422)
        payload = self.client.get("/api/v1/runs", params={"limit": 1}).json()
        self.assertEqual(len(payload["runs"]), 1)
        self.assertEqual(payload["total"], 2)

    def test_run_detail(self) -> None:
        payload = self.client.get(f"/api/v1/runs/{self.completed_run_id}").json()
        self.assertEqual(payload["state"], "COMPLETED")
        self.assertTrue(payload["is_terminal"])
        self.assertTrue(payload["transitions"])
        self.assertIn("worker", payload)
        self.assertIn("exit_code_description", payload)
        self.assertEqual(payload["identity"]["model_id"], "split_knet")
        self.assertEqual(payload["identity"]["paper_fidelity_status"], "partial")
        self.assertFalse(payload["identity"]["supports_exact_resume"])

    def test_unknown_run_is_404(self) -> None:
        from bench.control.identity import RunId

        self.assertEqual(self.client.get(f"/api/v1/runs/{RunId.new().value}").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/runs/not-a-uuid").status_code, 404)

    def test_event_pagination_with_cursor(self) -> None:
        """A-04: after_event_id, limit, ordering."""
        collected: list[int] = []
        cursor = 0
        for _ in range(20):
            page = self.client.get(
                f"/api/v1/runs/{self.completed_run_id}/events",
                params={"after_event_id": cursor, "limit": 5},
            ).json()
            collected.extend(event["event_id"] for event in page["events"])
            cursor = page["next_cursor"]
            if not page["has_more"]:
                break
        self.assertEqual(collected, sorted(collected))
        self.assertEqual(len(collected), len(set(collected)))
        self.assertGreater(len(collected), 10)

    def test_event_type_filter(self) -> None:
        page = self.client.get(
            f"/api/v1/runs/{self.completed_run_id}/events", params={"event_type": "metric"}
        ).json()
        self.assertTrue(page["events"])
        self.assertTrue(all(event["event_type"] == "metric" for event in page["events"]))

    def test_events_for_run_without_journal(self) -> None:
        page = self.client.get(f"/api/v1/runs/{self.failed_run_id}/events").json()
        self.assertEqual(page["events"], [])
        self.assertFalse(page["journal_present"])

    def test_metrics_endpoint_returns_named_series(self) -> None:
        payload = self.client.get(f"/api/v1/runs/{self.completed_run_id}/metrics").json()
        self.assertIn("loss/train_total", payload["series"])
        self.assertEqual(len(payload["series"]["loss/train_total"]), 10)
        self.assertIn("loss/validation_total", payload["series"])

    def test_resources_endpoint(self) -> None:
        payload = self.client.get(f"/api/v1/runs/{self.completed_run_id}/resources").json()
        self.assertEqual(len(payload["samples"]), 1)
        self.assertIsNone(payload["samples"][0]["gpu"])

    def test_artifacts_lists_disk_contents_and_hides_tmp(self) -> None:
        payload = self.client.get(f"/api/v1/runs/{self.failed_run_id}/artifacts").json()
        paths = {item["path"] for item in payload["on_disk"]}
        self.assertIn("artifacts/traceback.txt", paths)
        self.assertTrue(payload["failure_present"])
        self.assertFalse(any(path.startswith("tmp/") for path in paths), "tmp/ holds partial writes")

    def test_logs_are_bounded(self) -> None:
        """U-10: a huge log must not become a huge response."""
        record = self.registry.get_run(self.completed_run_id)
        big = Path(record.run_dir) / "stdout.log"
        big.write_text("x" * 500_000, encoding="utf-8")
        payload = self.client.get(
            f"/api/v1/runs/{self.completed_run_id}/logs", params={"stream": "stdout", "max_bytes": 5000}
        ).json()
        self.assertTrue(payload["truncated"])
        self.assertLessEqual(len(payload["text"]), 5100)
        self.assertEqual(payload["size_bytes"], 500_000)

    def test_stderr_is_retrievable(self) -> None:
        payload = self.client.get(
            f"/api/v1/runs/{self.failed_run_id}/logs", params={"stream": "stderr"}
        ).json()
        self.assertIn("Traceback", payload["text"])

    def test_no_write_endpoints_are_exposed(self) -> None:
        """DND-010: the UI cannot mutate run state, because there is no route to."""
        app = create_app()
        for route in app.routes:
            methods = set(getattr(route, "methods", set()) or set())
            self.assertFalse(
                methods & {"POST", "PUT", "PATCH", "DELETE"},
                f"unexpected write route: {getattr(route, 'path', route)} {methods}",
            )

    def test_orphan_candidates_endpoint(self) -> None:
        payload = self.client.get("/api/v1/orphan-candidates").json()
        self.assertIn("candidates", payload)
        self.assertIsInstance(payload["candidates"], list)


class BindSafetyTests(unittest.TestCase):
    def test_loopback_is_allowed(self) -> None:
        self.assertEqual(resolve_bind_host("127.0.0.1"), "127.0.0.1")

    def test_public_bind_is_refused_without_explicit_opt_in(self) -> None:
        os.environ.pop("BENCH_CONTROL_ALLOW_PUBLIC_BIND", None)
        with self.assertRaises(SystemExit):
            resolve_bind_host("0.0.0.0")

    def test_public_bind_allowed_with_opt_in(self) -> None:
        os.environ["BENCH_CONTROL_ALLOW_PUBLIC_BIND"] = "1"
        try:
            self.assertEqual(resolve_bind_host("0.0.0.0"), "0.0.0.0")
        finally:
            os.environ.pop("BENCH_CONTROL_ALLOW_PUBLIC_BIND", None)


class DashboardServerTests(ControlFixture):
    """Dash page + callback smoke, driven server-side (no browser on this host)."""

    @classmethod
    def _free_port(cls) -> int:
        import socket

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def setUp(self) -> None:
        super().setUp()
        from bench.ui.dash_app import create_dash_app

        # Serve the API in-process on a loopback port so the dashboard's HTTP
        # client exercises the real request path rather than a stub.
        import uvicorn

        self.api_port = self._free_port()
        config = uvicorn.Config(create_app(), host="127.0.0.1", port=self.api_port, log_level="error")
        self.api_server = uvicorn.Server(config)
        self.api_thread = threading.Thread(target=self.api_server.run, daemon=True)
        self.api_thread.start()
        self._wait_for_http(f"http://127.0.0.1:{self.api_port}/api/v1/system/health")

        self.app = create_dash_app(api_base_url=f"http://127.0.0.1:{self.api_port}")
        self.flask_client = self.app.server.test_client()
        self.addCleanup(self._stop_api)

    def _stop_api(self) -> None:
        self.api_server.should_exit = True
        self.api_thread.join(timeout=10)

    @staticmethod
    def _wait_for_http(url: str, timeout: float = 20.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2):
                    return
            except Exception:
                time.sleep(0.2)
        raise AssertionError(f"service at {url} did not become ready")

    def _callback(self, output: str, outputs: Any, inputs: list[dict], state: list[dict] | None = None) -> dict:
        response = self.flask_client.post(
            "/_dash-update-component",
            json={
                "output": output,
                "outputs": outputs,
                "inputs": inputs,
                "changedPropIds": [f"{item['id']}.{item['property']}" for item in inputs],
                "state": state or [],
            },
        )
        self.assertEqual(response.status_code, 200, response.data[:500])
        return json.loads(response.data)

    # -- page routes --------------------------------------------------------

    def test_all_page_routes_serve(self) -> None:
        for path in ("/", "/runs", "/system", f"/runs/{self.completed_run_id}"):
            response = self.flask_client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn(b"Benchmark Control Plane", response.data)

    def test_dash_layout_endpoint(self) -> None:
        self.assertEqual(self.flask_client.get("/_dash-layout").status_code, 200)

    # -- routing callback ---------------------------------------------------

    def test_routing_renders_each_page(self) -> None:
        """U-07: run_id is the route key; a deep link resolves in a fresh session."""
        for path, marker in (
            ("/runs", "Runs"),
            ("/system", "System"),
            (f"/runs/{self.completed_run_id}", self.completed_run_id),
        ):
            payload = self._callback(
                "page-content.children",
                {"id": "page-content", "property": "children"},
                [{"id": "url", "property": "pathname", "value": path}],
            )
            rendered = json.dumps(payload)
            self.assertIn(marker, rendered, f"page {path} did not render marker {marker!r}")

    def test_unknown_route_renders_not_found(self) -> None:
        payload = self._callback(
            "page-content.children",
            {"id": "page-content", "property": "children"},
            [{"id": "url", "property": "pathname", "value": "/nope"}],
        )
        self.assertIn("Not found", json.dumps(payload))

    # -- runs table ---------------------------------------------------------

    def test_runs_table_callback_lists_both_runs(self) -> None:
        payload = self._callback(
            "runs-table.children",
            {"id": "runs-table", "property": "children"},
            [
                {"id": "poll", "property": "n_intervals", "value": 1},
                {"id": "runs-state-filter", "property": "value", "value": ""},
                {"id": "runs-legacy-filter", "property": "value", "value": "all"},
                {"id": "runs-limit", "property": "value", "value": 50},
            ],
        )
        rendered = json.dumps(payload)
        self.assertIn(self.completed_run_id, rendered)
        self.assertIn(self.failed_run_id, rendered)
        # rows must link by run_id, not by a display label
        self.assertIn(f"/runs/{self.completed_run_id}", rendered)

    def test_runs_table_state_filter(self) -> None:
        payload = self._callback(
            "runs-table.children",
            {"id": "runs-table", "property": "children"},
            [
                {"id": "poll", "property": "n_intervals", "value": 1},
                {"id": "runs-state-filter", "property": "value", "value": "FAILED"},
                {"id": "runs-legacy-filter", "property": "value", "value": "all"},
                {"id": "runs-limit", "property": "value", "value": 50},
            ],
        )
        rendered = json.dumps(payload)
        self.assertIn(self.failed_run_id, rendered)
        self.assertNotIn(self.completed_run_id, rendered)

    # -- run detail ---------------------------------------------------------

    def test_run_detail_identity_and_badges(self) -> None:
        payload = self._callback(
            "..run-detail-identity.children...run-detail-progress.children..",
            [
                {"id": "run-detail-identity", "property": "children"},
                {"id": "run-detail-progress", "property": "children"},
            ],
            [{"id": "poll", "property": "n_intervals", "value": 1}],
            [{"id": "run-detail-id", "property": "data", "value": self.completed_run_id}],
        )
        rendered = json.dumps(payload)
        self.assertIn("COMPLETED", rendered)
        self.assertIn("split_knet", rendered)
        self.assertIn("fidelity: partial", rendered)
        self.assertIn("exact resume: not certified", rendered)

    def test_run_detail_charts_render_metrics(self) -> None:
        payload = self._callback(
            "run-detail-charts.children",
            {"id": "run-detail-charts", "property": "children"},
            [{"id": "poll", "property": "n_intervals", "value": 1}],
            [{"id": "run-detail-id", "property": "data", "value": self.completed_run_id}],
        )
        rendered = json.dumps(payload)
        self.assertIn("loss/train_total", rendered)
        self.assertIn("Metrics", rendered)
        self.assertIn("Resources", rendered)

    def test_run_detail_logs_render(self) -> None:
        payload = self._callback(
            "run-detail-logs.children",
            {"id": "run-detail-logs", "property": "children"},
            [{"id": "poll", "property": "n_intervals", "value": 1}],
            [{"id": "run-detail-id", "property": "data", "value": self.failed_run_id}],
        )
        rendered = json.dumps(payload)
        self.assertIn("Traceback", rendered)

    def test_run_detail_rest_sections(self) -> None:
        payload = self._callback(
            "run-detail-rest.children",
            {"id": "run-detail-rest", "property": "children"},
            [{"id": "poll", "property": "n_intervals", "value": 1}],
            [{"id": "run-detail-id", "property": "data", "value": self.failed_run_id}],
        )
        rendered = json.dumps(payload)
        self.assertIn("Checkpoints", rendered)
        # The empty-catalog message must still state the resume status explicitly,
        # so a weight file on disk is never mistaken for a resumable checkpoint.
        self.assertIn("resume-certified", rendered.lower())
        self.assertIn("Artifacts", rendered)
        self.assertIn("State transitions", rendered)

    # -- system page --------------------------------------------------------

    def test_system_page_callback(self) -> None:
        payload = self._callback(
            "..system-health.children...system-gpus.children...system-orphans.children..."
            "system-workers.children...system-capabilities.children..",
            [
                {"id": "system-health", "property": "children"},
                {"id": "system-gpus", "property": "children"},
                {"id": "system-orphans", "property": "children"},
                {"id": "system-workers", "property": "children"},
                {"id": "system-capabilities", "property": "children"},
            ],
            [{"id": "poll", "property": "n_intervals", "value": 1}],
        )
        rendered = json.dumps(payload)
        self.assertIn("registry: ok", rendered)
        self.assertIn("Model capability matrix", rendered)
        self.assertIn("exact resume: not certified", rendered)

    # -- gating -------------------------------------------------------------

    def test_no_stop_resume_or_launch_controls_exist_anywhere(self) -> None:
        """U-04 negative: a feature that does not exist must not be offered."""
        rendered_pages = []
        for path in ("/runs", "/system", f"/runs/{self.completed_run_id}"):
            payload = self._callback(
                "page-content.children",
                {"id": "page-content", "property": "children"},
                [{"id": "url", "property": "pathname", "value": path}],
            )
            rendered_pages.append(json.dumps(payload).lower())
        detail = json.dumps(
            self._callback(
                "run-detail-rest.children",
                {"id": "run-detail-rest", "property": "children"},
                [{"id": "poll", "property": "n_intervals", "value": 1}],
                [{"id": "run-detail-id", "property": "data", "value": self.completed_run_id}],
            )
        ).lower()
        rendered_pages.append(detail)

        for rendered in rendered_pages:
            self.assertNotIn('"type": "button"', rendered)
            for forbidden in ("stop run", "resume run", "warm start", "terminate", "kill run", "launch run"):
                self.assertNotIn(forbidden, rendered, f"UI offers {forbidden!r} but the feature does not exist")

    def test_dashboard_reports_an_unreachable_api_without_crashing(self) -> None:
        """A dead backend must degrade to an error panel, not a broken page."""
        from bench.ui.dash_app import create_dash_app

        offline = create_dash_app(api_base_url="http://127.0.0.1:9")
        client = offline.server.test_client()
        response = client.post(
            "/_dash-update-component",
            json={
                "output": "runs-table.children",
                "outputs": {"id": "runs-table", "property": "children"},
                "inputs": [
                    {"id": "poll", "property": "n_intervals", "value": 1},
                    {"id": "runs-state-filter", "property": "value", "value": ""},
                    {"id": "runs-legacy-filter", "property": "value", "value": "all"},
                    {"id": "runs-limit", "property": "value", "value": 50},
                ],
                "changedPropIds": ["poll.n_intervals"],
                "state": [],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("unreachable", response.data.decode("utf-8"))


class DashboardIsolationTests(unittest.TestCase):
    def test_ui_package_does_not_import_the_registry(self) -> None:
        """R-16/DND-010: the dashboard talks HTTP, it does not open the database."""
        import bench.ui.api_client as api_client
        import bench.ui.dash_app as dash_app
        import bench.ui.components as components

        for module in (api_client, dash_app, components):
            source = Path(module.__file__).read_text(encoding="utf-8")
            self.assertNotIn("SqliteRegistry", source, f"{module.__name__} must not touch the registry")
            self.assertNotIn("sqlite3", source, f"{module.__name__} must not open the database")

    def test_core_control_modules_do_not_import_ui_frameworks(self) -> None:
        """DND-009: domain objects must not depend on a frontend framework."""
        import bench.control as control

        root = Path(control.__file__).resolve().parent
        offenders: list[str] = []
        for path in root.rglob("*.py"):
            if "api" in path.relative_to(root).parts:
                continue  # the API layer may import fastapi
            source = path.read_text(encoding="utf-8")
            for framework in ("import dash", "import streamlit", "from dash", "from streamlit", "import flask"):
                if framework in source:
                    offenders.append(f"{path.relative_to(root)}: {framework}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
