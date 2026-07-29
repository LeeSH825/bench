from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Mapping

import numpy as np

from viz.app.components import overlay_picker
from viz.io.loader import VizRun, load_run
from viz.io.writer import write_viz_artifacts

try:
    from streamlit.testing.v1 import AppTest

    _HAVE_APP_TEST = True
except ImportError:  # pragma: no cover - environment without streamlit.testing
    _HAVE_APP_TEST = False

if _HAVE_APP_TEST:
    from tests.test_viz_cross_model_comparison import _write_run


REPO_ROOT = Path(__file__).resolve().parents[1]

_APP_SCRIPT = (
    "from viz.app.views.run_inspector import render_run_inspector\n"
    "import os\n"
    "render_run_inspector(os.environ['VIZ_TEST_RUNS_ROOT'])\n"
)


def _gain_payload(n_seq: int = 4, n_step: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(n_step, dtype=np.float32) * np.float32(0.5)
    x_true = np.zeros((n_seq, n_step, 9), dtype=np.float32)
    x_hat = x_true.copy()
    y = np.zeros((n_seq, n_step, 6), dtype=np.float32)
    return t, x_true, x_hat, y


def _write_gain_run(root: Path, name: str, *, model_id: str, with_g1g2: bool) -> Path:
    """Two-model gain fixture: one model exposes gain_g1/gain_g2, one only exposes `gain`.

    Used to reproduce and regression-guard the VIZ-R1.3.2 #3 fix: selecting a
    Gain source that a selected model does not provide must surface an
    explicit "not shown" reason instead of the model's trace silently
    disappearing (see `panels.add_overlay_traces`'s disabled-overlay path).
    """
    t, x_true, x_hat, y = _gain_payload()
    n_seq, n_step = x_true.shape[0], x_true.shape[1]
    innov = np.full((n_seq, n_step, 6), np.float32(2.0e-4), dtype=np.float32)
    gain = np.zeros((n_seq, n_step, 9, 6), dtype=np.float32)
    gain[:, :, 0:6, :] = np.eye(6, dtype=np.float32) * np.float32(0.1)
    diagnostics = {"innov": innov, "gain": gain}
    semantics = {"gain": "learned_combined_kalman_gain"}
    if with_g1g2:
        diagnostics["innov_valid"] = np.ones((n_seq, n_step), dtype=bool)
        diagnostics["gain_g1"] = np.broadcast_to(
            np.eye(9, dtype=np.float32) * np.float32(0.1), (n_seq, n_step, 9, 9)
        ).copy()
        diagnostics["gain_g2"] = np.broadcast_to(
            np.eye(6, dtype=np.float32) * np.float32(0.2), (n_seq, n_step, 6, 6)
        ).copy()
        semantics.update(
            {
                "gain_g1": "learned_split_factor_g1",
                "gain_g2": "learned_split_factor_g2",
                "validity_mask": "innov_valid",
            }
        )
    run_dir = root / name
    write_viz_artifacts(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        suite_name="gain_ui_ux_fixture",
        task_id="gain_ui_ux_fixture_v0",
        task_family="basilisk_imu_adcs_bias_v0",
        scenario_id="shared",
        model_id=model_id,
        seed=0,
        track_id="frozen",
        init_id="fixture",
        run_status="ok",
        time_s=t,
        time_meta={"time_source": "fixture", "time_unit": "s", "dt_s": 0.5},
        x_true=x_true,
        y_obs=y,
        x_hat=x_hat,
        split_extras={
            "gyro_bias_seq": x_true[:, :, 6:9].copy(),
            "event_flag_seq": np.zeros((n_seq, n_step), dtype=bool),
            "eclipse_flag_seq": np.zeros((n_seq, n_step), dtype=bool),
        },
        diagnostics=diagnostics,
        adapter_meta={"adapter_id": "fixture", "diagnostic_semantics": semantics},
        data_split="test",
        split_source="explicit",
        trajectory_ids=np.array([0, 5, 10, 15], dtype=np.int64),
        trajectory_id_source="test_split_row_index_fallback",
        k_traj=4,
    )
    return run_dir


class RunIndexScanCountTest(unittest.TestCase):
    """VIZ-R1.3.2 P0 #1/#2: discover_run_index must be called exactly once per rerun."""

    def setUp(self) -> None:
        self._calls: list[str] = []
        self._orig = overlay_picker.discover_run_index

        def counted(runs_root):
            self._calls.append(str(runs_root))
            return self._orig(runs_root)

        overlay_picker.discover_run_index = counted
        from viz.app.views import run_inspector as run_inspector_module

        self._run_inspector_module = run_inspector_module
        run_inspector_module.discover_run_index = counted

    def tearDown(self) -> None:
        overlay_picker.discover_run_index = self._orig
        self._run_inspector_module.discover_run_index = self._orig

    def test_single_scan_per_rerun_including_advanced_expander(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            _write_run(root, "mekf", physical=True, model_id="mekf")
            _write_run(root, "kalmannet", physical=False, model_id="kalmannet")
            _write_run(root, "split_knet", physical=False, model_id="split_knet")

            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])
            self.assertEqual(len(self._calls), 1, self._calls)

            # A rerun triggered by toggling every non-primary model on must
            # still be exactly one scan, even though this exercises all six
            # A-F panels plus the (always-executed) Advanced diagnostics
            # expander body.
            self._calls.clear()
            for checkbox in at.checkbox:
                if not checkbox.disabled:
                    checkbox.set_value(True)
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])
            self.assertEqual(len(self._calls), 1, self._calls)


@unittest.skipUnless(_HAVE_APP_TEST, "streamlit.testing.v1.AppTest is unavailable")
class DeadOverlayUIRemovedTest(unittest.TestCase):
    """VIZ-R1.3.2 P0 #3/#9: the legacy single-model Overlay artifact UI is gone."""

    def test_no_overlay_artifact_selectbox_or_banner(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            _write_run(root, "mekf", physical=True, model_id="mekf")
            _write_run(root, "kalmannet", physical=False, model_id="kalmannet")

            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])

            selectbox_labels = {s.label for s in at.selectbox}
            self.assertNotIn("Overlay artifact", selectbox_labels)
            for element in list(at.success) + list(at.error):
                self.assertNotIn("Overlay compatible", element.value)
                self.assertNotIn("Overlay blocked", element.value)

    def test_render_run_picker_returns_single_run_not_a_tuple(self) -> None:
        picker_script = (
            "import os\n"
            "import streamlit as st\n"
            "from viz.app.components.overlay_picker import discover_run_index, render_run_picker\n"
            "indexed_runs, index_errors, scan_seconds = discover_run_index(os.environ['VIZ_TEST_RUNS_ROOT'])\n"
            "result = render_run_picker(\n"
            "    indexed_runs, index_errors,\n"
            "    runs_root=os.environ['VIZ_TEST_RUNS_ROOT'], scan_seconds=scan_seconds,\n"
            ")\n"
            "st.text(type(result).__name__)\n"
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            _write_run(root, "mekf", physical=True, model_id="mekf")
            at = AppTest.from_string(picker_script)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])
            self.assertEqual(at.text[0].value, "VizRun")


@unittest.skipUnless(_HAVE_APP_TEST, "streamlit.testing.v1.AppTest is unavailable")
class PrimaryLabelAndSingleCandidateTest(unittest.TestCase):
    """VIZ-R1.3.2 P1 #6/#8: primary marker and single-candidate messaging."""

    def test_primary_checkbox_label_carries_marker(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            _write_run(root, "mekf", physical=True, model_id="mekf")
            _write_run(root, "kalmannet", physical=False, model_id="kalmannet")

            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            primary_boxes = [c for c in at.checkbox if c.disabled is False and c.value is True]
            self.assertTrue(any("· primary" in c.label for c in primary_boxes))
            non_primary = [c for c in at.checkbox if "· primary" not in c.label]
            self.assertTrue(non_primary)

    def test_single_candidate_context_shows_explicit_reason_not_unfounded_claims(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            _write_run(root, "only_model", physical=True, model_id="only_model")

            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])
            info_texts = [i.value for i in at.info]
            self.assertTrue(any("Only one run artifact is available" in t for t in info_texts))
            # Must not claim things the viz layer cannot know from run artifacts alone.
            for t in info_texts:
                self.assertNotIn("suite config", t.lower())
                self.assertNotIn("parser failed", t.lower())
            why_expanders = [e for e in at.expander if e.label == "Why only one candidate?"]
            self.assertTrue(why_expanders)
            stats = [c.value for c in at.caption if c.value.startswith("Indexed runs:")]
            self.assertTrue(stats)
            self.assertIn("Matching suite/task/scenario/split/seed/track: 1", stats[0])
            self.assertIn("Display candidates: 1", stats[0])


@unittest.skipUnless(_HAVE_APP_TEST, "streamlit.testing.v1.AppTest is unavailable")
class GainSourceExclusionTest(unittest.TestCase):
    """VIZ-R1.3.2 P0 #7: G1/G2 exclusion must be explicit, not silent."""

    def _setup_two_model_gain_suite(self, tmp: str) -> None:
        root = Path(tmp)
        _write_gain_run(root, "split_like", model_id="split_like", with_g1g2=True)
        _write_gain_run(root, "kalmannet_like", model_id="kalmannet_like", with_g1g2=False)

    def test_g1_dropdown_reachable_even_when_primary_lacks_g1(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            self._setup_two_model_gain_suite(tmp)
            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])
            # kalmannet_like sorts first alphabetically and has no gain_g1/g2.
            primary_boxes = [c for c in at.checkbox if c.value is True]
            self.assertIn("kalmannet_like", primary_boxes[0].label)

            for checkbox in at.checkbox:
                if not checkbox.disabled:
                    checkbox.set_value(True)
            at.run(timeout=60)
            gain_selectbox = next(s for s in at.selectbox if s.label == "Gain source")
            self.assertIn("Learned G1 factor", gain_selectbox.options)
            self.assertIn("Learned G2 factor", gain_selectbox.options)

    def test_selecting_g1_excludes_model_without_g1_with_explicit_reason(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            self._setup_two_model_gain_suite(tmp)
            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            for checkbox in at.checkbox:
                if not checkbox.disabled:
                    checkbox.set_value(True)
            at.run(timeout=60)
            gain_selectbox = next(s for s in at.selectbox if s.label == "Gain source")
            gain_selectbox.set_value("Learned G1 factor")
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])

            captions = [c.value for c in at.caption]
            self.assertTrue(
                any("kalmannet_like" in c and "not shown" in c and "gain_g1" in c for c in captions)
            )

            charts = at.get("plotly_chart")
            gain_specs = []
            for chart in charts:
                spec = json.loads(chart.proto.spec)
                names = [tr.get("name") for tr in spec.get("data", [])]
                if any("gain" in (n or "").lower() for n in names):
                    gain_specs.append(names)
            self.assertTrue(gain_specs)
            self.assertFalse(any("kalmannet_like" in (n or "") for names in gain_specs for n in names))
            self.assertTrue(any("split_like" in (n or "") for names in gain_specs for n in names))

    def test_combined_gain_still_shows_both_models(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            self._setup_two_model_gain_suite(tmp)
            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            for checkbox in at.checkbox:
                if not checkbox.disabled:
                    checkbox.set_value(True)
            at.run(timeout=60)
            charts = at.get("plotly_chart")
            gain_specs = []
            for chart in charts:
                spec = json.loads(chart.proto.spec)
                names = [tr.get("name") for tr in spec.get("data", [])]
                if any("gain" in (n or "").lower() for n in names):
                    gain_specs.append(names)
            self.assertTrue(any("split_like" in (n or "") for names in gain_specs for n in names))
            self.assertTrue(any("kalmannet_like" in (n or "") for names in gain_specs for n in names))


@unittest.skipUnless(_HAVE_APP_TEST, "streamlit.testing.v1.AppTest is unavailable")
class TerminologyAndLayoutTest(unittest.TestCase):
    """VIZ-R1.3.2 P1 #5/#9: no bare "Overlay" axis label, balanced checkbox grid."""

    def test_axis_mode_label_is_not_bare_overlay(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            _write_run(root, "mekf", physical=True, model_id="mekf")
            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            axis_radio = next(r for r in at.radio if r.label == "Axis mode")
            self.assertNotIn("Overlay", axis_radio.options)
            self.assertIn("Combined axes", axis_radio.options)

    def test_four_candidates_render_as_two_by_two(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            for name in ("additive_ekf", "kalmannet_mekf", "split_knet_mekf", "adaptive_kalmannet"):
                _write_run(root, name, physical=(name == "additive_ekf"), model_id=name)

            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])

            def walk(node, depth=0, collecting=[False]):
                node_type = getattr(node, "type", None)
                if node_type == "markdown":
                    try:
                        if "Models to display" in str(node.value):
                            collecting[0] = True
                    except Exception:
                        pass
                rows = []
                if collecting[0] and node_type == "flex_container":
                    checkbox_count = sum(
                        1 for child in getattr(node, "children", {}).values()
                        for grandchild in getattr(child, "children", {}).values()
                        if getattr(grandchild, "type", None) == "checkbox"
                    )
                    if checkbox_count:
                        rows.append(checkbox_count)
                for child in getattr(node, "children", {}).values() if hasattr(node, "children") else []:
                    rows.extend(walk(child, depth + 1, collecting))
                if node_type == "caption":
                    try:
                        if "Selected models:" in str(node.value):
                            collecting[0] = False
                    except Exception:
                        pass
                return rows

            row_sizes = walk(at.main)
            self.assertEqual(row_sizes, [2, 2])


@unittest.skipUnless(_HAVE_APP_TEST, "streamlit.testing.v1.AppTest is unavailable")
class LanguageAndDiagnosticsMatrixTest(unittest.TestCase):
    """VIZ-R1.3.2 P2 #14/#15: English-only UI text and the compatibility matrix."""

    def test_no_korean_word_in_english_ui(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            _write_run(root, "mekf", physical=True, model_id="mekf")
            _write_run(root, "kalmannet", physical=False, model_id="kalmannet")
            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            for checkbox in at.checkbox:
                if not checkbox.disabled:
                    checkbox.set_value(True)
            at.run(timeout=60)
            for c in at.caption:
                self.assertNotIn("기준", c.value)
            for s in at.subheader:
                self.assertNotIn("기준", s.value)

    def test_dataset_summary_title_names_primary_model(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            _write_run(root, "mekf", physical=True, model_id="mekf")
            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            titles = [s.value for s in at.subheader if s.value.startswith("Dataset Summary")]
            self.assertTrue(titles)
            self.assertIn("mekf", titles[0])

    def test_advanced_diagnostics_has_matrix_and_does_not_repeat_panel_reason_text(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            _write_run(root, "mekf", physical=True, model_id="mekf")
            _write_run(root, "kalmannet", physical=False, model_id="kalmannet")
            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            for checkbox in at.checkbox:
                if not checkbox.disabled:
                    checkbox.set_value(True)
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])

            markdowns = [m.value for m in at.markdown]
            self.assertTrue(any("Panel compatibility matrix" in m for m in markdowns))
            self.assertTrue(any("Dataset-average RMSE" in m for m in markdowns))
            matrix_tables = [m for m in markdowns if m.startswith("| Model |")]
            self.assertTrue(matrix_tables)
            self.assertIn("Available", matrix_tables[0])
            # The matrix must not restate a full physical-covariance reason
            # sentence — that already lives under the E. NEES/NIS panel above.
            self.assertNotIn("physical state covariance P is not provided", matrix_tables[0])


@unittest.skipUnless(_HAVE_APP_TEST, "streamlit.testing.v1.AppTest is unavailable")
class RegressionTest(unittest.TestCase):
    """VIZ-R1.3.2 completion criteria: prior guards/behavior still hold."""

    def test_legacy_artifact_without_comparison_spec_still_loads(self) -> None:
        real_root = REPO_ROOT / "runs" / "viz_attitude_bias_gpu_compare"
        if not real_root.exists():
            self.skipTest("actual runs/viz_attitude_bias_gpu_compare artifact not present")
        at = AppTest.from_string(_APP_SCRIPT)
        os.environ["VIZ_TEST_RUNS_ROOT"] = str(real_root)
        at.run(timeout=60)
        self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])

    def test_artifact_version_and_legacy_load_preserved(self) -> None:
        real_root = REPO_ROOT / "runs" / "viz_v4c_cross_models"
        if not real_root.exists():
            self.skipTest("actual runs/viz_v4c_cross_models artifact not present")
        for meta_path in real_root.rglob("meta.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta.get("artifact_version"), "1.1")
            break

    def test_no_third_party_import_added(self) -> None:
        source = (REPO_ROOT / "viz" / "app" / "views" / "run_inspector.py").read_text(encoding="utf-8")
        self.assertNotIn("third_party", source)


if __name__ == "__main__":
    unittest.main()
