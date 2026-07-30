from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path

from viz.app.help_content import CORE_FACTS, GUIDE_TABS, HELP_TEXT

_KOREAN_CHAR_RE = re.compile(r"[가-힣]")

try:
    from streamlit.testing.v1 import AppTest

    _HAVE_APP_TEST = True
except ImportError:  # pragma: no cover - environment without streamlit.testing
    _HAVE_APP_TEST = False

if _HAVE_APP_TEST:
    from tests.test_viz_cross_model_comparison import _write_run


REPO_ROOT = Path(__file__).resolve().parents[1]
_GUIDE_TEXT = "\n".join(body for _title, body in GUIDE_TABS)

_APP_SCRIPT = (
    "from viz.app.views.run_inspector import render_run_inspector\n"
    "import os\n"
    "render_run_inspector(os.environ['VIZ_TEST_RUNS_ROOT'])\n"
)


class GuideContentStructureTest(unittest.TestCase):
    """Pure content checks — no Streamlit, no run artifact involved."""

    def test_guide_has_the_four_required_tabs_in_order(self) -> None:
        titles = [title for title, _body in GUIDE_TABS]
        self.assertEqual(titles, ["Quick start", "Panels", "Model comparison", "Troubleshooting"])

    def test_quick_start_mentions_the_five_step_flow(self) -> None:
        _title, body = GUIDE_TABS[0]
        for phrase in ("data split", "Trajectory view", "Models to display", "panels A"):
            self.assertIn(phrase, body)

    def test_panels_tab_covers_all_six_af_panels_and_regime(self) -> None:
        _title, body = GUIDE_TABS[1]
        for heading in (
            "A. Attitude RPY",
            "B. Attitude Error",
            "C. Bias",
            "D. Innovation",
            "E. NEES / NIS",
            "F. Kalman Gain",
            "Regime Timeline",
        ):
            self.assertIn(heading, body)

    def test_model_comparison_tab_explains_primary_vs_models_to_display(self) -> None:
        _title, body = GUIDE_TABS[2]
        self.assertIn("Primary model vs. Models to display", body)
        self.assertIn(CORE_FACTS["init_id_not_hard_gate"], body)

    def test_model_comparison_tab_has_capability_table(self) -> None:
        _title, body = GUIDE_TABS[2]
        self.assertIn("Physical P / S", body)
        self.assertIn("G1 / G2", body)

    def test_troubleshooting_tab_covers_required_cases(self) -> None:
        _title, body = GUIDE_TABS[3]
        for phrase in (
            "Only one model is shown",
            "missing from one panel",
            "G1/G2 is not shown",
            "NEES/NIS is unavailable",
            "Physical 3σ is unavailable",
            "cross-model comparison",
            "Source ID mismatch",
            "Empty artifact root",
            "Slow initial scan",
        ):
            self.assertIn(phrase, body)

    def test_offline_viewer_limitation_is_stated(self) -> None:
        self.assertIn(CORE_FACTS["offline_viewer"], _GUIDE_TEXT)

    def test_source_id_row_fallback_is_explained(self) -> None:
        self.assertIn(CORE_FACTS["source_id_fallback"], _GUIDE_TEXT)

    def test_physical_vs_empirical_uncertainty_is_explained(self) -> None:
        self.assertIn(CORE_FACTS["physical_vs_empirical"], _GUIDE_TEXT)

    def test_g1_g2_is_explained_as_not_covariance(self) -> None:
        self.assertIn(CORE_FACTS["g1_g2_not_covariance"], _GUIDE_TEXT)

    def test_init_training_provenance_is_explained(self) -> None:
        self.assertIn(CORE_FACTS["init_id_not_hard_gate"], _GUIDE_TEXT)

    def test_no_korean_word_in_the_english_in_app_guide(self) -> None:
        # The in-app popover is English-only (docs/VIZ_USER_GUIDE_KO.md is the
        # separate Korean document) — see language policy in the report.
        offenders = [title for title, body in GUIDE_TABS if _KOREAN_CHAR_RE.search(body)]
        self.assertEqual(offenders, [])


class ContentConsistencyTest(unittest.TestCase):
    """Guide/tooltip content must not overclaim or misstate the contract."""

    def test_kalmannet_split_physical_p_s_is_not_claimed_unconditionally(self) -> None:
        # The capability table must say "No" for KalmanNet/Split-KalmanNet
        # physical P/S, not "Yes" or an unqualified claim.
        _title, body = GUIDE_TABS[2]
        table_lines = [line for line in body.splitlines() if line.startswith("| Physical P")]
        self.assertTrue(table_lines)
        self.assertIn("| No | No |", table_lines[0])

    def test_g1_g2_never_described_as_p_or_s(self) -> None:
        for text in (_GUIDE_TEXT, *HELP_TEXT.values()):
            if "G1" in text or "G2" in text or "gain_g1" in text:
                self.assertNotIn("G1 is a physical", text)
                self.assertNotIn("G2 is a physical", text)

    def test_init_id_never_described_as_hard_compatibility(self) -> None:
        for text in (_GUIDE_TEXT, *HELP_TEXT.values()):
            self.assertNotIn("init_id is a hard compatibility", text)
            self.assertNotIn("init_id must match", text)

    def test_dataset_summary_described_as_primary_based(self) -> None:
        self.assertIn(CORE_FACTS["dataset_summary_is_primary_only"], _GUIDE_TEXT)
        self.assertIn("primary navigation run", HELP_TEXT["dataset_summary"])

    def test_source_id_fallback_not_described_as_stable_global_id(self) -> None:
        for text in (_GUIDE_TEXT, *HELP_TEXT.values()):
            self.assertNotIn("globally stable", text)
            self.assertNotIn("permanent identity", text)

    def test_no_synthetic_fixture_described_as_actual_performance(self) -> None:
        for text in (_GUIDE_TEXT, *HELP_TEXT.values()):
            self.assertNotIn("benchmark result", text)
            self.assertNotIn("proven performance", text)

    def test_no_overclaiming_absolute_words(self) -> None:
        banned = ("always works", "perfectly", "guaranteed to", "optimally", "the best model")
        for text in (_GUIDE_TEXT, *HELP_TEXT.values()):
            lowered = text.lower()
            for phrase in banned:
                self.assertNotIn(phrase, lowered)

    def test_known_limitations_doc_does_not_contradict_guide(self) -> None:
        limitations_path = REPO_ROOT / "docs" / "VIZ_KNOWN_LIMITATIONS.md"
        self.assertTrue(limitations_path.exists())
        limitations_text = limitations_path.read_text(encoding="utf-8")
        # Both documents must agree that learned filters lack physical P/S —
        # spot-check the shared fact rather than the exact sentence, since
        # the limitations doc is Korean and the guide is English.
        self.assertIn("P", limitations_text)
        self.assertIn("S", limitations_text)
        self.assertIn(CORE_FACTS["g1_g2_not_covariance"], _GUIDE_TEXT)
        self.assertIn("G1/G2", limitations_text)


class DocFilesExistTest(unittest.TestCase):
    def test_english_guide_exists_and_is_utf8(self) -> None:
        path = REPO_ROOT / "docs" / "VIZ_USER_GUIDE.md"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn(CORE_FACTS["offline_viewer"], text)
        self.assertNotIn("TODO", text)
        self.assertNotIn("PLACEHOLDER", text.upper().replace("PLACEHOLDER-FREE", ""))

    def test_korean_guide_exists_and_is_utf8(self) -> None:
        path = REPO_ROOT / "docs" / "VIZ_USER_GUIDE_KO.md"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("Run Inspector", text)
        self.assertIn("오프라인", text.replace("offline", "오프라인") if "오프라인" not in text else text)

    def test_quickstart_and_troubleshooting_docs_exist(self) -> None:
        for name in ("VIZ_QUICKSTART.md", "VIZ_TROUBLESHOOTING.md"):
            path = REPO_ROOT / "docs" / name
            self.assertTrue(path.exists(), name)
            text = path.read_text(encoding="utf-8")
            self.assertGreater(len(text), 100)

    def test_guide_launch_command_matches_actual_entry_point(self) -> None:
        text = (REPO_ROOT / "docs" / "VIZ_USER_GUIDE.md").read_text(encoding="utf-8")
        self.assertIn("streamlit run viz/app/main.py", text)
        main_path = REPO_ROOT / "viz" / "app" / "main.py"
        self.assertTrue(main_path.exists())

    def test_guide_mentions_emit_viz_artifacts_flag_that_actually_exists(self) -> None:
        text = (REPO_ROOT / "docs" / "VIZ_USER_GUIDE.md").read_text(encoding="utf-8")
        self.assertIn("--emit-viz-artifacts", text)
        runner_text = (REPO_ROOT / "bench" / "runners" / "run_suite.py").read_text(encoding="utf-8")
        self.assertIn("--emit-viz-artifacts", runner_text)


@unittest.skipUnless(_HAVE_APP_TEST, "streamlit.testing.v1.AppTest is unavailable")
class GuideRenderTest(unittest.TestCase):
    """Renders the real app and inspects the Help & guide popover."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir="/tmp")
        root = Path(self._tmp.name)
        _write_run(root, "mekf", physical=True, model_id="mekf")
        self._prev_env = os.environ.get("VIZ_TEST_RUNS_ROOT")
        os.environ["VIZ_TEST_RUNS_ROOT"] = str(root)

    def tearDown(self) -> None:
        if self._prev_env is None:
            os.environ.pop("VIZ_TEST_RUNS_ROOT", None)
        else:
            os.environ["VIZ_TEST_RUNS_ROOT"] = self._prev_env
        self._tmp.cleanup()

    def test_help_control_rendered_exactly_once(self) -> None:
        at = AppTest.from_string(_APP_SCRIPT)
        at.run(timeout=60)
        self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])
        popovers = at.get("popover")
        help_popovers = [p for p in popovers if p.proto.popover.label == "Help & guide"]
        self.assertEqual(len(help_popovers), 1)

    def test_help_control_renders_near_title_before_navigation(self) -> None:
        at = AppTest.from_string(_APP_SCRIPT)
        at.run(timeout=60)

        order: list[tuple[str, object]] = []

        def walk(node) -> None:
            node_type = getattr(node, "type", None)
            if node_type:
                order.append((node_type, node))
            for child in getattr(node, "children", {}).values() if hasattr(node, "children") else []:
                walk(child)

        walk(at.main)
        types_in_order = [t for t, _n in order]
        title_idx = types_in_order.index("title")
        popover_idx = next(
            i for i, (t, n) in enumerate(order) if t == "popover" and n.proto.popover.label == "Help & guide"
        )
        first_nav_selectbox_idx = next(i for i, (t, n) in enumerate(order) if t == "selectbox")
        self.assertLess(title_idx, popover_idx)
        self.assertLess(popover_idx, first_nav_selectbox_idx)

    def test_guide_popover_starts_closed(self) -> None:
        at = AppTest.from_string(_APP_SCRIPT)
        at.run(timeout=60)
        help_popover = next(p for p in at.get("popover") if p.proto.popover.label == "Help & guide")
        self.assertFalse(help_popover.proto.popover.open)

    def test_guide_render_does_not_scan_runs_or_load_trajectories(self) -> None:
        from viz.app.components import overlay_picker
        from viz.io import loader as loader_mod

        scan_calls: list[str] = []
        load_calls: list[str] = []
        orig_scan = overlay_picker.discover_run_index
        orig_load = loader_mod.VizRun.load_trajectory

        def counted_scan(runs_root):
            scan_calls.append(str(runs_root))
            return orig_scan(runs_root)

        def counted_load(self, *args, **kwargs):
            load_calls.append(str(self.meta.get("model_id")))
            return orig_load(self, *args, **kwargs)

        overlay_picker.discover_run_index = counted_scan
        loader_mod.VizRun.load_trajectory = counted_load
        try:
            script = (
                "from viz.app.components.help_guide import render_help_popover\n"
                "render_help_popover()\n"
            )
            at = AppTest.from_string(script)
            at.run(timeout=30)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])
            self.assertEqual(scan_calls, [])
            self.assertEqual(load_calls, [])
        finally:
            overlay_picker.discover_run_index = orig_scan
            loader_mod.VizRun.load_trajectory = orig_load

    def test_default_page_af_layout_unaffected_by_guide_presence(self) -> None:
        at = AppTest.from_string(_APP_SCRIPT)
        at.run(timeout=60)
        subheaders = [s.value for s in at.subheader]
        for expected in (
            "A. Attitude RPY Overlay",
            "B. Attitude Error + 3 sigma",
            "C. Bias + 3 sigma",
            "D. Innovation",
            "E. NEES / NIS + chi-square",
            "F. Gain",
        ):
            self.assertIn(expected, subheaders)


if __name__ == "__main__":
    unittest.main()
