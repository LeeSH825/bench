from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path

from viz.app.help_content import HELP_TEXT

try:
    from streamlit.testing.v1 import AppTest

    _HAVE_APP_TEST = True
except ImportError:  # pragma: no cover - environment without streamlit.testing
    _HAVE_APP_TEST = False

if _HAVE_APP_TEST:
    from tests.test_viz_cross_model_comparison import _write_run


_APP_SCRIPT = (
    "from viz.app.views.run_inspector import render_run_inspector\n"
    "import os\n"
    "render_run_inspector(os.environ['VIZ_TEST_RUNS_ROOT'])\n"
)

_MAX_TOOLTIP_LENGTH = 320  # ~250 chars recommended; generous ceiling, not the target.
_BANNED_ABSOLUTE_WORDS = ("always", "perfectly", "guaranteed", "optimal", "never fails")
_KOREAN_CHAR_RE = re.compile(r"[가-힣]")


class HelpTextInventoryTest(unittest.TestCase):
    """Pure content checks on the HELP_TEXT mapping itself."""

    _REQUIRED_KEYS = (
        "data_split",
        "suite",
        "task",
        "scenario",
        "model",
        "seed",
        "track",
        "init_checkpoint",
        "trajectory_view",
        "models_to_display",
        "model_checkbox",
        "model_checkbox_primary",
        "provenance",
        "axis_mode",
        "transient_window",
        "gain_source",
        "gain_display",
        "gain_row_col",
        "dataset_summary",
        "selected_trajectory",
        "attitude_rpy",
        "attitude_error",
        "bias",
        "innovation",
        "nees_nis",
        "kalman_gain",
        "regime_timeline",
        "advanced_diagnostics",
    )

    def test_all_required_keys_present(self) -> None:
        missing = [key for key in self._REQUIRED_KEYS if key not in HELP_TEXT]
        self.assertEqual(missing, [])

    def test_no_tooltip_is_empty(self) -> None:
        for key, text in HELP_TEXT.items():
            self.assertTrue(text.strip(), key)

    def test_no_tooltip_exceeds_length_ceiling(self) -> None:
        too_long = {key: len(text) for key, text in HELP_TEXT.items() if len(text) > _MAX_TOOLTIP_LENGTH}
        self.assertEqual(too_long, {})

    def test_no_tooltip_uses_banned_absolute_words(self) -> None:
        offenders = {}
        for key, text in HELP_TEXT.items():
            lowered = text.lower()
            hits = [word for word in _BANNED_ABSOLUTE_WORDS if word in lowered]
            if hits:
                offenders[key] = hits
        self.assertEqual(offenders, {})

    def test_no_korean_word_in_english_tooltips(self) -> None:
        offenders = [key for key, text in HELP_TEXT.items() if _KOREAN_CHAR_RE.search(text)]
        self.assertEqual(offenders, [])

    def test_no_custom_javascript_tooltip_markup(self) -> None:
        for key, text in HELP_TEXT.items():
            self.assertNotIn("<script", text.lower(), key)
            self.assertNotIn("onmouseover", text.lower(), key)

    def test_no_unsupported_capability_claims(self) -> None:
        # Physical 3-sigma / NEES-NIS help text must not claim a learned
        # model has physical P/S unconditionally.
        for key in ("attitude_error", "nees_nis"):
            text = HELP_TEXT[key].lower()
            self.assertNotIn("kalmannet has physical p", text)
            self.assertNotIn("split-kalmannet has physical p", text)

    def test_gain_help_does_not_call_g1_g2_a_covariance(self) -> None:
        text = HELP_TEXT["gain_source"]
        self.assertIn("not a physical covariance", text)


@unittest.skipUnless(_HAVE_APP_TEST, "streamlit.testing.v1.AppTest is unavailable")
class WidgetTooltipRenderTest(unittest.TestCase):
    """Confirms native `help=` reaches the real widgets in the rendered app."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir="/tmp")
        root = Path(self._tmp.name)
        _write_run(root, "mekf", physical=True, model_id="mekf")
        _write_run(root, "kalmannet", physical=False, model_id="kalmannet")
        self._prev_env = os.environ.get("VIZ_TEST_RUNS_ROOT")
        os.environ["VIZ_TEST_RUNS_ROOT"] = str(root)

    def tearDown(self) -> None:
        if self._prev_env is None:
            os.environ.pop("VIZ_TEST_RUNS_ROOT", None)
        else:
            os.environ["VIZ_TEST_RUNS_ROOT"] = self._prev_env
        self._tmp.cleanup()

    def _run(self) -> "AppTest":
        at = AppTest.from_string(_APP_SCRIPT)
        at.run(timeout=60)
        self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])
        return at

    def test_navigation_selectboxes_have_help(self) -> None:
        at = self._run()
        required_labels = {
            "Data split",
            "Suite",
            "Task",
            "Scenario",
            "Model",
            "Seed",
            "Track",
            "Init/checkpoint",
            "Trajectory view",
        }
        found = {s.label: s.proto.help for s in at.selectbox if s.label in required_labels}
        self.assertEqual(set(found.keys()), required_labels)
        for label, help_text in found.items():
            self.assertTrue(help_text, label)

    def test_navigation_selectbox_labels_are_not_collapsed(self) -> None:
        at = self._run()
        required_labels = {"Data split", "Suite", "Task", "Scenario", "Model", "Seed", "Track", "Init/checkpoint"}
        for s in at.selectbox:
            if s.label in required_labels:
                self.assertEqual(s.proto.label_visibility.value, 0)  # LabelVisibility.VISIBLE

    def test_models_to_display_and_checkboxes_have_help(self) -> None:
        at = self._run()
        markdown_help = {m.value: m.proto.help for m in at.markdown}
        self.assertIn("**Models to display**", markdown_help)
        self.assertTrue(markdown_help["**Models to display**"])
        for c in at.checkbox:
            self.assertTrue(c.proto.help, c.label)
            self.assertEqual(c.proto.label_visibility.value, 0)  # LabelVisibility.VISIBLE

    def test_dataset_summary_and_selected_trajectory_headings_have_help(self) -> None:
        at = self._run()
        subheader_help = {s.value: s.proto.help for s in at.subheader}
        matched_summary = [v for k, v in subheader_help.items() if k.startswith("Dataset Summary")]
        matched_traj = [v for k, v in subheader_help.items() if k.startswith("Selected Trajectory")]
        self.assertTrue(matched_summary and matched_summary[0])
        self.assertTrue(matched_traj and matched_traj[0])

    def test_af_panel_headings_have_help(self) -> None:
        at = self._run()
        subheader_help = {s.value: s.proto.help for s in at.subheader}
        for prefix in ("A. Attitude RPY", "B. Attitude Error", "C. Bias", "D. Innovation", "E. NEES", "F. Gain"):
            matched = [v for k, v in subheader_help.items() if k.startswith(prefix)]
            self.assertTrue(matched, prefix)
            self.assertTrue(matched[0], prefix)

    def test_regime_timeline_heading_has_help(self) -> None:
        at = self._run()
        subheader_help = {s.value: s.proto.help for s in at.subheader}
        self.assertIn("Regime Timeline", subheader_help)
        self.assertTrue(subheader_help["Regime Timeline"])

    def test_axis_and_transient_controls_have_help(self) -> None:
        at = self._run()
        radio_help = {r.label: r.proto.help for r in at.radio}
        self.assertTrue(radio_help.get("Axis mode"))
        self.assertTrue(radio_help.get("Transient window"))

    def test_gain_controls_have_help(self) -> None:
        at = self._run()
        selectbox_help = {s.label: s.proto.help for s in at.selectbox}
        radio_help = {r.label: r.proto.help for r in at.radio}
        self.assertTrue(selectbox_help.get("Gain source"))
        self.assertTrue(radio_help.get("Gain display"))

    def test_advanced_diagnostics_and_provenance_sections_have_a_help_bearing_caption(self) -> None:
        at = self._run()
        for checkbox in at.checkbox:
            if not checkbox.disabled:
                checkbox.set_value(True)
        at.run(timeout=60)
        self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])
        captions_with_help = [c.value for c in at.caption if c.proto.help]
        self.assertTrue(any("Physical Outputs compares" in c for c in captions_with_help))
        self.assertTrue(any(c == "What is this?" for c in captions_with_help))


if __name__ == "__main__":
    unittest.main()
