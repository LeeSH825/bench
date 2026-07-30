from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import plotly.graph_objects as go

from viz.app.components.model_toggle_picker import (
    candidate_label,
    model_context_key,
    reconcile_selection,
    suite_candidates,
)
from viz.figures.panels import PanelResult, add_overlay_traces, label_model_traces, model_color
from viz.io import loader as loader_mod

try:
    from streamlit.testing.v1 import AppTest

    _HAVE_APP_TEST = True
except ImportError:  # pragma: no cover - environment without streamlit.testing
    _HAVE_APP_TEST = False

if _HAVE_APP_TEST:
    from tests.test_viz_cross_model_comparison import _write_run


class _ManifestOnlyRun:
    def __init__(self, run_dir: str, meta: dict, source_ids: set[int] | None = None) -> None:
        self.run_dir = run_dir
        self.meta = meta
        self.source_ids = source_ids or {0}
        self.npz_loads = 0

    def trajectory_by_source_id(self, source_id: int):
        if source_id not in self.source_ids:
            raise KeyError(source_id)
        return SimpleNamespace(source_trajectory_id=source_id)

    def load_trajectory(self, **_kwargs):
        self.npz_loads += 1
        raise AssertionError("candidate discovery must not load a trajectory NPZ")


def _meta(model_id: str, *, suite: str = "suite-a", seed: int = 0) -> dict:
    return {
        "suite": suite,
        "task": "task-a",
        "scenario_id": "scenario-a",
        "model_id": model_id,
        "seed": seed,
        "track_id": "frozen",
        "init_id": "pretrained",
        "data_spec": {"split": "test"},
    }


class SuiteModelToggleTest(unittest.TestCase):
    def test_candidates_are_same_context_and_manifest_only(self) -> None:
        primary = _ManifestOnlyRun("/primary", _meta("kalmannet"))
        split = _ManifestOnlyRun("/split", _meta("split_knet"))
        other_suite = _ManifestOnlyRun("/other", _meta("mekf", suite="suite-b"))
        other_seed = _ManifestOnlyRun("/seed", _meta("adaptive", seed=1))
        missing_source = _ManifestOnlyRun("/missing", _meta("missing"), source_ids={1})
        candidates = suite_candidates(
            primary,
            [split, other_suite, other_seed, missing_source],
            source_trajectory_id=0,
        )
        self.assertEqual([candidate.model_id for candidate in candidates], ["kalmannet", "split_knet"])
        self.assertEqual(primary.npz_loads, 0)
        self.assertEqual(split.npz_loads, 0)

    def test_initial_selection_is_primary_only_and_context_changes_drop_stale(self) -> None:
        primary = _ManifestOnlyRun("/primary", _meta("kalmannet"))
        split = _ManifestOnlyRun("/split", _meta("split_knet"))
        candidates = suite_candidates(primary, [split], source_trajectory_id=0)
        selected = reconcile_selection(candidates, primary_run_dir="/primary", previous=None)
        self.assertEqual(selected, {"/primary"})
        selected = reconcile_selection(candidates, primary_run_dir="/primary", previous=["/primary", "/split"])
        self.assertEqual(selected, {"/primary", "/split"})
        changed = _ManifestOnlyRun("/primary", _meta("kalmannet", suite="suite-b"))
        changed_candidates = suite_candidates(changed, [], source_trajectory_id=0)
        self.assertEqual(
            reconcile_selection(changed_candidates, primary_run_dir="/primary", previous=selected),
            {"/primary"},
        )

    def test_primary_can_be_excluded_once_another_model_is_selected(self) -> None:
        primary = _ManifestOnlyRun("/primary", _meta("kalmannet"))
        split = _ManifestOnlyRun("/split", _meta("split_knet"))
        candidates = suite_candidates(primary, [split], source_trajectory_id=0)
        # The caller (live toggle interaction) explicitly dropped the primary
        # while keeping another model selected; reconcile must not force it
        # back on, since that would prevent the "primary OFF" UI flow.
        selected = reconcile_selection(candidates, primary_run_dir="/primary", previous=["/split"])
        self.assertEqual(selected, {"/split"})

    def test_selection_entirely_invalidated_by_context_change_defaults_to_primary(self) -> None:
        primary = _ManifestOnlyRun("/primary", _meta("kalmannet"))
        candidates = suite_candidates(primary, [], source_trajectory_id=0)
        # previous selection excluded the primary and only referenced a run
        # that is no longer a valid candidate in this context; nothing valid
        # survives the intersection, so the default (primary only) applies.
        selected = reconcile_selection(candidates, primary_run_dir="/primary", previous=["/stale"])
        self.assertEqual(selected, {"/primary"})

    def test_context_includes_seed_and_split(self) -> None:
        self.assertNotEqual(model_context_key(_meta("a", seed=0)), model_context_key(_meta("a", seed=1)))
        changed = _meta("a")
        changed["data_spec"]["split"] = "train"
        self.assertNotEqual(model_context_key(_meta("a")), model_context_key(changed))

    def test_candidate_label_exposes_init_track_and_seed(self) -> None:
        self.assertEqual(
            candidate_label(_meta("split_knet")),
            "split_knet · init=pretrained · frozen / seed 0",
        )

    def test_model_color_and_trace_labels_are_stable(self) -> None:
        ordered = ["kalmannet", "split_knet"]
        self.assertEqual(model_color("kalmannet", ordered), model_color("kalmannet", ordered))
        result = label_model_traces(
            PanelResult(
                go.Figure(
                    data=[
                        go.Scatter(x=[0, 1], y=[0, 1], name="truth roll"),
                        go.Scatter(x=[0, 1], y=[1, 2], name="estimate roll"),
                    ]
                )
            ),
            model_id="kalmannet",
            ordered_model_ids=ordered,
        )
        self.assertEqual([trace.name for trace in result.figure.data], ["truth roll", "kalmannet · estimate roll"])
        self.assertEqual(result.figure.data[1].line.color, model_color("kalmannet", ordered))

    def test_enabled_overlay_survives_disabled_primary_panel(self) -> None:
        primary = PanelResult(go.Figure(), disabled_reason="primary has no physical P")
        overlay = PanelResult(go.Figure(data=[go.Scatter(x=[0, 1], y=[1, 2], name="model · NEES")]))
        result = add_overlay_traces(primary, overlay, overlay_label="model")
        self.assertIsNone(result.disabled_reason)
        self.assertEqual(len(result.figure.data), 1)


_APP_SCRIPT = (
    "from viz.app.views.run_inspector import render_run_inspector\n"
    "import os\n"
    "render_run_inspector(os.environ['VIZ_TEST_RUNS_ROOT'])\n"
)


@unittest.skipUnless(_HAVE_APP_TEST, "streamlit.testing.v1.AppTest is unavailable")
class SuiteModelToggleAppIntegrationTest(unittest.TestCase):
    """Drives the real Run Inspector Streamlit script headlessly (no browser).

    streamlit.testing.v1.AppTest executes run_inspector.py end to end and
    exposes the resulting checkbox/plotly_chart element tree, so trace names
    and NPZ load counts can be asserted directly instead of only exercising
    the pure helper functions above.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir="/tmp")
        root = Path(self._tmp.name)
        _write_run(root, "mekf", physical=True, model_id="mekf")
        _write_run(root, "kalmannet", physical=False, model_id="kalmannet")
        _write_run(root, "split_knet", physical=False, model_id="split_knet")
        self._prev_env = os.environ.get("VIZ_TEST_RUNS_ROOT")
        os.environ["VIZ_TEST_RUNS_ROOT"] = str(root)
        self._loads: list[str] = []
        self._orig_load_trajectory = loader_mod.VizRun.load_trajectory

        def counted(run_self, *args, **kwargs):
            self._loads.append(str(run_self.meta.get("model_id")))
            return self._orig_load_trajectory(run_self, *args, **kwargs)

        loader_mod.VizRun.load_trajectory = counted

    def tearDown(self) -> None:
        loader_mod.VizRun.load_trajectory = self._orig_load_trajectory
        if self._prev_env is None:
            os.environ.pop("VIZ_TEST_RUNS_ROOT", None)
        else:
            os.environ["VIZ_TEST_RUNS_ROOT"] = self._prev_env
        self._tmp.cleanup()

    @staticmethod
    def _trace_names(at) -> list[list[str | None]]:
        names = []
        for chart in at.get("plotly_chart"):
            spec = json.loads(chart.proto.spec)
            names.append([trace.get("name") for trace in spec.get("data", [])])
        return names

    def _run(self, at) -> None:
        at.run(timeout=60)
        self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])

    @staticmethod
    def _document_order(at) -> list[tuple[str, str]]:
        """Flatten the render tree in document order as (type, value) pairs."""
        flat: list[tuple[str, str]] = []

        def walk(node) -> None:
            node_type = getattr(node, "type", None)
            if node_type:
                try:
                    value = str(node.value)
                except Exception:
                    value = ""
                flat.append((node_type, value))
            for child in getattr(node, "children", {}).values() if hasattr(node, "children") else []:
                walk(child)

        walk(at.main)
        return flat

    def test_models_to_display_renders_before_dataset_summary(self) -> None:
        at = AppTest.from_string(_APP_SCRIPT)
        self._run(at)
        order = self._document_order(at)
        trajectory_view_idx = next(i for i, (t, v) in enumerate(order) if t == "selectbox" and "TrajectoryInfo" in v)
        # Exact match, not substring: the Help & guide popover's own content
        # (rendered right after the title) also mentions "Models to display"
        # in prose, so a substring match could find that instead of the
        # real "**Models to display**" heading further down the page.
        models_idx = next(i for i, (t, v) in enumerate(order) if t == "markdown" and v == "**Models to display**")
        dataset_summary_idx = next(i for i, (t, v) in enumerate(order) if t == "subheader" and "Dataset Summary" in v)
        selected_trajectory_idx = next(i for i, (t, v) in enumerate(order) if t == "subheader" and "Selected Trajectory" in v)
        diagnostics_idx = next(i for i, (t, v) in enumerate(order) if t == "expander")
        self.assertLess(trajectory_view_idx, models_idx)
        self.assertLess(models_idx, dataset_summary_idx)
        self.assertLess(dataset_summary_idx, selected_trajectory_idx)
        self.assertLess(selected_trajectory_idx, diagnostics_idx)
        # Rendered exactly once — no duplicate "Models to display" block.
        models_blocks = [i for i, (t, v) in enumerate(order) if t == "markdown" and v == "**Models to display**"]
        self.assertEqual(len(models_blocks), 1)

    def test_default_selection_is_primary_only_and_others_are_not_loaded(self) -> None:
        at = AppTest.from_string(_APP_SCRIPT)
        self._run(at)
        labels = {c.label: (c.value, c.disabled) for c in at.checkbox}
        # Primary is checked by default but its checkbox is enabled (VIZ-R1.3.1)
        # and its label carries a "· primary" marker (VIZ-R1.3.2 #6): the user
        # may turn it off as long as another model stays selected.
        self.assertEqual(labels["kalmannet · init=fixture · frozen / seed 0 · primary"], (True, False))
        self.assertEqual(labels["mekf · init=fixture · frozen / seed 0"], (False, False))
        self.assertEqual(labels["split_knet · init=fixture · frozen / seed 0"], (False, False))
        self.assertTrue(self._loads)
        self.assertTrue(all(model_id == "kalmannet" for model_id in self._loads))

    def test_primary_off_with_others_on_removes_only_primary_trace(self) -> None:
        at = AppTest.from_string(_APP_SCRIPT)
        self._run(at)
        for checkbox in at.checkbox:
            if "kalmannet" in checkbox.label:
                checkbox.set_value(False)
            else:
                checkbox.set_value(True)
        self._run(at)

        labels = {c.label: c.value for c in at.checkbox}
        self.assertEqual(labels["kalmannet · init=fixture · frozen / seed 0 · primary"], False)
        self.assertEqual(labels["mekf · init=fixture · frozen / seed 0"], True)
        self.assertEqual(labels["split_knet · init=fixture · frozen / seed 0"], True)

        names = self._trace_names(at)
        self.assertFalse(any("kalmannet" in (n or "") for panel in names for n in panel))
        rpy_traces = names[0]
        self.assertTrue(any("mekf" in (n or "") for n in rpy_traces))
        self.assertTrue(any("split_knet" in (n or "") for n in rpy_traces))

        dataset_summary_titles = [s.value for s in at.subheader if "Dataset Summary" in s.value]
        self.assertTrue(dataset_summary_titles)

    def test_last_selected_model_cannot_be_turned_off(self) -> None:
        at = AppTest.from_string(_APP_SCRIPT)
        self._run(at)
        for checkbox in at.checkbox:
            if "kalmannet" in checkbox.label:
                checkbox.set_value(False)
        self._run(at)

        labels = {c.label: c.value for c in at.checkbox}
        self.assertEqual(labels["kalmannet · init=fixture · frozen / seed 0 · primary"], True)
        warnings = [w.value for w in at.warning]
        self.assertTrue(any("At least one model must remain selected" in w for w in warnings))

    def test_toggle_on_adds_traces_and_loads_only_selected_models(self) -> None:
        at = AppTest.from_string(_APP_SCRIPT)
        self._run(at)
        self._loads.clear()
        for checkbox in at.checkbox:
            if not checkbox.disabled:
                checkbox.set_value(True)
        self._run(at)
        self.assertIn("mekf", self._loads)
        self.assertIn("split_knet", self._loads)

        names = self._trace_names(at)
        rpy_traces = names[0]
        self.assertTrue(any("kalmannet" in (n or "") for n in rpy_traces))
        self.assertTrue(any("mekf" in (n or "") for n in rpy_traces))
        self.assertTrue(any("split_knet" in (n or "") for n in rpy_traces))

        consistency_traces = names[4]
        self.assertTrue(any("mekf" in (n or "") and "NEES" in (n or "") for n in consistency_traces))
        self.assertFalse(any("kalmannet" in (n or "") for n in consistency_traces))
        self.assertFalse(any("split_knet" in (n or "") for n in consistency_traces))
        # kalmannet and split_knet both lack physical P, so two models are
        # excluded from this panel: a summary caption plus a "Why?" expander
        # with one detail line per excluded model (VIZ-R1.3.2 #7 / #4).
        captions = [c.value for c in at.caption]
        self.assertTrue(any("not shown in E. NEES" in c for c in captions))
        self.assertTrue(any("split_knet" in c and "NEES" in c for c in captions))
        why_expanders = [e for e in at.expander if e.label == "Why?"]
        self.assertTrue(why_expanders)

    def test_toggle_off_removes_traces_without_reloading(self) -> None:
        at = AppTest.from_string(_APP_SCRIPT)
        self._run(at)
        for checkbox in at.checkbox:
            if not checkbox.disabled:
                checkbox.set_value(True)
        self._run(at)

        for checkbox in at.checkbox:
            if "split_knet" in checkbox.label:
                checkbox.set_value(False)
        self._loads.clear()
        self._run(at)

        self.assertNotIn("split_knet", self._loads)
        names = self._trace_names(at)
        self.assertFalse(any("split_knet" in (n or "") for panel in names for n in panel))
        rpy_traces = names[0]
        self.assertTrue(any("kalmannet" in (n or "") for n in rpy_traces))
        self.assertTrue(any("mekf" in (n or "") for n in rpy_traces))


if __name__ == "__main__":
    unittest.main()
