from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from viz.app.components.model_toggle_picker import (
    CONTEXT_FIELDS,
    model_context_key,
    reconcile_selection,
    suite_candidates,
    variant_label,
)
from viz.io.writer import write_viz_artifacts

try:
    from streamlit.testing.v1 import AppTest

    _HAVE_APP_TEST = True
except ImportError:  # pragma: no cover - environment without streamlit.testing
    _HAVE_APP_TEST = False

if _HAVE_APP_TEST:
    from tests.test_viz_cross_model_comparison import _diagnostics, _payload


REPO_ROOT = Path(__file__).resolve().parents[1]

_APP_SCRIPT = (
    "from viz.app.views.run_inspector import render_run_inspector\n"
    "import os\n"
    "render_run_inspector(os.environ['VIZ_TEST_RUNS_ROOT'])\n"
)


def _write_variant_run(
    root: Path,
    name: str,
    *,
    model_id: str,
    init_id: str,
    physical: bool = False,
    seed: int = 0,
    track_id: str = "frozen",
) -> Path:
    """Same suite/task/scenario/split/seed/track, only init_id/model_id/physical vary.

    Mirrors `tests.test_viz_cross_model_comparison._write_run` but exposes
    `init_id` — that helper hardcodes `init_id="fixture"`, which cannot
    reproduce the "same model_id, different init_id" or "model-based KF vs
    trained learned filter" scenarios this suite exists to test.
    """
    t, x_true, x_hat, y = _payload()
    b_true = x_true[:, :, 6:9].copy()
    event_flag = np.zeros((x_true.shape[0], x_true.shape[1]), dtype=bool)
    eclipse_flag = np.zeros_like(event_flag)
    event_flag[:, 3:5] = True
    eclipse_flag[:, 7:10] = True
    run_dir = root / name
    write_viz_artifacts(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        suite_name="init_provenance_fixture",
        task_id="attitude_bias_comparison_v0",
        task_family="basilisk_imu_adcs_bias_v0",
        scenario_id="physical_scenario_shared",
        model_id=model_id,
        seed=seed,
        track_id=track_id,
        init_id=init_id,
        run_status="ok",
        time_s=t,
        time_meta={"time_source": "fixture", "time_unit": "s", "dt_s": 0.5},
        x_true=x_true,
        y_obs=y,
        x_hat=x_hat,
        split_extras={
            "gyro_bias_seq": b_true,
            "event_flag_seq": event_flag,
            "eclipse_flag_seq": eclipse_flag,
        },
        diagnostics=_diagnostics(physical=physical),
        adapter_meta={
            "adapter_id": "fixture",
            "gain_semantics": "model_based_kalman_gain" if physical else "learned_kalman_gain",
        },
        data_split="test",
        split_source="explicit",
        trajectory_ids=np.array([0, 5, 10, 15], dtype=np.int64),
        trajectory_id_source="test_split_row_index_fallback",
        k_traj=4,
    )
    return run_dir


class _ManifestOnlyRun:
    """Metadata-only stand-in identical to the one in test_viz_suite_model_toggles.py."""

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


def _meta(
    model_id: str,
    init_id: str,
    *,
    suite: str = "suite-a",
    task: str = "task-a",
    scenario_id: str = "scenario-a",
    seed: int = 0,
    track_id: str = "frozen",
    split: str = "test",
) -> dict:
    return {
        "suite": suite,
        "task": task,
        "scenario_id": scenario_id,
        "model_id": model_id,
        "seed": seed,
        "track_id": track_id,
        "init_id": init_id,
        "data_spec": {"split": split},
    }


class ContextFieldsTest(unittest.TestCase):
    """Candidate filtering: init_id is not a hard-gate field."""

    def test_context_fields_excludes_init_id(self) -> None:
        self.assertNotIn("init_id", CONTEXT_FIELDS)
        self.assertEqual(CONTEXT_FIELDS, ("suite", "task", "scenario_id", "split", "seed", "track_id"))

    def test_context_key_ignores_init_id_difference(self) -> None:
        trained = _meta("kalmannet_tsp", "trained")
        pretrained = _meta("oracle_kf", "pretrained")
        self.assertEqual(model_context_key(trained), model_context_key(pretrained))

    def test_init_id_mismatch_alone_does_not_exclude_candidate(self) -> None:
        primary = _ManifestOnlyRun("/kalmannet", _meta("kalmannet_tsp", "trained"))
        oracle = _ManifestOnlyRun("/oracle", _meta("oracle_kf", "pretrained"))
        candidates = suite_candidates(primary, [oracle], source_trajectory_id=0)
        self.assertEqual({c.run_dir for c in candidates}, {"/kalmannet", "/oracle"})
        self.assertEqual(primary.npz_loads, 0)
        self.assertEqual(oracle.npz_loads, 0)

    def test_trained_split_knet_primary_still_admits_pretrained_oracle_kf(self) -> None:
        primary = _ManifestOnlyRun("/split", _meta("split_knet", "trained"))
        oracle = _ManifestOnlyRun("/oracle", _meta("oracle_kf", "pretrained"))
        candidates = suite_candidates(primary, [oracle], source_trajectory_id=0)
        self.assertEqual({c.model_id for c in candidates}, {"split_knet", "oracle_kf"})

    def test_task_mismatch_still_excludes(self) -> None:
        primary = _ManifestOnlyRun("/a", _meta("kalmannet_tsp", "trained"))
        other_task = _ManifestOnlyRun("/b", _meta("oracle_kf", "pretrained", task="task-b"))
        candidates = suite_candidates(primary, [other_task], source_trajectory_id=0)
        self.assertEqual({c.run_dir for c in candidates}, {"/a"})

    def test_scenario_mismatch_still_excludes(self) -> None:
        primary = _ManifestOnlyRun("/a", _meta("kalmannet_tsp", "trained"))
        other = _ManifestOnlyRun("/b", _meta("oracle_kf", "pretrained", scenario_id="scenario-b"))
        candidates = suite_candidates(primary, [other], source_trajectory_id=0)
        self.assertEqual({c.run_dir for c in candidates}, {"/a"})

    def test_split_mismatch_still_excludes(self) -> None:
        primary = _ManifestOnlyRun("/a", _meta("kalmannet_tsp", "trained"))
        other = _ManifestOnlyRun("/b", _meta("oracle_kf", "pretrained", split="train"))
        candidates = suite_candidates(primary, [other], source_trajectory_id=0)
        self.assertEqual({c.run_dir for c in candidates}, {"/a"})

    def test_seed_mismatch_still_excludes(self) -> None:
        primary = _ManifestOnlyRun("/a", _meta("kalmannet_tsp", "trained"))
        other = _ManifestOnlyRun("/b", _meta("oracle_kf", "pretrained", seed=1))
        candidates = suite_candidates(primary, [other], source_trajectory_id=0)
        self.assertEqual({c.run_dir for c in candidates}, {"/a"})

    def test_track_mismatch_still_excludes(self) -> None:
        primary = _ManifestOnlyRun("/a", _meta("kalmannet_tsp", "trained"))
        other = _ManifestOnlyRun("/b", _meta("oracle_kf", "pretrained", track_id="budgeted"))
        candidates = suite_candidates(primary, [other], source_trajectory_id=0)
        self.assertEqual({c.run_dir for c in candidates}, {"/a"})

    def test_missing_source_id_still_excludes(self) -> None:
        primary = _ManifestOnlyRun("/a", _meta("kalmannet_tsp", "trained"))
        other = _ManifestOnlyRun("/b", _meta("oracle_kf", "pretrained"), source_ids={99})
        candidates = suite_candidates(primary, [other], source_trajectory_id=0)
        self.assertEqual({c.run_dir for c in candidates}, {"/a"})


class VariantIdentityTest(unittest.TestCase):
    """Same model_id, different init_id: distinct candidates, distinct identity."""

    def test_same_model_id_multiple_init_variants_all_present(self) -> None:
        primary = _ManifestOnlyRun("/untrained", _meta("kalmannet_tsp", "untrained"))
        trained = _ManifestOnlyRun("/trained", _meta("kalmannet_tsp", "trained"))
        adapted = _ManifestOnlyRun("/adapted", _meta("kalmannet_tsp", "adapted"))
        candidates = suite_candidates(primary, [trained, adapted], source_trajectory_id=0)
        self.assertEqual({c.run_dir for c in candidates}, {"/untrained", "/trained", "/adapted"})
        # All three share model_id — candidate uniqueness must be run_dir, not model_id.
        self.assertEqual([c.model_id for c in candidates], ["kalmannet_tsp"] * 3)

    def test_candidates_not_deduplicated_by_model_id(self) -> None:
        primary = _ManifestOnlyRun("/a", _meta("kalmannet_tsp", "trained"))
        same_model_other_init = _ManifestOnlyRun("/b", _meta("kalmannet_tsp", "pretrained"))
        candidates = suite_candidates(primary, [same_model_other_init], source_trajectory_id=0)
        self.assertEqual(len(candidates), 2)

    def test_variant_label_disambiguates_same_model_id(self) -> None:
        trained = variant_label(_meta("kalmannet_tsp", "trained"))
        pretrained = variant_label(_meta("kalmannet_tsp", "pretrained"))
        self.assertNotEqual(trained, pretrained)
        self.assertIn("kalmannet_tsp", trained)
        self.assertIn("init=trained", trained)

    def test_candidate_labels_include_init_id_and_stay_distinct(self) -> None:
        primary = _ManifestOnlyRun("/untrained", _meta("kalmannet_tsp", "untrained"))
        trained = _ManifestOnlyRun("/trained", _meta("kalmannet_tsp", "trained"))
        candidates = suite_candidates(primary, [trained], source_trajectory_id=0)
        labels = {c.label for c in candidates}
        self.assertEqual(len(labels), 2)
        self.assertTrue(any("init=untrained" in label for label in labels))
        self.assertTrue(any("init=trained" in label for label in labels))

    def test_legacy_missing_init_id_shows_as_unknown_and_is_not_excluded(self) -> None:
        legacy_meta = _meta("oracle_kf", "trained")
        del legacy_meta["init_id"]
        primary = _ManifestOnlyRun("/a", _meta("kalmannet_tsp", "trained"))
        legacy = _ManifestOnlyRun("/legacy", legacy_meta)
        candidates = suite_candidates(primary, [legacy], source_trajectory_id=0)
        self.assertEqual({c.run_dir for c in candidates}, {"/a", "/legacy"})
        legacy_candidate = next(c for c in candidates if c.run_dir == "/legacy")
        self.assertIn("init=unknown", legacy_candidate.label)

    def test_duplicate_label_gets_short_run_disambiguator(self) -> None:
        # Same model_id, same init_id, same track/seed, different run_dir —
        # the only remaining disambiguator is a short run identifier.
        primary = _ManifestOnlyRun("/ckpt_a", _meta("kalmannet_tsp", "trained"))
        other = _ManifestOnlyRun("/ckpt_b", _meta("kalmannet_tsp", "trained"))
        candidates = suite_candidates(primary, [other], source_trajectory_id=0)
        labels = [c.label for c in candidates]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(any("run=" in label for label in labels))


class SessionContextTest(unittest.TestCase):
    """Model-toggle session-state context hash must not key off init_id."""

    def test_primary_init_variant_change_keeps_same_context_key(self) -> None:
        trained_primary = _meta("kalmannet_tsp", "trained")
        pretrained_primary = _meta("kalmannet_tsp", "pretrained")
        self.assertEqual(model_context_key(trained_primary), model_context_key(pretrained_primary))

    def test_selected_variants_survive_primary_init_change(self) -> None:
        trained_primary = _ManifestOnlyRun("/trained", _meta("kalmannet_tsp", "trained"))
        oracle = _ManifestOnlyRun("/oracle", _meta("oracle_kf", "pretrained"))
        candidates = suite_candidates(trained_primary, [oracle], source_trajectory_id=0)
        selected = reconcile_selection(candidates, primary_run_dir="/trained", previous=["/trained", "/oracle"])
        self.assertEqual(selected, {"/trained", "/oracle"})

        # Primary switches to a different init variant of the same model_id;
        # same suite/task/scenario/split/seed/track -> pool/selection survive.
        pretrained_primary = _ManifestOnlyRun("/pretrained_variant", _meta("kalmannet_tsp", "pretrained"))
        new_candidates = suite_candidates(
            pretrained_primary, [oracle, trained_primary], source_trajectory_id=0
        )
        new_selected = reconcile_selection(
            new_candidates, primary_run_dir="/pretrained_variant", previous=selected
        )
        # /oracle is still a valid run_dir in the same context -> preserved.
        self.assertIn("/oracle", new_selected)

    def test_context_change_still_drops_invalid_selection(self) -> None:
        primary = _ManifestOnlyRun("/a", _meta("kalmannet_tsp", "trained"))
        candidates = suite_candidates(primary, [], source_trajectory_id=0)
        changed = _ManifestOnlyRun("/a", _meta("kalmannet_tsp", "trained", suite="suite-b"))
        changed_candidates = suite_candidates(changed, [], source_trajectory_id=0)
        selected = reconcile_selection(
            changed_candidates, primary_run_dir="/a", previous={"/a", "/stale-from-other-context"}
        )
        self.assertEqual(selected, {"/a"})


@unittest.skipUnless(_HAVE_APP_TEST, "streamlit.testing.v1.AppTest is unavailable")
class ActualScenarioFixtureTest(unittest.TestCase):
    """End-to-end: model-based KF (pretrained) vs. trained learned filters.

    UI/contract verification only — not a claim about model performance.
    """

    def _write_three_model_suite(self, tmp: str) -> None:
        root = Path(tmp)
        _write_variant_run(root, "oracle_kf", model_id="oracle_kf", init_id="pretrained", physical=True)
        _write_variant_run(root, "kalmannet_tsp", model_id="kalmannet_tsp", init_id="trained", physical=False)
        _write_variant_run(root, "split_knet", model_id="split_knet", init_id="trained", physical=False)

    def test_oracle_kf_is_a_candidate_under_trained_kalmannet_primary(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            self._write_three_model_suite(tmp)
            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])
            labels = [c.label for c in at.checkbox]
            self.assertTrue(any("kalmannet_tsp" in l and "init=trained" in l for l in labels))
            self.assertTrue(any("oracle_kf" in l and "init=pretrained" in l for l in labels))
            self.assertTrue(any("split_knet" in l and "init=trained" in l for l in labels))
            # No caption anywhere blames init_id mismatch for an exclusion.
            for c in at.caption:
                self.assertNotIn("init_id mismatch", c.value)

    def test_oracle_kf_togglable_on_and_traces_appear_in_compatible_panels(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            self._write_three_model_suite(tmp)
            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            for checkbox in at.checkbox:
                if "oracle_kf" in checkbox.label:
                    checkbox.set_value(True)
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])

            charts = at.get("plotly_chart")
            names = []
            for chart in charts:
                spec = json.loads(chart.proto.spec)
                names.extend(tr.get("name") for tr in spec.get("data", []))
            self.assertTrue(any("oracle_kf" in (n or "") for n in names))

    def test_provenance_notice_and_table_shown_when_init_differs(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            self._write_three_model_suite(tmp)
            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            for checkbox in at.checkbox:
                if not checkbox.disabled:
                    checkbox.set_value(True)
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])

            warnings = [w.value for w in at.warning]
            self.assertTrue(any("different initialization/training labels" in w for w in warnings))
            self.assertTrue(any("does not" not in w and "not as identical training conditions" in w for w in warnings))

            expanders = [e.label for e in at.expander]
            self.assertIn("Run provenance", expanders)
            markdowns = [m.value for m in at.markdown]
            table = [m for m in markdowns if m.startswith("| Run |")]
            self.assertTrue(table)
            self.assertIn("pretrained", table[0])
            self.assertIn("trained", table[0])
            self.assertIn("oracle_kf", table[0])

    def test_dataset_summary_stays_primary_only_regardless_of_variant_selection(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            self._write_three_model_suite(tmp)
            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            primary_label = next(c.label for c in at.checkbox if c.value and c.disabled is False)
            self.assertTrue(primary_label.startswith("kalmannet_tsp"))
            for checkbox in at.checkbox:
                if not checkbox.disabled:
                    checkbox.set_value(True)
            at.run(timeout=60)
            titles = [s.value for s in at.subheader if s.value.startswith("Dataset Summary")]
            self.assertTrue(titles)
            self.assertIn("kalmannet_tsp", titles[0])
            self.assertNotIn("oracle_kf", titles[0])

    def test_toggle_off_variant_npz_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            self._write_three_model_suite(tmp)
            from viz.io import loader as loader_mod

            loads: list[str] = []
            orig = loader_mod.VizRun.load_trajectory

            def counted(self, *args, **kwargs):
                loads.append(str(self.meta.get("model_id")))
                return orig(self, *args, **kwargs)

            loader_mod.VizRun.load_trajectory = counted
            try:
                at = AppTest.from_string(_APP_SCRIPT)
                os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
                at.run(timeout=60)
                self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])
                # oracle_kf and split_knet stay OFF by default; only the
                # primary (kalmannet_tsp) trajectory should ever be loaded.
                self.assertTrue(loads)
                self.assertTrue(all(model_id == "kalmannet_tsp" for model_id in loads))
            finally:
                loader_mod.VizRun.load_trajectory = orig

    def test_run_index_scan_stays_one_per_rerun_with_more_candidates(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            self._write_three_model_suite(tmp)
            from viz.app.components import overlay_picker
            from viz.app.views import run_inspector as run_inspector_module

            calls: list[str] = []
            orig = overlay_picker.discover_run_index

            def counted(runs_root):
                calls.append(str(runs_root))
                return orig(runs_root)

            overlay_picker.discover_run_index = counted
            run_inspector_module.discover_run_index = counted
            try:
                at = AppTest.from_string(_APP_SCRIPT)
                os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
                at.run(timeout=60)
                self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])
                self.assertEqual(len(calls), 1, calls)

                calls.clear()
                for checkbox in at.checkbox:
                    if not checkbox.disabled:
                        checkbox.set_value(True)
                at.run(timeout=60)
                self.assertEqual(len(calls), 1, calls)
            finally:
                overlay_picker.discover_run_index = orig
                run_inspector_module.discover_run_index = orig


@unittest.skipUnless(_HAVE_APP_TEST, "streamlit.testing.v1.AppTest is unavailable")
class TrainedUntrainedAblationFixtureTest(unittest.TestCase):
    """Same model_id, three init variants, identical evaluation context."""

    def _write_ablation_suite(self, tmp: str) -> None:
        root = Path(tmp)
        _write_variant_run(root, "untrained", model_id="kalmannet_tsp", init_id="untrained")
        _write_variant_run(root, "trained", model_id="kalmannet_tsp", init_id="trained")
        _write_variant_run(root, "adapted", model_id="kalmannet_tsp", init_id="adapted")

    def test_all_three_variants_are_independent_candidates(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            self._write_ablation_suite(tmp)
            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])
            self.assertEqual(len(at.checkbox), 3)
            inits = {"untrained", "trained", "adapted"}
            for expected in inits:
                self.assertTrue(any(f"init={expected}" in c.label for c in at.checkbox))

    def test_trained_and_untrained_selected_together_no_collision(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            self._write_ablation_suite(tmp)
            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            for checkbox in at.checkbox:
                if "init=untrained" in checkbox.label or "init=trained" in checkbox.label:
                    checkbox.set_value(True)
                else:
                    checkbox.set_value(False)
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])

            labels = {c.label: c.value for c in at.checkbox}
            self.assertTrue(any("init=untrained" in l and v for l, v in labels.items()))
            self.assertTrue(any("init=trained" in l and v for l, v in labels.items()))
            self.assertFalse(any("init=adapted" in l and v for l, v in labels.items()))

            charts = at.get("plotly_chart")
            rpy_names = json.loads(charts[0].proto.spec).get("data", [])
            rpy_names = [tr.get("name") for tr in rpy_names]
            self.assertTrue(any("init=trained" in (n or "") for n in rpy_names))
            self.assertTrue(any("init=untrained" in (n or "") for n in rpy_names))
            self.assertFalse(any("init=adapted" in (n or "") for n in rpy_names))

    def test_turning_off_one_variant_removes_only_that_traces(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            self._write_ablation_suite(tmp)
            at = AppTest.from_string(_APP_SCRIPT)
            os.environ["VIZ_TEST_RUNS_ROOT"] = tmp
            at.run(timeout=60)
            for checkbox in at.checkbox:
                checkbox.set_value(True)
            at.run(timeout=60)

            for checkbox in at.checkbox:
                if "init=trained" in checkbox.label and "init=untrained" not in checkbox.label:
                    checkbox.set_value(False)
            at.run(timeout=60)
            self.assertFalse(list(at.exception), msg=[str(e) for e in at.exception])

            charts = at.get("plotly_chart")
            names = []
            for chart in charts:
                spec = json.loads(chart.proto.spec)
                names.extend(tr.get("name") for tr in spec.get("data", []))
            self.assertFalse(any(n and "init=trained" in n and "init=untrained" not in n for n in names))
            self.assertTrue(any("init=untrained" in (n or "") for n in names))
            self.assertTrue(any("init=adapted" in (n or "") for n in names))


if __name__ == "__main__":
    unittest.main()
