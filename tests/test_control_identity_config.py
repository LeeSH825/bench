"""Identity and typed-config acceptance tests (design doc 06, C-01 … C-07)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

from bench.control.canonical import CanonicalizationError, canonical_json, content_hash
from bench.control.config.compatibility import (
    MODEL_SUPPORTED_KEYS,
    draft_from_suite,
    drafts_from_suite,
)
from bench.control.config.resolver import (
    draft_from_dict,
    resolve_run_spec,
    resolved_from_json,
    validate_draft,
)
from bench.control.config.schema import (
    ConfigValidationError,
    DatasetSection,
    ExperimentSection,
    InitializationSection,
    OptimizerSection,
    ResumeSection,
    RunSpecDraft,
    RuntimeSection,
    SystemSection,
    TelemetrySection,
    TrainingSection,
    UnknownKeyPolicy,
    operational_config_hash,
    structural_config_hash,
)
from bench.control.identity import (
    ExperimentId,
    IdentityError,
    ImplementationId,
    InitId,
    ModelId,
    RunId,
    compute_variant_id,
    uuid7,
    uuid7_timestamp_ms,
    variant_label,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = REPO_ROOT / "bench" / "configs" / "gpu_figure_pack_smoke.yaml"


def make_draft(**overrides) -> RunSpecDraft:
    base = dict(
        experiment=ExperimentSection(experiment_id=ExperimentId.new().value, name="unit-test"),
        model_id=ModelId("kalmannet_tsp"),
        implementation_id=ImplementationId("bench_kalmannet_tsp_adapter_v1"),
        system=SystemSection(task_id="t", scenario_id="s", state_dim=2, observation_dim=2),
        dataset=DatasetSection(dataset_id="ds"),
        training=TrainingSection(enabled=True, max_updates=10, batch_size=4, validation_interval_updates=2),
        runtime=RuntimeSection(device="cpu", seed=0),
    )
    base.update(overrides)
    return RunSpecDraft(**base)


class CanonicalJsonTests(unittest.TestCase):
    def test_key_order_does_not_affect_encoding(self) -> None:
        self.assertEqual(canonical_json({"b": 1, "a": 2}), canonical_json({"a": 2, "b": 1}))

    def test_nan_is_rejected(self) -> None:
        with self.assertRaises(CanonicalizationError):
            canonical_json({"x": float("nan")})

    def test_sets_are_rejected_because_ordering_is_unstable(self) -> None:
        with self.assertRaises(CanonicalizationError):
            canonical_json({"x": {1, 2, 3}})

    def test_float_and_int_are_distinct_inputs(self) -> None:
        self.assertNotEqual(content_hash({"x": 1}), content_hash({"x": 1.0}))


class IdentityTests(unittest.TestCase):
    """C-01 stable identity, C-03 variant collision regression."""

    def test_uuid7_is_unique(self) -> None:
        values = [uuid7() for _ in range(2000)]
        self.assertEqual(len(set(values)), 2000)

    def test_uuid7_timestamps_are_non_decreasing(self) -> None:
        """UUIDv7 orders by embedded millisecond timestamp.

        Within a single millisecond the remaining bits are random, so ids minted
        in the same millisecond have no defined order — only the timestamp
        prefix is monotonic. Asserting full lexicographic ordering would be
        asserting more than the format guarantees.
        """
        stamps = [uuid7_timestamp_ms(uuid7()) for _ in range(500)]
        self.assertEqual(stamps, sorted(stamps))

    def test_uuid7_version_and_variant_bits(self) -> None:
        import uuid as uuid_module

        parsed = uuid_module.UUID(uuid7())
        self.assertEqual(parsed.version, 7)
        self.assertEqual(parsed.variant, uuid_module.RFC_4122)

    def test_same_inputs_give_same_variant_id(self) -> None:
        kwargs = dict(
            model_id=ModelId("split_knet"),
            implementation_id=ImplementationId("bench_split_adapter_v1"),
            init=InitId(mode="trained"),
            structural_config_hash="sha256:" + "a" * 64,
        )
        self.assertEqual(compute_variant_id(**kwargs), compute_variant_id(**kwargs))

    def test_same_model_different_init_is_a_different_variant(self) -> None:
        """The exact collision the audit flagged: init provenance must separate variants."""
        common = dict(
            model_id=ModelId("split_knet"),
            implementation_id=ImplementationId("bench_split_adapter_v1"),
            structural_config_hash="sha256:" + "a" * 64,
        )
        trained = compute_variant_id(init=InitId(mode="trained"), **common)
        untrained = compute_variant_id(init=InitId(mode="untrained"), **common)
        pretrained_a = compute_variant_id(
            init=InitId(mode="pretrained", source_checkpoint_hash="sha256:aa"), **common
        )
        pretrained_b = compute_variant_id(
            init=InitId(mode="pretrained", source_checkpoint_hash="sha256:bb"), **common
        )
        self.assertEqual(len({trained, untrained, pretrained_a, pretrained_b}), 4)

    def test_same_model_different_implementation_is_a_different_variant(self) -> None:
        common = dict(
            model_id=ModelId("split_knet"),
            init=InitId(mode="trained"),
            structural_config_hash="sha256:" + "a" * 64,
        )
        first = compute_variant_id(implementation_id=ImplementationId("impl_v1"), **common)
        second = compute_variant_id(implementation_id=ImplementationId("impl_v2"), **common)
        self.assertNotEqual(first, second)

    def test_variant_id_is_stable_across_process_restart(self) -> None:
        """Python's hash() is salted per process; a persistent id must not be."""
        script = (
            "from bench.control.identity import *;"
            "print(compute_variant_id("
            "model_id=ModelId('split_knet'),"
            "implementation_id=ImplementationId('bench_split_adapter_v1'),"
            "init=InitId(mode='trained'),"
            "structural_config_hash='sha256:'+'a'*64).value)"
        )
        outputs = set()
        for seed in ("0", "1", "12345"):
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
                check=True,
            )
            outputs.add(result.stdout.strip())
        self.assertEqual(len(outputs), 1, f"variant id varied across PYTHONHASHSEED values: {outputs}")

    def test_variant_label_is_presentation_only(self) -> None:
        label = variant_label(
            model_id=ModelId("split_knet"),
            implementation_id=ImplementationId("bench_split_adapter_v1"),
            init=InitId(mode="trained"),
        )
        self.assertIn("split_knet", label)
        # A label is not an identity: it must not be accepted as a VariantId.
        from bench.control.identity import VariantId

        with self.assertRaises(IdentityError):
            VariantId(label)

    def test_malformed_identifiers_are_rejected(self) -> None:
        for bad in ("not-a-uuid", "", "12345"):
            with self.assertRaises(IdentityError):
                RunId(bad)
        with self.assertRaises(IdentityError):
            ModelId("Has Spaces")
        with self.assertRaises(IdentityError):
            InitId(mode="teleported")


class RunAllocationIdentityTests(unittest.TestCase):
    """C-02 unique run allocation."""

    def test_resolving_the_same_draft_twice_gives_distinct_run_ids(self) -> None:
        draft = make_draft()
        first = resolve_run_spec(draft)
        second = resolve_run_spec(draft)
        self.assertNotEqual(first.run_id, second.run_id)
        # ... but the same variant and structural identity: they are comparable runs.
        self.assertEqual(first.variant_id, second.variant_id)
        self.assertEqual(first.structural_config_hash, second.structural_config_hash)


class ConfigValidationTests(unittest.TestCase):
    """C-05 unknown/invalid fields, C-06 structural hash."""

    def test_field_level_errors_are_all_reported_at_once(self) -> None:
        draft = make_draft(
            training=TrainingSection(enabled=True, max_updates=-1, batch_size=0),
            optimizer=OptimizerSection(name="not_an_optimizer", learning_rate=-1.0),
            runtime=RuntimeSection(device="tpu", precision="fp8", seed=0),
            telemetry=TelemetrySection(enabled=True, interval_seconds=0.0),
        )
        issues = validate_draft(draft)
        paths = {issue.path for issue in issues}
        for expected in (
            "training.max_updates",
            "training.batch_size",
            "optimizer.name",
            "optimizer.learning_rate",
            "runtime.device",
            "runtime.precision",
            "telemetry.interval_seconds",
        ):
            self.assertIn(expected, paths)
        for issue in issues:
            self.assertTrue(issue.message)
            self.assertTrue(issue.code)

    def test_cross_field_validation(self) -> None:
        draft = make_draft(training=TrainingSection(enabled=True, max_updates=0))
        codes = {issue.code for issue in validate_draft(draft)}
        self.assertIn("cross_field", codes)

    def test_resume_mode_other_than_none_is_refused(self) -> None:
        """A capability that does not exist must not be accepted in a config."""
        draft = make_draft(resume=ResumeSection(mode="exact"))
        with self.assertRaises(ConfigValidationError) as ctx:
            resolve_run_spec(draft)
        self.assertIn("unsupported_capability", {issue.code for issue in ctx.exception.issues})

    def test_pretrained_init_requires_a_checkpoint_reference(self) -> None:
        draft = make_draft(
            initialization=InitializationSection(mode="pretrained"),
            training=TrainingSection(enabled=False, max_updates=0),
        )
        with self.assertRaises(ConfigValidationError):
            resolve_run_spec(draft)

    def test_structural_hash_ignores_operational_changes(self) -> None:
        base = make_draft()
        operational = make_draft(
            telemetry=TelemetrySection(enabled=False, interval_seconds=30.0),
            experiment=ExperimentSection(experiment_id=base.experiment.experiment_id, name="renamed"),
            runtime=RuntimeSection(device="cuda:0", seed=999),
        )
        self.assertEqual(structural_config_hash(base), structural_config_hash(operational))
        self.assertNotEqual(operational_config_hash(base), operational_config_hash(operational))

    def test_structural_hash_changes_with_structural_fields(self) -> None:
        base = make_draft()
        for changed in (
            make_draft(optimizer=OptimizerSection(name="sgd", learning_rate=1e-3)),
            make_draft(system=SystemSection(task_id="t", scenario_id="s", state_dim=6, observation_dim=2)),
            make_draft(runtime=RuntimeSection(device="cpu", precision="fp64", seed=0)),
            make_draft(model_config_extra={"hidden": 128}),
        ):
            self.assertNotEqual(structural_config_hash(base), structural_config_hash(changed))

    def test_precision_is_structural_even_though_it_looks_operational(self) -> None:
        base = make_draft()
        other = make_draft(runtime=RuntimeSection(device="cpu", precision="fp16", seed=0))
        self.assertNotEqual(structural_config_hash(base), structural_config_hash(other))

    def test_newer_schema_version_is_refused_not_guessed(self) -> None:
        spec = resolve_run_spec(make_draft())
        document = spec.as_dict()
        document["schema_version"] = 999
        with self.assertRaises(ConfigValidationError):
            draft_from_dict(document)


class ConfigRoundTripTests(unittest.TestCase):
    """C-04 config round-trip."""

    def test_resolved_spec_json_round_trip_is_byte_stable(self) -> None:
        spec = resolve_run_spec(make_draft())
        again = resolved_from_json(spec.to_json())
        self.assertEqual(again.to_json(), spec.to_json())
        self.assertEqual(again.run_id, spec.run_id)
        self.assertEqual(again.variant_id, spec.variant_id)

    def test_round_trip_preserves_resolved_semantics(self) -> None:
        spec = resolve_run_spec(make_draft(model_config_extra={"a": 1}, task_config_extra={"b": [1, 2]}))
        again = resolved_from_json(spec.to_json())
        self.assertEqual(structural_config_hash(again.draft), spec.structural_config_hash)
        self.assertEqual(operational_config_hash(again.draft), spec.operational_config_hash)


@unittest.skipUnless(SUITE_PATH.exists(), f"suite fixture missing: {SUITE_PATH}")
class SuiteCompatibilityTests(unittest.TestCase):
    """The compatibility layer must resolve what the existing CLI would run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))

    def test_draft_from_suite_matches_runner_scenario_id(self) -> None:
        from bench.runners.run_suite import _build_scenario_cfg_basis, _canonicalize_scenario_id

        task = self.suite["tasks"][0]
        model = self.suite["models"][1]
        draft = draft_from_suite(self.suite, task=task, model=model, seed=0, track_id="frozen", init_id="trained")
        expected = _canonicalize_scenario_id(
            str(task["task_id"]), _build_scenario_cfg_basis(dict(task), {})
        )
        self.assertEqual(draft.system.scenario_id, expected)

    def test_supported_fields_are_mapped_onto_typed_sections(self) -> None:
        task = self.suite["tasks"][0]
        model = self.suite["models"][1]
        draft = draft_from_suite(self.suite, task=task, model=model, seed=3, track_id="frozen", init_id="trained")
        self.assertEqual(draft.system.state_dim, int(task["x_dim"]))
        self.assertEqual(draft.system.observation_dim, int(task["y_dim"]))
        self.assertEqual(draft.runtime.seed, 3)
        self.assertEqual(draft.optimizer.learning_rate, float(model["lr"]))
        self.assertEqual(draft.training.batch_size, int(model["batch_size"]))
        self.assertEqual(
            draft.training.max_updates, int(self.suite["runner"]["budget"]["train_max_updates"])
        )

    def test_unsupported_keys_are_captured_not_dropped(self) -> None:
        task = self.suite["tasks"][0]
        model = self.suite["models"][1]
        draft = draft_from_suite(self.suite, task=task, model=model, seed=0, track_id="frozen", init_id="trained")
        self.assertTrue(draft.unsupported_fields, "expected some unmodelled suite keys")
        # every reported path must actually survive in the preserved extras
        for path in draft.unsupported_fields:
            section, _, key = path.partition(".")
            source = draft.model_config_extra if section == "model" else draft.task_config_extra
            self.assertIn(key, source, f"{path} was reported unsupported but not preserved")
        # and they participate in the structural hash, because they change results
        import dataclasses

        stripped = dataclasses.replace(draft, model_config_extra={})
        self.assertNotEqual(structural_config_hash(draft), structural_config_hash(stripped))

    def test_error_policy_rejects_unknown_keys(self) -> None:
        with self.assertRaises(ConfigValidationError) as ctx:
            draft_from_suite(
                self.suite,
                task=self.suite["tasks"][0],
                model=self.suite["models"][1],
                seed=0,
                track_id="frozen",
                init_id="trained",
                unknown_key_policy=UnknownKeyPolicy.ERROR,
            )
        self.assertTrue(all(issue.code == "unknown_key" for issue in ctx.exception.issues))

    def test_original_config_survives_the_spec_round_trip(self) -> None:
        """Regression: the worker reads only resolved_run_spec.json.

        The suite executor reconstructs its task/model entries from the attached
        original config, so dropping it from as_dict() made real suite runs fail.
        """
        draft = draft_from_suite(
            self.suite,
            task=self.suite["tasks"][0],
            model=self.suite["models"][1],
            seed=0,
            track_id="frozen",
            init_id="trained",
        )
        spec = resolve_run_spec(draft)
        restored = resolved_from_json(spec.to_json())
        self.assertIsNotNone(restored.draft.original_config)
        self.assertEqual(
            [t["task_id"] for t in restored.draft.original_config["tasks"]],
            [t["task_id"] for t in self.suite["tasks"]],
        )

    def test_suite_expansion_honours_enabled_policy(self) -> None:
        drafts = drafts_from_suite(self.suite, init_id="trained")
        expected = (
            len(self.suite["tasks"])
            * len(self.suite["models"])
            * len(self.suite["runner"]["tracks"])
            * len(self.suite["seeds"])
        )
        self.assertEqual(len(drafts), expected)

    def test_training_enabled_follows_runner_init_semantics(self) -> None:
        """Only init_id='trained' triggers a training phase (audit §6)."""
        task, model = self.suite["tasks"][0], self.suite["models"][1]
        trained = draft_from_suite(self.suite, task=task, model=model, seed=0, track_id="frozen", init_id="trained")
        untrained = draft_from_suite(self.suite, task=task, model=model, seed=0, track_id="frozen", init_id="untrained")
        self.assertTrue(trained.training.enabled)
        self.assertFalse(untrained.training.enabled)


class CapabilityDeclarationTests(unittest.TestCase):
    def test_no_implementation_claims_exact_resume(self) -> None:
        """R-05: an uncertified resume claim is a research-integrity hazard."""
        from bench.control.capabilities import all_capabilities

        for capability in all_capabilities():
            self.assertFalse(
                capability["supports_exact_resume"],
                f"{capability['model_id']} claims exact resume without a parity test",
            )

    def test_paper_fidelity_is_independent_of_trainability(self) -> None:
        from bench.control.capabilities import capabilities_for

        split = capabilities_for("split_knet")
        self.assertTrue(split.trainable)
        self.assertEqual(split.paper_fidelity_status, "partial")
        self.assertIn("alternating", split.paper_fidelity_note.lower())

    def test_aliases_share_an_implementation_id_but_keep_their_model_id(self) -> None:
        from bench.control.capabilities import capabilities_for

        for alias in ("oracle_kf", "nominal_kf", "mb_kf_oracle"):
            capability = capabilities_for(alias)
            self.assertEqual(capability.model_id, alias)
            self.assertEqual(capability.implementation_id, "bench_mb_kf_adapter_v1")

    def test_undeclared_model_gets_a_conservative_declaration(self) -> None:
        from bench.control.capabilities import capabilities_for

        capability = capabilities_for("a_model_that_does_not_exist")
        self.assertFalse(capability.trainable)
        self.assertFalse(capability.supports_exact_resume)
        self.assertEqual(capability.paper_fidelity_status, "unverified")

    def test_every_registered_bench_model_has_a_capability_row(self) -> None:
        """A newly registered adapter must show up as 'undeclared', not vanish."""
        from bench.models.registry import list_model_ids
        from bench.control.capabilities import capabilities_for

        for model_id in list_model_ids():
            capability = capabilities_for(model_id)
            self.assertEqual(capability.model_id, model_id)
            self.assertTrue(capability.implementation_id)


class PathSafetyTests(unittest.TestCase):
    """C-07 safe path."""

    def test_absolute_and_traversal_paths_are_rejected(self) -> None:
        from bench.control.paths import UnsafePathError, safe_relative_path

        base = Path(__file__).resolve().parent
        for bad in ("/etc/passwd", "../../etc/passwd", "a/../../b", "/"):
            with self.assertRaises(UnsafePathError, msg=f"{bad!r} should be rejected"):
                safe_relative_path(base, bad)

    def test_normal_relative_path_is_accepted(self) -> None:
        from bench.control.paths import safe_relative_path

        base = Path(__file__).resolve().parent
        self.assertEqual(safe_relative_path(base, "sub/file.txt"), base / "sub" / "file.txt")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
