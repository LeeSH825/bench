"""Preset catalog safety, schema descriptor, validation and diff.

The catalog is an allowlist, not a filesystem browser: these tests pin that
property, because the GUI is the first surface that lets a user point the
runner at a config.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from bench.control.config.descriptor import (
    descriptor_document,
    field_by_path,
    supported_paths,
)
from bench.control.config.gui_service import (
    apply_overrides,
    parse_submitted_yaml,
    validate_config,
)
from bench.control.config.presets import (
    MAX_PRESET_DEPTH,
    MAX_PRESET_NODES,
    PresetCatalog,
    PresetError,
    PresetNotFound,
    PresetUnsafe,
    preset_id_for,
    safe_load_preset_text,
)

SMOKE = "suite_train_smoke.yaml"


@pytest.fixture(scope="module")
def catalog() -> PresetCatalog:
    return PresetCatalog()


@pytest.fixture(scope="module")
def smoke(catalog):
    entry = next(e for e in catalog.list() if e.relative_path.endswith(SMOKE))
    return entry, catalog.get(entry.preset_id)[1]


# -- catalog safety ----------------------------------------------------------


def test_catalog_lists_only_tracked_presets(catalog) -> None:
    import subprocess

    entries = catalog.list()
    assert entries, "expected tracked presets"
    tracked = set(subprocess.run(
        ["git", "ls-files", "--", "bench/configs"], cwd=str(catalog.root),
        capture_output=True, text=True).stdout.split())
    for entry in entries:
        assert entry.relative_path in tracked, f"{entry.relative_path} is not tracked"
        assert entry.relative_path.startswith("bench/configs/")


def test_untracked_config_is_not_exposed(catalog, tmp_path) -> None:
    """A file dropped into the config root must not appear in the catalog."""
    intruder = Path(catalog.root) / "bench/configs/_untracked_probe.yaml"
    intruder.write_text("suite: {name: probe}\n", encoding="utf-8")
    try:
        paths = {e.relative_path for e in catalog.list()}
        assert "bench/configs/_untracked_probe.yaml" not in paths
        with pytest.raises(PresetNotFound):
            catalog.resolve_id(preset_id_for("bench/configs/_untracked_probe.yaml"))
    finally:
        intruder.unlink(missing_ok=True)


@pytest.mark.parametrize("hostile", [
    "../../../etc/passwd",
    "/etc/passwd",
    "bench/configs/../../etc/passwd",
    "bench/configs/../../../root/.ssh/id_rsa",
    "",
])
def test_hostile_ids_are_not_resolvable(catalog, hostile: str) -> None:
    """preset_id is opaque: a path-shaped id simply does not match anything."""
    with pytest.raises(PresetNotFound):
        catalog.resolve_id(hostile)
    with pytest.raises(PresetNotFound):
        catalog.resolve_id(preset_id_for(hostile))


def test_preset_id_is_not_a_path(catalog) -> None:
    entries = catalog.list()
    for entry in entries[:5]:
        assert "/" not in entry.preset_id
        assert not os.path.isabs(entry.preset_id)


def test_symlink_escaping_the_root_is_refused(catalog, tmp_path) -> None:
    outside = tmp_path / "secret.yaml"
    outside.write_text("suite: {name: secret}\n", encoding="utf-8")
    link = Path(catalog.root) / "bench/configs/_probe_link.yaml"
    try:
        link.symlink_to(outside)
        # Not tracked, so not listed; and reading it directly is refused.
        assert "bench/configs/_probe_link.yaml" not in {
            e.relative_path for e in catalog.list()}
    finally:
        link.unlink(missing_ok=True)


# -- parser hardening --------------------------------------------------------

def test_custom_yaml_tags_are_refused() -> None:
    with pytest.raises(PresetError):
        safe_load_preset_text("!!python/object/apply:os.system ['echo pwned']\n")


def test_oversized_preset_is_refused() -> None:
    with pytest.raises(PresetUnsafe, match="bytes"):
        safe_load_preset_text("a: " + "x" * (512 * 1024 + 10))


def test_deeply_nested_preset_is_refused() -> None:
    document = "a: 1"
    for _ in range(MAX_PRESET_DEPTH + 5):
        document = "n:\n  " + document.replace("\n", "\n  ")
    with pytest.raises(PresetUnsafe):
        safe_load_preset_text(document)


def test_alias_expansion_bomb_is_refused() -> None:
    """Small on disk, enormous once resolved — must fail as validation."""
    bomb = "a: &a [1,1,1,1,1,1,1,1,1]\nb: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]\n" \
           "c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b]\nd: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c]\n" \
           "e: [*d,*d,*d,*d,*d,*d,*d,*d,*d]\n"
    with pytest.raises(PresetUnsafe):
        safe_load_preset_text(bomb)


def test_non_mapping_preset_is_refused() -> None:
    with pytest.raises(PresetError):
        safe_load_preset_text("- just\n- a list\n")


def test_yaml_syntax_error_reports_line_and_column() -> None:
    with pytest.raises(PresetError) as excinfo:
        safe_load_preset_text("a: [1, 2\nb: 3\n")
    assert "line" in str(excinfo.value) and "column" in str(excinfo.value)


# -- schema descriptor -------------------------------------------------------


def test_descriptor_declares_classification_for_every_field() -> None:
    document = descriptor_document()
    assert document["fields"]
    for field in document["fields"]:
        assert field["classification"] in ("structural", "operational", "identity")
        assert field["label"] and field["path"]


def test_descriptor_only_offers_the_certified_envelope() -> None:
    """The form must not advertise an envelope the backend refuses."""
    assert field_by_path("runtime.device").enum == ("cpu",)
    assert field_by_path("runtime.precision").enum == ("fp32",)
    assert field_by_path("runtime.num_workers").maximum == 0
    assert field_by_path("training.gradient_accumulation_steps").maximum == 1


def test_training_fields_are_conditional() -> None:
    for path in ("training.max_updates", "training.batch_size", "optimizer.learning_rate"):
        assert field_by_path(path).visible_when == "model_trainable"


# -- overrides ---------------------------------------------------------------


def test_unknown_override_path_is_refused(smoke) -> None:
    _entry, text = smoke
    document = safe_load_preset_text(text)
    with pytest.raises(PresetError, match="unsupported override"):
        apply_overrides(document, {"runner.adapter": "evil:Adapter"})
    with pytest.raises(PresetError):
        apply_overrides(document, {"models.0.repo.path": "/etc"})


def test_overrides_do_not_mutate_the_source_document(smoke) -> None:
    _entry, text = smoke
    document = safe_load_preset_text(text)
    before = yaml.safe_dump(document, sort_keys=True)
    apply_overrides(document, {"training.max_updates": 999})
    assert yaml.safe_dump(document, sort_keys=True) == before


def test_original_preset_file_is_never_modified(catalog, smoke) -> None:
    entry, _text = smoke
    path = Path(catalog.root) / entry.relative_path
    before = path.read_bytes()
    validate_config(suite_document=safe_load_preset_text(before.decode()),
                    model_id="kalmannet_tsp", init_id="trained",
                    overrides={"training.max_updates": 3})
    assert path.read_bytes() == before
    assert catalog.digest_of(entry.preset_id) == entry.content_digest


# -- validation and diff -----------------------------------------------------


def test_valid_knet_config_resolves_to_the_certified_path(smoke) -> None:
    _entry, text = smoke
    result = validate_config(suite_document=safe_load_preset_text(text),
                             model_id="kalmannet_tsp", init_id="trained")
    assert result.valid, [i.as_dict() for i in result.issues]
    assert result.training_path_id == "control_resumable_v1"
    assert result.launch_eligibility["eligible"] is True
    assert result.launch_eligibility["stop_resume_available"] is True
    assert result.structural_config_hash and result.variant_id


def test_structural_edit_changes_identity_and_is_reported(smoke) -> None:
    """A budget change must move the hash *and* be listed field-by-field."""
    _entry, text = smoke
    document = safe_load_preset_text(text)
    baseline = validate_config(suite_document=document, model_id="kalmannet_tsp",
                               init_id="trained")
    edited = validate_config(suite_document=document, model_id="kalmannet_tsp",
                             init_id="trained", overrides={"training.max_updates": 25},
                             baseline_document=document)
    assert edited.valid
    assert edited.structural_config_hash != baseline.structural_config_hash
    assert edited.diff["structural_changed"] is True
    assert edited.diff["variant_changed"] is True
    changed = {c["path"]: c for c in edited.diff["changed_fields"]}
    assert "training.max_updates" in changed, edited.diff
    assert changed["training.max_updates"]["after"] == 25
    assert changed["training.max_updates"]["classification"] == "structural"


def test_identical_config_reports_no_change(smoke) -> None:
    _entry, text = smoke
    document = safe_load_preset_text(text)
    result = validate_config(suite_document=document, model_id="kalmannet_tsp",
                             init_id="trained", baseline_document=document)
    assert result.diff["changed_fields"] == []
    assert result.diff["structural_changed"] is False


def test_unknown_model_is_rejected_without_side_effects(smoke) -> None:
    _entry, text = smoke
    result = validate_config(suite_document=safe_load_preset_text(text),
                             model_id="does_not_exist")
    assert not result.valid
    assert result.issues[0].code == "UNKNOWN_MODEL"
    assert result.resolved_run_spec is None


def test_preview_hashes_are_stable_across_repeated_validation(smoke) -> None:
    _entry, text = smoke
    document = safe_load_preset_text(text)
    first = validate_config(suite_document=document, model_id="kalmannet_tsp",
                            init_id="trained", overrides={"training.max_updates": 7})
    second = validate_config(suite_document=document, model_id="kalmannet_tsp",
                             init_id="trained", overrides={"training.max_updates": 7})
    assert first.structural_config_hash == second.structural_config_hash
    assert first.operational_config_hash == second.operational_config_hash
    assert first.variant_id == second.variant_id


def test_uncertified_adapter_is_not_launchable(catalog) -> None:
    """Adaptive/MAML/ME-Split may be visible but must not be launchable."""
    entry = next((e for e in catalog.list()
                  if "adaptive_knet" in e.model_ids), None)
    if entry is None:
        pytest.skip("no preset in this repository declares adaptive_knet")
    result = validate_config(suite_document=safe_load_preset_text(catalog.get(entry.preset_id)[1]),
                             model_id="adaptive_knet", init_id="trained")
    if result.valid:
        assert result.launch_eligibility["eligible"] is False
        assert result.launch_eligibility["reason_code"] == "ADAPTER_NOT_GUI_LAUNCH_CERTIFIED"


def test_form_yaml_round_trip_preserves_the_draft(smoke) -> None:
    """Editing via YAML and via overrides must resolve identically."""
    _entry, text = smoke
    document = safe_load_preset_text(text)
    via_form = validate_config(suite_document=document, model_id="kalmannet_tsp",
                               init_id="trained", overrides={"training.max_updates": 11})
    edited_yaml = apply_overrides(document, {"training.max_updates": 11})
    via_yaml = validate_config(
        suite_document=parse_submitted_yaml(yaml.safe_dump(edited_yaml)),
        model_id="kalmannet_tsp", init_id="trained")
    assert via_form.valid and via_yaml.valid
    assert via_form.structural_config_hash == via_yaml.structural_config_hash
    assert via_form.variant_id == via_yaml.variant_id
