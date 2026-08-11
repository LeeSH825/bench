#!/usr/bin/env python3
"""Fail-closed contract and mutation validator for side-gyro-mag-comp-v2."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
STUDY_ID = "side-gyro-mag-comp-v2"
BASE = "235619cbd7b7af7dcc24db89c247673cd72a0363"
ARCHIVE = "9cf80cc85f2a01297cfd7932c1ce3cfcd87a15c0"
BRANCH = "codex/side-gyro-mag-comp-v1-terminal-archive"
V2_PREFIXES = (
    "agent_system/side_gyro_mag_comp_v2/",
    "docs/research/side_gyro_mag_comp_v2/",
    "experiments/side_gyro_mag_comp_v2/",
)
REQUIRED_DESCRIPTOR_FIELDS = {
    "producer", "sample_membership", "trajectory_aggregation", "population",
    "comparison_direction", "split", "record_uniqueness",
    "record_completeness",
}
PRODUCER_FIELDS = {"machine_path", "callable", "output_schema", "value_field"}
UNIQUE = "exactly one record per full record key"
COMPLETE = "complete declared Cartesian population; any missing or duplicate record invalidates the entire dataset"
EXPECTED_G1_METRICS = [
    "gyro_corrected_rate_rmse_rad_s",
    "gyro_corrected_rate_rmse_axis_mean_rad_s",
    "gyro_integrated_increment_error_path_rms_rad",
    "gyro_integrated_increment_error_terminal_rad",
    "mag_corrected_vector_angular_error_mean_rad",
    "mag_corrected_vector_angular_error_rms_rad",
    "attitude_geodesic_rmse_rad",
]
EXPECTED_G1_AGGREGATIONS = [
    "sqrt((1/T)*sum_t ||e_t||^2)",
    "(1/3)*sum_i sqrt((1/T)*sum_t e_t,i^2)",
    "sqrt((1/T)*sum_t ||S_t||^2)",
    "||S_T||",
    "(1/T)*sum_t theta_t",
    "sqrt((1/T)*sum_t theta_t^2)",
    "sqrt(mean(phi_t^2))",
]
EXPECTED_CALLABLES = {
    "attitude_geodesic_rmse_rad": "produce_attitude_geodesic_rmse_rad",
    "gyro_corrected_rate_rmse_rad_s": "produce_gyro_corrected_rate_rmse_rad_s",
    "gyro_corrected_rate_rmse_axis_mean_rad_s": "produce_gyro_corrected_rate_rmse_axis_mean_rad_s",
    "gyro_integrated_increment_error_path_rms_rad": "produce_gyro_integrated_increment_error_path_rms_rad",
    "gyro_integrated_increment_error_terminal_rad": "produce_gyro_integrated_increment_error_terminal_rad",
    "mag_corrected_vector_angular_error_mean_rad": "produce_mag_corrected_vector_angular_error_mean_rad",
    "mag_corrected_vector_angular_error_rms_rad": "produce_mag_corrected_vector_angular_error_rms_rad",
    "association_contrast_T_rad": "produce_association_contrast_T_rad",
    "divergence_flag": "produce_divergence_flag",
    "residual_gyro_bias_rmse_rad_s": "produce_residual_gyro_bias_rmse_rad_s",
    "attitude_geodesic_p95_rad": "produce_attitude_geodesic_p95_rad",
    "gyro_corrected_rate_rmse_rad_s_R4": "produce_gyro_corrected_rate_rmse_rad_s",
    "gyro_integrated_increment_error_path_rms_rad_R4": "produce_gyro_integrated_increment_error_path_rms_rad",
    "magnetic_axis_weak_error_rms_rad": "produce_observability_metrics_from_diagnostic_truth_sidecar",
    "observable_plane_error_rms_rad": "produce_observability_metrics_from_diagnostic_truth_sidecar",
}
EXPECTED_STAGE_ORDER = [
    "CONTRACT", "CONTRACT_INDEPENDENT_AUDIT", "IMPLEMENTATION",
    "IMPLEMENTATION_INDEPENDENT_AUDIT", "ORACLE_HEADROOM_G0",
    "LEARNED_COMPENSATION_G1", "FEATURE_INCREMENT_G2",
    "ASSOCIATION_FALSIFICATION_G3", "NOMINAL_HARMLESSNESS_G4",
    "EVIDENCE_INDEPENDENT_AUDIT", "FINAL_SYNTHESIS",
    "FINAL_INDEPENDENT_AUDIT", "COMPLETE",
]
REQUIRED_EXCLUSIONS = {
    "SNN", "SoW", "reliability_gating", "attention", "Transformer",
    "learned_Q", "learned_R", "uncertainty_head", "extra_runtime_sensors",
    "closed_loop", "FPGA", "automated_sweeps", "broad_comparisons",
    "test_driven_tuning", "new_variant", "new_regime", "new_ablation",
    "rescue_experiment",
}
REQUIRED_INVARIANTS = {
    *(f"INV-R{i}" for i in range(1, 14)),
    "INV-METRIC-SCHEMA", "INV-WEAK-PLANE",
}


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def path_sha(path: Path) -> str:
    return sha(path.read_bytes())


def git(*args: str) -> bytes:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise ContractError(f"git failed: {' '.join(args)}") from exc


def git_paths(commit: str) -> list[str]:
    text = git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).decode()
    return [line for line in text.splitlines() if line]


def validate_archive(c: dict[str, Any], checks: list[str]) -> None:
    a = c.get("archive_provenance", {})
    require(a.get("archive_branch") == BRANCH, "archive branch mismatch")
    require(a.get("base_evidence_commit") == BASE, "base commit mismatch")
    require(a.get("self_contained_archive_commit") == ARCHIVE, "archive commit mismatch")
    require(a.get("self_contained_archive_parent") == BASE, "archive parent mismatch")
    require(git("rev-parse", BRANCH).decode().strip() == ARCHIVE, "archive branch tip mismatch")
    require(git("show", "-s", "--format=%P", ARCHIVE).decode().strip() == BASE, "archive parent is not exact base")
    base_paths = git_paths(BASE)
    require(len(base_paths) == 84, "base does not have exactly 84 paths")
    require(a.get("base_parent_path_count") == 84, "declared base count mismatch")
    require(a.get("base_parent_staged_paths") == base_paths, "base path list mismatch")
    require(git_paths(ARCHIVE) == [".codex/config.toml"], "companion path set mismatch")
    require(a.get("companion_staged_paths") == [".codex/config.toml"], "declared companion path mismatch")
    require(a.get("companion_path_count") == 1 and a.get("archive_total_unique_path_count") == 85, "archive path counts mismatch")
    manifest = json.loads(git("show", f"{ARCHIVE}:{a['final_manifest_path']}"))
    require(manifest.get("file_count") == 9 and len(manifest.get("files", [])) == 9, "final archive manifest is not 9")
    require(a.get("final_manifest_members") == manifest["files"], "archived final member declaration mismatch")
    require(a.get("final_manifest_self_containment") == "9/9_VERIFIED", "9/9 self-containment marker missing")
    for member in manifest["files"]:
        require(sha(git("show", f"{ARCHIVE}:{member['path']}")) == member["sha256"], f"archived member drift: {member['path']}")
    checks.append("archive_identity_84_plus_1_and_9_of_9")


def validate_pins(c: dict[str, Any], checks: list[str]) -> None:
    sources = {
        "experiments/side_gyro_mag_comp/audits/SC_IMPL_ROUND1_COUNTERPROPOSAL.json": "087713e7c05b4166af3bc6e144bb98dac46c713e94c12dc402c07300b9064839",
        "experiments/side_gyro_mag_comp/audits/SC_IMPLEMENTATION_AUDIT.json": "74c0df75107af1a869794ba9310ee98cd64a33053626c31f89ec3d3875685d49",
        "experiments/side_gyro_mag_comp/audits/SC_IMPLEMENTATION_AUDIT_ADDENDUM.md": "e4f74facdce348790b3703b5884d50c879281530b76be14923df08b730e2e9d2",
        "experiments/side_gyro_mag_comp/audits/SC_IMPL_R9_CLARIFICATION.json": "32d5d2d8c3435c6bea8e9ccbb3ab4e32357c4347fc2cbaef65e40b3729ea1dd1",
    }
    immutable = {
        ".codex/config.toml": "315ec7d2282939ea0344b6de5ec5dc2c6dbab3bbee91fa3d1e63912b29a2c20d",
        "bench/estimators/mekf.py": "2b857fe358deeb7af7427d13c79beced4f84f4598014cc8b562d9e7d60fea63e",
        "bench/metrics/mekf.py": "8fd7ea2838a415a60a54e28142293803505f3e214b2c46624ede43f8f89f5b09",
        "bench/tasks/generator/mekf_fusion_events.py": "b79d2f6d7e5f7a92daa2330d9222540d6ebaa3647ae3fb3badf4ece995cd70b4",
    }
    require({x["path"]: x["sha256"] for x in c.get("v1_repair_source_hashes", [])} == sources, "four source pins mismatch")
    require({x["path"]: x["sha256"] for x in c.get("immutable_path_pins", [])} == immutable, "immutable pins mismatch")
    for path, expected in {**sources, **immutable}.items():
        require((ROOT / path).is_file(), f"pinned path missing: {path}")
        require(path_sha(ROOT / path) == expected, f"pinned path drift: {path}")
    checks.append("source_and_immutable_pins")


def validate_source_projections(c: dict[str, Any], checks: list[str]) -> None:
    p = c.get("canonical_source_projections", {})
    counter = json.loads((ROOT / "experiments/side_gyro_mag_comp/audits/SC_IMPL_ROUND1_COUNTERPROPOSAL.json").read_text())
    r9 = json.loads((ROOT / "experiments/side_gyro_mag_comp/audits/SC_IMPL_R9_CLARIFICATION.json").read_text())["authorized_action"]
    require(p.get("required_repairs_source") == "experiments/side_gyro_mag_comp/audits/SC_IMPL_ROUND1_COUNTERPROPOSAL.json#/required_repairs", "repair projection pointer mismatch")
    require(p.get("required_repairs_verbatim") == counter["required_repairs"], "R1-R13 projection is not exact deep equality")
    keys = ["common_quantities", "metric_definitions", "record_and_window_aggregation", "conservative_dominant_rule", "required_red_path"]
    require(p.get("r9_clarification_projection") == {k: r9[k] for k in keys}, "R9 clarification projection is not exact deep equality")
    expected_overrides = [
        {"id":"R8_WEAK_PLANE_ONLY","source_phrase":"replace the degenerate weak-axis population counters with a frozen-threshold definition","effective_replacement":"replace the degenerate weak-axis population counters with the user-authored all-valid observability semantics in /observability_semantics","scope":"this phrase only; every other required_repairs byte remains authoritative"},
        {"id":"R11_NAMESPACE_ONLY","source_path":"experiments/side_gyro_mag_comp/implementation/RED_PATH_MATRIX.json","effective_v2_path":"experiments/side_gyro_mag_comp_v2/implementation/RED_PATH_MATRIX.json","scope":"namespace substitution only; evidence content obligation is unchanged"},
    ]
    require(p.get("application_overrides") == expected_overrides, "source projection overrides exceed exact R8/path substitutions")
    checks.append("exact_r1_r13_and_r9_source_projections")


def validate_method_observability(c: dict[str, Any], checks: list[str]) -> None:
    m = c.get("frozen_method", {})
    require(m.get("runtime_sensors") == ["gyro", "magnetometer"], "sensor set drift")
    require(m.get("feature_dim_per_sensor") == 8, "feature dimension drift")
    require(m.get("encoders") == "separate causal gyro and magnetometer encoders", "separate encoders missing")
    require(m.get("feature_off") == "exact identity gamma=1 beta=0", "FiLM identity missing")
    require(m.get("event_order") == ["gyro_compensation", "propagation", "magnetometer_compensation", "magnetometer_update"], "event order drift")
    boundary = c.get("runtime_information_boundary", {})
    require({"truth", "true bias", "oracle corrections", "event labels", "future samples", "evaluation metrics"} <= set(boundary.get("forbidden_inputs", [])), "runtime boundary incomplete")
    data = c.get("frozen_data_contract", {})
    require(data.get("same_raw_realization") is True, "same raw realization missing")
    require(data.get("training_seeds") == [31001, 31002, 31003], "seed drift")
    require(data.get("smoke_population_per_regime") == {"train":4,"validation":2,"test":4}, "smoke population drift")
    require(data.get("method_lock_population_per_regime") == {"train":40,"validation":10,"test":30}, "method-lock population drift")
    o = c.get("observability_semantics", {})
    exact = {
        "status":"USER_AUTHORED_EFFECTIVE_RULE",
        "posterior_timestamp":"same-timestamp posterior q_hat_plus",
        "attitude_error":"e_theta=Log(inverse(q_hat_NB_plus) tensor_product q_true_NB)",
        "field_direction":"u_m=m_true_B/||m_true_B||",
        "weak_component":"e_weak=u_m^T e_theta",
        "plane_component":"e_plane=(I-u_m u_m^T)e_theta",
        "trajectory_weak_metric":"sqrt(mean(e_weak^2))",
        "trajectory_plane_metric":"sqrt(mean(||e_plane||^2))",
        "membership":"Every valid magnetometer-update sample contributes to both trajectory metrics.",
        "thresholded_subgroups":"FORBIDDEN",
        "geometric_membership_threshold":"FORBIDDEN",
        "population":"all declared test trajectories",
        "truth_boundary":"diagnostic truth sidecar only",
        "metric_semantics":"descriptive_only",
        "decision_firewall":"Never an entry, gate, selection, early-stopping, or stopping input.",
        "dataset_invalidation":"Zero valid magnetometer-update samples, any non-finite value, zero or non-finite true field norm, or a duplicate or missing update invalidates the entire dataset; no trajectory may be dropped.",
    }
    for key, value in exact.items():
        require(o.get(key) == value, f"observability mismatch: {key}")
    checks.append("method_boundary_identity_and_all_valid_observability")


def expected_gate_arithmetic() -> list[dict[str, Any]]:
    return [
        {"aggregate":"unweighted arithmetic mean over all declared trajectory IDs and all three training seeds","reference_denominator":"mean_N0_R3 must be finite and strictly greater than 0","fractional_change_formula":"(mean_N1_R3-mean_N0_R3)/mean_N0_R3","fractional_predicate":"fractional_change <= -0.10","paired_contrast":"N1-N0","ci_predicate":"percentile_95_paired_bootstrap_upper < 0"},
        {"sensor_aggregate":"for each of the six separately addressed metrics, unweighted arithmetic mean over all 30 trajectory IDs and all three training seeds","sensor_predicate":"mean_N2 < mean_N0 for each of all six metrics; conjunction has no CI and no two-of-three rule","sensor_conjunct_metrics":EXPECTED_G1_METRICS[:6],"primary_aggregate":"unweighted arithmetic mean over all 30 R4 trajectory IDs and all three training seeds","primary_reference_denominator":"mean_N0_R4 must be finite and strictly greater than 0","primary_fractional_change_formula":"(mean_N2_R4-mean_N0_R4)/mean_N0_R4","primary_fractional_predicate":"fractional_change <= -0.05","primary_paired_contrast":"N2-N0","primary_ci_predicate":"percentile_95_paired_bootstrap_upper < 0","primary_seed_rule":"strictly negative mean paired difference over the 30 R4 IDs in at least 2 of 3 seeds"},
        {"aggregate":"unweighted arithmetic mean over all 30 R4 trajectory IDs and all three training seeds","reference_denominator":"mean_N2_R4 must be finite and strictly greater than 0","fractional_change_formula":"(mean_N3_R4-mean_N2_R4)/mean_N2_R4","fractional_predicate":"fractional_change <= -0.05","paired_contrast":"N3-N2","ci_predicate":"percentile_95_paired_bootstrap_upper < 0","seed_rule":"strictly negative mean paired difference over the 30 R4 IDs in at least 2 of 3 seeds"},
        {"per_trajectory_formula":"T=RMSE_N3S-0.5*RMSE_N2-0.5*RMSE_N3","weights":{"N3S":1.0,"N2":-0.5,"N3":-0.5},"paired_bootstrap":"cluster by trajectory_id with all three seed values as one block; average within ID over seeds then over sampled IDs","ci_predicate":"percentile_95_paired_bootstrap_lower > 0","ci_crossing_zero":"INCONCLUSIVE_UNDERPOWERED","no_disjunct":True},
        {"rmse_aggregate":"unweighted arithmetic mean over all 30 R0 trajectory IDs and all three training seeds","reference_denominator":"mean_N0_R0 must be finite and strictly greater than 0","ratio_formula":"(mean_N3_R0-mean_N0_R0)/mean_N0_R0","ratio_predicate":"ratio <= 0.03","divergence_formula":"sum_over_R0_IDs(divergence_N3)-sum_over_R0_IDs(divergence_N0)","divergence_predicate":"added_divergence <= 0 in every individual seed","divergence_definition":"1 iff any estimate or per-trajectory metric is non-finite or max_t(phi_t) > 1.0 rad; else 0"},
    ]


def all_descriptors(c: dict[str, Any]) -> list[dict[str, Any]]:
    return [d for g in c.get("gates", []) for d in g.get("metric_descriptors", [])] + c.get("descriptive_metrics", [])


def validate_metrics(c: dict[str, Any], checks: list[str]) -> None:
    schema = c.get("metric_schema", {})
    require(set(schema.get("required_fields", [])) == REQUIRED_DESCRIPTOR_FIELDS, "metric required fields mismatch")
    require(set(schema.get("producer_required_fields", [])) == PRODUCER_FIELDS, "producer fields mismatch")
    require(schema.get("record_key") == "{experiment,split,regime,model,window,metric,seed,trajectory_id}", "record key lacks exact split-aware key")
    require(schema.get("record_window") == "whole_trajectory", "record window drift")
    require(schema.get("point_estimate") == "seed mean", "point estimate drift")
    require(schema.get("paired_bootstrap") == {"cluster":"unique test trajectory_id carrying all three seed values as one block","statistic":"average paired contrast over seeds within ID, then over sampled IDs","resamples":10000,"interval":"percentile 95%","seed":45173,"below_zero":"upper endpoint < 0"}, "bootstrap drift")
    gates = c.get("gates", [])
    require([g.get("id") for g in gates] == ["G0","G1","G2","G3","G4"], "gate order mismatch")
    require([len(g.get("metric_descriptors", [])) for g in gates] == [1,7,1,1,2], "gate descriptor counts mismatch")
    require([d.get("metric") for d in gates[1]["metric_descriptors"]] == EXPECTED_G1_METRICS, "G1 is not six addressable sensor metrics plus R4 primary")
    require([d.get("trajectory_aggregation") for d in gates[1]["metric_descriptors"]] == EXPECTED_G1_AGGREGATIONS, "G1 formula projection mismatch")
    desc_names = {d.get("metric") for d in c.get("descriptive_metrics", [])}
    require(desc_names == {"residual_gyro_bias_rmse_rad_s","attitude_geodesic_p95_rad","gyro_corrected_rate_rmse_rad_s_R4","gyro_integrated_increment_error_path_rms_rad_R4","magnetic_axis_weak_error_rms_rad","observable_plane_error_rms_rad"}, "descriptive metric set mismatch")
    for d in all_descriptors(c):
        name = d.get("metric", "<unnamed>")
        require(REQUIRED_DESCRIPTOR_FIELDS <= d.keys(), f"descriptor field missing: {name}")
        require(isinstance(d.get("producer"), dict) and set(d["producer"]) == PRODUCER_FIELDS, f"producer binding incomplete: {name}")
        require(d["producer"]["machine_path"] == "bench/side_gyro_mag_comp_v2/evaluation.py", f"producer machine path drift: {name}")
        require(d["producer"]["output_schema"] == "side-gyro-mag-comp-v2-metric-record-v1", f"producer schema drift: {name}")
        require(d["producer"]["value_field"] == "value", f"producer value field drift: {name}")
        require(d["producer"]["callable"] == EXPECTED_CALLABLES[name], f"producer callable drift: {name}")
        require(d["split"] == "test", f"descriptor split drift: {name}")
        require(d["record_uniqueness"] == UNIQUE, f"uniqueness drift: {name}")
        require(d["record_completeness"] == COMPLETE, f"completeness drift: {name}")
        has_threshold, has_desc = "threshold" in d, "descriptive_only" in d
        require(has_threshold ^ has_desc, f"threshold/descriptive xor failed: {name}")
        if has_threshold:
            require(isinstance(d["threshold"], dict) and d["threshold"], f"empty threshold: {name}")
        else:
            require(d["descriptive_only"] is True, f"descriptive_only must be true: {name}")
    require([g.get("gate_arithmetic") for g in gates] == expected_gate_arithmetic(), "G0-G4 arithmetic/formula exact-match failed")
    expected_thresholds = [
        [{"fractional_reduction_min":0.10,"paired_bootstrap_ci_upper_strictly_less_than":0}],
        *[],
    ]
    require(gates[0]["metric_descriptors"][0]["threshold"] == expected_thresholds[0][0], "G0 threshold drift")
    require([d["threshold"] for d in gates[1]["metric_descriptors"][:6]] == [{"strict_reduction":True}] * 6, "G1 sensor threshold/conjunction drift")
    primary = {"fractional_reduction_min":0.05,"paired_bootstrap_ci_upper_strictly_less_than":0,"seed_direction_required":"negative mean paired difference in at least 2 of 3 seeds"}
    require(gates[1]["metric_descriptors"][6]["threshold"] == primary, "G1 primary threshold drift")
    require(gates[2]["metric_descriptors"][0]["threshold"] == primary, "G2 threshold drift")
    require(gates[3]["metric_descriptors"][0]["threshold"] == {"paired_bootstrap_ci_lower_strictly_greater_than":0,"weights":[1.0,-0.5,-0.5],"ci_crossing_zero_outcome":"INCONCLUSIVE_UNDERPOWERED"}, "G3 threshold drift")
    require(gates[4]["metric_descriptors"][0]["threshold"] == {"maximum_fractional_increase":0.03}, "G4 ratio drift")
    require(gates[4]["metric_descriptors"][1]["threshold"] == {"maximum_added_divergence":0,"must_hold_in_every_seed":True}, "G4 divergence drift")
    checks.append("split_aware_metric_bindings_and_exact_g0_g4_arithmetic")


def validate_policy(c: dict[str, Any], checks: list[str]) -> None:
    repairs = c.get("repair_obligations", [])
    require([x.get("id") for x in repairs] == [f"R{i}" for i in range(1,14)], "effective R1-R13 missing")
    r8 = repairs[7]
    require("frozen-threshold definition" in r8.get("inherited_v1_change", ""), "R8 inherited phrase missing")
    require("frozen-threshold definition" not in r8.get("effective_v2_change", ""), "obsolete R8 phrase remains effective")
    require("all-valid observability semantics" in r8.get("effective_v2_change", ""), "R8 all-valid override missing")
    invariants = c.get("invariant_registry", [])
    require({x.get("id") for x in invariants} == REQUIRED_INVARIANTS, "invariant registry mismatch")
    require(all(x.get("required_mutations") for x in invariants), "empty invariant mutation list")
    policy = c.get("sequential_stage_policy", {})
    require(policy.get("stage_order") == EXPECTED_STAGE_ORDER, "stage order mismatch")
    require(policy.get("gate_order") == ["G0","G1","G2","G3","G4"], "gate order mismatch")
    require("at most one minimum repair" in policy.get("one_repair_rule", "") and "second failure closes" in policy.get("one_repair_rule", ""), "one-repair rule missing")
    require(policy.get("no_test_tuning") is True, "no-test-tuning missing")
    require(REQUIRED_EXCLUSIONS <= set(c.get("hard_exclusions", [])), "hard exclusions incomplete")
    checks.append("repair_invariants_stage_stop_and_exclusions")


def actual_v2_changed_paths() -> list[str]:
    tracked = git("diff", "--name-only", ARCHIVE, "--", *V2_PREFIXES).decode().splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *V2_PREFIXES).decode().splitlines()
    return sorted(set(filter(None, tracked + untracked)))


def validate_changed_path_list(declared: list[str]) -> None:
    actual = actual_v2_changed_paths()
    require(sorted(declared) == actual, "CHANGED_PATHS does not equal actual v2 git changes from archive baseline")
    require(len(declared) == len(set(declared)), "duplicate changed path")
    require(all(path.startswith(V2_PREFIXES) for path in declared), "changed path outside v2 namespaces")
    forbidden = []
    for path in declared:
        low = path.lower()
        if path.startswith(("bench/", "tests/")) or low.endswith((".yaml", ".yml", ".toml")) or "/model" in low or "/data" in low or "/tests/" in low or "/test_" in low:
            forbidden.append(path)
    require(not forbidden, f"forbidden contract-stage code class: {forbidden}")


def validate_command_access_records(commands: dict[str, Any], validation: dict[str, Any], state: dict[str, Any]) -> None:
    require(commands.get("performance_commands") == [], "performance command recorded")
    require(commands.get("smoke_commands") == [], "smoke command recorded")
    require(commands.get("held_out_or_test_payload_commands") == [], "payload-access command recorded")
    forbidden_terms = ["pytest", "unittest", "smoke run", "performance run", "held-out payload", "test payload", "method-lock experiment"]
    for record in commands.get("commands", []):
        text = json.dumps(record).lower()
        require(not any(term in text for term in forbidden_terms), "command record implies forbidden contract-stage execution")
    require(validation.get("test_payload_accessed") is False, "validation records test payload access")
    require(validation.get("performance_or_smoke_run_executed") is False, "validation records performance/smoke")
    require(state.get("test_payload_accessed") is False, "state records test payload access")
    require(state.get("performance_or_smoke_run_executed") is False, "state records performance/smoke")


def validate_workspace(c: dict[str, Any], checks: list[str]) -> None:
    changed = json.loads((ROOT / "experiments/side_gyro_mag_comp_v2/contract/CHANGED_PATHS.json").read_text())
    require(changed.get("path_count") == len(changed.get("paths", [])), "changed path count mismatch")
    validate_changed_path_list(changed["paths"])
    v1_pathspec = c["archive_provenance"]["base_parent_staged_paths"] + [".codex/config.toml"]
    require(not git("diff", "--name-only", ARCHIVE, "--", *v1_pathspec).decode().strip(), "v1/config drift from archive baseline")
    commands = json.loads((ROOT / "experiments/side_gyro_mag_comp_v2/contract/COMMANDS.json").read_text())
    validation = json.loads((ROOT / "experiments/side_gyro_mag_comp_v2/contract/VALIDATION_RESULT.json").read_text())
    state = json.loads((ROOT / "agent_system/side_gyro_mag_comp_v2/state/STAGE_STATE.json").read_text())
    validate_command_access_records(commands, validation, state)
    require(state.get("current_stage") == "CONTRACT_INDEPENDENT_AUDIT" and state.get("stage_status") == "WAITING_FOR_PEER" and state.get("next_actor") == "CLAUDE", "state handoff fields mismatch")
    require(state.get("repair_round_by_stage", {}).get("CONTRACT") == 1, "CONTRACT repair count must be 1")
    require(not (ROOT / "experiments/side_gyro_mag_comp_v2/handoffs/claude/CLAUDE_TO_CODEX_PREREGISTERED_CRITERIA.json").exists(), "Codex-created artifact remains in Claude-owned handoff path")
    checks.append("actual_changed_paths_v1_drift_and_execution_firewalls")


def validate_contract_data(c: dict[str, Any], workspace: bool = True) -> list[str]:
    checks: list[str] = []
    require(c.get("study_id") == STUDY_ID, "study_id mismatch")
    require(c.get("contract_stage_only") is True, "contract-stage marker missing")
    validate_archive(c, checks)
    validate_pins(c, checks)
    validate_source_projections(c, checks)
    validate_method_observability(c, checks)
    validate_metrics(c, checks)
    validate_policy(c, checks)
    require(c.get("authorized_v2_paths") == ["agent_system/side_gyro_mag_comp_v2/**","docs/research/side_gyro_mag_comp_v2/**","experiments/side_gyro_mag_comp_v2/**"], "authorized namespaces mismatch")
    if workspace:
        validate_workspace(c, checks)
    return checks


def mutate_value(value: Any) -> Any:
    if isinstance(value, bool): return not value
    if isinstance(value, (int, float)): return value + 1
    if isinstance(value, str): return value + "__MUTATED"
    if isinstance(value, list): return value + ["__MUTATED"]
    if isinstance(value, dict): return {**value, "__MUTATED": True}
    return None


def run_mutation_matrix(c: dict[str, Any]) -> dict[str, Any]:
    labels: list[str] = []
    categories: dict[str, int] = {}

    def rejected(label: str, mutated: dict[str, Any], fn: Callable[[dict[str, Any], list[str]], None], category: str) -> None:
        try:
            fn(mutated, [])
        except ContractError:
            labels.append(label)
            categories[category] = categories.get(category, 0) + 1
            return
        raise ContractError(f"mutation survived validator: {label}")

    descriptor_locations: list[tuple[str, int, int | None]] = []
    for gi, gate in enumerate(c["gates"]):
        for di, _ in enumerate(gate["metric_descriptors"]):
            descriptor_locations.append(("gate", gi, di))
    for di, _ in enumerate(c["descriptive_metrics"]):
        descriptor_locations.append(("descriptive", di, None))

    for kind, a, b in descriptor_locations:
        base_name = c["gates"][a]["metric_descriptors"][b]["metric"] if kind == "gate" else c["descriptive_metrics"][a]["metric"]
        for field in sorted(REQUIRED_DESCRIPTOR_FIELDS):
            m = copy.deepcopy(c)
            d = m["gates"][a]["metric_descriptors"][b] if kind == "gate" else m["descriptive_metrics"][a]
            del d[field]
            rejected(f"descriptor.{base_name}.remove.{field}", m, validate_metrics, "descriptor_required_fields")
        for field in sorted(PRODUCER_FIELDS):
            m = copy.deepcopy(c)
            d = m["gates"][a]["metric_descriptors"][b] if kind == "gate" else m["descriptive_metrics"][a]
            del d["producer"][field]
            rejected(f"descriptor.{base_name}.producer.remove.{field}", m, validate_metrics, "producer_fields")
            m = copy.deepcopy(c)
            d = m["gates"][a]["metric_descriptors"][b] if kind == "gate" else m["descriptive_metrics"][a]
            d["producer"][field] = mutate_value(d["producer"][field])
            rejected(f"descriptor.{base_name}.producer.mutate.{field}", m, validate_metrics, "producer_value_bindings")
        for field in ["split", "record_uniqueness", "record_completeness"]:
            m = copy.deepcopy(c)
            d = m["gates"][a]["metric_descriptors"][b] if kind == "gate" else m["descriptive_metrics"][a]
            d[field] = mutate_value(d[field])
            rejected(f"descriptor.{base_name}.mutate.{field}", m, validate_metrics, "descriptor_value_bindings")
        m = copy.deepcopy(c)
        d = m["gates"][a]["metric_descriptors"][b] if kind == "gate" else m["descriptive_metrics"][a]
        if "threshold" in d:
            del d["threshold"]
        else:
            del d["descriptive_only"]
        rejected(f"descriptor.{base_name}.remove.semantics", m, validate_metrics, "exclusive_semantics")
        m = copy.deepcopy(c)
        d = m["gates"][a]["metric_descriptors"][b] if kind == "gate" else m["descriptive_metrics"][a]
        if "threshold" in d: d["descriptive_only"] = True
        else: d["threshold"] = {"invalid": 1}
        rejected(f"descriptor.{base_name}.add.both_semantics", m, validate_metrics, "exclusive_semantics")

    for i, metric in enumerate(EXPECTED_G1_METRICS):
        m = copy.deepcopy(c)
        del m["gates"][1]["metric_descriptors"][i]
        rejected(f"g1.remove_conjunct.{metric}", m, validate_metrics, "g1_six_conjuncts_plus_primary")

    def mutate_leaves(obj: Any, prefix: str, apply: Callable[[list[Any]], None], path: list[Any] | None = None) -> None:
        path = [] if path is None else path
        if isinstance(obj, dict):
            for key, value in obj.items(): mutate_leaves(value, prefix, apply, path + [key])
        else:
            apply(path)

    for gi, gate in enumerate(c["gates"]):
        def arithmetic_mutation(path: list[Any], gi: int = gi) -> None:
            m = copy.deepcopy(c)
            target = m["gates"][gi]["gate_arithmetic"]
            for key in path[:-1]: target = target[key]
            target[path[-1]] = mutate_value(target[path[-1]])
            rejected(f"gate.{gi}.arithmetic.{'.'.join(map(str,path))}", m, validate_metrics, "gate_arithmetic_leaves")
        mutate_leaves(gate["gate_arithmetic"], f"gate.{gi}", arithmetic_mutation)
        for di, d in enumerate(gate["metric_descriptors"]):
            def threshold_mutation(path: list[Any], gi: int = gi, di: int = di) -> None:
                m = copy.deepcopy(c)
                target = m["gates"][gi]["metric_descriptors"][di]["threshold"]
                for key in path[:-1]: target = target[key]
                target[path[-1]] = mutate_value(target[path[-1]])
                rejected(f"gate.{gi}.descriptor.{di}.threshold.{'.'.join(map(str,path))}", m, validate_metrics, "threshold_leaves")
            mutate_leaves(d["threshold"], f"gate.{gi}.descriptor.{di}", threshold_mutation)

    for i in range(13):
        m = copy.deepcopy(c)
        m["canonical_source_projections"]["required_repairs_verbatim"][i]["title"] += "__MUTATED"
        rejected(f"source_projection.R{i+1}", m, validate_source_projections, "r1_r13_source_projection")
    for key in ["common_quantities","metric_definitions","record_and_window_aggregation","conservative_dominant_rule","required_red_path"]:
        m = copy.deepcopy(c)
        del m["canonical_source_projections"]["r9_clarification_projection"][key]
        rejected(f"source_projection.R9.{key}", m, validate_source_projections, "r9_source_projection")
    for key in ["status","posterior_timestamp","attitude_error","field_direction","weak_component","plane_component","trajectory_weak_metric","trajectory_plane_metric","membership","thresholded_subgroups","geometric_membership_threshold","population","truth_boundary","metric_semantics","decision_firewall","dataset_invalidation"]:
        m = copy.deepcopy(c)
        m["observability_semantics"][key] = mutate_value(m["observability_semantics"][key])
        rejected(f"observability.{key}", m, validate_method_observability, "observability_fields")

    declared = json.loads((ROOT / "experiments/side_gyro_mag_comp_v2/contract/CHANGED_PATHS.json").read_text())["paths"]
    path_cases = [declared[:-1], declared + ["outside_v2.txt"], declared + ["tests/side_gyro_mag_comp_v2/test_forbidden.py"]]
    for i, case in enumerate(path_cases):
        try: validate_changed_path_list(case)
        except ContractError:
            labels.append(f"changed_paths.case_{i}"); categories["changed_paths"] = categories.get("changed_paths",0)+1
        else: raise ContractError(f"changed-path mutation survived: {i}")
    commands = json.loads((ROOT / "experiments/side_gyro_mag_comp_v2/contract/COMMANDS.json").read_text())
    validation = json.loads((ROOT / "experiments/side_gyro_mag_comp_v2/contract/VALIDATION_RESULT.json").read_text())
    state = json.loads((ROOT / "agent_system/side_gyro_mag_comp_v2/state/STAGE_STATE.json").read_text())
    access_cases = []
    for field in ["performance_commands","smoke_commands","held_out_or_test_payload_commands"]:
        x = copy.deepcopy(commands); x[field] = [["forbidden"]]; access_cases.append((field,x,validation,state))
    x = copy.deepcopy(commands); x["commands"].append({"purpose":"smoke run","argv":["pytest"]}); access_cases.append(("implied",x,validation,state))
    v = copy.deepcopy(validation); v["test_payload_accessed"] = True; access_cases.append(("payload_boolean",commands,v,state))
    for name, cmd, val, st in access_cases:
        try: validate_command_access_records(cmd,val,st)
        except ContractError:
            labels.append(f"access.{name}"); categories["execution_access_firewall"] = categories.get("execution_access_firewall",0)+1
        else: raise ContractError(f"access mutation survived: {name}")

    label_blob = "\n".join(labels).encode()
    return {
        "mutation_count": len(labels),
        "all_mutations_rejected": True,
        "unrejected_mutations": [],
        "category_counts": categories,
        "mutation_labels_sha256": sha(label_blob),
        "coverage": {
            "descriptor_instances": len(descriptor_locations),
            "required_descriptor_fields_each": len(REQUIRED_DESCRIPTOR_FIELDS),
            "producer_fields_each": len(PRODUCER_FIELDS),
            "g1_addressable_metrics": len(EXPECTED_G1_METRICS),
            "r1_r13_source_projection_rows": 13,
            "r9_projection_sections": 5,
            "observability_fields": 16,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--mutation-matrix", action="store_true")
    args = parser.parse_args()
    path = (ROOT / args.contract).resolve()
    try:
        require(path.is_relative_to(ROOT), "contract outside repository")
        c = json.loads(path.read_text())
        checks = validate_contract_data(c, workspace=True)
        result: dict[str, Any] = {
            "schema_version":"side-gyro-mag-comp-v2-contract-validation-result-v2",
            "study_id":STUDY_ID,
            "decision":"PASS",
            "contract_path":str(path.relative_to(ROOT)),
            "contract_sha256":path_sha(path),
            "checks_passed":checks,
            "check_count":len(checks),
            "errors":[],
        }
        if args.mutation_matrix:
            result["mutation_matrix"] = run_mutation_matrix(c)
    except (ContractError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        result = {"schema_version":"side-gyro-mag-comp-v2-contract-validation-result-v2","study_id":STUDY_ID,"decision":"FAIL","errors":[str(exc)]}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
