"""Execute the frozen Step 1 oracle sensor-decomposition diagnostic."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from bench.estimators.mekf import propagate_state
from bench.side_gyro_mag_comp_pilot.data import (
    freeze_train_normalization,
    generate_dataset,
    strip_runtime_normalization,
)
from bench.side_gyro_mag_comp_pilot.model import (
    FEATURE_DIM,
    SideEstimator,
    classical_vector_update,
    mekf_reset_state_digest,
)
from bench.side_gyro_mag_comp_pilot.runner import trajectory_metrics
from bench.side_gyro_mag_comp_pilot.study import (
    REGIME_NAMES,
    ReplayResult,
    _assemble_replay,
    _event_pairs,
    _initial_state,
    load_config,
    state_dict_digest,
)


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 45_173
REGIMES = ("R0", "R1", "R2", "R3")
SEEDS = (31_001, 31_002, 31_003)
SPEC_COMMIT = "c93d764"
PILOT_SPEC_SHA256 = "a10d1de963b27606e924657565716afbd7d85aea516f919a26875dd472c15410"
RETAINED_SHA256 = {
    "PER_TRAJECTORY_RECORDS.jsonl": "da1ff1654fc9a12e024d00fe3ad9ebd6cb53f0c5b00f1fe439f7831c407222ae",
    "PAIRED_COMPARISONS.json": "ffe421232cee164bb87d9d626fd7b3c5c3dde84487fafa425a8c9710909520ad",
    "GATE_RESULTS.json": "16c15db75f45ee0a67cbbd3b6fe8d52485b0383c9b4103a4fa3ed4f93140a401",
}
METRICS = {
    "attitude_geodesic_rmse_rad": "attitude_geodesic_rmse_rad",
    "residual_gyro_bias_rmse": "residual_gyro_bias_rmse_rad_s",
    "corrected_gyro_rate_rmse_rad_s": "corrected_gyro_rate_rmse_rad_s",
    "integrated_gyro_increment_rmse_rad": "integrated_gyro_increment_rmse_rad",
    "corrected_magnetometer_angular_error_rad": "corrected_magnetometer_angular_error_rad",
    "weak_axis_rmse": "weak_axis_rmse_rad",
    "observable_plane_rmse": "observable_plane_rmse_rad",
    "divergence_count": "divergence_count",
}
FOUR_ARM_COMPARISONS = {
    "C1_minus_C0": ("C0", "C1"),
    "N1_minus_N0": ("N0", "N1"),
    "N0_minus_C0": ("C0", "N0"),
    "N1_minus_C1": ("C1", "N1"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _bootstrap_ci(values: np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size != 30 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap input must contain 30 finite trajectory contrasts")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, array.size, size=(BOOTSTRAP_RESAMPLES, array.size))
    statistics = np.mean(array[sampled], axis=1)
    return [float(value) for value in np.percentile(statistics, [2.5, 97.5])]


def _existing_values(
    records: list[dict[str, Any]], variant: str, regime: str, metric: str,
) -> tuple[list[int], np.ndarray]:
    full_regime = REGIME_NAMES[regime]
    source_metric = METRICS[metric]
    selected = [
        record for record in records
        if record["variant_code"] == variant and record["regime"] == full_regime
    ]
    ids = sorted({int(record["trajectory_id"]) for record in selected})
    if len(ids) != 30:
        raise ValueError(f"existing {variant}/{regime} population is incomplete")
    if variant.startswith("C"):
        values = []
        for trajectory_id in ids:
            matches = [record for record in selected if record["trajectory_id"] == trajectory_id]
            if len(matches) != 1 or matches[0]["training_seed"] is not None:
                raise ValueError("classical record must be deterministic and unique")
            values.append(matches[0]["metrics"][source_metric])
        return ids, np.asarray(values, dtype=np.float64)[:, None]
    values = np.empty((30, 3), dtype=np.float64)
    for row, trajectory_id in enumerate(ids):
        for column, seed in enumerate(SEEDS):
            matches = [
                record for record in selected
                if record["trajectory_id"] == trajectory_id and record["training_seed"] == seed
            ]
            if len(matches) != 1:
                raise ValueError("neural seed record must be unique")
            values[row, column] = matches[0]["metrics"][source_metric]
    return ids, values


def reconstruct_four_arm(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "oracle-decomposition-existing-four-arm-v1",
        "source": "committed PER_TRAJECTORY_RECORDS.jsonl; no arm rerun",
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "contrast": "candidate_minus_reference"},
        "comparisons": {},
    }
    for comparison, (reference, candidate) in FOUR_ARM_COMPARISONS.items():
        result["comparisons"][comparison] = {}
        for regime in REGIMES:
            regime_result: dict[str, Any] = {}
            for metric in METRICS:
                ref_ids, ref = _existing_values(records, reference, regime, metric)
                cand_ids, cand = _existing_values(records, candidate, regime, metric)
                if ref_ids != cand_ids:
                    raise ValueError("existing comparison trajectory IDs are not paired")
                ref_cluster = np.mean(ref, axis=1)
                cand_cluster = np.mean(cand, axis=1)
                contrast = cand_cluster - ref_cluster
                seed_details = []
                if reference.startswith("N") or candidate.startswith("N"):
                    for column, seed in enumerate(SEEDS):
                        ref_seed = ref[:, column] if ref.shape[1] == 3 else ref[:, 0]
                        cand_seed = cand[:, column] if cand.shape[1] == 3 else cand[:, 0]
                        mean = float(np.mean(cand_seed - ref_seed))
                        seed_details.append({"seed": seed, "mean_contrast": mean, "direction": "negative" if mean < 0 else "positive" if mean > 0 else "zero"})
                regime_result[metric] = {
                    "reference": reference,
                    "candidate": candidate,
                    "trajectory_count": 30,
                    "mean_reference": float(np.mean(ref_cluster)),
                    "mean_candidate": float(np.mean(cand_cluster)),
                    "mean_contrast": float(np.mean(contrast)),
                    "paired_bootstrap_95_ci": _bootstrap_ci(contrast),
                    "seed_directions": seed_details,
                    "previous_gate_status": "NOT_PREVIOUSLY_GATED" if comparison != "N1_minus_N0" else "G0_GATED_ONLY_ON_R3_ATTITUDE",
                }
            result["comparisons"][comparison][REGIME_NAMES[regime]] = regime_result
    return result


def _classical_intervention(
    dataset: Any, trajectory_id: int, *, oracle_gyro: bool, oracle_mag: bool, variant: str,
) -> ReplayResult:
    trajectory = dataset.sensor[trajectory_id]
    sidecar = dataset.oracle[trajectory_id]
    state, current_time = _initial_state(), 0.0
    initial_digest = mekf_reset_state_digest(state, current_time)
    lineage = hashlib.sha256(b"oracle-decomposition-classical-v1\0")
    lineage.update(trajectory.realization_id.encode() + b"\0")
    q_c = np.diag(np.r_[np.full(3, 1e-8), np.full(3, 1e-12)]).astype(np.float64)
    q_hist, b_hist, gyro_hist, mag_hist, timestamps = [], [], [], [], []
    for index, (gyro, mag) in enumerate(_event_pairs(trajectory)):
        gyro_value = sidecar.gyro_target_B_rad_s[index] if oracle_gyro else gyro.measurement_S
        mag_value = sidecar.mag_target_B[index] if oracle_mag else mag.measurement_S
        state = propagate_state(state, gyro_value, gyro.timestamp_s - current_time, q_c).state
        current_time = gyro.timestamp_s
        state = classical_vector_update(state, mag_value, dataset.m_model_N_onboard)
        lineage.update(f"{variant}/{trajectory_id}/{index}\n".encode())
        q_hist.append(state.q_NB); b_hist.append(state.b_g)
        gyro_hist.append(gyro_value); mag_hist.append(mag_value); timestamps.append(current_time)
    count = len(timestamps)
    return ReplayResult(
        trajectory_id, variant, trajectory.realization_id, np.asarray(timestamps),
        np.stack(q_hist), np.stack(b_hist), np.stack(gyro_hist), np.stack(mag_hist),
        np.zeros((count, FEATURE_DIM)), np.zeros((count, FEATURE_DIM)),
        tuple(("gyro_compensation", "propagation", "mag_compensation", "mag_update") for _ in range(count)),
        initial_digest, trajectory.realization_id, lineage.hexdigest(),
    )


def _neural_intervention(
    dataset: Any, trajectory_id: int, state_dict: Mapping[str, torch.Tensor], normalization: Any,
    *, oracle_gyro: bool, oracle_mag: bool, variant: str,
) -> ReplayResult:
    trajectory = dataset.sensor[trajectory_id]
    sidecar = dataset.oracle[trajectory_id]
    estimator = SideEstimator("raw", feature_enabled=False)
    estimator.load_state_dict(state_dict, strict=True)
    estimator.install_normalization(
        normalization.gyro_mean, normalization.gyro_std,
        normalization.mag_mean, normalization.mag_std,
    )
    estimator.reset_trajectory(
        _initial_state(), 0.0, trajectory_owner_token=trajectory.realization_id,
    )
    steps, timestamps = [], []
    for index, (gyro, mag) in enumerate(_event_pairs(trajectory)):
        gyro_value = sidecar.gyro_target_B_rad_s[index] if oracle_gyro else gyro.measurement_S
        mag_value = sidecar.mag_target_B[index] if oracle_mag else mag.measurement_S
        steps.append(estimator.step_pair(
            gyro_value, mag_value, gyro.timestamp_s, dataset.m_model_N_onboard,
            gyro_valid=gyro.valid, mag_valid=mag.valid,
        ))
        timestamps.append(gyro.timestamp_s)
    runtime_stub = type("RuntimeStub", (), {
        "trajectory_id": trajectory_id,
        "realization_sha256": trajectory.realization_id,
    })()
    return _assemble_replay(runtime_stub, variant, timestamps, steps, estimator)


def _metric_subset(dataset: Any, replay: ReplayResult, threshold: float) -> dict[str, float | int]:
    canonical = trajectory_metrics(dataset, replay, threshold)
    return {name: canonical[source] for name, source in METRICS.items()}


def _new_values(
    records: list[dict[str, Any]], variant: str, regime: str, metric: str,
) -> tuple[list[int], np.ndarray]:
    selected = [record for record in records if record["variant"] == variant and record["regime"] == REGIME_NAMES[regime]]
    ids = sorted({record["trajectory_id"] for record in selected})
    columns = 1 if variant.startswith("C") else 3
    if len(ids) != 30:
        raise ValueError("new intervention population is incomplete")
    values = np.empty((30, columns), dtype=np.float64)
    for row, trajectory_id in enumerate(ids):
        if columns == 1:
            matches = [record for record in selected if record["trajectory_id"] == trajectory_id]
            if len(matches) != 1:
                raise ValueError("classical intervention record must be unique")
            values[row, 0] = matches[0]["metrics"][metric]
        else:
            for column, seed in enumerate(SEEDS):
                matches = [record for record in selected if record["trajectory_id"] == trajectory_id and record["training_seed"] == seed]
                if len(matches) != 1:
                    raise ValueError("neural intervention seed record must be unique")
                values[row, column] = matches[0]["metrics"][metric]
    return ids, values


def _effect_summary(values: np.ndarray) -> dict[str, Any]:
    cluster = np.mean(values, axis=1) if values.ndim == 2 else values
    ci = _bootstrap_ci(cluster)
    result: dict[str, Any] = {
        "mean": float(np.mean(cluster)),
        "paired_bootstrap_95_ci": ci,
        "resolved_positive": bool(ci[0] > 0),
        "resolved_negative": bool(ci[1] < 0),
    }
    if values.ndim == 2 and values.shape[1] == 3:
        result["seed_directions"] = [
            {
                "seed": seed,
                "mean": float(np.mean(values[:, column])),
                "direction": "positive" if np.mean(values[:, column]) > 0 else "negative" if np.mean(values[:, column]) < 0 else "zero",
            }
            for column, seed in enumerate(SEEDS)
        ]
    return result


def summarize_effects(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    arms = {
        "classical_mekf": {"raw": "C0", "gyro": "CG_ORACLE_GYRO_ONLY_MEKF", "mag": "CM_ORACLE_MAG_ONLY_MEKF", "combined": "C1"},
        "fixed_n0_split_knet": {"raw": "N0", "gyro": "NG_INT_ORACLE_GYRO_ONLY_SPLIT_KNET", "mag": "NM_INT_ORACLE_MAG_ONLY_SPLIT_KNET", "combined": "NGM_INT_ORACLE_GYRO_MAG_SPLIT_KNET"},
    }
    for backend, names in arms.items():
        summary[backend] = {}
        for regime in REGIMES:
            metrics: dict[str, Any] = {}
            for metric in METRICS:
                ids, raw = _existing_values(existing, names["raw"], regime, metric)
                if backend == "classical_mekf":
                    gyro_ids, gyro = _new_values(new, names["gyro"], regime, metric)
                    mag_ids, mag = _new_values(new, names["mag"], regime, metric)
                    combined_ids, combined = _existing_values(existing, names["combined"], regime, metric)
                else:
                    gyro_ids, gyro = _new_values(new, names["gyro"], regime, metric)
                    mag_ids, mag = _new_values(new, names["mag"], regime, metric)
                    combined_ids, combined = _new_values(new, names["combined"], regime, metric)
                if ids != gyro_ids or ids != mag_ids or ids != combined_ids:
                    raise ValueError("effect trajectory populations are not paired")
                e_g, e_m, e_gm = raw - gyro, raw - mag, raw - combined
                interaction = e_gm - e_g - e_m
                metrics[metric] = {
                    "raw_mean": float(np.mean(raw)),
                    "gyro_oracle_mean": float(np.mean(gyro)),
                    "mag_oracle_mean": float(np.mean(mag)),
                    "combined_oracle_mean": float(np.mean(combined)),
                    "E_G": _effect_summary(e_g),
                    "E_M": _effect_summary(e_m),
                    "E_GM": _effect_summary(e_gm),
                    "interaction_I": _effect_summary(interaction),
                    "E_M_minus_E_G": _effect_summary(e_m - e_g),
                    "E_G_minus_E_M": _effect_summary(e_g - e_m),
                }
            summary[backend][REGIME_NAMES[regime]] = metrics
    return summary


def diagnostic_conclusion(effects: dict[str, Any]) -> str:
    key = "attitude_geodesic_rmse_rad"
    r3 = REGIME_NAMES["R3"]
    values = [effects[backend][r3][key] for backend in ("classical_mekf", "fixed_n0_split_knet")]
    if all(value["E_M"]["resolved_positive"] and value["E_M_minus_E_G"]["resolved_positive"] for value in values):
        return "MAG_DOMINANT_HEADROOM"
    if all(value["E_G"]["resolved_positive"] and value["E_G_minus_E_M"]["resolved_positive"] for value in values):
        return "GYRO_DOMINANT_HEADROOM"
    if all(value["E_G"]["resolved_positive"] and value["E_M"]["resolved_positive"] for value in values):
        return "GYRO_AND_MAG_HEADROOM"
    if all(
        not value["E_G"]["resolved_positive"] and not value["E_M"]["resolved_positive"]
        and value["E_GM"]["resolved_positive"] and value["interaction_I"]["resolved_positive"]
        for value in values
    ):
        return "COMBINED_INTERACTION_ONLY"
    if all(not value[name]["resolved_positive"] for value in values for name in ("E_G", "E_M", "E_GM")):
        return "NO_RESOLVED_SENSOR_SPECIFIC_HEADROOM"
    return "INCONCLUSIVE_OR_IMPLEMENTATION_BLOCKED"


def _provenance(training_manifest: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    records = []
    for variant in ("N0", "N1"):
        for seed in SEEDS:
            item = next(record for record in training_manifest if record["variant"] == variant and record["training_seed"] == seed)
            checkpoint = root / item["checkpoint_path"]
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if _sha256(checkpoint) != item["checkpoint_sha256"] or payload["variant"] != variant:
                raise ValueError("checkpoint provenance mismatch")
            records.append({
                "variant": variant, "seed": seed, "selected_epoch": item["selected_epoch"],
                "checkpoint_path": item["checkpoint_path"], "checkpoint_file_sha256": item["checkpoint_sha256"],
                "state_dict_sha256": state_dict_digest(payload["state_dict"]),
                "normalization_sha256": item["normalization_sha256"],
                "normalization_source_ids": item["normalization_source_ids"],
                "training_sensor_inputs": "raw_gyro_and_raw_mag" if variant == "N0" else "oracle_gyro_and_oracle_mag",
            })
    return {
        "same_checkpoint": False,
        "separate_checkpoints_trained_on_different_sensor_inputs": True,
        "same_normalization_statistics": len({record["normalization_sha256"] for record in records}) == 1,
        "same_normalization_source_ids": len({tuple(record["normalization_source_ids"]) for record in records}) == 1,
        "records": records,
    }


def _report_four_arm_rows(comparison: dict[str, Any]) -> list[str]:
    rows = ["| Comparison | Regime | Metric | Reference | Candidate | Contrast | 95% CI | Seed mean directions |", "|---|---|---|---:|---:|---:|---|---|"]
    for name, regimes in comparison["comparisons"].items():
        for regime, metrics in regimes.items():
            for metric, value in metrics.items():
                directions = ", ".join(f"{item['seed']}:{item['direction']}" for item in value["seed_directions"]) or "deterministic"
                rows.append(
                    f"| {name} | {regime} | {metric} | {value['mean_reference']:.9g} | {value['mean_candidate']:.9g} | {value['mean_contrast']:+.9g} | [{value['paired_bootstrap_95_ci'][0]:+.9g}, {value['paired_bootstrap_95_ci'][1]:+.9g}] | {directions} |"
                )
    return rows


def _report_effect_rows(effects: dict[str, Any]) -> list[str]:
    rows = ["| Backend | Regime | Metric | E_G | E_M | E_GM | I |", "|---|---|---|---:|---:|---:|---:|"]
    for backend, regimes in effects.items():
        for regime, metrics in regimes.items():
            for metric, value in metrics.items():
                rows.append(
                    f"| {backend} | {regime} | {metric} | {value['E_G']['mean']:+.9g} | {value['E_M']['mean']:+.9g} | {value['E_GM']['mean']:+.9g} | {value['interaction_I']['mean']:+.9g} |"
                )
    return rows


def run(repo_root: Path, output_dir: Path, config_path: Path) -> dict[str, Any]:
    pilot_root = repo_root / "experiments/side_gyro_mag_comp_pilot"
    if _sha256(pilot_root / "PILOT_SPEC.md") != PILOT_SPEC_SHA256:
        raise ValueError("PILOT_SPEC.md changed")
    for name, digest in RETAINED_SHA256.items():
        if _sha256(pilot_root / name) != digest:
            raise ValueError(f"retained pilot artifact changed: {name}")
    existing = [json.loads(line) for line in (pilot_root / "PER_TRAJECTORY_RECORDS.jsonl").read_text().splitlines()]
    training_manifest = json.loads((pilot_root / "TRAINING_MANIFEST.json").read_text())["records"]
    provenance = _provenance(training_manifest, repo_root)
    four_arm = reconstruct_four_arm(existing)
    four_arm["checkpoint_and_normalization_provenance"] = provenance
    _write_json(output_dir / "EXISTING_FOUR_ARM_COMPARISON.json", four_arm)

    config = load_config(config_path)
    dataset = generate_dataset(
        population=config["data"]["method_lock_population_per_regime"],
        sequence_length=int(config["data"]["sequence_length"]), dt_s=float(config["data"]["dt_s"]),
        generation_seed=int(config["data"]["generation_seed"]), split_seed=int(config["data"]["split_seed"]),
    )
    manifest = json.loads((pilot_root / "DATASET_MANIFEST.json").read_text())
    manifest_by_id = {item["trajectory_id"]: item for item in manifest["trajectories"]}
    for trajectory_id, trajectory in dataset.sensor.items():
        if manifest_by_id[trajectory_id]["realization_sha256"] != trajectory.realization_id:
            raise ValueError("regenerated raw realization differs from committed manifest")
    normalization = freeze_train_normalization(dataset)
    if normalization.sha256 != provenance["records"][0]["normalization_sha256"]:
        raise ValueError("regenerated normalization differs from checkpoint provenance")
    runtime_normalization = strip_runtime_normalization(normalization)
    threshold = float(config["evaluation"]["divergence_threshold_rad"])

    n0_states: dict[int, Mapping[str, torch.Tensor]] = {}
    n0_checkpoint_sha: dict[int, str] = {}
    for seed in SEEDS:
        record = next(item for item in provenance["records"] if item["variant"] == "N0" and item["seed"] == seed)
        payload = torch.load(repo_root / record["checkpoint_path"], map_location="cpu", weights_only=False)
        n0_states[seed] = copy.deepcopy(payload["state_dict"])
        n0_checkpoint_sha[seed] = record["checkpoint_file_sha256"]

    new_records: list[dict[str, Any]] = []
    for trajectory_id in dataset.split.test_ids:
        regime = dataset.sensor[trajectory_id].regime
        if regime not in REGIMES:
            continue
        classical_arms = (
            ("CG_ORACLE_GYRO_ONLY_MEKF", True, False),
            ("CM_ORACLE_MAG_ONLY_MEKF", False, True),
        )
        for variant, oracle_gyro, oracle_mag in classical_arms:
            replay = _classical_intervention(
                dataset, trajectory_id, oracle_gyro=oracle_gyro, oracle_mag=oracle_mag, variant=variant,
            )
            new_records.append({
                "record_origin": "new_classical_oracle_decomposition",
                "backend": "classical_mekf", "variant": variant,
                "sensor_intervention": {"oracle_gyro": oracle_gyro, "oracle_magnetometer": oracle_mag},
                "training_seed": None, "trajectory_id": trajectory_id, "regime": REGIME_NAMES[regime],
                "realization_sha256": replay.realization_id, "split": "test", "window": "whole_trajectory",
                "metrics": _metric_subset(dataset, replay, threshold),
            })
        neural_arms = (
            ("NG_INT_ORACLE_GYRO_ONLY_SPLIT_KNET", True, False),
            ("NM_INT_ORACLE_MAG_ONLY_SPLIT_KNET", False, True),
            ("NGM_INT_ORACLE_GYRO_MAG_SPLIT_KNET", True, True),
        )
        for seed in SEEDS:
            for variant, oracle_gyro, oracle_mag in neural_arms:
                replay = _neural_intervention(
                    dataset, trajectory_id, n0_states[seed], runtime_normalization,
                    oracle_gyro=oracle_gyro, oracle_mag=oracle_mag, variant=variant,
                )
                new_records.append({
                    "record_origin": "new_fixed_n0_checkpoint_sensor_intervention",
                    "backend": "fixed_n0_split_knet", "variant": variant,
                    "sensor_intervention": {"oracle_gyro": oracle_gyro, "oracle_magnetometer": oracle_mag},
                    "training_seed": seed, "n0_checkpoint_file_sha256": n0_checkpoint_sha[seed],
                    "trajectory_id": trajectory_id, "regime": REGIME_NAMES[regime],
                    "realization_sha256": replay.realization_id, "split": "test", "window": "whole_trajectory",
                    "metrics": _metric_subset(dataset, replay, threshold),
                })

    if len(new_records) != 1320:
        raise ValueError("unexpected new intervention record count")
    with (output_dir / "ORACLE_DECOMPOSITION_RECORDS.jsonl").open("w") as handle:
        for record in new_records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    # Exact R0 no-op: compare all requested metrics against committed raw records.
    for metric in METRICS:
        _, c0 = _existing_values(existing, "C0", "R0", metric)
        _, n0 = _existing_values(existing, "N0", "R0", metric)
        _, c1 = _existing_values(existing, "C1", "R0", metric)
        for arm in ("CG_ORACLE_GYRO_ONLY_MEKF", "CM_ORACLE_MAG_ONLY_MEKF"):
            _, value = _new_values(new_records, arm, "R0", metric)
            np.testing.assert_array_equal(value, c0)
        np.testing.assert_array_equal(c1, c0)
        for arm in ("NG_INT_ORACLE_GYRO_ONLY_SPLIT_KNET", "NM_INT_ORACLE_MAG_ONLY_SPLIT_KNET", "NGM_INT_ORACLE_GYRO_MAG_SPLIT_KNET"):
            _, value = _new_values(new_records, arm, "R0", metric)
            np.testing.assert_array_equal(value, n0)

    effects = summarize_effects(existing, new_records)
    conclusion = diagnostic_conclusion(effects)
    summary = {
        "schema_version": "oracle-decomposition-summary-v1",
        "diagnostic_conclusion": conclusion,
        "step2_authorized": False,
        "existing_four_arm_source": "committed machine records; no rerun",
        "new_record_count": len(new_records),
        "new_classical_record_count": sum(record["backend"] == "classical_mekf" for record in new_records),
        "new_fixed_n0_record_count": sum(record["backend"] == "fixed_n0_split_knet" for record in new_records),
        "r0_exact_noop": True,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "checkpoint_and_normalization_provenance": provenance,
        "effects": effects,
        "retained_artifact_sha256": RETAINED_SHA256,
    }
    _write_json(output_dir / "ORACLE_DECOMPOSITION_SUMMARY.json", summary)

    report = [
        "# Oracle Compensation Decomposition Report", "",
        f"Diagnostic conclusion: `{conclusion}`", "",
        "This report separates committed four-arm evidence, new classical sensor interventions, fixed-N0-checkpoint neural interventions, and separately trained N1 performance. It does not authorize Step 2.", "",
        "## Checkpoint provenance", "",
        "N0 and N1 use separate checkpoints trained on different sensor inputs for every seed. They share one normalization digest and the same normalization source IDs. N1 is not a fixed-N0 intervention.", "",
        "## Existing committed C0/C1/N0/N1 comparison", "",
        "The following values were reconstructed from committed records; no old arm was rerun. Negative contrast means lower candidate error. C0/C1 and cross-backend comparisons were not previous pilot gates.", "",
        *_report_four_arm_rows(four_arm), "",
        "## Oracle sensor decomposition", "",
        "Positive E values mean lower error after intervention. Positive I means combined improvement exceeds the sum of isolated improvements; negative I indicates overlap or redundancy.", "",
        *_report_effect_rows(effects), "",
        "Full paired confidence intervals and seed directions are in `ORACLE_DECOMPOSITION_SUMMARY.json`.", "",
        "## Scope", "",
        "Classical CG/CM keep the MEKF fixed. NG/NM/NGM use the exact frozen N0 checkpoint for each seed and change only selected sensor values. Existing N1 instead uses separately trained oracle-input checkpoints. No interaction is called causal synergy without an interval excluding zero and these fixed-intervention assumptions.",
    ]
    (output_dir / "ORACLE_DECOMPOSITION_REPORT.md").write_text("\n".join(report) + "\n")

    r3_attitude = {
        backend: effects[backend][REGIME_NAMES["R3"]]["attitude_geodesic_rmse_rad"]
        for backend in effects
    }
    decision = {
        "schema_version": "oracle-decomposition-step01-final-decision-v1",
        "decision": conclusion,
        "step0_verdict": "PASS",
        "step1_verdict": "COMPLETE",
        "step2_authorized": False,
        "n1_provenance": "separately_trained_oracle_input_checkpoints",
        "fixed_n0_intervention_provenance": "exact_committed_N0_checkpoint_per_seed",
        "r3_attitude_effects": r3_attitude,
    }
    _write_json(output_dir / "STEP01_FINAL_DECISION.json", decision)
    final_lines = [
        "# Step 0-1 Final Result", "", "Step 0: `PASS`.", "",
        f"Step 1 diagnostic conclusion: `{conclusion}`.", "",
        "The existing C0/C1/N0/N1 matrix was reconstructed from committed records. New evaluation was limited to CG/CM classical arms and NG/NM/NGM fixed-N0 sensor interventions on R0-R3. N0/N1 were not retrained and G0/G1 were not rerun.", "",
    ]
    for backend, value in r3_attitude.items():
        final_lines.append(
            f"- {backend}: E_G={value['E_G']['mean']:+.9g}, E_M={value['E_M']['mean']:+.9g}, E_GM={value['E_GM']['mean']:+.9g}, I={value['interaction_I']['mean']:+.9g} rad."
        )
    final_lines.extend(["", "Step 2 is not authorized or started."])
    (output_dir / "STEP01_FINAL_RESULT.md").write_text("\n".join(final_lines) + "\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.repo_root.resolve(), args.output_dir.resolve(), args.config.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
