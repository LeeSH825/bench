"""Sequential executor for the preregistered gyro/magnetometer pilot.

This is a direct study runner, not an autonomous control plane.  It implements
the one-way G0--G4 stop logic frozen in PILOT_SPEC.md.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bench.metrics.mekf import attitude_geodesic_error_rad, right_local_state_error
from bench.side_gyro_mag_comp_pilot.data import (
    REGIMES,
    freeze_train_normalization,
    generate_dataset,
    strip_runtime_normalization,
    strip_runtime_trajectory,
)
from bench.side_gyro_mag_comp_pilot.model import SideEstimator
from bench.side_gyro_mag_comp_pilot.study import (
    REGIME_NAMES,
    VARIANT_NAMES,
    _classical_replay,
    _event_pairs,
    deployable_replay,
    diagnostic_oracle_replay,
    evaluate_g0_gate,
    evaluate_g1_gate,
    evaluate_g2_gate,
    evaluate_g3_gate,
    evaluate_g4_gate,
    load_config,
    n3s_replay_namespace,
    protected_replay_hashes,
    run_tiny_smoke,
    state_dict_digest,
    train_variant,
    verify_n3s_bridge,
    weak_observable_metrics,
)


SPEC_COMMIT = "a7ebd8247bf00cbca888c08feb6dafa6ce6ebe40"
BASE_COMMIT = "052d2f7217b964b1fa4e80bd643716b433780f08"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _valid_mag_mask(dataset: Any, trajectory_id: int) -> np.ndarray:
    return np.asarray(
        [mag.valid for _, mag in _event_pairs(dataset.sensor[trajectory_id])], dtype=np.bool_,
    )


def trajectory_metrics(dataset: Any, replay: Any, divergence_threshold_rad: float) -> dict[str, Any]:
    """Produce every frozen per-trajectory endpoint from one replay."""

    truth = dataset.truth[replay.trajectory_id]
    oracle = dataset.oracle[replay.trajectory_id]
    attitude = attitude_geodesic_error_rad(
        replay.q_hat_NB.astype(np.float64), truth.q_true_NB.astype(np.float64),
    )
    gyro_error = replay.corrected_gyro_B - oracle.gyro_target_B_rad_s
    dt = np.diff(np.r_[0.0, replay.timestamp_s])
    integrated = np.cumsum(gyro_error * dt[:, None], axis=0)
    corrected_mag = replay.corrected_mag_B
    target_mag = oracle.mag_target_B
    corrected_norm = np.linalg.norm(corrected_mag, axis=1)
    target_norm = np.linalg.norm(target_mag, axis=1)
    if np.any(corrected_norm <= 0) or np.any(target_norm <= 0):
        raise ValueError("magnetometer angular metric received a zero vector")
    left = corrected_mag / corrected_norm[:, None]
    right = target_mag / target_norm[:, None]
    mag_angle = np.arctan2(np.linalg.norm(np.cross(left, right), axis=1), np.sum(left * right, axis=1))
    state_error = right_local_state_error(
        replay.q_hat_NB.astype(np.float64), replay.b_hat_B_rad_s.astype(np.float64),
        truth.q_true_NB.astype(np.float64), truth.residual_bias_B_rad_s.astype(np.float64),
    )
    weak = weak_observable_metrics(
        replay, truth.q_true_NB, truth.residual_bias_B_rad_s, truth.m_true_N,
        _valid_mag_mask(dataset, replay.trajectory_id),
    )
    values = {
        "attitude_geodesic_rmse_rad": float(np.sqrt(np.mean(attitude ** 2))),
        "corrected_gyro_rate_rmse_rad_s": float(np.sqrt(np.mean(np.sum(gyro_error ** 2, axis=1)))),
        "integrated_gyro_increment_rmse_rad": float(np.sqrt(np.mean(np.sum(integrated ** 2, axis=1)))),
        "corrected_magnetometer_angular_error_rad": float(np.mean(mag_angle)),
        "residual_gyro_bias_rmse_rad_s": float(np.sqrt(np.mean(np.sum(state_error.delta_bias_rad_s ** 2, axis=1)))),
        "attitude_p95_rad": float(np.percentile(attitude, 95.0)),
        "divergence_count": int((not np.all(np.isfinite(replay.q_hat_NB))) or np.max(attitude) > divergence_threshold_rad),
        **weak,
    }
    numeric = [value for value in values.values() if isinstance(value, float)]
    if not np.all(np.isfinite(numeric)):
        raise ValueError("a required per-trajectory metric is non-finite")
    return values


def _dataset_manifest(dataset: Any, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "side-gyro-mag-comp-pilot-dataset-v1",
        "spec_commit": SPEC_COMMIT,
        "base_commit": BASE_COMMIT,
        "generation_seed": dataset.split.data_generation_seed,
        "split_seed": dataset.split.split_seed,
        "sequence_length": config["data"]["sequence_length"],
        "dt_s": config["data"]["dt_s"],
        "whole_trajectory_split": {
            "train_ids": list(dataset.split.train_ids),
            "validation_ids": list(dataset.split.validation_ids),
            "test_ids": list(dataset.split.test_ids),
        },
        "trajectories": [
            {
                "trajectory_id": item,
                "regime": REGIME_NAMES[dataset.sensor[item].regime],
                "split": (
                    "train" if item in dataset.split.train_ids else
                    "validation" if item in dataset.split.validation_ids else "test"
                ),
                "realization_sha256": dataset.sensor[item].realization_id,
                "valid_magnetometer_samples": int(np.sum(_valid_mag_mask(dataset, item))),
            }
            for item in sorted(dataset.sensor)
        ],
        "same_realization_key": "trajectory_id+realization_sha256",
        "r4_train_or_validation_count": sum(
            dataset.sensor[item].regime == "R4"
            for item in dataset.split.train_ids + dataset.split.validation_ids
        ),
    }


class PilotRun:
    def __init__(self, config_path: Path, output_dir: Path) -> None:
        self.config_path = config_path
        self.output_dir = output_dir
        self.config = load_config(config_path)
        self.records: list[dict[str, Any]] = []
        self.training: list[dict[str, Any]] = []
        self.comparisons: dict[str, Any] = {}
        self.gates: dict[str, Any] = {
            name: {"status": "NOT_RUN", "authorized_next": False} for name in ("G0", "G1", "G2", "G3", "G4")
        }

    def _flush(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "PER_TRAJECTORY_RECORDS.jsonl").open("w") as handle:
            for record in self.records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        _write_json(self.output_dir / "TRAINING_MANIFEST.json", {
            "schema_version": "side-gyro-mag-comp-pilot-training-v1",
            "spec_commit": SPEC_COMMIT,
            "records": self.training,
        })
        _write_json(self.output_dir / "PAIRED_COMPARISONS.json", self.comparisons)
        _write_json(self.output_dir / "GATE_RESULTS.json", self.gates)

    def _append_replay(self, dataset: Any, replay: Any, variant: str, seed: int | None) -> None:
        trajectory = dataset.sensor[replay.trajectory_id]
        if replay.realization_id != trajectory.realization_id:
            raise ValueError("replay raw-realization provenance changed")
        self.records.append({
            "trajectory_id": replay.trajectory_id,
            "regime": REGIME_NAMES[trajectory.regime],
            "variant": VARIANT_NAMES[variant],
            "variant_code": variant,
            "training_seed": seed,
            "realization_sha256": replay.realization_id,
            "window": "whole_trajectory",
            "metrics": trajectory_metrics(
                dataset, replay, float(self.config["evaluation"]["divergence_threshold_rad"]),
            ),
        })

    def _train_and_evaluate(
        self,
        dataset: Any,
        normalization: Any,
        runtime_normalization: Any,
        runtime_by_id: dict[int, Any],
        variant: str,
    ) -> dict[int, tuple[SideEstimator, dict[str, torch.Tensor], str]]:
        trained: dict[int, tuple[SideEstimator, dict[str, torch.Tensor], str]] = {}
        for seed in self.config["training"]["seeds"]:
            checkpoint = self.output_dir / "checkpoints" / variant / f"seed_{seed}.pt"
            estimator, result = train_variant(
                dataset, variant, normalization, self.config, int(seed), checkpoint, smoke=False,
            )
            self.training.append(asdict(result))
            state = copy.deepcopy(estimator.state_dict())
            trained[int(seed)] = (estimator, state, result.checkpoint_sha256)
            for trajectory_id in dataset.split.test_ids:
                if variant == "N1":
                    current = SideEstimator("raw", feature_enabled=False)
                    current.load_state_dict(state, strict=True)
                    replay = diagnostic_oracle_replay(dataset, trajectory_id, "N1", current)
                else:
                    current = SideEstimator(
                        "learned" if variant in ("N2", "N3") else "raw",
                        feature_enabled=(variant == "N3"),
                    )
                    current.load_state_dict(state, strict=True)
                    replay = deployable_replay(
                        runtime_by_id[trajectory_id], current, runtime_normalization,
                        dataset.m_model_N_onboard, variant=variant,
                    )
                self._append_replay(dataset, replay, variant, int(seed))
            self._flush()
        return trained

    def _matrix(self, variant: str, regime: str, metric: str) -> np.ndarray:
        ids = sorted(
            record["trajectory_id"] for record in self.records
            if record["variant_code"] == variant and record["regime"] == REGIME_NAMES[regime]
            and record["training_seed"] == self.config["training"]["seeds"][0]
        )
        if len(ids) != self.config["data"]["method_lock_population_per_regime"]["test"]:
            raise ValueError(f"incomplete paired population for {variant}/{regime}/{metric}")
        rows = []
        for trajectory_id in ids:
            row = []
            for seed in self.config["training"]["seeds"]:
                matches = [
                    record for record in self.records
                    if record["variant_code"] == variant and record["regime"] == REGIME_NAMES[regime]
                    and record["trajectory_id"] == trajectory_id and record["training_seed"] == seed
                ]
                if len(matches) != 1:
                    raise ValueError("paired metric population has a missing or duplicate member")
                row.append(matches[0]["metrics"][metric])
            rows.append(row)
        value = np.asarray(rows, dtype=np.float64)
        if value.shape != (len(ids), 3) or not np.all(np.isfinite(value)):
            raise ValueError("paired metric matrix is invalid")
        return value

    def _record_comparison(self, gate: str, arrays: dict[str, np.ndarray], result: dict[str, Any]) -> None:
        self.comparisons[gate] = {
            "cluster_unit": "trajectory_id_with_three_training_seed_values",
            "bootstrap_resamples": self.config["evaluation"]["bootstrap_resamples"],
            "bootstrap_seed": self.config["evaluation"]["bootstrap_seed"],
            "arrays": {name: value.tolist() for name, value in arrays.items()},
            "result": result,
        }

    def _finalize(self, decision: str, reason: str) -> dict[str, Any]:
        self._flush()
        payload = {
            "schema_version": "side-gyro-mag-comp-pilot-final-decision-v1",
            "decision": decision,
            "reason": reason,
            "spec_commit": SPEC_COMMIT,
            "base_commit": BASE_COMMIT,
            "gates": self.gates,
            "performance_claim_scope": "this preregistered synthetic pilot only",
            "covariance_claim_valid": False,
        }
        _write_json(self.output_dir / "FINAL_DECISION.json", payload)
        lines = [
            "# Final Gyro-Magnetometer Compensation Pilot Result", "",
            f"Final decision: `{decision}`", "", reason, "", "## Gate decisions", "",
        ]
        for gate in ("G0", "G1", "G2", "G3", "G4"):
            lines.append(f"- {gate}: `{self.gates[gate]['status']}`")
        lines.extend([
            "", "## Interpretation", "",
            "This is a bounded result for the frozen synthetic gyro/magnetometer pilot. It does not establish calibrated covariance, flight performance, hardware efficiency, or generality beyond the declared regimes.",
            "", "All metrics use whole held-out trajectories and identical raw realizations within paired comparisons. Weak-axis and observable-plane results are descriptive only.",
        ])
        (self.output_dir / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n")
        return payload

    def run(self) -> dict[str, Any]:
        population = self.config["data"]["method_lock_population_per_regime"]
        dataset = generate_dataset(
            population=population,
            sequence_length=int(self.config["data"]["sequence_length"]),
            dt_s=float(self.config["data"]["dt_s"]),
            generation_seed=int(self.config["data"]["generation_seed"]),
            split_seed=int(self.config["data"]["split_seed"]),
        )
        _write_json(self.output_dir / "DATASET_MANIFEST.json", _dataset_manifest(dataset, self.config))
        (self.output_dir / "PILOT_SPEC_COMMIT.txt").write_text(SPEC_COMMIT + "\n")
        normalization = freeze_train_normalization(dataset)
        runtime_normalization = strip_runtime_normalization(normalization)
        runtime_by_id = {
            item: strip_runtime_trajectory(trajectory) for item, trajectory in dataset.sensor.items()
        }

        # Deterministic classical references are evaluated once; they do not enter a gate.
        for trajectory_id in dataset.split.test_ids:
            self._append_replay(dataset, _classical_replay(dataset, trajectory_id, oracle_enabled=False), "C0", None)
            self._append_replay(dataset, _classical_replay(dataset, trajectory_id, oracle_enabled=True), "C1", None)
        self._flush()

        self._train_and_evaluate(dataset, normalization, runtime_normalization, runtime_by_id, "N0")
        self._train_and_evaluate(dataset, normalization, runtime_normalization, runtime_by_id, "N1")
        g0_arrays = {
            "N1_R3_attitude": self._matrix("N1", "R3", "attitude_geodesic_rmse_rad"),
            "N0_R3_attitude": self._matrix("N0", "R3", "attitude_geodesic_rmse_rad"),
        }
        g0 = evaluate_g0_gate(g0_arrays["N1_R3_attitude"], g0_arrays["N0_R3_attitude"])
        self._record_comparison("G0", g0_arrays, g0)
        self.gates["G0"] = {"status": "PASS" if g0["passed"] else "FAIL", "authorized_next": bool(g0["passed"]), **g0}
        self._flush()
        if not g0["passed"]:
            return self._finalize("STOP_NO_COMPENSATION_HEADROOM", "G0 failed; learned-compensation and feature experiments were not run.")

        self._train_and_evaluate(dataset, normalization, runtime_normalization, runtime_by_id, "N2")
        g1_arrays = {
            "N2_R1_gyro_rate": self._matrix("N2", "R1", "corrected_gyro_rate_rmse_rad_s"),
            "N0_R1_gyro_rate": self._matrix("N0", "R1", "corrected_gyro_rate_rmse_rad_s"),
            "N2_R1_increment": self._matrix("N2", "R1", "integrated_gyro_increment_rmse_rad"),
            "N0_R1_increment": self._matrix("N0", "R1", "integrated_gyro_increment_rmse_rad"),
            "N2_R2_mag_angle": self._matrix("N2", "R2", "corrected_magnetometer_angular_error_rad"),
            "N0_R2_mag_angle": self._matrix("N0", "R2", "corrected_magnetometer_angular_error_rad"),
            "N2_R3_attitude": self._matrix("N2", "R3", "attitude_geodesic_rmse_rad"),
            "N0_R3_attitude": self._matrix("N0", "R3", "attitude_geodesic_rmse_rad"),
        }
        g1 = evaluate_g1_gate(*g1_arrays.values())
        self._record_comparison("G1", g1_arrays, g1)
        self.gates["G1"] = {"status": "PASS" if g1["passed"] else "FAIL", "authorized_next": bool(g1["passed"]), **g1}
        self._flush()
        if not g1["passed"]:
            return self._finalize("REJECT_LEARNED_COMPENSATION", "G1 failed; the feature claim was not tested.")

        n3_trained = self._train_and_evaluate(dataset, normalization, runtime_normalization, runtime_by_id, "N3")
        g2_arrays = {
            "N3_R4_attitude": self._matrix("N3", "R4", "attitude_geodesic_rmse_rad"),
            "N2_R4_attitude": self._matrix("N2", "R4", "attitude_geodesic_rmse_rad"),
        }
        g2 = evaluate_g2_gate(g2_arrays["N3_R4_attitude"], g2_arrays["N2_R4_attitude"])
        self._record_comparison("G2", g2_arrays, g2)
        self.gates["G2"] = {"status": "PASS" if g2["passed"] else "FAIL", "authorized_next": bool(g2["passed"]), **g2}
        self._flush()
        if not g2["passed"]:
            return self._finalize("LOCK_COMPENSATION_ONLY_REJECT_FEATURE_PATH", "G2 failed; G3 and G4 were not run because the feature-increment premise failed.")

        r4_ids = tuple(item for item in dataset.split.test_ids if dataset.sensor[item].regime == "R4")
        n3s_evidence: list[dict[str, Any]] = []
        for seed in self.config["training"]["seeds"]:
            _, n3_state, checkpoint_sha = n3_trained[int(seed)]
            for trajectory_id in r4_ids:
                # Run the paired N3 reference first so its post-replay state digest is
                # independent evidence for the N3S checkpoint identity check.
                reference_estimator = SideEstimator("learned", feature_enabled=True)
                reference_estimator.load_state_dict(n3_state, strict=True)
                n3_reference_replay = deployable_replay(
                    runtime_by_id[trajectory_id], reference_estimator, runtime_normalization,
                    dataset.m_model_N_onboard, variant="N3",
                )
                replay, evidence = n3s_replay_namespace(
                    runtime_by_id, r4_ids, REGIMES.index("R4"), dataset.m_model_N_onboard,
                    trajectory_id, int(seed), n3_state, checkpoint_sha,
                    state_dict_digest(reference_estimator.state_dict()), runtime_normalization,
                )
                own = [
                    record for record in self.records
                    if record["variant_code"] == "N3" and record["training_seed"] == seed
                    and record["trajectory_id"] == trajectory_id
                ]
                if len(own) != 1:
                    raise ValueError("N3S could not locate the paired N3 record")
                n3_hashes = protected_replay_hashes(
                    runtime_by_id[trajectory_id], n3_reference_replay,
                )
                n3s_hashes = protected_replay_hashes(runtime_by_id[trajectory_id], replay)
                verify_n3s_bridge(
                    n3_hashes, n3s_hashes, evidence,
                    n3s_recurrent_owner_token=replay.recurrent_history_owner_token,
                    n3s_recurrent_history_sha256=replay.recurrent_history_provenance_sha256,
                )
                if evidence["n3_state_dict_sha256"] != state_dict_digest(n3_state):
                    raise ValueError("N3S did not use the exact N3 state dictionary")
                evidence["n3_protected_hashes"] = n3_hashes
                evidence["n3s_protected_hashes"] = n3s_hashes
                n3s_evidence.append(evidence)
                self._append_replay(dataset, replay, "N3S", int(seed))
        _write_json(self.output_dir / "N3S_ASSOCIATION_EVIDENCE.json", n3s_evidence)
        g3_arrays = {
            "N2_R4_attitude": self._matrix("N2", "R4", "attitude_geodesic_rmse_rad"),
            "N3_R4_attitude": self._matrix("N3", "R4", "attitude_geodesic_rmse_rad"),
            "N3S_R4_attitude": self._matrix("N3S", "R4", "attitude_geodesic_rmse_rad"),
        }
        g3 = evaluate_g3_gate(g3_arrays["N2_R4_attitude"], g3_arrays["N3_R4_attitude"], g3_arrays["N3S_R4_attitude"])
        self._record_comparison("G3", g3_arrays, g3)
        self.gates["G3"] = {"status": "PASS" if g3["passed"] else "FAIL", "authorized_next": bool(g3["passed"]), **g3}
        self._flush()
        if not g3["passed"]:
            return self._finalize("LOCK_COMPENSATION_ONLY_REJECT_FEATURE_PATH", "G3 failed; the incremental feature claim was rejected and G4 was not run.")

        g4_arrays = {
            "N3_R0_attitude": self._matrix("N3", "R0", "attitude_geodesic_rmse_rad"),
            "N0_R0_attitude": self._matrix("N0", "R0", "attitude_geodesic_rmse_rad"),
            "N3_R0_divergence": self._matrix("N3", "R0", "divergence_count"),
            "N0_R0_divergence": self._matrix("N0", "R0", "divergence_count"),
        }
        g4 = evaluate_g4_gate(*g4_arrays.values())
        self._record_comparison("G4", g4_arrays, g4)
        self.gates["G4"] = {"status": "PASS" if g4["passed"] else "FAIL", "authorized_next": False, **g4}
        self._flush()
        if not g4["passed"]:
            return self._finalize("LOCK_COMPENSATION_ONLY_REJECT_FEATURE_PATH", "G4 failed; learned compensation passed, but the feature-conditioned path was not nominally harmless.")
        return self._finalize("LOCK_COMPENSATION_CONDITIONED_SPLIT_MEKF_KALMANNET", "All preregistered G0-G4 gates passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tiny-smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.tiny_smoke:
        result = run_tiny_smoke(args.config, args.output_dir)
    else:
        result = PilotRun(args.config, args.output_dir).run()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
