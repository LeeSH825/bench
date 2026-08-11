# SC-02 Repository Reuse Map

The new family is isolated; frozen sources are imported or used as semantic references only.

| Contract | Canonical source | Decision |
|---|---|---|
| Right-local state, propagation, injection/reset | `bench/estimators/mekf.py` | reuse public math functions unchanged |
| Typed gyro/mag event and same-time order pattern | `bench/tasks/generator/mekf_fusion_events.py` | reuse schema mechanics; create new family |
| Sensor model/frame pattern | `bench/tasks/generator/phase1b_sensor_fusion.py` | reference only; Phase 1 stays frozen |
| Magnetometer replay/update and truth join | `bench/experiments/phase1b_sensor_fusion_c4.py` | reuse semantics/import helpers only |
| Geodesic and bias metrics | `bench/metrics/mekf.py` | reuse public metrics |
| Weak/observable decomposition | `bench/experiments/phase1b_sensor_fusion_c4.py` | reproduce in isolated evaluation code |
| Split factor construction/reset | `third_party/Split_KalmanNet/GSSFiltering/` | semantic/formula reference only |
| Divergence semantics | this SC-01 contract | non-finite output or max geodesic error `>1.0 rad`; do not inherit a different threshold |

The Euclidean `bench/models/split_knet.py` shell is not manifold/asynchronous compatible. The frozen main-Phase-2 `bench/phase2/spikera_dt_cm_knet_v1_1/` implementation is monolithic gyro+star-tracker and its checkpoints are incompatible. Historical Split checkpoints are ambiguous and prohibited.

The later implementation allowlist is exactly `bench/side_gyro_mag_comp_v1/**`, `tests/side_gyro_mag_comp_v1/**`, `bench/configs/side_gyro_mag_comp_v1.yaml`, and `experiments/side_gyro_mag_comp/**`, plus stage state updates through the supplied script. `.codex/config.toml`, Phase 0-1, main Phase 2, SpikeRA code/config/results, third-party code, and unrelated dirty paths are frozen.

Exact content pins for the three reused modules and `.codex/config.toml` are in `experiments/side_gyro_mag_comp/design_review/REUSE_PIN.json`; later stages must fail closed on drift.
