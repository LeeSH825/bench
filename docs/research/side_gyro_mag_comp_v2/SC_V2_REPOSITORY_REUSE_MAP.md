# Side Gyro–Mag Compensation v2 Repository Reuse Map

Status: contract-stage reference map, sealed before independent audit.

The v2 family is isolated under the three `side_gyro_mag_comp_v2` namespaces. No v1 path is edited. The v1 terminal archive is evidence and semantic provenance only; it is not an implementation namespace for v2.

| Contract subject | Frozen source | v2 decision |
|---|---|---|
| Right-local state, propagation, injection, reset | `bench/estimators/mekf.py` | immutable public-math reference; sha256 pinned in the canonical contract |
| Geodesic and bias metrics | `bench/metrics/mekf.py` | immutable public-metric reference; sha256 pinned |
| Typed gyro/mag event ordering | `bench/tasks/generator/mekf_fusion_events.py` | immutable schema-mechanics reference; sha256 pinned |
| Existing agent configuration | `.codex/config.toml` | immutable, tracked in the self-contained archive companion commit; sha256 `315ec7d...` |
| v1 R1–R13 repair obligations | four exact v1 audit sources listed in the canonical contract | copied into the v2 machine contract without weakening |
| Weak/plane observability | user-authored v2 rule | supersedes only R8's obsolete frozen-threshold phrase; diagnostic truth-sidecar semantics only |

The Euclidean Split shell, monolithic gyro–star-tracker Phase-2 implementation, historical checkpoints, and v1 implementation namespace are not v2 implementation targets. This contract stage creates no estimator, model, data, test, config, checkpoint, or evaluation code.

Canonical machine authority: `docs/research/side_gyro_mag_comp_v2/SC_V2_CANONICAL_CONTRACT.json`.
