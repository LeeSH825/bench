# Phase 1A Gate B2 Validation Report

## Decision

**PASS_GATE_B2 — Gate B2 GO.** Gate C is authorized by this gate result but was
not started. The validated Gate A core and Gate B1 schema/serialization/replay
source were not modified.

## Created and corrected files

Created exact-allowlist targets:

- `bench/tasks/generator/basilisk_unit_st.py`
- `tests/test_basilisk_unit_st_generator.py`
- `docs/research/phase1a/P1A_BASILISK_FRAME_CONVENTION_PROOF.md`
- `docs/research/phase1a/P1A_BASILISK_UNIT_ST_CONTRACT.md`
- `docs/research/phase1a/P1A_GATE_B2_TEST_MATRIX.md`
- `experiments/phase1a/reports/P1A_GATE_B2_VALIDATION_REPORT.md`

The 03B2 overlay additionally authorized one documentation-only correction to
`docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md`. Only the two
`star_tracker_q_NB`/`q_true_NB` passive descriptions were corrected to
active body-to-navigation meaning, and a Gate B2 convention erratum was
appended. No source, NPZ schema, serializer, version, event order, replay API, or
hash domain changed.

Required provenance was written below
`experiments/phase1a/agent_logs/03B_*` and the recoverable preflight snapshot is
`experiments/phase1a/preflight_snapshots/03B_20260801T100923Z/`.

## Runtime identity

| Component | Actual |
|---|---|
| Python | 3.10.13 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| Basilisk | 2.10.2 |
| bsk distribution | 2.10.2 |
| Basilisk package | `/home/dss-pc-05/.pyenv/versions/3.10.13/lib/python3.10/site-packages/Basilisk/__init__.py` |

All Python commands used
`/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python` and disabled bytecode.
Pytest commands also disabled the cache provider.

## Frame conversion and q_NB erratum

The final adapter is:

```text
q_NB = normalize(Basilisk.RigidBodyKinematics.MRP2EP(sigma_BN))
R_NB = GateA.quat_to_dcm(q_NB)
C_BN = Basilisk.RigidBodyKinematics.MRP2C(sigma_BN)
R_NB = C_BN.T
```

`R_NB` is active body-to-navigation. The recorder-backed proof included
identity, each axis at ±90 degrees, ten deterministic arbitrary attitudes, and
all three body basis columns. Positive 90 degrees about z maps body +x to
navigation +y.

| Static proof quantity | Actual |
|---|---:|
| Cases / arbitrary cases | 17 / 10 |
| Max body-basis error | `4.440892098500626e-16` |
| Max `R_NB-C_BN.T` error | `4.440892098500626e-16` |
| Max recorder MRP error | `5.551115123125783e-17` |
| Rejected inverse-candidate error | `2.0` |
| Max shadow-set DCM error | `4.85722573273506e-16` |
| Min shadow `abs(q1 dot q2)` | `1.0` |

The Gate B1 synthetic semantic probe independently produced maximum right-
propagation log error `1.3036998028387272e-16 rad`, exact zero-noise ST
physical equality, and final replay log error
`6.265750595619259e-17 rad`. This confirms that the old passive wording was
only a documentation defect.

## omega_BN_B and constant-rate dynamics

A representative `10 kg`, spherical `7 kg m^2` hub with zero torque and no
environment was run for zero rate, six signed axis rates, and ten arbitrary
rates. Norms were at most `0.2 rad/s`, duration was `1 s`, and the primary
error was `norm(Log(q_reference^-1 otimes q_basilisk))`.

| Dynamic quantity | Coarse `dt=.01 s` | Fine `dt=.005 s` |
|---|---:|---:|
| Max attitude log error | `4.998209174226825e-16 rad` | `4.872566201647101e-16 rad` |
| Max local rate-increment error | `1.5855372570428017e-14 rad/s` | `3.219646771412954e-14 rad/s` |
| Max recorded component error | `0 rad/s` | `0 rad/s` |
| Max rate-norm drift | `0 rad/s` | `0 rad/s` |

The fine attitude error did not increase and is below the predeclared
`1e-8 rad` target. Therefore `omega_BN_B` is body-frame rad/s with the same
sign and right-local meaning as Gate A:

```text
q_NB(t+dt) = q_NB(t) otimes Exp_q(omega_BN_B*dt)
```

## Truth and project-owned sensor layer

Basilisk supplies recorder `sigma_BN` and `omega_BN_B`. The adapter supplies
active, representation-continuous `q_NB_true`. Constant gyro bias is a
project-owned sensor truth parameter, not a Basilisk rigid-body state.

The sensor equations are:

```text
omega_m = omega_true_B + b_g_true + n_g
q_ST    = q_NB_true otimes Exp_q(n_ST)
```

The quaternion star tracker is a project-owned parameterized wrapper. A
Basilisk built-in star tracker was not used and is not claimed. `n_ST` is
right-local, the recorded `R_ST` is strict SPD float64, and raw q/-q
representation uses an independent deterministic stream.

All events are valid and zero-latency. Star-tracker timestamps are gyro
timestamp subsets; shared timestamps order gyro before star tracker.

## Gate B1 reuse, manifest, hashes, seeds, and split

The implementation reuses Gate B1 dataclasses, validation, semantic hashes,
whole-trajectory split, serializer/loader, and direct replay. It defines no
duplicate event schema, serializer, hash engine, splitter, or replay engine.

The manifest records `generator_id=basilisk-unit-st-v1`, strict schema and
convention IDs, simulator/sensor/proof IDs, full config, named seeds, runtime
identity, source/proof fingerprints, and split membership. Strict
`load_event_dataset(..., expected_generator_id="basilisk-unit-st-v1")` passed;
the synthetic expected ID mismatch and simulator-identity tamper both failed
loudly.

The five-seed property sweep used three trajectories per seed:

| Property | Actual |
|---|---:|
| Same-seed semantic-hash reproduction | 5/5 |
| Strict serialization round trip | 5/5 |
| q/-q sign-paired replay | 5/5 |
| Disjoint whole-trajectory split | 5/5 |
| Max quaternion norm deviation | `0.0` |
| Minimum posterior P eigenvalue | `1.8336798328931196e-06` |
| Minimum ST S eigenvalue | `1.2649972746123421e-05` |

Dedicated tests also confirmed truth/noise/bias/sign/split seed isolation. A
split-only change preserved truth, sensor, and event-order hashes and changed
only manifest membership identity.

## Direct replay and boundary

Zero-noise replay began from exact true q/b, used zero sampled gyro/ST noise with
strictly SPD nominal ST covariance, and matched final Basilisk truth with
attitude log error `3.903274402329114e-15 rad` and bias error
`5.64988976823351e-16 rad/s`.

Representative noisy replay stayed finite, unit-quaternion, symmetric, and SPD.
Repeated and serialized round-trip replay traces were exact. Negating all raw ST
quaternions produced exact-equal q/b/P/residual/S. Replay accepts no truth,
oracle, label, or future input and does not mutate truth/events/prior state.
Gate A state q/b/P arrays remained read-only.

## Tests

| Suite | Result |
|---|---|
| New Gate B2 | `67 passed in 2.74s` |
| Gate A | `55 passed in 0.86s` |
| Gate B1 Amendment A1 | `55 passed in 1.48s` |
| Legacy | `18 passed, 5 subtests passed in 2.81s` |

No tolerance was relaxed after failure. No skip, xfail, pseudo-inverse, explicit
inverse fallback, covariance jitter, or eigenvalue clipping was introduced.

## Dirty-tree integrity

The approved current-tree baseline contained 1,394 status paths. The 1,393
paths outside the explicitly authorized event-contract erratum retained their
content/deletion fingerprints. Gate A's seven frozen fingerprints and the ten
frozen Gate B1 fingerprints other than that erratum remained exact. Staged diff
stayed empty.

Nine untracked `artifacts/benchmark_write_api_ui/20260801T102738Z/**` files
appeared concurrently. They are non-source external artifacts; only their paths
and status were ledgered. They were not read, imported, executed, or modified,
and the execution contract explicitly permits such external non-source
artifacts without failing Gate B2. No concurrent source change was detected.

Agent-only changed targets consist of six new B2 allowlist files plus the one
authorized event-contract documentation correction: 7 files, 1,988 insertions,
and 2 deletions. The retry's timestamped evidence is in
`03B_20260801T104729Z_agent_only.patch`,
`03B_20260801T104729Z_agent_only_stat.txt`, and
`03B_20260801T104729Z_changed_paths.txt`. Untimestamped files with similar
names are preserved provenance from the earlier blocked attempt.

## Blocking and deferred items

Blocking issues: none.

Deferred by contract: nonzero latency, outage/false solution, magnetometer, sun
sensor, orbit/environment, canonical metrics, runner/registry integration,
visualization, and neural models.

## Final gate

- Runtime identity: PASS
- Static `sigma_BN -> q_NB` frame proof: PASS
- MRP shadow invariance: PASS
- `q_NB` active/passive executable resolution: PASS
- `omega_BN_B` sign/frame/unit proof: PASS
- Constant-rate dynamics/convergence: PASS
- Basilisk truth generation: PASS
- Gyro sensor layer: PASS
- Star-tracker sensor layer: PASS
- Gate B1 schema/serialization reuse: PASS
- Determinism/semantic hashes: PASS
- Seed isolation: PASS
- Trajectory split: PASS
- Direct replay/truth boundary/numerical safety: PASS
- Gate A/B1/legacy regressions: PASS
- Gate B1 convention documentation erratum: PASS
- Dirty-tree integrity: PASS

**Gate B2: GO**

**Gate C authorized: YES — not executed.**

Next stage title only: **Phase 1A Gate C — Canonical MEKF
Geodesic/Bias/NIS/NEES/SPD Metrics**
