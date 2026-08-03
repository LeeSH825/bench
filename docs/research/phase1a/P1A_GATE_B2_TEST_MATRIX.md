# Phase 1A Gate B2 Test Matrix

Runtime for every executable row: Python 3.10.13 at
`/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python`, with
`PYTHONDONTWRITEBYTECODE=1`. The new suite contains 67 pytest cases covering
the 39 required logical areas below.

| Test ID | Requirement | Input | Expected behavior | Tolerance | Actual result | Evidence log | Status |
|---|---|---|---|---|---|---|---|
| B2-01 | Explicit Basilisk runtime | Version/path probe | Import bsk 2.10.2 on required interpreter | Exact IDs | Python 3.10.13; Basilisk/bsk 2.10.2 | `03B_runtime_versions.txt` | PASS |
| B2-02 | Import boundary | AST and isolated import | No runner/model/metric/torch/viz import | Exact module set | Forbidden set empty | `03B_new_tests.txt` | PASS |
| B2-03 | Missing runtime | Forced Basilisk import failure | Fail; no synthetic fallback | Exact exception | RuntimeError with no-fallback message | `03B_new_tests.txt` | PASS |
| B2-04 | Identity frame | Recorder `sigma=[0,0,0]` | `q_NB=[1,0,0,0]` | Exact | Exact identity | `03B_frame_proof.txt` | PASS |
| B2-05 | Axis ±90 basis | Six closed-form MRPs | All body bases map to closed form | `5e-15` | Max `2.22e-16` for axis cases | `03B_frame_proof.txt` | PASS |
| B2-06 | Arbitrary attitudes | 10 seeded axis-angle cases | Recorder basis matches Rodrigues | `5e-15` | Max `4.44e-16` | `03B_frame_proof.txt` | PASS |
| B2-07 | DCM direction | `MRP2C`, selected candidate | `R_NB=C_BN.T`; inverse rejected | `5e-15` | Relation error `4.44e-16`; inverse error `2.0` | `03B_frame_proof.txt` | PASS |
| B2-08 | MRP shadow | `-sigma/(sigma.T sigma)` | Same DCM and `abs(dot)=1` | `1e-14` | DCM `4.86e-16`; min dot `1.0` | `03B_shadow_proof.txt` | PASS |
| B2-09 | Invalid MRP | NaN, Inf, wrong ranks | Fail loudly | Exact exception | All rejected | `03B_new_tests.txt` | PASS |
| B2-10 | Unit/deterministic adapter | Repeated finite MRP | Exact same unit quaternion | Norm `2e-15` | Exact repeat; unit | `03B_new_tests.txt` | PASS |
| B2-11 | Zero rate | Nonidentity initial attitude | Attitude and rate stay constant | `1e-14 rad` | Roundoff-only error | `03B_dynamic_proof.txt` | PASS |
| B2-12 | Axis ±rate | Six `0.2 rad/s` cases | Gate A right propagation | `1e-14 rad` | All six pass | `03B_dynamic_proof.txt` | PASS |
| B2-13 | Arbitrary rates | 10 seeded vectors | Analytic propagation agreement | Fine `1e-8 rad` | Max fine `4.87e-16` | `03B_dynamic_proof.txt` | PASS |
| B2-14 | Local rate meaning | Quaternion-log increments | Match recorder frame/sign/unit | `5e-13 rad/s` | Max `3.22e-14 rad/s` | `03B_dynamic_proof.txt` | PASS |
| B2-15 | Constant-rate truth | Spherical inertia, zero torque | Recorder rate drift zero | Float64 exact | Component/norm drift `0` | `03B_dynamic_proof.txt` | PASS |
| B2-16 | Grid convergence | `dt=.01/.005 s` | Fine attitude error not larger | Direct comparison | `4.87e-16 <= 5.00e-16` | `03B_dynamic_proof.txt` | PASS |
| B2-17 | Fine error target | 17 dynamic cases, 1 s | Max log error at most `1e-8` | `1e-8 rad` | `4.87e-16 rad` | `03B_dynamic_proof.txt` | PASS |
| B2-18 | Schema dtype/shape | Generated truth/events/payloads | Exact Gate B1 types/ranks | Exact | All exact | `03B_new_tests.txt` | PASS |
| B2-19 | Zero latency | All nominal events | Arrival equals measurement time | Exact float64 | Array-equal | `03B_new_tests.txt` | PASS |
| B2-20 | Same-time order | Shared gyro/ST epochs | Gyro then star tracker | Exact code/order | All shared epochs pass | `03B_new_tests.txt` | PASS |
| B2-21 | ST cadence | Per-trajectory timestamps | ST set is gyro subset | Exact set | All trajectories pass | `03B_new_tests.txt` | PASS |
| B2-22 | Event validity | Nominal table | Every event valid | Exact bool | All true | `03B_new_tests.txt` | PASS |
| B2-23 | Regeneration | Same config/seed twice | All semantic hashes equal | Exact strings | 5/5 property seeds plus unit case | `03B_hash_seed_property_sweep.txt` | PASS |
| B2-24 | Simulator identity | Manifest inspection | All versions/config/fingerprints present | Exact keys | Complete | `03B_manifest_compatibility_after_a1.txt` | PASS |
| B2-25 | Sensor seed isolation | Gyro/ST noise namespace changes | Truth same; sensor hash changes | Exact arrays/hashes | Both streams pass | `03B_hash_seed_property_sweep.txt` | PASS |
| B2-26 | Truth seed isolation | Truth attitude namespace change | Truth hash changes | Exact hash | Changed | `03B_hash_seed_property_sweep.txt` | PASS |
| B2-27 | Bias seed isolation | Bias namespace change | q/rate same; bias/gyro change | Exact arrays | Required boundary observed | `03B_sensor_equation_proof.txt` | PASS |
| B2-28 | ST sign stream | Sign namespace and full negation | Physical measurement/replay same | `abs(dot)` `2e-15`; replay exact | 5/5 sign pairs exact | `03B_hash_seed_property_sweep.txt` | PASS |
| B2-29 | Whole-trajectory split | 3–8 trajectory configs | Unique IDs; disjoint/nonempty sets | Exact sets | 5/5 sweep seeds pass | `03B_hash_seed_property_sweep.txt` | PASS |
| B2-30 | Serialization round trip | Three-file artifact | Arrays/hashes exact | Exact bytes/strings | 5/5 sweep seeds pass | `03B_hash_seed_property_sweep.txt` | PASS |
| B2-31 | Identity/hash rejection | Wrong expected ID; manifest tamper | Fail loudly | Exact exception | Both rejected | `03B_manifest_compatibility_after_a1.txt` | PASS |
| B2-32 | Zero-noise replay | Exact initial q/b, zero sampled noise | Match Basilisk truth | `2e-12` | Attitude `3.90e-15`; bias `5.65e-16` | `03B_sensor_equation_proof.txt` | PASS |
| B2-33 | Noisy safety | Perturbed state, nominal noise | Finite/unit; P and S SPD | Norm `2e-14`; Cholesky | All pass | `03B_new_tests.txt` | PASS |
| B2-34 | Replay determinism | Same stream/state twice | Exact trace equality | Exact arrays | Exact | `03B_new_tests.txt` | PASS |
| B2-35 | Round-trip replay | Before/after strict load | Exact trace equality | Exact arrays | Exact | `03B_new_tests.txt` | PASS |
| B2-36 | q/-q replay | All ST quaternions negated | Same q/b/P/residual/S | Exact arrays | 5/5 plus suite case exact | `03B_hash_seed_property_sweep.txt` | PASS |
| B2-37 | Truth boundary | Replay signature and input snapshots | No truth input or mutation | Exact API/arrays | Pass | `03B_new_tests.txt` | PASS |
| B2-38 | Gate A immutability | Direct writes to q/b/P | Arrays remain read-only | Exact exception | All writes rejected | `03B_new_tests.txt` | PASS |
| B2-39 | Malformed config/SPD | Bad time/rate/mass/inertia/noise/R | Fail loudly; no repair | Exact exception | All cases rejected | `03B_new_tests.txt` | PASS |

## Regression matrix

| Suite | Required command result | Actual | Status |
|---|---|---|---|
| Gate B2 new | Exit 0 | 67 passed | PASS |
| Gate A | Exit 0 | 55 passed | PASS |
| Gate B1 Amendment A1 | Exit 0 | 55 passed | PASS |
| Legacy | Exit 0 | 18 passed, 5 subtests | PASS |

No skip, xfail, jitter, pseudo-inverse, covariance repair, or post-failure
tolerance change was used.

