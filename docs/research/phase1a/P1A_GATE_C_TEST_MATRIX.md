# Phase 1A Gate C Test Matrix

All executable rows use Python 3.10.13 at
`/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python`,
`PYTHONDONTWRITEBYTECODE=1`, and pytest with `-p no:cacheprovider`. The Gate C
suite contains 43 pytest cases for the 16 required logical areas plus the
deterministic property sweep.

| ID | Contract | Input | Expected | Tolerance | Actual | Evidence | Status |
|---|---|---|---|---|---|---|---|
| C-01 | Identity and sign invariance | Identity and independently signed q pairs | Zero geodesic error; exact q/-q equality | Exact | Exact zero/equality | `04_20260801T143933Z_closed_form.txt` | PASS |
| C-02 | Known rotations | x/y/z and arbitrary axis-angle | Log norm equals angle | `2e-15 rad` | All four cases pass | `04_20260801T143933Z_closed_form.txt` | PASS |
| C-03 | Boundary stability | `1e-13`, exact pi, pi-minus-`1e-12` | Stable magnitude and exact sign equality | `3e-15 rad` | All boundaries pass | `04_20260801T143933Z_closed_form.txt` | PASS |
| C-04 | Right-local recovery | `q_true=q_hat*Exp(delta)` and known bias delta | Recover `[delta_theta,delta_b]` | attitude `3e-16`; bias exact | Pass | `04_20260801T143933Z_closed_form.txt` | PASS |
| C-05 | Bias summary | Two closed-form signed bias rows | Axis errors/norms/RMSE agree | `2e-18 rad/s` | Pass | `04_20260801T143933Z_closed_form.txt` | PASS |
| C-06 | NIS | Diagonal and full 3x3 SPD | Gate A result equals independent Cholesky/triangular reference | rel `2e-15`, abs `2e-16` | Pass; independent solve probe max diff `1.30e-18` | `04_20260801T143933Z_final_property_sweep.txt`, numeric evidence | PASS |
| C-07 | NEES | Identity and full 6x6 SPD | Right-local result equals independent Cholesky/triangular reference | rel `3e-15`, abs `3e-16` | Pass; independent solve probe max diff `1.11e-16` | `04_20260801T143933Z_final_property_sweep.txt`, numeric evidence | PASS |
| C-08 | NEES q/-q | Independent estimate/truth sign flips | Exact physical equality | Exact | Max difference `0` | `04_20260801T143933Z_numeric_evidence.txt` | PASS |
| C-09 | Consistency summary | `[1,2,3,6]`, dof 3, confidence .95 | count 4, sum 12, mean 3, normalized mean 1, ordered bounds | Exact scalar summaries | Pass | `04_20260801T143933Z_closed_form.txt` | PASS |
| C-10 | SPD fail-loud | SPD batch; asymmetric, indefinite, nonfinite P/S | Diagnostics; invalid matrices rejected without repair | Gate A `1e-12` symmetry policy; exact rejection | All pass | `04_20260801T143933Z_new_tests_second.txt` | PASS |
| C-11 | Validation | float32, empty, shape/count/batch/dof errors | TypeError/ValueError | Exact exception class | All rejected | `04_20260801T143933Z_new_tests_second.txt` | PASS |
| C-12 | Immutability | Snapshotted q/b/P/S and returned records | Inputs exact; output arrays read-only; records frozen | Exact | All pass | `04_20260801T143933Z_new_tests_second.txt` | PASS |
| C-13 | Import boundary | AST plus isolated interpreter import | Only Gate A/numerical standard dependencies; no Basilisk/task/runner/model/viz | Exact module set | Forbidden set empty | `04_20260801T143933Z_new_tests_second.txt` | PASS |
| C-14 | Forbidden source path | Source token scan | No matrix inverse, pseudo-inverse, least-squares fallback, perturbation, clipping | Exact absence | All tokens absent | `04_20260801T143933Z_new_tests_second.txt` | PASS |
| C-15 | B2 replay smoke | Seeds 801, 802, 803; typed direct replay | finite/nonnegative metrics, SPD P/S, q/-q equality | Exact sign equality; Cholesky success | 3/3; min P `2.245e-6`, min S `1.042e-3`, sign diffs `0` | `04_20260801T143933Z_b2_metric_smoke.txt`, numeric evidence | PASS |
| C-16 | Pairing fail-loud | Time, posterior, trajectory-ID, and partial-metadata mismatches | Exact mismatch rejection; no interpolation/alignment inference | Exact exception | All rejected | `04_20260801T143933Z_new_tests_second.txt` | PASS |

## Property and regression matrix

| Suite | Expected | Actual | Evidence | Status |
|---|---|---|---|---|
| Gate C full | Exit 0 | 43 passed | `04_20260801T143933Z_final_gate_c.txt` | PASS |
| Deterministic analytic property sweep | At least 10 cases | 10 passed; max state recovery `3.54e-16` | `04_20260801T143933Z_property_sweep.txt`, numeric evidence | PASS |
| B2 metric replay sweep | At least 3 seeds | 3 passed | `04_20260801T143933Z_b2_metric_smoke.txt` | PASS |
| Gate A baseline | Exit 0 | 55 passed | `04_20260801T143933Z_baseline_gate_a.txt` | PASS |
| Gate B1 baseline | Exit 0 | 55 passed | `04_20260801T143933Z_baseline_gate_b1.txt` | PASS |
| Gate B2 baseline | Exit 0 | 67 passed | `04_20260801T143933Z_baseline_gate_b2.txt` | PASS |
| Legacy baseline | Exit 0 | 18 passed, 5 subtests | `04_20260801T143933Z_baseline_legacy.txt` | PASS |

No skip, xfail, inverse fallback, covariance repair, or post-failure tolerance
change is used.
