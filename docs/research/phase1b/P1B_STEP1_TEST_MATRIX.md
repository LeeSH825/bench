# Phase 1B Step 1 Test Matrix

Evidence command:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_phase1b_unit_st_regimes.py tests/test_phase1b_unit_st_classical.py
```

Actual: `52 passed in 0.97s`. Exact/bitwise checks use `np.array_equal`; quaternion
representation uses `atol=4e-16`; statistical fixture uses the locked 5% RMS match
and at least 2,000 bootstrap resamples.

## Regime generator

| ID | Input / expected evidence | Actual | Status |
|---|---|---|---|
| R01 stationary scales | stationary sidecar; all alpha exactly one | exact | PASS |
| R02 C2 isolation | paired C1/C2; event gyro only changes | exact | PASS |
| R03 C3 isolation | paired C1/C3; event ST quaternion only changes | exact | PASS |
| R04 truth/order pairing | C2 and C3 parameterized; truth/order unchanged | 2 cases exact | PASS |
| R05 sqrt scale | alpha 1/2/5; increment 0/1/2 | exact | PASS |
| R06 event-only scale | alpha=4 only inside window and correct sensor side | exact | PASS |
| R07 bias lock | alpha_b=1 in all regimes | exact | PASS |
| R08 timing hidden | sensor manifest forbidden-key scan | none present | PASS |
| R09 sidecar immutability | direct mutation of every field | ValueError | PASS |
| R10 forward-only cursor | future/repeated order consumption | rejected | PASS |
| R11 determinism | same config/seed twice | raw/oracle hashes equal | PASS |
| R12 whole split | paired split IDs and disjointness | exact | PASS |
| R13 zero latency/order | arrival=time; gyro precedes ST | exact | PASS |
| R14 q sign representation | randomized vs positive representation | physical dots abs≈1 | PASS |
| R15 q normalization | every ST quaternion | norm tolerance met | PASS |
| R16 artifact separation | sensor three files, oracle two files | exact file sets | PASS |
| R17 strict round trip | save/load/hash | both hashes preserved | PASS |
| R18 expected generator ID | wrong strict ID | rejected | PASS |
| R19 oracle corruption | modified alpha NPZ | semantic mismatch rejected | PASS |
| R20 wrong pairing | unrelated raw hash | rejected | PASS |
| R21 semantic domain | C1/C2/C3 | truth same, raw/oracle hashes distinct | PASS |
| R22 sidecar shape | shortened alpha array | rejected | PASS |
| R23 base Qg relation | sigma²/rate | exact | PASS |
| R24 dependency boundary | source scan | no neural/viz/sensor expansion | PASS |

## Replay, tuning, statistics and end-to-end

| ID | Input / expected evidence | Actual | Status |
|---|---|---|---|
| P01 fixed API | signature scan | no oracle/context/label/window | PASS |
| P02 deployable artifact | F-TUNED serialization | fixed scales only | PASS |
| P03 all-one replay | five trajectories | Phase 1A arrays bit-exact | PASS |
| P04 raw immutability | replay then compare payloads | exact unchanged | PASS |
| P05 fixed scale mapping | Qg=2,Qb=.5,R=4 capture | Gate A calls exact | PASS |
| P06 oracle mapping | C2/C3 parameterized | current alpha correct side, 2 cases | PASS |
| P07 wrong-side mapping | C2/C3 parameterized | swapped action, 2 cases | PASS |
| P08 cursor consumption | every event then overrun | strict/rejected | PASS |
| P09 q/-q replay | negate all ST q | q/b/P/r/S bit-exact | PASS |
| P10 SPD | C2/C3 × three policies | Cholesky success, 2 cases | PASS |
| P11 canonical evaluation | exact truth join | finite Gate C metrics/SPD | PASS |
| P12 fixed timing blindness | same raw data, changed sidecar window | q/P exact | PASS |
| P13 recovery known | sustained fixture | 1.0 s exact | PASS |
| P14 no recovery | interrupted fixture | None | PASS |
| P15 bootstrap | same seed twice; 1999 rejection | deterministic/guarded | PASS |
| P16 pairing | IDs 5/7 | paired mean exact | PASS |
| P17 tuning access/budget | mocked records | 42 candidates, no test ID | PASS |
| P18 tuning tie-break | all objectives tied | deterministic `(0.125)^3` | PASS |
| P19 C5 matching | train/val fixture | alpha frozen, test untouched | PASS |
| P20 divergence | threshold below fixture error | reported true | PASS |
| P21 workload | locked YAML | required N=50, nine scenarios | PASS |
| P22 small report E2E | three synthetic records | canonical summary written | PASS |
| P23 source safety | source scan | no inverse/pinv/clip/hiding/neural | PASS |
| P24 prior immutability | direct q/b/P mutation | ValueError | PASS |

Parameterized R04/P06/P07/P10 account for four additional pytest cases, giving
52 total executions from 48 named test functions.
