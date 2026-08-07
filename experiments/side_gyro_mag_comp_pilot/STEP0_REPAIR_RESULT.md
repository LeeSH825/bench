# Step 0 Integrity and Report Repair Result

Verdict: `PASS`

The R-1 through R-6 repair was applied on `codex/side-gyro-mag-comp-pilot` against reviewed commit `ba312f10559935f7f60782771f8de64ab77787af`.

- The count remains exactly six integrity checks.
- All six strengthened checks passed against the unchanged estimator implementation.
- Each check now rejects the focused counterexample required by the repair request.
- The training-time torch one-step filter agrees with the canonical numpy/runtime estimator to `1e-14` in the strengthened right-error check.
- The canonical MEKF regression suite passed 98 tests.
- `PILOT_SPEC.md`, G0/G1 machine records, paired comparisons, gate results, and all retained pilot checkpoints are byte-identical.
- G0 and G1 were not rerun.
- `FINAL_RESULT.md` changed only to correct the integrity-verification statement and add the four requested scope disclosures.

Machine evidence is in `TEST_EVIDENCE.json` and `STEP0_REPAIR_RESULT.json`.
