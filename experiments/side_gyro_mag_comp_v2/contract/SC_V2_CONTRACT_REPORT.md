# Side Gyro–Mag Compensation v2 Contract-Stage Report

Decision after the single minimum CONTRACT repair: `GO_TO_CONTRACT_INDEPENDENT_AUDIT`.

The canonical v2 contract is complete and machine-valid. It preserves the gyro/magnetometer-only, separate-causal-encoder, 8D-per-sensor, branch-specific-FiLM, right-local 6D correction method and all v1 R1–R13 obligations. The only precedence change is the user's all-valid weak/plane definition replacing R8's obsolete frozen-threshold phrase; no other obligation changes.

Archive provenance is self-contained. The archive branch resolves to companion commit `9cf80cc85f2a01297cfd7932c1ce3cfcd87a15c0`, whose only staged path is `.codex/config.toml` and whose parent is the 84-path base evidence commit `235619cbd7b7af7dcc24db89c247673cd72a0363`. All 9 final-manifest members were found at the archive tip and matched their raw-byte SHA-256 values.

The repaired machine contract includes complete descriptors for G0–G4 and all required descriptive metrics. Its full key is `{experiment,split,regime,model,window,metric,seed,trajectory_id}`. Every descriptor binds an exact producer machine path, callable, output schema, value field, split, sample membership, trajectory aggregation, population, comparison direction, uniqueness rule, completeness rule, and exactly one of threshold or `descriptive_only`. G1 has six separately addressable R9 sensor metrics plus the R4 primary metric. All G0–G4 formulas, denominator-positivity guards, fractional-change arithmetic, CI strictness, seed rules, G3 weights, and G4 divergence arithmetic are exact-matched by the validator.

The validator also deep-equals the complete R1–R13 array and the R9 formula/conservative-conjunction projection against the four immutable hash-pinned v1 sources. Only the explicit R8 phrase replacement and R11 v2 namespace substitution are stored outside the verbatim source copy. Actual v2 git changes from archive baseline `9cf80cc...` must exactly equal `CHANGED_PATHS.json`; outside-namespace and forbidden contract-stage code classes, v1/config drift, and command/access records implying tests, smoke, performance, or payload access fail closed.

The weak/plane metrics use every valid same-timestamp posterior magnetometer update in every declared test trajectory. Any zero-valid-sample, non-finite, zero/non-finite-field, duplicate-update, or missing-update condition invalidates the entire dataset. The truth required for these metrics is confined to a diagnostic sidecar, and the metrics are forbidden from entry, gates, selection, early stopping, or stopping.

Validation command:

```text
python3 agent_system/side_gyro_mag_comp_v2/control/validate_contract.py --contract docs/research/side_gyro_mag_comp_v2/SC_V2_CANONICAL_CONTRACT.json
```

Result: `PASS`, 7/7 fail-closed control groups. The in-memory mutation matrix rejected 485/485 mutations across every descriptor field, every frozen producer/split/uniqueness/completeness value binding, threshold/descriptive semantic, G1 conjunct, gate arithmetic leaf, threshold leaf, R1–R13 source row, R9 projection section, observability field, changed-path rule, and execution/access firewall. No held-out/test payload was opened, and no performance or smoke test was run. Implementation remains unauthorized pending an independent Claude contract audit.
