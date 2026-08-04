# P1 Exit Closure Validation Report

Implementation: `PASS`; search: `COMPLETE`; confirmation: `COMPLETE`.
Runtime confirmation seconds: `285.103`; records written/reused: `528` / `22`.

## Regression and integrity evidence

- All required regression groups passed: `True`.
- Pytest/legacy counts: `{'closure_new': 17, 'cp4_integration': 22, 'd1_bridge': 24, 'gate_a': 55, 'gate_b1': 55, 'gate_b2': 67, 'gate_c': 43, 'legacy_passed': 18, 'legacy_subtests_passed': 5, 'phase1b_step1': 52, 'phase1b_step2': 38}`.
- Phase 1A fresh/cache smoke: `{'basilisk': {'6101': '25c591f24b83a47de3e0b07010fdddbd644860836dabb3ea2b97c34d5a080a66', '6102': '4a700930dd4604b9496b5a9b252ab12b6721af6ba614581b02ef337fe6608dcf', '6103': '321876d8584b39db6a337d7e65286442f8ab18266df9eba03c06a1fe15e51134'}, 'fresh_generation_count': 6, 'synthetic': {'6101': 'b039abe71d0863965f3be5b1d390c99ebb097bf8d769d4e857941dcfb2384eeb', '6102': 'd79391e4a803a64298e4451654515fb8fd58a84ac13045b0baed8c6438cc4ec9', '6103': '9ab654fa04480cd479b45a7929192e24e2a766499a4b8711750fe8e1f084bc01'}, 'verified_cache_hit_count': 6}`.
- Frozen files checked/mismatched: `2879` / `0`.
- Final tracked/staged patch equality, allowlist classification, and the ignored smoke-output note are recorded in `preflight_snapshots/03_20260802T032016Z/FINAL_INTEGRITY.md`.

## Blocking and deferred items

There is no implementation or sample-count blocker. The scientific result remains conditional: F-CALIBRATED-v1 closes stationary posterior consistency but distorts C4 bias accuracy and fixed-policy sensor NIS, so it must not replace F-BASE for C4. Phase 2 and learned/FPGA/closed-loop work remain explicitly deferred.
