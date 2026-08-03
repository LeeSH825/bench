# research/mekf-attitude

Phase 0a + Phase 1A: the scalar-first Hamilton, right-error 6D kinematic MEKF,
its typed gyro/star-tracker event schema, the synthetic and Basilisk UNIT-ST
generators, and the canonical attitude/bias/NIS/NEES/SPD metrics.

Status per `P1A_FOUNDATION_FINAL_APPROVAL_AND_P1B_HANDOFF.md` (2026-08-02):
**Phase 1A COMPLETE**. Integration ids:

```
task_family = mekf_unit_st_v1
model_id    = mekf_event_replay_v1
```

Base: `research/common-foundation`. Phase 1B (sensor fusion) is stacked on top
of this branch, matching the documented Phase 1A → Phase 1B handoff.

## Verification on this branch

```
tests/test_mekf_metrics.py, test_mekf_adapter.py,
tests/test_basilisk_unit_st_generator.py     134 passed
bench.estimators.mekf / metrics.mekf / models.mekf   import OK
registry.list_typed_event_bridge_ids()       ['mekf_event_replay_v1']
third-party tracked diff                     empty
```

## Known limitations

- `bench/models/registry.py` carries **only** the typed-event-bridge hunk from
  the working tree. The spike-adapter entries from the same file are on
  `research/spike-snn`; the two hunks are in different regions of the file and
  were split on that evidence.
- `bench/runners/run_suite.py`, `bench/tasks/bench_generated.py` and
  `bench/tests/run_all.py` are **not** here. Their working-tree changes mix
  MEKF and Vizard/replay work inside single hunks, so a suite run driven
  through `run_suite` will not yet see the MEKF plans on this branch. The
  complete versions are preserved on `archive/research-wip-20260803`.
- No experiment output is committed. Phase 1A results live outside Git, under
  `experiments/phase1a/` (468 MB), archived at
  `bench-backups/research-commit-migration/20260803T031651Z/`.
