# Research branch layout

This branch (`research/common-foundation`) carries the research material that
is not specific to one hypothesis: the Phase 0/1 evidence index, the
source-of-truth research documents, and the validator that checks the index.

## Where each hypothesis lives

```
research/common-foundation      ⟵ integration/benchmark-platform
  research/mekf-attitude        ⟵ research/common-foundation
    research/phase1b-fusion     ⟵ research/mekf-attitude
  research/vizard-replay        ⟵ research/common-foundation
  research/spike-snn            ⟵ research/common-foundation
```

`phase1b-fusion` is stacked on `mekf-attitude` because
`docs/research/phase1a/P1A_FOUNDATION_FINAL_APPROVAL_AND_P1B_HANDOFF.md`
declares Phase 1A **COMPLETE** and hands off to Phase 1B Step 1 — Phase 1B
extends that filter, it does not replace it.

## Known limitations of this branch

`tests/test_phase0_1_evidence_index.py` **fails here on purpose.** The index it
validates spans the whole research corpus, including
`docs/research/phase0a/` and `docs/research/phase1b/`, which live on
`research/mekf-attitude` and `research/phase1b-fusion`. The test passes on
`research/phase1b-fusion`, where all three document sets are present, and it
will pass on `integration/benchmark-platform` once the research branches are
merged there. It is kept with the index rather than duplicated per branch.

## What is deliberately absent

Four files that the whole research corpus modifies could not be attributed to
one hypothesis and are **not** on any research branch:

```
bench/runners/run_suite.py        +547   plan wiring for MEKF and Vizard/replay in the same hunks
bench/tasks/bench_generated.py    +379   4 hunks Vizard, 1 hunk MEKF
bench/tests/run_all.py            +243   one hunk lists MEKF, Vizard and spike tests together
DECISIONS.md → see research/vizard-replay (D20-D34 are the Phase 2-7 chain)
bench/models/registry.py → split by hunk: spike entries on research/spike-snn,
                           typed-event bridge on research/mekf-attitude
```

They are preserved complete on `archive/research-wip-20260803` and described in
the migration report's `shared_files_decision.md`. Splitting them requires the
researcher, because the hunks mix hypotheses.
