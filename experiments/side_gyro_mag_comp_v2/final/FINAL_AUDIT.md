# Final Independent Audit — side-gyro-mag-comp-v2

Auditor: Claude, sole independent final auditor, fresh audit context, no subagents used.
Audit stage: `FINAL_INDEPENDENT_AUDIT`.
Single audit target: `experiments/side_gyro_mag_comp_v2/final/CHECKPOINT_MANIFEST.json`.
Target digest before audit: `eb6421fbbf31fd0fc412280d2be604261a7ffba5553b065fd1bad807077dff4b`.
Target digest after audit: `eb6421fbbf31fd0fc412280d2be604261a7ffba5553b065fd1bad807077dff4b` — **stable**.

## Decision

`FAIL_FINAL_REPORT_MISMATCH`

The terminal blocked status is **upheld and not softened**. The study remains
`BLOCKED_CONTRACT_INCOMPLETE_AFTER_ONE_REPAIR`, G0–G4 remain `NOT_RUN`, no scientific
conclusion is available and none is inferred here, and implementation remains unauthorized with
zero CONTRACT repair rounds remaining. Nothing in this audit relaxes any of that.

The audit fails because the sealed bundle's changed-path and command accounting does not
disclose a deletion inside the Claude-owned handoff namespace that the sealed validator
machine-requires to be absent as a precondition of the recorded 7/7 PASS. A terminal bundle
whose own evidence chain conditions its PASS on the removal of a peer-owned artifact, while
recording neither the removal nor the condition, is not an accurate final report.

## What verified cleanly

**Checkpoint integrity.** The target digest matched the handoff declaration at audit open and
again at audit close, unchanged. All 9 declared members hashed exactly to their recorded
`sha256`, with `file_count: 9` consistent with the member array — **9/9, 0 mismatches, 0
missing**. The contract checkpoint `2aa9a457…` self-verified at 10/10 members, including the
pinned validator `483889…` and canonical contract `50a1c4a…`.

**No forbidden execution or access.** `bench/side_gyro_mag_comp_v2/` does not exist,
`tests/side_gyro_mag_comp_v2/` does not exist, and
`experiments/side_gyro_mag_comp_v2/implementation/` and `.../results/` contain **zero files**.
`COMMANDS.json` records empty `implementation_commands`, `test_commands`, `smoke_commands`,
`performance_commands`, `G0_G4_commands`, and `test_or_held_out_payload_commands`. Nothing in
the workspace contradicts the declared firewall. No sealed main Phase 2 evaluation payload was
inspected during this audit.

**No Codex-fabricated final audit.** `FINAL_AUDIT.md` and `FINAL_AUDIT.json` were both absent
at audit open, matching `independent_audit_created_by_codex: false`. Codex did not fabricate
independent audit evidence. (See the chain-of-custody section for what appeared later.)

**Wrong-namespace cleanup.** All three recorded duplicate-authority files under
`agent_system/side_gyro_mag_comp/state/` were absent when observed, and their pre-cleanup hashes
are recorded in `FINAL_DECISION.json`. *(This observation was superseded after audit close — see
the post-audit-close addendum.)*

**v1 archive, namespace, and config preservation — independently recomputed, not adopted from
report.** Archive branch tip equals `9cf80cc8…`; its parent is exactly the base evidence commit
`235619cb…`; the base commit stages exactly 84 paths and the archive tip stages exactly
`.codex/config.toml`, for 85 unique paths. Every one of the 85 is present in the working tree
and byte-identical to `git show 9cf80cc8:<path>` — **0 missing, 0 byte mismatches**. A full
recursive enumeration of the v1 namespaces including ignored files found **0 extra** paths. The
archived v1 final manifest `experiments/side_gyro_mag_comp/final/CHECKPOINT_MANIFEST.json`
verifies **9/9**, and `.codex/config.toml` hashes to the pinned `315ec7d2…`.

One reported value is not reproducible from its own description: `path_byte_set_sha256`
`d9668c8976…` is an opaque set digest whose serialization no sealed artifact specifies. My
independent recomputation under my own serialization yields `587e629b79e5…`. This is a
documentation-precision gap, not a contradiction — every substantive claim the digest
summarizes verified directly. It is **not** a basis for this audit's failure.

## The decisive red check, reproduced

The most load-bearing of the four recorded second-review defects is that the effective
`repair_obligations` array is not bound to the immutable source projection and can be weakened
while validation stays green. I reproduced it directly, in-process, **without editing any
repository source**.

Command: `python3 /tmp/claude_redcheck.py`

Method: load `agent_system/side_gyro_mag_comp_v2/control/validate_contract.py` by `importlib`
from its real path, deep-copy the pristine sealed contract, and call
`validate_contract_data(contract, workspace=False)` on a control and a treatment. The workspace
conjunct is excluded deliberately: it is a time-of-check assertion over live git state that has
legitimately drifted since the contract seal, and excluding it isolates the contract-data
predicate under test.

- **Control** — pristine contract: `PASS`, 6/6 data-level checks, including
  `repair_invariants_stage_stop_and_exclusions`.
- **Treatment** — every `effective_v2_change` in `repair_obligations` emptied to `""` (except
  R8, held at the literal `"all-valid observability semantics"` the validator substring-checks),
  and every other obligation field deleted: **`PASS`, the same 6/6 checks.**

The weakened contract is not deep-equal to the immutable source
(`SC_IMPL_ROUND1_COUNTERPROPOSAL.json#/required_repairs`, 13 rows), yet
`exact_r1_r13_and_r9_source_projections` still passes — because that check binds
`canonical_source_projections.required_repairs_verbatim`, a *separate declarative field* that is
deep-equal to the source, while `validate_policy` checks the *effective* array only for
`[R1..R13]` id equality plus three R8 substring assertions (`validate_contract.py:281-286`).
Nothing joins the two. Confirmed numerically: the projection field is deep-equal to the source
(`True`), the effective array is not (`False`) — in both the pristine and the weakened contract.

Gutting every operative repair obligation therefore escapes validation entirely. **Defect 2 is
CONFIRMED.** The run re-hashed the validator and contract before and after: unchanged.

**Why 485/485 cannot cover this.** `category_counts` in `CONTRACT_MUTATION_RESULT.json` contains
no `repair_obligations` category, and `repair_obligations` appears exactly once in the entire
validator, at line 281. The matrix never mutates the effective obligations array. The recorded
qualification `INSUFFICIENT_NOT_A_PASS` is accurate and, on this evidence, understated rather
than overstated.

## The other three recorded defects

**Defect 4 — producer path outside the frozen allowlist: CONFIRMED, decisively.** All 15 metric
descriptors declare exactly one `producer.machine_path`,
`bench/side_gyro_mag_comp_v2/evaluation.py`. That path is outside every entry of
`authorized_v2_paths`, and `validate_changed_path_list` (`validate_contract.py:313`)
affirmatively rejects any declared changed path beginning with `bench/`. The contract
pre-registers a producer its own machine-enforced allowlist forbids the implementation stage
from ever creating. The path does not exist and cannot lawfully be made to exist.

**Defect 1 — producer evidence binding: CONFIRMED structurally.** `PRODUCER_FIELDS` is exactly
`{machine_path, callable, output_schema, value_field}`. `output_schema` holds a schema *name*
(`side-gyro-mag-comp-v2-metric-record-v1`), not an emitted artifact path. No producer carries
any emitted-evidence path, evidence-file schema, or report-resolution field.

**Defect 3 — wrong-namespace escape: CONFIRMED structurally.** `V2_PREFIXES` covers only the
three `*_v2` namespaces, so `actual_v2_changed_paths()` is structurally blind to
`agent_system/side_gyro_mag_comp/`. The v1-freeze conjunct (`validate_contract.py:337`) uses
`git diff --name-only` against a tracked pathspec, which does not report untracked files. An
untracked artifact placed in the v1 namespace passes both boundaries unseen — which is precisely
how the three duplicate-authority files arose.

All four recorded defects are represented in `FINAL_DECISION.json` and `FINAL_RESULT.md`
without weakening. Codex's characterization of its own two review rounds is faithful, and the
`INSUFFICIENT_NOT_A_PASS` qualification is correct.

## Why the audit fails

**Finding A (decisive): an undisclosed deletion in the Claude-owned handoff namespace is a
machine precondition of the sealed PASS.**

`validate_contract.py:344`, a member of the contract checkpoint which is itself a member of the
sealed final manifest, reads:

```python
require(not (ROOT / "experiments/side_gyro_mag_comp_v2/handoffs/claude/CLAUDE_TO_CODEX_PREREGISTERED_CRITERIA.json").exists(),
        "Codex-created artifact remains in Claude-owned handoff path")
```

The sealed 7/7 PASS is conditional on the *absence* of a named file in the Claude-owned handoff
directory. Three independent lines of evidence establish that the file existed and was removed:

1. **The check itself.** A require-absent assertion naming an exact path, with an error message
   asserting the file "remains", is not written for a path that never existed.
2. **Filesystem state.** `experiments/side_gyro_mag_comp_v2/handoffs/claude/` is empty, yet its
   mtime is `2026-08-06 23:08:20 +0900` while the directory was created at `23:00:44`. An empty
   directory's mtime advances only when an entry is added or removed.
3. **The contract-stage independent audit** records the file at `23:03:49` as the auditor's
   pre-registered acceptance criteria.

No sealed artifact discloses this. `COMMANDS.json` logs four records — three cleanup operations
and one preservation proof — none touching this path. Neither the final `CHANGED_PATHS.json`
(8 finalization paths, 3 duplicate-authority deletions, 8 bytecode deletions) nor the
contract-stage `CHANGED_PATHS.json` (13 paths, `independent_audit_paths_created: []`) records
any deletion in the Claude-owned namespace. A targeted search across all sealed artifacts
returns no disclosure.

I cannot adjudicate authorship of the deleted file — the validator's error string frames it as a
stray Codex artifact, the contract-stage audit claims it as the auditor's own pre-registered
criteria, and the bytes are gone. **Authorship is not what fails this audit.** What fails it is
that a deletion inside a peer-owned namespace became a hardcoded pass condition of the sealed
validator, and the final report accounts for neither. The bundle's deleted-path accounting is
presented as complete and is not.

**Finding B (corroborating): the sealed defect set is known to be incomplete.**
`experiments/side_gyro_mag_comp_v2/audits/SC_V2_CONTRACT_INDEPENDENT_AUDIT.json`
(sha256 `5fbafc4c8840…`) records a fifth blocking contract gap — absence of the v1 gauge-fixing
and identifiability anchors, explicitly flagged as appearing in neither Codex review — plus
governance findings. That artifact is timestamped one minute after the Codex final seal, so
Codex could not have incorporated it; its omission is not a misrepresentation by Codex. It is
recorded because it bears on terminality: it **strengthens** the blocked finding.

**Finding C (governance, disclosed): concurrent Claude-role writer.** I was designated the sole
independent final auditor. During my audit window, a concurrent actor also claiming
`auditor: "claude-independent"` wrote to paths I do not own or own exclusively:

| Path | sha256 | Appeared |
|---|---|---|
| `experiments/side_gyro_mag_comp_v2/audits/SC_V2_CONTRACT_INDEPENDENT_AUDIT.json` | `5fbafc4c884086d414263f14425b6c4165551d716ed094cad0298b3b66612e75` | 23:24 |
| `experiments/side_gyro_mag_comp_v2/audits/SC_V2_CONTRACT_INDEPENDENT_AUDIT.md` | `99299a8696da67444085231496b97766c7338762a0ce4921cde23126985f12c8` | 23:25 |
| `experiments/side_gyro_mag_comp_v2/final/FINAL_AUDIT.json` | `c8d89d4126f6c63d818d147b7b19abc4dfef85a079fe7dc290f17b4766532623` | 23:27:48 |
| `experiments/side_gyro_mag_comp_v2/final/FINAL_AUDIT.md` | `04ddb4f711653993648e0b50fe86cecf5d6caf02ec3b735c1b16dae3668e989f` | 23:28:43 |

All four were absent when this audit opened. I did not write them. The last two occupied my
mandated output paths; that actor's decision there was `CONFIRM_BLOCKED_ON_ENLARGED_GROUNDS`
with `predicate_result: false`, and it declared use of two subagents. I could not adopt those
files as my own independent audit without committing precisely the adoption failure this
governance structure exists to prevent, so I authored this audit in their place under explicit
principal instruction. **Their digests are recorded above so the supersession is disclosed and
attributable rather than silent** — the distinction from Finding A that I hold Codex to, I hold
myself to.

That actor's core conclusion and mine converge independently: the blocked outcome is right and
the bundle's process accounting is not truthful. Its `GV-1` "deleted_without_record" is the same
artifact as my Finding A, reached separately. This is corroboration, not adoption; every finding
in this report was derived from my own machine checks. Claims in those artifacts that I could
not reproduce — the 2650-file fingerprint taken before my session, the contract-seal digest
timeline, and the assertion that the validator did not exist on disk at 23:11:09 — are **not**
relied upon here and are recorded as unverified by me.

**On the non-reproducibility of the sealed validator PASS.** Re-executing the exact declared
command today yields `decision: FAIL`, exit 1,
`CHANGED_PATHS does not equal actual v2 git changes from archive baseline`. I do **not** treat
this as evidence of fabrication. The failing conjunct is a time-of-check assertion over live git
state, and the v2 changed-path set has legitimately grown since the contract seal by the six
`final/` artifacts, the final handoff, and the four artifacts in Finding C. My control run
confirms all six contract-data checks still pass on the pristine contract. The sealed 7/7 is
plausible as of its seal time, and I record **no finding against it**.

## Audit predicate qualification

The handoff's `audit_predicate` is **nonempty**, **unambiguous**, and a **single predicate** —
one interrogative in conjunctive form whose five conjuncts are each independently decidable
against machine evidence, over a nonempty population (9 manifest members, 15 metric descriptors,
13 repair obligations, 85 archive paths). I evaluated all five. Four hold: the two local review
outcomes are preserved faithfully, the 485/485 insufficiency is explained correctly, the stop
after the exhausted repair is real with no implementation or G0–G4 execution, and no independent
audit evidence was fabricated by Codex. The fifth — preservation of the archive and boundary —
holds for the v1 archive and config but fails at the Claude-owned handoff boundary per Finding A.

I decline `required_checks` item 8 only insofar as it asks me to ratify the *validity* of the
peer's adjudication of Claude-owned artifacts. I confirm the verifiable part: the three files
are absent and their pre-cleanup hashes are recorded. I do not certify that their deletion was
authorized; that is a governance question, not an audit finding I can settle from bytes.

## Paths changed by this audit

Exactly four, all Claude-owned or Claude-owned transition state:

- `experiments/side_gyro_mag_comp_v2/final/FINAL_AUDIT.md`
- `experiments/side_gyro_mag_comp_v2/final/FINAL_AUDIT.json`
- `experiments/side_gyro_mag_comp_v2/handoffs/claude/CLAUDE_FINAL_TO_CODEX.json`
- `agent_system/side_gyro_mag_comp_v2/state/STAGE_STATE.json`

No Codex final artifact, contract, validator, v1 path, config, implementation, test, checkpoint,
or result was edited, created, or deleted. The red check ran entirely in memory against deep
copies; a throwaway script was written to `/tmp`, outside the repository.

## Post-audit-close addendum

A claim recorded during the audit window ceased to hold after this audit's outputs were written.
I correct it here rather than leave it stale.

At `23:29:14`–`23:30:02` KST — after I had observed them absent, and after `FINAL_AUDIT.md` and
`FINAL_AUDIT.json` were written — the concurrent writer of Finding C recreated the three
v1-namespace governance files:

| Path | Codex pre-cleanup sha256 | sha256 now | Byte-identical |
|---|---|---|---|
| `agent_system/side_gyro_mag_comp/state/STAGE_STATE.v2.json` | `6b439b77…` | `ac097d06…` | no |
| `agent_system/side_gyro_mag_comp/state/DECISION_LEDGER.v2.md` | `d9803219…` | `53c2e48a…` | no |
| `agent_system/side_gyro_mag_comp/state/DEFERRED_REGISTER.v2.md` | `a1230cd7…` | `44bb40be…` | no |

They are **new reconstructions, not byte-restorations** of the deleted originals. Two statements
above are therefore corrected: the three files were absent *when observed* and are present as of
audit close; and `extra_versionable_paths` was 0 *when observed*, with 3 extra untracked files in
the v1 state namespace as of close.

What is unaffected, re-verified after the recreation: **85/85 archive paths present, 0 missing, 0
byte mismatches**; `.codex/config.toml` unchanged at `315ec7d2…`; archived v1 final manifest
**9/9**; final checkpoint digest stable with **9/9** members matching. The three recreated files
are untracked governance artifacts in the v1 state namespace, not members of the v1 archive or of
any checkpoint manifest.

This changes no part of the decision. It further evidences Finding C. I recorded it only — I did
not create, modify, or delete these files.

## Consequence

The study is terminal and blocked. No repair round remains; implementation, G0–G4, and any
scientific claim remain unauthorized and unsupported. Nothing here may be read as evidence about
gyro–magnetometer compensation — no hypothesis was tested.

The final *report* additionally fails independent audit for incomplete disclosure of its own
evidence chain. Because the CONTRACT repair budget is exhausted, this defect is not repairable by
Codex within the current authorization, and it is not mine to repair — it lies in artifacts I am
forbidden to edit. Combined with the concurrent-role-writer collision in Finding C, resolution
requires principal governance adjudication. `human_review_mode` is `FINAL_ONLY`; that review is
now due.
