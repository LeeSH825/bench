# Claude Independent Implementation Audit — Side Gyro/Mag Compensation

- **Study**: `side-gyro-mag-comp-v1`
- **Stage audited**: IMPLEMENTATION (sealed by Codex)
- **Audit target**: `experiments/side_gyro_mag_comp/implementation/CHECKPOINT_MANIFEST.json`
- **Digest before and after audit**: `68a08285878fa484291697c69e93ea855afe51521d90cc86061bc362b90a90a5` (stable)
- **Decision**: `FAIL_RED_PATH` — repairable, first failure, one repair round remains
- **Counterproposal**: `SC_IMPL_ROUND1_COUNTERPROPOSAL.json`

## Audited predicate

> Every named red-path test in the sealed checkpoint actually becomes red under its own
> counterexample, so that the CF-1 information boundary, the CF-2 split firewall, the frozen
> right-error math, and the repaired SC-01 G3 predicate are verified rather than merely asserted.

**Result: false.**

## Method

I did not adopt the Codex report's conclusions. I re-derived from canonical sources, machine
records and source code, and then applied **24 mutations to an isolated `/tmp` copy** of the
sealed sources. The repository was never modified. Every mutation ran the full 42-test suite
against a verified green baseline.

Independent reproductions, all matching:

| Check | Result |
|---|---|
| Manifest self-digest and all 28 member hashes | match, 0 mismatches |
| Test suite | 42 passed, as claimed |
| Tiny smoke | **bit-exact** — 140 records, 20 N3S blocks, normalization and config hashes identical |
| Smoke checkpoints N0/N1/N2/N3 | all four sha256 identical to sealed manifest |
| Train-only selection | earliest-minimum validation epoch confirmed for all four variants |
| Split | 16/8/20, disjoint, R4 only in test, normalization sources equal the training set |
| Reuse pins and `.codex/config.toml` | byte-identical; no Phase 0-1, main Phase 2 or SpikeRA path touched |

The implementation is genuinely deterministic and reproducible, and the sealed run contains
**no realized contamination**. That is a real strength and I record it plainly.

## Why this still fails

The problem is not what the code does. It is that the suite cannot tell when the code stops
doing it. **18 mutations of invariants the sealed claims assert left all 42 tests green.**

### CF-2 — the split firewall is a declaration no test can falsify

`train_variant` builds its `SplitFirewallRecord` from the literals `train_ids`/`val_ids`
rather than from the IDs the epochs actually consumed, so `validate_firewall` compares those
fields to themselves.

- Checkpoint selection and early stopping evaluated on the **test split** → 42 passed.
- Training loss computed over the **test split** → 42 passed.

CF-1/CF-2 were mandatory blocking carry-forwards from the DR0 re-audit. CF-2 states in terms
that the named test *must become red on any such contribution*. It does not. Only the
normalization channel is genuinely protected — mutating that does go red.

### CF-1 — the vocabulary is protected, the enforcement placement is not

Disabling the forbidden-symbol logic turns two tests red, and the symbol list is recursive and
alias-aware. But all three enforcement call sites can be deleted independently with the suite
green, and `deployable_replay` can be opened to the oracle variants N1/C1 with the suite green.
The only test exercising the validator calls it directly rather than through the deployable
entry points.

### The training-path math is entirely unverified

`study.py` re-implements the whole recursion in torch for training and is never cross-checked
against the numpy runtime operator. Each of these left the suite green:

- propagation with `(gyro + bias)` instead of `(gyro − bias)`
- **left** quaternion injection instead of right
- flipped innovation sign

The equations themselves are correct — I re-derived the right-error definition, the reset
Jacobian, the error dynamics, the magnetometer Jacobian sign and the `K = G1 Hᵀ G2` shape, and
the runtime-side red paths for these are genuine. But the operator that actually trains the
network carries no such protection.

### N3S checkpoint identity is tautological

Both sides of every digest equality are emitted from the same variable, so `verify_n3s_bridge`
compares a value to itself. The consequence is concrete: the N3S target estimator can **skip
loading the N3 checkpoint entirely** and replay an untrained model, with all 42 tests green and
the emitted evidence still asserting identical digests.

This corrected my own earlier reading. I had verified the digest equality across all 20 records
and initially treated it as confirmation; the mutation showed the check could not fail.

The rest of N3S is sound and red-path protected: zero fixed points, preserved raw/corrected/
timestamp/init/owner hashes, a distinct target-owned recurrent lineage, and a non-degenerate
effect on all 20 trajectories.

### Other unprotected invariants

- Removing the residual gyro bias from the sensor packet entirely → green.
- Removing the backbone recurrent reset, in both the runtime and training paths → green.
- Relaxing the repaired G3 strict inequality, and altering its 0.5/0.5 contrast weights → green.
  This is the exact predicate that consumed the single DR0 repair round.
- The feature-dimension test asserts against the implementation's own constant, so it cannot
  detect a change to the frozen 8.

Six further assertions are vacuous, including a causal-prefix test that compares two identical
clones on identical inputs, a gain-shape test whose adversary raises its own error, and a
weak-axis population certificate that cannot fail for any non-empty trajectory.

## G1 and G4 are not decidable by the sealed code

`evaluate_g1_gate` consumes six sensor-metric arrays and `evaluate_g4_gate` consumes two
divergence arrays. **Nothing in the repository produces any of them**, and the frozen
`divergence_threshold_rad = 1.0` is validated by `load_config` and then never read. The gate
arithmetic matches canonical SC-01 and no threshold drifted, but G1's three strict sensor
sub-predicates and G4's no-added-divergence condition cannot currently be resolved. Deferring
these producers to the experiment stage would place their definition at the moment test access
begins, which is exactly what pre-registration exists to prevent.

## Scope

No scope creep. Every hard exclusion is absent from code, config and tests — no attention,
Transformer, SNN, SoW, reliability gate, learned Q/R, uncertainty head, extra sensor,
closed-loop or FPGA path, and no automated search. Every gate constant, seed and population is
frozen in config and re-checked on load. `full_pilot_executed` is false, no gate has seen test
data, and the smoke correctly claims neither performance nor physical covariance validity.

## Governance

Stage state and handoff both validate; the handoff is fresh and names exactly one target; the
one-writer rule held on all data paths; frozen boundaries and reuse pins are intact.

Two defects. `prompts/.../05_CODEX_MINIMAL_EXPERIMENT_EXECUTION.md` was modified at 21:06:34,
inside the implementation window in which Codex was the sole active writer, between two Codex
source edits; every other prompt retains its original 18:48 mtime and the directory is
untracked, so no authorship record exists. The sealed report asserts that "the orchestrator
amended prompt05 G3 under its own authority" and that "this implementation did not edit
prompt05", but no artifact anywhere records such an amendment. That is an unsupported
provenance claim. Substantively nothing drifted — the amended text matches canonical SC-01, and
SC-01 governs gates regardless — so this is a provenance and ownership defect, not a threshold
defect. Relatedly, `CHANGED_PATHS.json` declares no out-of-allowlist modification while listing
that same file. The decision ledger is also missing rows for the DR0 reseal, the DR0 re-audit
PASS and the implementation seal.

## Why `FAIL_RED_PATH` and not `FAIL_INFORMATION_LEAKAGE`

CF-2 nominates `FAIL_INFORMATION_LEAKAGE` as its failure consequence. I did not use it, and
this is not a softening. I positively established by bit-exact reproduction that **no leakage
exists in the sealed artifacts** — normalization came from the 16 training IDs, selection from
the 8 validation IDs, and R4 never entered training or validation. Asserting leakage I had
disproven would itself be an unsupported claim. The honest single predicate is that the
guarantee is unverifiable: CF-1 and CF-2 both remain **NOT SATISFIED and blocking**, and the
firewall is currently a declaration no test can falsify.

## Disposition

- Decision: `FAIL_RED_PATH`, repairable.
- Implementation repair rounds consumed before this audit: 0. One round is now authorized.
- State: `IMPLEMENTATION_REPAIR`, next actor Codex.
- Advance to `ORACLE_HEADROOM` is **not** authorized.
- A second failed implementation audit closes the study with a final blocked result.

The counterproposal is a single class-level repair — bind every declaration to the execution it
describes, then prove each binding with a mutation that must go red — plus the metric producers
needed to keep G1 and G4 decidable and the declaration corrections. No threshold, architecture,
split, seed or gate value may change, and no new experiment is authorized.
