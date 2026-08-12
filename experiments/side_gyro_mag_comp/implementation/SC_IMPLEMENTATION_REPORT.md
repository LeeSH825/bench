# SC Minimal Pilot Implementation Report

Decision: `READY_FOR_IMPLEMENTATION_AUDIT`.

The isolated `side_gyro_mag_comp_v1` family implements R0--R4 generation,
whole-trajectory splits, train-only normalization, separate causal gyro and
magnetometer encoders, corrected 3-vectors plus exactly 8D features,
branch-specific FiLM, and the frozen right-local `K = G1 H.T G2` update.

Deployable N0/N2/N3/N3S paths use stripped allowlisted runtime packet,
trajectory-batch, and normalization records. Recursive dataclass, mapping,
list, config, and checkpoint-key inspection rejects regime/event labels,
truth, oracle, future data, all six calibration symbols, aliases, and inverses.
C1/N1 oracle replay and generator calibration remain diagnostic-only.

Training uses a differentiable causal right-local trajectory unroll through
gyro correction, right propagation, magnetometer correction, Split gain,
right injection, and bias update. All four frozen loss weights are consumed.
Checkpoint selection uses actual R0--R3 validation attitude geodesic RMSE only,
with the earliest-minimum tie break. A ranking-disagreement red test prevents
selection by an auxiliary sensor loss.

Realization identity is derived from raw packet values, timestamps, sensor
role, event order, and validity bytes. Duplicate IDs, empty per-regime split
populations, same-label/different-byte realizations, and test-derived
normalization or selection fail closed. Normalization is hashed before test
replay and recorded in each checkpoint. Every normalization, training-loss,
and threshold source must be the exact nonempty unique 16-ID training set;
every stopping and checkpoint-selection source must be the exact nonempty
unique 8-ID validation set. Empty sets, strict subsets, duplicates, and test
contamination are rejected.

The final targeted run passed 42 tests. Direct adversaries include wrong gain
shape and identity reset through `SideEstimator`, trained-checkpoint residual
bias removal and nominal drift, recursive runtime/checkpoint leakage, split and
byte-provenance mutations, an altered reset state, substitution of source
recurrent history for target history, and each protected N3S bridge field.
N3S reset hashes are captured from the actual `MEKFState` and time supplied to
reset. Recurrent hashes are built from transition events emitted by the actual
target-owned Split backbone execution. Every transition contributes the dtype,
shape, and bytes of both post-transition GRU hidden tensors; they are not
reconstructed from an owner, index, shape, trajectory ID, or sequence length.
A same-owner, same-length adversary with different hidden histories produces
different lineage digests. Actual
weak-axis and observable-plane metrics are finite and nonempty.

Executable G0--G4 arithmetic is encoded and boundary-tested. For G3, repaired
and independently audited SC-01 is canonical and supersedes prompt05 only for
this predicate: `T=N3S-0.5*N2-0.5*N3`; PASS requires the paired CI lower
endpoint `>0`; a CI crossing zero is `INCONCLUSIVE_UNDERPOWERED`; the old
disjunct is forbidden.
The paired CI implementation resamples whole trajectory-ID rows across all
three seeds. A heterogeneous multi-seed red fixture deterministically
distinguishes this clustered result from independent-cell and independently
per-seed resampling.

The two-epoch tiny smoke produced 140 finite long-form records with keys
`{experiment,regime,split,model,window,metric,seed,trajectory_id}`: four test
trajectories per R0--R4 for C0/C1/N0/N1/N2/N3/N3S. Each of 20 N3S records
proves identical N3/N3S checkpoint-file and state-dict digests plus protected
hashes for raw packets, corrected values, timestamps, initialization, and
target ownership. Recurrent digests are intentionally excluded from generic
N3/N3S equality because the feature intervention may change later backbone
hidden states. Instead, each N3S replay independently emits its target-owned
hidden-byte lineage digest and transition count, and the bridge verifies those
values directly.

This is interface and numerical smoke evidence only. No full pilot, G0--G4
scientific decision, comparative performance claim, physical covariance
claim, NIS/NEES claim, or test-driven tuning occurred.

Control provenance: the orchestrator amended prompt05 G3 under its own
authority after the repaired SC-01 audit. This implementation did not edit
prompt05; its final bytes are included as a read-only pinned manifest member.
