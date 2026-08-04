# Exact-Resume Certification Matrix

`supports_exact_resume` is not a property of a model name. It is a property of
the tuple below, and it is true only where a parity test has actually been run
(ADR-CSR-013).

```
model_id | implementation_id | checkpoint_schema_version | resume_boundary
         | precision | device_class | num_workers | training_mode
```

**Anything not listed as certified is uncertified.** The default answer is
"no", not "probably".

## 1. Certified

| model_id | implementation_id | boundary | precision | device | workers | training mode |
|---|---|---|---|---|---|---|
| `kalmannet_tsp` | `bench_kalmannet_tsp_adapter_v1` | `optimizer_update` | fp32 | cpu | 0 | `supervised_single_optimizer` |
| `split_knet` | `bench_split_knet_adapter_v1` | `optimizer_update` | fp32 | cpu | 0 | `supervised_single_optimizer_split_deviation` |

Schema version 1 for both.

### What "certified" means here

Training straight through *N* updates and training *K* updates → interrupt
checkpoint → **fresh process** → restore → finish, produce **bitwise identical**:

- final model `state_dict` (sha256 over tensor bytes)
- optimizer state (Adam moments)
- the full per-update training-loss sequence
- the full validation history
- optimizer update count and batch-plan position
- best step and best metric

Equality is bitwise, not `allclose`. No tolerance was introduced.

Evidence: `tests/test_control_exact_resume_certification.py`.

### The tests are shown to be sensitive

A parity test that cannot fail proves nothing. Three mutation probes are part
of the suite, and each must fail:

| Probe | Result |
|---|---|
| Drop Adam's optimizer state before resuming | detected — weights diverge |
| Shift the batch cursor by one | detected — weights diverge |
| Drop Split's GRU initial hidden state | refused at restore; and without the guard, weights diverge |

Every resume in the suite also constructs the resuming adapter with a
**different seed** than the interrupted run, so any state being carried
implicitly rather than through the checkpoint shows up as a failure.

## 2. Not certified

| model_id | Why |
|---|---|
| `adaptive_knet` | Separate adapt-phase lifecycle and update counter, not modelled by this schema |
| `maml_knet` | Meta inner/outer-loop cursor and task sampling not captured |
| `me_split_knet_v0` | Measurement-enhancer lifecycle unaudited |
| `mb_kf` | No learning lifecycle; resume is not a meaningful concept |

These adapters do not inherit `CheckpointableAdapterMixin`, so they cannot
acquire the capability by accident (DND-CSR-013). A test asserts this.

## 3. Uncertified envelopes — for every model

A CPU/fp32/single-worker result says nothing about these, and is not
generalised to them (DND-CSR-008):

| Dimension | Uncertified values |
|---|---|
| `device_class` | `cuda`, `gpu`, `mps` |
| `precision` | `fp16`, `bf16`, AMP |
| `num_workers` | anything other than `0` |
| distributed | any |
| gradient accumulation | any |
| resume boundary | anything other than `optimizer_update` |

`is_certified()` returns `False` for these even when the model is otherwise
certified — verified by test.

## 4. Certification ≠ paper fidelity

Split-KalmanNet's adapter uses **one optimizer slot**, not the paper's
alternating optimization. Certifying exact resume says the *implementation*
resumes exactly; it says nothing about whether the implementation matches the
paper. `paper_fidelity_status` remains a separate field and is unchanged by
this tranche.

## 5. Where to read it at runtime

```bash
curl -s http://127.0.0.1:8765/api/v1/capabilities/exact-resume | jq
```

Read-only. Certification rows are also stored in the registry's
`exact_resume_certifications` table (`seed_certifications()`).
