# KalmanNet / Split-KalmanNet Training-State Audit

Written **before** any checkpoint code, because the checkpoint schema is only
correct if it matches the lifecycle that actually exists. Nothing here is
inferred from the papers; every row was read out of the adapter and the
third-party module at the pinned submodule revision.

Submodules audited:

| Repo | Revision |
|---|---|
| `third_party/KalmanNet_TSP` | `828a2cf529bc84f43b37d543d916fe5858054457` |
| `third_party/Split_KalmanNet` | `0d6265668e58e6f934a09212b465a82666e544a6` |

## 1. Side-by-side

| | `kalmannet_tsp` | `split_knet` |
|---|---|---|
| Adapter | `bench/models/kalmannet_tsp.py` → `KalmanNetTSPAdapter` | `bench/models/split_knet.py` → `SplitKNetAdapter` |
| Third-party class | `KNet.KalmanNet_nn.KalmanNetNN` | `GSSFiltering.filtering.Split_KalmanNet_Filter` |
| Weight-holding module | `self.model` | `self._filter_obj.kf_net` (`DNN_SKalmanNet_GSS`) |
| Model state slots | 1 | 1 |
| Optimizer slots | 1 — `Adam(self.model.parameters())` | 1 — `Adam(kf_net trainable params)` |
| Scheduler | none | none |
| GradScaler / AMP | none | none |
| `optimizer.step()` site | after `loss.backward()` and optional `clip_grad_norm_` | same, plus finite-gradient and finite-parameter guards |
| Update counter | `updates_used += 1` immediately after `step()` | same |
| Validation trigger | `updates_used % eval_interval == 0 or updates_used == max_updates` | identical |
| Early stopping | `no_improve_evals` vs `patience_evals`, `min_delta` | identical |
| Best state location | **in memory** — `best_state` dict of CPU clones | **in memory** — same |
| Best state persisted? | only at the very end, into `model.pt` | same |
| Batch source | `while updates < max: for batch in train_dl` | identical |
| Loader | `DataLoader(shuffle=True, generator=g, num_workers=0)`, `g.manual_seed(seed)` | identical |
| Recurrent hidden state across updates | **no** — `init_hidden_KNet()` + `InitSequence()` at the top of every `_forward_sequence` | **no** — `_rollout_one` calls `filter.reset(clean_history=True)` per sequence |
| Extra numerical state outside `state_dict()` | none found | **yes — see §3** |
| `model.pt` today | `{state_dict, best_step, best_val_mse, train_updates_used, model_class}` | same shape |

## 2. Where the safe boundary is

Both loops have the same shape, so both have the same safe boundary:

```
zero_grad → forward → loss → backward → clip → optimizer.step()
→ updates_used += 1
→ observer.metric("loss/train_total", step=updates_used)
→ if due: validation, best/early-stop update
→ ← checkpoint-safe boundary
```

The boundary is placed *after* the validation block deliberately. Checkpointing
between the update and the validation that update was due to trigger would
produce a resumed run whose validation history has an extra or missing entry —
which is precisely the discrepancy the first draft of the parity harness
produced before the stop was modelled correctly.

`model.pt` is **not** this state. It has no optimizer, no RNG, and no cursor:
loading it is a warm start. Checkpoint v1 is therefore a new artifact rather
than a rename (DND-CSR-001).

## 3. The finding that changed the design

Split-KalmanNet has numerically-significant state that `state_dict()` does not
contain.

`third_party/Split_KalmanNet/GSSFiltering/dnn.py`, `DNN_SKalmanNet_GSS.__init__`:

```python
self.hn1 = torch.randn(self.gru_n_layer, self.batch_size, self.gru_hidden_dim)
self.hn1_init = self.hn1.detach().clone()
...
self.hn2 = torch.randn(...)
self.hn2_init = self.hn2.detach().clone()

def initialize_hidden(self):
    self.hn1 = self.hn1_init.detach().clone()
    self.hn2 = self.hn2_init.detach().clone()
```

and `Split_KalmanNet_Filter.reset()` calls `kf_net.initialize_hidden()`.

So `hn1_init` / `hn2_init` are **random constants drawn at construction from
the process RNG**, reused as the GRU's initial hidden state for *every*
sequence of *every* update. They are plain attributes, not registered buffers,
so they never appear in `state_dict()`.

How this was found: the first resume harness deliberately built the resuming
adapter with a *different seed*. KalmanNet matched bitwise; Split diverged from
the first post-resume update. Re-running with the same seed made Split match,
which localised the cause to seed-dependent setup state rather than to the
cursor or the optimizer.

Consequences:

* Split's exact-resume claim requires capturing these tensors. They are carried
  in the checkpoint's `extra_state` via `_ckpt_extra_state` /
  `_ckpt_restore_extra`, and declared in `required_conditional_state`.
* A checkpoint that lacks them is **refused** at restore rather than resumed
  approximately — a resume that loads the right weights and still drifts is the
  worst possible failure mode (ADR-CSR §3.2).
* This is handled entirely in the adapter layer. No third-party file is
  modified, so no exception record is needed (ADR-CSR-015).
* Worth noting separately: because `model.pt` also omits these tensors, a
  warm start from `model.pt` does not reproduce Split's evaluation behaviour
  bitwise either. That is pre-existing and out of scope here, but it is a real
  reproducibility caveat for anyone comparing loaded-weight results.

## 4. Why the DataLoader is replaced rather than checkpointed

With `shuffle=True` and a `generator`, the batch order is an emergent property
of a `RandomSampler` consuming that generator. Landing on update *N* means
replaying every draw before it, and the resume claim would then depend on
torch's sampler internals and on worker semantics this tranche has not
certified. Skipping *K* batches from a fresh iterator is the same bet in
different clothing.

Instead the schedule is made explicit (`BatchPlan`, §7 option B): the batch
order is a pure function of `(dataset_length, batch_size, seed, drop_last)`,
each epoch's permutation is drawn from its own generator seeded from
`(seed, epoch)`, and an arbitrary global position is therefore seekable in
O(1). The cursor is two integers plus a plan id.

This is **opt-in**. `train()` is untouched and still uses the existing
DataLoader path, so no existing numerical result moves and the
observer/telemetry inertness certified in the previous tranche still holds.
Resumable training is a second entry point (`resumable_train`), and the
exact-resume certification applies to runs that use it.

## 5. Uncertified by construction

| Model | Why not certified |
|---|---|
| `adaptive_knet` | Separate adapt-phase lifecycle and its own update counter; not modelled by this schema |
| `maml_knet` | Meta inner/outer-loop cursor and task sampling are not captured |
| `me_split_knet_v0` | Measurement-enhancer lifecycle unaudited |
| `mb_kf` | No learning lifecycle; resume is not a meaningful concept |

These adapters do not inherit `CheckpointableAdapterMixin`, so they cannot
accidentally acquire a capability they have not earned (DND-CSR-013).
