# research/spike-snn

> **Legacy scope warning:** this directory documents the Euclidean
> `x=[sigma, omega]`, six-dimensional `y=[gyro, delta-angle]` observation
> benchmark. It is not the current right-local Phase 2 gyro-process-input
> architecture, authorization, or evidence. Historical phase/stage names below
> are labels from the source branch only.

Spike-Split KalmanNet: a spiking module in the innovation-covariance branch of
the existing Split-KalmanNet ADCS benchmark, per
`spike_split_kalmannet_minimal_abstract_experiment_plan.md` (target: IAA
Conference on AI for Space).

The plan is explicitly additive — it reuses the existing benchmark, NTD dataset
format, Basilisk generation, EKF and Split-KalmanNet baselines — so this branch
contains only the spike-specific pieces:

```
bench/models/spike_split_knet.py    G2-SNN innovation-covariance branch
bench/models/g1_snn_split_knet.py   G1-SNN ablation
bench/models/spike_ra_knet.py       SNN reliability-attention variant
bench/tasks/generator/basilisk_imu_adcs.py   event-aware measurement
                                    disturbance layer (+166 lines)
12 suite configs, 3 tests, the plan document, the AR0 rebaseline review
```

Base: `research/common-foundation`.

## Why the generator change is here

`bench/tasks/generator/basilisk_imu_adcs.py` gained `event_disturbance` /
`measurement_gyro_bias_jump` support. That key is referenced by exactly two
things in the whole repository: the twelve `suite_basilisk_spike_*` configs and
`bench/tests/test_basilisk_imu_measurement_event.py`. No MEKF or Vizard
document mentions it, and the spike plan lists "event-aware measurement
disturbance layer" as a required component. Hence this branch, on evidence
rather than convenience.

## Verification on this branch

Run as the repository runs `bench/tests` (scripts, not pytest collection):

```
PYTHONPATH=$PWD python bench/tests/test_spike_ra_knet.py
    SpikeRA module, trace, event segmentation, smoke, and metrics checks passed
PYTHONPATH=$PWD python bench/tests/test_spike_split_knet_snn.py
    Split/G2-SNN/G1-SNN module, registry, checkpoint, spike-stat and CPU
    pipeline checks passed
PYTHONPATH=$PWD python bench/tests/test_basilisk_imu_measurement_event.py
    gyro std event=2.83e-02 vs non-event=9.63e-03 — disturbance layer active

registry: ['g1_snn_split_knet', 'spike_ra_knet', 'spike_split_knet'] registered
12 spike suite configs parse
```

## Known limitations

- `bench/models/registry.py` carries **only** the two spike hunks (imports and
  `_REGISTRY` entries). The typed-event-bridge hunk from the same file is on
  `research/mekf-attitude`; the hunks are in different regions, so they were
  split on that evidence and should merge cleanly.
- `bench/tests/run_all.py` is not here — its added block lists MEKF, Vizard and
  spike tests in one hunk, so the spike tests are not yet wired into the
  aggregate runner on this branch. Run them directly, as above.
- These tests are not pytest-collectible; that is the existing `bench/tests`
  convention in this repository, not something this branch introduced.
- No experiment output committed.
