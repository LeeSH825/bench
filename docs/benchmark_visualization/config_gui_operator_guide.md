# Starting a Run from the Browser

For the read-only dashboard and the Stop/Resume controls, see
`operator_quickstart.md` and `write_control_operator_guide.md`. This page is
only about the **New run** workflow.

## 1. Turn on write mode

The New run page exists in the read-only build, but it will not offer a Launch
button — it tells you the build is read-only and stops there.

```bash
export BENCH_CONTROL_ENABLE_WRITES=1
export BENCH_CONTROL_ROOT=~/bench-control

python -m bench.control.api.app --port 8765 --control-root "$BENCH_CONTROL_ROOT"
python -m bench.ui.dash_app     --port 8766 --api http://127.0.0.1:8765
```

Both must bind loopback. Write mode refuses a public bind at startup, because
there is no authentication.

Open <http://127.0.0.1:8766/new-run>.

## 2. The five steps

**1 · Choose preset.** The dropdown lists only configs Git tracks under
`bench/configs`. Dropping a file there does not make it appear; commit it, or
use one of the existing presets. The summary shows the file, its digest, and
which of its models can actually be launched here.

**2 · Configure.** Two tabs over one draft — a form, and the raw YAML. Editing
either produces the same run; use whichever you prefer.

Fields are labelled **structural** or **operational**:

- *structural* (budget, batch size, learning rate, precision, determinism)
  changes `variant_id`. The result is a **different experiment**, not a rerun.
- *operational* (description, telemetry, artifact emission) does not.

Device, precision, workers and gradient accumulation are fixed to the certified
envelope (CPU / fp32 / 0 / 1). They are shown so you know what you are getting,
not so you can change them.

Anything the schema does not model stays in the YAML untouched and is listed as
"preserved but unmanaged".

**3 · Validate.** Nothing is created. You get the resolved spec, the hashes,
the `training_path_id`, and a field-by-field diff against the preset. If the
config cannot run, you get the reason and the Launch button stays disabled.

**4 · Review.** Read the identity block. If `structural_config_hash` changed
from the preset, you are starting something new — the page says so.

**5 · Launch.** One click. The button disables itself, and the request carries
a stable idempotency key, so a double-click cannot produce two runs. The status
panel links to the run.

## 3. What "launched" means

The `LAUNCH_RUN` action reaching COMPLETED means **the worker started** — not
that the benchmark finished. Follow the run's own state:

```
CREATED → VALIDATING → QUEUED → STARTING → RUNNING → COMPLETED
```

If the action reports FAILED, no worker started; the allocated run (if there
was one) is CANCELLED with `exit_code 52`, and the failure reason is on the
action.

## 4. Stopping and resuming what you launched

A run launched here is an ordinary control-plane run. If its
`training_path_id` is `control_resumable_v1` you get **Stop safely** and
**Resume training** on its page, with the same guarantees as a CLI-launched
run: the stop lands on an optimizer-update boundary, writes a checkpoint, and
the child resumes from it exactly.

Model-based baselines (`mb_kf_oracle`, `nominal_kf`, …) are launchable but have
no learning lifecycle, so they show no Stop/Resume controls. That is a property
of the filter, not a missing feature.

## 5. Things that are not there

No sweeps, no batch launch, no GPU queue, no scheduling. A GPU run started here
would contend with whatever else is on the device — the page says so, and the
certified envelope is CPU anyway.

Adaptive-KNet, MAML-KalmanNet and ME-Split are visible in preset summaries but
cannot be launched from the browser; they are not certified for this path yet.
Use the CLI.

## 6. When something is refused

| Message | What happened | What to do |
|---|---|---|
| `UNKNOWN_PRESET` | the preset id no longer resolves | reload the page |
| preset changed since preview | someone edited the file after you previewed | revalidate and re-read the diff |
| resolved hash changed since preview | the config resolves differently now | revalidate; do not launch what you did not review |
| idempotency key already used with different configuration | the page was reused across two different drafts | reload `/new-run` to get a fresh key |
| `ADAPTER_NOT_GUI_LAUNCH_CERTIFIED` | that model has no certified GUI launch path | use the CLI |
