# Write-Control UI Contract

## 1. Layering

Dash callbacks run **server-side** and talk only to `ApiClient`. No callback
imports a registry, adapter, trainer or `WorkerManager`, and the browser never
calls the API directly — the control header and idempotency key are set in the
Dash server process, where page JS cannot forge them.

## 2. Controls

Rendered only when the API reports `write_control_enabled`. Run Detail only;
no bulk actions in the runs table.

| Control | Shown when |
|---|---|
| Stop safely | `action_eligibility.stop_action.eligible` |
| Resume training | `action_eligibility.resume_action.eligible` |

Every condition and reason comes from the API. The UI never recomputes them.

## 3. Stop safely

> 현재 optimizer update를 완료한 뒤 검증된 interrupt checkpoint를 저장하고
> 종료합니다. 즉시 종료되지 않을 수 있습니다.

Also stated: this is a graceful stop, not a kill; if the checkpoint cannot be
written the run becomes `FAILED` (exit 50) and is **not** resumable.

## 4. Resume training

> 검증된 checkpoint에서 새 child run을 생성해 학습을 이어갑니다.
> 기존 parent run과 checkpoint는 변경되지 않습니다.

Shows the target checkpoint, and states that this is an exact resume rather
than a warm start, and that it creates a new child run.

## 5. Idempotency in the UI

A stable per-(run, action) key lives in a `dcc.Store`, so a double-click or a
re-render reuses it. Verified in a real browser: two rapid clicks on Stop
produce **one** action row. Buttons disable on click; a retry reuses the key
rather than minting a new request.

## 6. Action panel lifetime

The action status panel lives **outside** the polled controls block. It was
inside at first, and every poll re-rendered the block and destroyed the panel
the action callback had just written into — the status and the child link both
disappeared within a second. Anything a callback writes into must not be a
child of something another callback re-renders.

## 7. Durable state is the authority

The panel polls `GET /api/v1/actions/{id}`. Browser memory is never the source
of truth, so a refresh or an API restart shows the same action. If the API is
unreachable the UI reports it and reuses the same key on retry — it never
mints a new one.

## 8. Launch vs training completion

The child link is accompanied by: *"The launch action is complete; the child's
own state tells you whether its training finished."* The two are never
collapsed into one status.

## 9. Ineligible states

Reasons are shown, not hidden:

```
Safe stop unavailable: this run used legacy_train_v1. …
Exact resume unavailable: This checkpoint predates the training-path contract …
```

## 10. Not present

No Force terminate, Warm start, Evaluate checkpoint, Delete, or Clone button.
Asserted in the browser test. The navbar badge reads "write control enabled" in
write mode and "read-only" otherwise.

## 11. Launch (added by the config-GUI tranche)

There is now one launch affordance: the **New run** wizard at `/new-run`
(`config_gui_operator_guide.md`). It is a five-step page, not a button on the
run list, and in the read-only build it renders no Launch control at all.

The same rules as Stop/Resume apply to it: server-side callbacks only, the
browser never calls the API directly, the idempotency key is minted once per
page load and held in a `dcc.Store`, and the button disables itself on click so
a double-click cannot produce two runs. The launch status panel lives outside
the polled block, for the same reason the action panel does.
