# Automatic Blocker Repair — One Round Only

Run only when a Claude audit returns a repairable failure and the current stage has repair count 0.

1. Read the complete audit finding set.
2. Group findings by failure class; do not repair one example at a time.
3. Apply only the minimum counterproposal that covers the full class.
4. Do not change frozen thresholds, model family, feature dimension, conditioning mechanism, dataset split, or test endpoint.
5. Add or strengthen a red-path test for every repaired claim.
6. Increment `repair_round_by_stage` atomically.
7. Seal a new checkpoint and return it to Claude for one re-audit.

If repair count is already 1, do not edit. Finalize the appropriate blocked or rejected decision.

Do not ask the user for authorization.
