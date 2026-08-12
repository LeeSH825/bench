# Portable integration status

Status date: 2026-08-11

This file records repository integration and portability checks only. It does
not reopen, revise, or supersede any frozen side-study scientific decision.

## Side gyro-mag lineage integration

- Canonical v1/v2 merge commit:
  `e3946a826030fd854165855821baaba6cace1015`, with source tip
  `7c0979cc605e1111da4f57d936a42d2812cd328f`.
- Canonical pilot merge commit:
  `63e86703fb936646c2a60f5bbac2e86b8996d55d`, with source tip
  `360b7887b23db81d450ac83bb0d361b6ded7bc8c`.
- The pilot ancestry from `a7ebd82` through `360b788` is preserved. The
  duplicate historical portability commit `7b794b` is excluded.
- The terminal v2 decision remains
  `BLOCKED_CONTRACT_INCOMPLETE_AFTER_ONE_REPAIR`. Integration does not change
  that negative status or authorize training, experiments, repair, or rescue.

## Read-only integrity evidence

- All requested canonical commits are ancestors of the integration branch.
- The v2 namespace has no byte difference from terminal commit `7c0979c` to
  the integrated `HEAD`.
- The frozen v1 archive members plus `.codex/config.toml` have no byte
  difference from archive commit `9cf80cc` to the integrated `HEAD`.
- The terminal v2 namespace is an exact 25-path closure relative to `9cf80cc`.
- All 10 contract checkpoint-manifest members and all 9 final
  checkpoint-manifest members match their recorded SHA-256 digests.
- The v1 red tests and pilot integrity tests passed 48 tests. This is
  portability/integrity evidence only, not a new scientific run or result.

The intrinsic v2 contract validation passes all 6 checks when invoked with the
workspace time-of-check comparison disabled. The full historical CLI validator
returns the exact expected limitation:

`EXPECTED_HISTORICAL_SCOPE_FAIL: CHANGED_PATHS does not equal actual v2 git changes from archive baseline`

The limitation is expected because the frozen contract-stage changed-path list
predates the terminal audit/final/handoff closure in the same v2 namespace. It
is not merge corruption and must not be repaired by changing frozen v2 files.

## Portable package boundary

The wheel includes both side implementation namespaces and their YAML
contracts:

- `bench.side_gyro_mag_comp_v1`
- `bench.side_gyro_mag_comp_pilot`
- `bench/configs/side_gyro_mag_comp_v1.yaml`
- `bench/configs/side_gyro_mag_comp_pilot.yaml`

Frozen evidence under `experiments/`, side-study control records, generated
root `runs/`, root `reports/`, and data caches are not wheel payloads.
