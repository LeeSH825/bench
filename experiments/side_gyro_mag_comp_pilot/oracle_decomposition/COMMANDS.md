# Oracle Decomposition Command Log

All commands ran from `/home/dss-pc-05/bench` on branch `codex/side-gyro-mag-comp-pilot` with `/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python`.

## Step 0 verification and commit

```text
TMPDIR=/tmp /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest -q -s tests/test_side_gyro_mag_comp_pilot_integrity.py
TMPDIR=/tmp /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest -q -s tests/test_mekf_conventions.py tests/test_mekf_core.py tests/test_mekf_metrics.py
git commit -m "repair(side-gyro-mag-comp): strengthen pilot evidence without changing G0/G1"
```

Final results: six strengthened integrity checks passed; 98 canonical MEKF tests passed. Step 0 commit: `e622ca0e247b8a8e41d9f670eeb93f30a87d84a1`.

## Checkpoint provenance inspection

The training manifest and all N0/N1 checkpoint payloads were read with `torch.load(..., map_location="cpu", weights_only=False)`. Checkpoint file SHA-256 and state-dictionary SHA-256 were recomputed. The inspection established separate N0/N1 checkpoints and identical normalization digest/source IDs.

## Specification freeze

```text
git add -- experiments/side_gyro_mag_comp_pilot/oracle_decomposition/ORACLE_DECOMPOSITION_SPEC.md
git commit -m "preregister gyro-mag oracle decomposition diagnostic"
```

Spec-only commit: `c93d764`.

## New evaluation

```text
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m py_compile bench/side_gyro_mag_comp_pilot/oracle_decomposition.py
TMPDIR=/tmp /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m bench.side_gyro_mag_comp_pilot.oracle_decomposition --repo-root /home/dss-pc-05/bench --output-dir /home/dss-pc-05/bench/experiments/side_gyro_mag_comp_pilot/oracle_decomposition --config /home/dss-pc-05/bench/bench/configs/side_gyro_mag_comp_pilot.yaml
```

The evaluator reconstructed C0/C1/N0/N1 from committed records, regenerated the committed dataset deterministically, verified every raw-realization digest and N0 checkpoint, evaluated only CG/CM and NG/NM/NGM on R0-R3, enforced exact R0 no-op, and wrote 1,320 new records. No network was trained.

## Preservation checks

```text
sha256sum experiments/side_gyro_mag_comp_pilot/PILOT_SPEC.md
sha256sum experiments/side_gyro_mag_comp_pilot/PER_TRAJECTORY_RECORDS.jsonl
sha256sum experiments/side_gyro_mag_comp_pilot/PAIRED_COMPARISONS.json
sha256sum experiments/side_gyro_mag_comp_pilot/GATE_RESULTS.json
sha256sum .codex/config.toml
```

Expected digests remained respectively `a10d1de9...c15410`, `da1ff165...c407222ae`, `ffe42123...109520ad`, `16c15db7...93140a401`, and `315ec7d2...2c20d`.
