# Pilot Command Log

All commands were run from `/home/dss-pc-05/bench`. The Python runtime was `/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python`.

## Preservation and branch setup

```text
git show-ref --verify refs/heads/codex/side-gyro-mag-comp-v1-terminal-archive
git cat-file -t 235619cbd7b7af7dcc24db89c247673cd72a0363
git cat-file -t 9cf80cc85f2a01297cfd7932c1ce3cfcd87a15c0
git add -- agent_system/side_gyro_mag_comp_v2 docs/research/side_gyro_mag_comp_v2 experiments/side_gyro_mag_comp_v2
git commit -m "archive terminal side gyro-mag compensation v2 evidence"
git switch -c codex/side-gyro-mag-comp-pilot 052d2f7217b964b1fa4e80bd643716b433780f08
git restore --source=9cf80cc85f2a01297cfd7932c1ce3cfcd87a15c0 --worktree -- .codex/config.toml
sha256sum .codex/config.toml
```

The v2 archive commit is `7c0979c`. The preregistered specification was separately committed before any test or data access:

```text
git add -- experiments/side_gyro_mag_comp_pilot/PILOT_SPEC.md
git commit -m "preregister side gyro-mag compensation pilot"
git rev-parse a7ebd82
```

Result: `a7ebd8247bf00cbca888c08feb6dafa6ce6ebe40`.

## Sanity and integrity

```text
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m py_compile bench/side_gyro_mag_comp_pilot/data.py bench/side_gyro_mag_comp_pilot/model.py bench/side_gyro_mag_comp_pilot/study.py bench/side_gyro_mag_comp_pilot/runner.py tests/test_side_gyro_mag_comp_pilot_integrity.py
TMPDIR=/tmp /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest -q -s tests/test_mekf_conventions.py tests/test_mekf_core.py tests/test_mekf_metrics.py
TMPDIR=/tmp /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest -q -s tests/test_side_gyro_mag_comp_pilot_integrity.py
```

Results: 98 canonical MEKF tests passed; 6 pilot integrity tests passed.

## Smoke and gated pilot

```text
TMPDIR=/tmp /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m bench.side_gyro_mag_comp_pilot.runner --config bench/configs/side_gyro_mag_comp_pilot.yaml --output-dir experiments/side_gyro_mag_comp_pilot/smoke --tiny-smoke
TMPDIR=/tmp /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m bench.side_gyro_mag_comp_pilot.runner --config bench/configs/side_gyro_mag_comp_pilot.yaml --output-dir experiments/side_gyro_mag_comp_pilot --pilot
```

The pilot runner executed G0, continued to G1 after G0 PASS, and stopped after G1 FAIL. It did not train or evaluate N3 and did not run G2-G4.

## Final preservation verification

```text
git show-ref --verify refs/heads/codex/side-gyro-mag-comp-v1-terminal-archive
git cat-file -t 235619cbd7b7af7dcc24db89c247673cd72a0363
git cat-file -t 9cf80cc85f2a01297cfd7932c1ce3cfcd87a15c0
git branch --contains 7c0979c
sha256sum .codex/config.toml
git branch --show-current
git merge-base --is-ancestor 052d2f7217b964b1fa4e80bd643716b433780f08 HEAD
```

The configuration digest remained `315ec7d2282939ea0344b6de5ec5dc2c6dbab3bbee91fa3d1e63912b29a2c20d`.
