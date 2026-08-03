# Phase 1A Gate B2 Retry Execution Contract
## Post–Gate B1 Amendment A1 Overlay

당신은 `/home/dss-pc-05/bench` repository에서 **Phase 1A Gate B2만 재시도**하는 구현 agent다.

이번 파일은 기존 Gate B2 실행 계약을 폐기하지 않는다. 다음 원본 계약을 먼저
처음부터 끝까지 읽고, 그 전체 내용을 기본 계약으로 사용하라.

```text
docs/research/phase1a/prompts/03B_CODE_AGENT_GATE_B2_BASILISK_FRAME_UNIT_ST_PROMPT.md
```

그 다음 이 파일을 끝까지 읽어라.

이 파일과 원본 Gate B2 계약이 충돌하면 **이 파일의 규칙이 최우선**이다.

이번 재시도의 목적은 Gate B1 Amendment A1으로 해소된 manifest compatibility를
사용하여, 중단됐던 Gate B2를 처음부터 완전하게 수행하는 것이다.

---

# 1. 반드시 추가로 읽을 Amendment 문서와 증거

원본 03B 계약이 요구한 문서에 더하여 다음을 모두 읽어라.

```text
docs/research/phase1a/P1A_GATE_B1_AMENDMENT_A1_CONTRACT.md
experiments/phase1a/reports/P1A_GATE_B1_AMENDMENT_A1_REPORT.md
docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
docs/research/phase1a/P1A_GATE_B1_TEST_MATRIX.md
experiments/phase1a/reports/P1A_GATE_B1_VALIDATION_REPORT.md
```

다음 이전 Gate B2 blocker evidence가 존재하면 원인 확인용으로 읽어라.

```text
experiments/phase1a/agent_logs/03B_manifest_compatibility_probe.txt
experiments/phase1a/agent_logs/03B_blocker_report.txt
가장 최근 experiments/phase1a/agent_logs/03B_*_final.md
```

이전 실행은 구현 실패가 아니라 다음 synthetic-only manifest 검증 때문에
중단됐다.

```text
manifest["generator_id"] = "basilisk-unit-st-v1"

ValueError:
manifest generator_id must equal 'synthetic-unit-st-v1'
```

Gate B1 Amendment A1에서 이 blocker는 해소됐다.

현재 공통 serializer는 다음을 구분한다.

```text
schema_version = p1a-mekf-events-v1
generator_id   = <versioned dataset-generator identity>
```

Gate B2의 정확한 generator identity는 다음이다.

```text
basilisk-unit-st-v1
```

---

# 2. 현재 승인된 Gate B1 Amendment 상태

다음 결과를 현재 baseline으로 사용하라.

```text
Gate A: 55 passed
Gate B1 after Amendment A1: 55 passed
Legacy: 18 passed, 5 subtests passed
Gate B1 reapproval: GO
Gate B2 retry authorized: YES
```

원본 03B 계약에 기록된 `Gate B1: 39 passed`는 Amendment 이전의 과거 기록이다.
이번 실행의 기대 baseline은 **55 passed**다.

시험 개수 자체를 하드코딩한 pass/fail 조건으로 사용하지 말고 exit code 0과
시험 내용의 유지 여부를 판정하되, 실제 실행 개수를 보고서에 기록하라.

---

# 3. Current-tree 및 dirty-tree 정책

실행 시작 시점의 현재 working tree 전체를 사용자가 승인한 기준선으로 사용하라.

다음을 승인 조건으로 검토하거나 비교하지 마라.

```text
branch 이름
HEAD
commit history
과거 commit delta
merge-base
repository 전체 whitespace
기존 visualization 문서
```

다음을 수행하지 마라.

```text
git reset
git restore
git clean
git stash
git add
git commit
git push
git merge
git rebase
git switch
git checkout
```

원본 03B 계약의 recoverable snapshot, 기존 dirty fingerprint, frozen-file
fingerprint, allowlist 및 무결성 검사는 유지하라.

이전 차단 실행이 만든 다음 evidence는 기존 provenance이며 target collision이 아니다.

```text
experiments/phase1a/agent_logs/03B_*
experiments/phase1a/preflight_snapshots/03B_*/**
```

새 재시도 evidence도 같은 prefix 아래 timestamp로 분리하여 생성할 수 있다.

---

# 4. Gate A 및 Gate B1 동결 범위

Gate A source/test/contract는 계속 수정 금지다.

Gate B1 Amendment 이후 다음 파일도 동결 source of truth다.

```text
bench/tasks/generator/mekf_events.py
bench/tasks/generator/unit_st_synthetic.py

tests/test_mekf_events.py
tests/test_unit_st_synthetic.py
tests/test_mekf_replay.py

docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
docs/research/phase1a/P1A_SYNTHETIC_UNIT_ST_CONTRACT.md
docs/research/phase1a/P1A_GATE_B1_TEST_MATRIX.md
docs/research/phase1a/P1A_GATE_B1_AMENDMENT_A1_CONTRACT.md

experiments/phase1a/reports/P1A_GATE_B1_VALIDATION_REPORT.md
experiments/phase1a/reports/P1A_GATE_B1_AMENDMENT_A1_REPORT.md
```

이 파일들은 읽고 import할 수 있으나 수정하지 마라.

Gate B1 serializer/API 변경이 다시 필요하다고 판단되면 수정하지 말고:

```text
BLOCKED_GATE_B1_INTERFACE_CHANGE_REQUIRED
```

로 중단하고 정확한 반례와 최소 변경 범위를 보고하라.

---

# 5. Manifest/serializer 사용 계약

Gate B2 generator는 공통 serializer를 복사하거나 별도 구현해서는 안 된다.

반드시 다음 source of truth를 import하여 사용하라.

```text
bench/tasks/generator/mekf_events.py
```

Gate B2 dataset manifest에는 정확히 다음 identity를 기록하라.

```text
schema_version = "p1a-mekf-events-v1"
generator_id   = "basilisk-unit-st-v1"
```

저장 후 strict load에서는 가능한 경우 반드시 다음과 동등한 검증을 사용하라.

```python
load_event_dataset(
    artifact_path,
    expected_generator_id="basilisk-unit-st-v1",
)
```

다음을 모두 검증하라.

1. recorded generator ID가 `basilisk-unit-st-v1`
2. expected generator ID와 exact match
3. supported schema identity
4. canonical manifest
5. truth/sensor/event/manifest/dataset semantic hash
6. exact three-file artifact
7. `allow_pickle=False`
8. NPZ field/dtype/rank 불변

다음 우회는 금지한다.

```text
generator_id를 synthetic-unit-st-v1로 위장
expected_generator_id 검증 생략
Basilisk 전용 serializer 복사
manifest/hash 검증 생략
Gate B1 파일 수정
```

---

# 6. Exact allowlist와 target collision

원본 03B 계약의 exact allowlist를 그대로 적용하라.

핵심 Gate B2 target은 다음이다.

```text
bench/tasks/generator/basilisk_unit_st.py
tests/test_basilisk_unit_st_generator.py

docs/research/phase1a/P1A_BASILISK_FRAME_CONVENTION_PROOF.md
docs/research/phase1a/P1A_BASILISK_UNIT_ST_CONTRACT.md
docs/research/phase1a/P1A_GATE_B2_TEST_MATRIX.md

experiments/phase1a/reports/P1A_GATE_B2_VALIDATION_REPORT.md
```

이전 차단 보고에 따르면 이 target들은 생성되지 않았다. 실행 시 다시 확인하라.

target이 이미 존재하면 내용을 덮어쓰지 말고, 다음을 구분하라.

- 비어 있거나 이전 차단 실행의 불완전 target
- 다른 agent가 만든 실제 구현
- 이번 재시도 전에 사용자가 의도적으로 배치한 파일

출처가 불명확하거나 비어 있지 않은 구현 target이 존재하면:

```text
BLOCKED_TARGET_EXISTS
```

로 중단하라.

---

# 7. Basilisk frame proof의 현재 예비 근거

이전 차단 실행의 exploratory probe에서는 다음 관계가 관측됐다.

```text
q_NB = normalize(MRP2EP(sigma_BN))
R_NB = quat_to_dcm(q_NB)
R_NB = MRP2C(sigma_BN).T
```

따라서 예비 해석은 다음이다.

```text
MRP2C(sigma_BN) = C_BN
quat_to_dcm(q_NB) = R_NB = C_BN.T
```

하지만 이 결과를 그대로 사실로 복사하거나 PASS 처리하지 마라.

반드시 원본 03B 계약의 executable proof를 완성하라.

필수:

```text
identity
각 축 +90°, -90°
최소 10개 deterministic arbitrary attitude
MRP shadow-set physical invariance
body basis-vector mapping
time-series quaternion continuity
```

최종 문서에는 다음을 명시하라.

```text
MRP2C(sigma_BN)의 검증된 물리적 방향
R_NB/C_BN transpose 관계
최종 sigma_BN -> q_NB 변환식
closed-form basis-vector evidence
shadow-set evidence
```

---

# 8. 동적 proof의 수치 판정 변경

이전 exploratory run에서 `acos` 기반 quaternion geodesic은 동일하거나 거의 동일한
자세에도 약 `2.98e-8 rad`의 float64 수치 바닥을 보였다.

따라서 원본 03B 계약의 동적 proof에서 다음을 최우선 오차로 사용하라.

```text
delta_q = q_reference^{-1} ⊗ q_basilisk
attitude_error_rad = ||Log_q(delta_q)||
```

또는 Gate A의 동등한 shortest-arc quaternion-log helper를 사용하라.

다음도 보조 증거로 기록할 수 있다.

```text
DCM Frobenius difference
basis-vector mapping difference
abs(q_reference dot q_basilisk)
```

`2*acos(abs(dot))`는 보고용 보조값으로 사용할 수 있지만,
machine-roundoff 구간의 최종 pass/fail 단독 기준으로 사용하지 마라.

수치 실패를 tolerance 완화로 숨기지 마라.

필수 동적 proof:

```text
zero rate
각 축 ± 단일 rate
최소 10개 arbitrary body-rate vector
coarse dt
fine dt = coarse / 2
quaternion-log attitude error
local rate-increment error
fine-step convergence
```

이전 예비 결과:

```text
quaternion-log attitude error ≲ 4.7e-16 rad
fine-step local rate-increment max error ≈ 3.67e-14 rad/s
```

이 값들은 참고일 뿐 expected value를 하드코딩하지 마라.
새 실행의 측정값을 기록하라.

---

# 9. Baseline 명령

모든 Python 명령은 다음 interpreter를 명시적으로 사용하라.

```text
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python
```

## 9.1 Gate A

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_conventions.py \
  tests/test_mekf_core.py
```

현재 기준: `55 passed`, exit code 0.

## 9.2 Gate B1 Amendment 이후 전체

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_events.py \
  tests/test_unit_st_synthetic.py \
  tests/test_mekf_replay.py
```

현재 기준: `55 passed`, exit code 0.

## 9.3 Legacy

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_imu_generator.py \
  tests/test_basilisk_mrp_ekf.py \
  bench/tests/test_generator_contract_tg0.py \
  bench/tests/test_adcs_event_metrics.py
```

현재 기준: `18 passed, 5 subtests passed`, exit code 0.

baseline 중 하나라도 실패하면 기존 source/test를 고치지 말고:

```text
BLOCKED_BASELINE_REGRESSION
```

으로 중단하라.

---

# 10. Gate B2 generator 요구사항

원본 03B 계약의 요구사항을 모두 유지한다.

구조:

```text
Basilisk rigid-body truth
├─ sigma_BN
└─ omega_BN_B
        ↓ executable frame/rate proof
locked truth
├─ q_NB_true
├─ omega_true_B
└─ b_g_true
        ↓ project-owned parameterized sensor-output layer
gyro:
  omega_m = omega_true_B + b_g_true + n_g

star tracker:
  q_ST = q_NB_true ⊗ Exp_q(n_ST)
        ↓
Gate B1 typed schema
        ↓
common serializer
generator_id = basilisk-unit-st-v1
        ↓
strict load expected_generator_id=basilisk-unit-st-v1
        ↓
whole-trajectory split
        ↓
Gate B1 direct MEKF replay
```

최초 B2는 계속 zero latency다.

```text
arrival_time_s == measurement_time_s
```

동일 timestamp ordering은 계속:

```text
gyro → star tracker
```

이다.

truth, event label, oracle 값은 replay 공개 입력으로 전달하지 마라.

---

# 11. Determinism과 seed isolation

최소 다음 namespace를 분리하고 manifest에 기록하라.

```text
truth initial condition
gyro bias
gyro white noise
ST tangent noise
ST quaternion representation sign
trajectory split
```

필수 검증:

1. 동일 config/seed 재생성 → 모든 semantic hash 동일
2. gyro-noise seed만 변경 → truth hash 동일, sensor hash 변경
3. ST-noise seed만 변경 → truth hash 동일, sensor hash 변경
4. ST-sign seed만 변경 → physical quaternion 동일, raw representation 필요 시 변경
5. truth seed 변경 → truth hash 변경
6. split seed 변경 → dataset physical hashes 동일, split membership 변경
7. serialization round trip → arrays/hash/replay 동일
8. `q/-q` ST stream → posterior physical q/b/P 및 residual/S 동일
9. strict loader expected-ID mismatch → fail-loud
10. recorded ID tamper/hash corruption → fail-loud

---

# 12. Required evidence files

원본 03B evidence에 더하여 최소 다음을 명확히 남겨라.

```text
experiments/phase1a/agent_logs/03B_manifest_compatibility_after_a1.txt
experiments/phase1a/agent_logs/03B_frame_proof.txt
experiments/phase1a/agent_logs/03B_shadow_proof.txt
experiments/phase1a/agent_logs/03B_dynamic_proof.txt
experiments/phase1a/agent_logs/03B_sensor_equation_proof.txt
experiments/phase1a/agent_logs/03B_hash_seed_property_sweep.txt
experiments/phase1a/agent_logs/03B_new_tests.txt
experiments/phase1a/agent_logs/03B_gate_a_regression.txt
experiments/phase1a/agent_logs/03B_gate_b1_regression.txt
experiments/phase1a/agent_logs/03B_legacy_regression.txt
experiments/phase1a/agent_logs/03B_dirty_tree_integrity.txt
experiments/phase1a/agent_logs/03B_agent_only.patch
experiments/phase1a/agent_logs/03B_agent_only_stat.txt
```

`03B_manifest_compatibility_after_a1.txt`에는 최소 다음을 기록하라.

```text
recorded generator_id
strict expected generator_id
save/load result
schema_version
manifest hash
dataset hash
identity mismatch negative result
```

---

# 13. 완료 판정

정상 완료 시 다음 형식으로 보고하라.

```text
Status: PASS_GATE_B2

Runtime identity: PASS
Static sigma_BN -> q_NB frame proof: PASS
MRP shadow invariance: PASS
omega_BN_B sign/frame/unit proof: PASS
Constant-rate dynamics/convergence: PASS
Basilisk truth generation: PASS
Gyro sensor layer: PASS
Star-tracker sensor layer: PASS
Gate B1 schema/serialization reuse: PASS
Amendment A1 generator identity compatibility: PASS
Strict expected-generator-ID load: PASS
Determinism/semantic hashes: PASS
Seed isolation: PASS
Trajectory split: PASS
Direct replay: PASS
Truth boundary: PASS
Numerical/replay safety: PASS
Gate A regression: PASS
Gate B1 regression: PASS
Legacy regression: PASS
Dirty-tree integrity: PASS

Gate B2: GO
Gate C authorized: YES
```

실패 또는 차단 시 다음을 명확히 보고하라.

```text
Status
최초 실패한 계약
실제 반례/예외
생성·수정된 파일
Gate A/B1 동결 파일 변화 여부
dirty-tree integrity
추가 승인이 필요한 exact scope
```

Gate C로 자동 진행하지 마라.

---

# 14. 종료 조건

이번 실행은 Gate B2에서 종료한다.

다음은 시작하지 마라.

```text
Gate C canonical metric
Gate D adapter/runner/registry
nonzero latency/OOSM
outage/false ST/outlier
magnetometer/sun sensor
orbit/environment
controller/closed loop
KalmanNet
ANN
SNN
FPGA
Package C
```
