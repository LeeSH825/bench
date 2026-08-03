# Phase 1A Gate B2 Retry — q_NB Convention Erratum Overlay

당신은 `/home/dss-pc-05/bench` repository에서 Phase 1A Gate B2를 재시도하는 구현 agent다.

먼저 다음 두 실행 계약을 순서대로 처음부터 끝까지 읽어라.

1. `docs/research/phase1a/prompts/03B_CODE_AGENT_GATE_B2_BASILISK_FRAME_UNIT_ST_PROMPT.md`
2. `docs/research/phase1a/prompts/03B1_CODE_AGENT_GATE_B2_RETRY_AFTER_MANIFEST_AMENDMENT_PROMPT.md`

그 다음 이 파일을 끝까지 읽어라.

충돌 시 우선순위는 다음과 같다.

```text
이 파일
> 03B1 retry overlay
> 원본 03B Gate B2 계약
```

이번 파일은 Gate B1 Amendment A1 이후 Gate B2 재시도를 승인하면서,
Gate B1 event-contract 문서에서 발견된 `q_NB` active/passive 용어 충돌을
실행 가능한 frame proof로 해결하기 위한 추가 계약이다.

---

# 1. 현재 승인 상태

다음 baseline을 사용하라.

```text
Gate A: 55 passed
Gate B1 after Amendment A1: 55 passed
Legacy: 18 passed, 5 subtests passed
Gate B1 reapproval: GO
Gate B2 retry authorized: YES
```

Gate B1 Amendment A1에서 다음 계약은 이미 통과했다.

```text
schema_version = p1a-mekf-events-v1
generator_id   = basilisk-unit-st-v1
strict expected-generator-ID load
```

manifest identity blocker는 해소됐다.

---

# 2. 발견된 문서 용어 충돌

Gate A source of truth는 다음을 고정한다.

```text
q_NB:
- scalar-first Hamilton
- active body-to-navigation quaternion
- R_NB(q) maps B-frame coordinates to N-frame coordinates
- C_BN = R_NB.T
```

그러나 현재 다음 문서의 field table에는 두 곳이 `passive q_NB`로 적혀 있다.

```text
docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md

star_tracker_q_NB
q_true_NB
```

이 용어를 근거로 quaternion 방향을 재해석하지 마라.

Gate A active B-to-N 계약이 상위 source of truth다.

이번 Gate B2에서 Basilisk executable frame proof를 수행하여 실제 generated
arrays가 active B-to-N `q_NB`인지 확인하라.

---

# 3. 문서 정정 허용 조건

다음 조건을 모두 만족한 뒤에만
`docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md`를 문서 수준에서 수정할 수 있다.

1. identity 및 각 축 ±90° basis-vector proof 통과
2. 최소 10개 arbitrary attitude proof 통과
3. MRP shadow-set physical invariance 통과
4. `R_NB = quat_to_dcm(q_NB)`가 B 좌표를 N 좌표로 매핑함을 증명
5. `MRP2C(sigma_BN) = C_BN = R_NB.T` 관계를 증명
6. synthetic Gate B1 generator/replay가 동일 active convention을 사용함을
   기존 tests 또는 추가 read-only probe로 확인
7. Gate A와 Gate B1 regression 모두 통과

조건이 통과하면 다음 두 문구를 정확한 의미로 정정하라.

```text
Scalar-first Hamilton passive q_NB
```

→

```text
Scalar-first Hamilton active body-to-navigation q_NB
```

정정 대상은 다음 두 field description으로 제한한다.

```text
star_tracker_q_NB
q_true_NB
```

문서에 `Gate B2 convention erratum` 절을 append하여 다음을 기록하라.

- 기존 `passive` 표현은 documentation defect였음
- array key, dtype, rank, byte representation, serializer, hash domain은 변경 없음
- code behavior와 physical dataset semantics는 변경 없음
- executable basis-vector proof로 active B-to-N 의미를 확인함
- `R_NB`와 `C_BN` 관계

이 문서 정정은 schema migration이 아니다.

다음을 변경하지 마라.

```text
schema_version
generator_id
NPZ keys
dtype
rank
event ordering
hash domains
replay API
Gate B1 source code
Gate B1 tests
```

---

# 4. 추가 allowlist

원본 03B/03B1 allowlist에 더하여 다음 한 파일의 문서-only 수정만 허용한다.

```text
docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
```

다음 provenance 문서도 생성할 수 있다.

```text
experiments/phase1a/agent_logs/03B_qnb_convention_erratum.txt
```

`03B_qnb_convention_erratum.txt`에는 다음을 기록하라.

```text
Gate A locked convention
기존 B1 문서 문구
static basis-vector proof 결과
arbitrary attitude max error
shadow-set max error
verified R_NB/C_BN relation
synthetic semantic probe 결과
문서 변경 전후 exact diff
code/schema/hash behavior unchanged 여부
```

실행 가능한 proof가 active convention과 일치하지 않으면 문서를 수정하지 말고:

```text
BLOCKED_QNB_CONVENTION_CONFLICT
```

로 중단하라.

---

# 5. Gate B2 frame proof

원본 03B/03B1의 모든 frame proof 요구를 유지한다.

최종적으로 다음 관계를 증명하거나 반증하라.

```text
q_NB = normalize(MRP2EP(sigma_BN))
R_NB = quat_to_dcm(q_NB)
MRP2C(sigma_BN) = C_BN
R_NB = C_BN.T
```

필수 case:

```text
identity
+x90, -x90
+y90, -y90
+z90, -z90
최소 10개 deterministic arbitrary attitude
MRP shadow set
```

단순 변환 round trip만으로 PASS하지 마라.
closed-form expected basis mapping을 사용하라.

---

# 6. 동적 proof

`omega_BN_B`가 Gate A propagation의 body-frame angular rate와 같은 sign,
frame, unit을 사용하는지 다음으로 증명하라.

```text
q_NB(t+dt) ≈ q_NB(t) ⊗ Exp_q(omega_BN_B * dt)
```

주 attitude error는 quaternion-log로 계산하라.

```text
delta_q = q_reference^-1 ⊗ q_basilisk
error_rad = ||Log_q(delta_q)||
```

`acos(abs(dot))`는 machine-roundoff 근처의 단독 pass/fail 기준으로 사용하지 마라.

---

# 7. Manifest 및 serializer

Gate B2 dataset은 반드시 다음을 사용한다.

```text
schema_version = p1a-mekf-events-v1
generator_id   = basilisk-unit-st-v1
```

공통 serializer:

```text
bench/tasks/generator/mekf_events.py
```

strict load:

```python
load_event_dataset(
    artifact_path,
    expected_generator_id="basilisk-unit-st-v1",
)
```

producer-specific serializer를 만들지 마라.

---

# 8. 완료 판정

정상 완료 시 기존 Gate B2 PASS 항목에 다음을 추가하라.

```text
q_NB active/passive executable resolution: PASS
Gate B1 convention documentation erratum: PASS
Event schema physical meaning unchanged: PASS
```

최종 형식:

```text
Status: PASS_GATE_B2

Runtime identity: PASS
Static sigma_BN -> q_NB frame proof: PASS
MRP shadow invariance: PASS
q_NB active/passive executable resolution: PASS
omega_BN_B sign/frame/unit proof: PASS
Constant-rate dynamics/convergence: PASS
Basilisk truth generation: PASS
Gyro sensor layer: PASS
Star-tracker sensor layer: PASS
Gate B1 schema/serialization reuse: PASS
Amendment A1 generator identity compatibility: PASS
Strict expected-generator-ID load: PASS
Gate B1 convention documentation erratum: PASS
Event schema physical meaning unchanged: PASS
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

Gate C로 자동 진행하지 마라.
