# Phase 1A Chat–Code CLI 실행 방식과 저장소 배치

## 1. 최종 권장안

Phase 1A는 다음 혼합 방식으로 수행한다.

- Chat: 연구·수학 계약 검토, repository audit 결과 검토, 구현 범위 승인, test 결과 해석, 다음 Gate 판단
- Code CLI agent: 실제 repository 탐색, 코드 작성·수정, test 실행, config와 report 생성, diff 요약
- 저장소: 별도 standalone repository를 만들지 않는다. 기존 benchmark repository와 같은 Git history를 사용한다.
- 작업 폴더: 현재 working tree를 직접 변경하지 않고 새 Git worktree `AI-ADCS-SNN-phase1a`를 만든다.
- 구현 위치: 기존 legacy MRP/EKF 파일을 직접 MEKF로 변형하지 않고, 같은 repository 안에 새 attitude math, MEKF core, sensor-packet, Phase 1A runner/test 모듈을 추가한다.

즉, 물리적으로는 새 폴더에서 작업하지만 논리적으로는 기존 benchmark를 확장한다.

## 2. 권장 Git 준비

현재 repository가 깨끗한지 먼저 확인한다.

```bash
cd <EXISTING_REPO_ROOT>
git status --short --branch
git rev-parse HEAD
```

현재 변경사항이 있다면 필요한 source 변경만 검토하여 checkpoint commit으로 남기거나 patch로 백업한다. `runs/`, cache, log, virtual environment, generated plot은 checkpoint source commit에 포함하지 않는다.

```bash
git diff > ../pre_phase1a_worktree.patch
git diff --cached > ../pre_phase1a_index.patch
```

검증된 base commit에서 새 worktree를 만든다.

```bash
git worktree add ../AI-ADCS-SNN-phase1a -b feat/phase1a-mekf <BASE_COMMIT>
cd ../AI-ADCS-SNN-phase1a
```

Windows PowerShell에서도 동일하게 실행할 수 있다. 경로에 공백이 있으면 따옴표로 감싼다.

## 3. 연구 문서 배치

다음 문서는 Code CLI agent가 repository 내부에서 항상 읽을 수 있게 둔다.

```text
<REPO_ROOT>/
└─ docs/
   └─ research/
      ├─ source/
      │  ├─ 01_AI_ADCS_KalmanNet_Research_Evaluation_Human_Readable.md
      │  ├─ 02_AI_ADCS_KalmanNet_Research_Action_Guidelines.md
      │  └─ 03_AI_ADCS_KalmanNet_Detailed_Phase_Step_Roadmap.md
      ├─ audits/
      │  └─ 2026-07-27/
      │     └─ AUDIT_CURRENT_STATE.md
      ├─ phase0a/
      │  └─ decision_lock/
      │     ├─ README.md
      │     ├─ P0A_PHASE_0A_SYNTHESIS.md
      │     ├─ P0A_REFERENCE_REGISTER.md
      │     ├─ P0A_MANIFEST_AND_QA.md
      │     ├─ P0_00_EVIDENCE_REGISTER.md
      │     ├─ P0_00_DEPRECATED_ASSUMPTIONS.md
      │     ├─ P0_01_DECISION_LEDGER.md
      │     ├─ P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md
      │     ├─ P0_03_TRUTH_MODEL_SPEC.md
      │     ├─ P0_04_SENSOR_ROLE_AND_MODEL_SPEC.md
      │     ├─ P0_04_SENSOR_ERROR_CATALOG.md
      │     ├─ P0_05_MEKF_MATH_CONTRACT.md
      │     ├─ P0_05_MEKF_CONVENTION_TEST_VECTORS.md
      │     ├─ P0_06_NEURAL_INSERTION_OPTIONS.md
      │     ├─ P0_07_CONTEXT_CONTRACT.md
      │     └─ P0A_IMMEDIATE_TEST_SPEC.md
      └─ phase1a/
         ├─ P1A_REPOSITORY_AUDIT.md
         ├─ P1A_IMPLEMENTATION_CONTRACT.md
         ├─ P1A_IMPLEMENTATION_MAP.md
         ├─ P1A_TEST_MATRIX.md
         └─ P1A_GATE_REPORT.md
```

ZIP 자체는 repository에 넣지 않아도 된다. 압축을 풀어 Markdown 원본만 넣는다. Phase 0A 문서는 read-only source-of-truth로 취급한다.

## 4. 권장 코드 목표 구조

아래는 책임 기준의 목표 구조다. read-only audit에서 기존 package convention과 충돌하면 경로 이름은 조정할 수 있지만 책임 분리는 유지한다.

```text
<REPO_ROOT>/
├─ bench/
│  ├─ attitude/
│  │  ├─ __init__.py
│  │  ├─ conventions.py
│  │  ├─ quaternion.py
│  │  └─ so3.py
│  ├─ filters/
│  │  └─ mekf/
│  │     ├─ __init__.py
│  │     ├─ types.py
│  │     ├─ discretization.py
│  │     ├─ measurements.py
│  │     ├─ kinematic.py
│  │     └─ diagnostics.py
│  ├─ models/
│  │  └─ kinematic_mekf_adapter.py
│  ├─ tasks/
│  │  └─ generator/
│  │     ├─ sensor_packets.py
│  │     ├─ basilisk_phase1a.py
│  │     └─ phase1a_profiles.py
│  ├─ metrics/
│  │  ├─ attitude_geodesic.py
│  │  └─ consistency.py
│  ├─ runners/
│  │  └─ run_phase1a.py
│  └─ configs/
│     └─ phase1a/
│        ├─ unit_st_zero_noise.yaml
│        ├─ unit_st_constant_bias.yaml
│        ├─ unit_st_noisy.yaml
│        └─ c1_matched_baseline.yaml
├─ tests/
│  ├─ unit/
│  │  ├─ attitude/
│  │  └─ mekf/
│  └─ integration/
│     └─ phase1a/
├─ experiments/
│  └─ phase1a/
│     ├─ manifests/
│     ├─ reports/
│     └─ scripts/
└─ docs/research/...
```

### 초기에는 연결하지 않을 것

- `third_party/` 수정
- 기존 `basilisk_mrp_ekf.py`를 MEKF로 변형
- 기존 MRP+angular-rate task의 output schema 변경
- 기존 `basilisk_adcs.py` 또는 `basilisk_imu_adcs.py`의 public behavior 변경
- neural model registry 연결
- Split-KalmanNet 재학습
- SNN/FPGA 구현

### 재사용할 가능성이 높은 것

- existing suite/config parser
- deterministic seed와 cache infrastructure
- generated-task contract
- Basilisk dependency/bootstrap code
- report location과 run metadata
- 일부 attitude metric scaffold

### 새로 만들어야 하는 것

- locked quaternion/SO(3) algebra
- 6D `[delta_theta, delta_b_g]` Kinematic MEKF
- multiplicative injection/reset
- star-tracker tangent update
- sensor packet timestamp/arrival/validity contract
- geodesic attitude, bias, NIS, NEES, SPD diagnostics
- Phase 1A dedicated runner와 regression tests

## 5. 기존 benchmark에 통합하는 시점

### Stage A — 독립 수학 core

기존 model registry와 runner에 연결하지 않는다. B1/B3/B4/B5/B6 test만 통과시킨다.

### Stage B — Phase 1A 전용 runner

동일 repository 안에서 `run_phase1a.py`를 사용한다. pre-generated sensor packet replay, UNIT-ST, B2/B7/B8, C1을 검증한다.

### Stage C — 공통 benchmark 연결

Phase 1A Gate 통과 후 `kinematic_mekf_adapter`와 task registration을 기존 suite/registry에 연결한다. 이 시점이 Phase 2 common-shell baseline의 출발점이다.

이 순서가 필요한 이유는 MEKF core 오류가 기존 training runner, cache, legacy MRP task 문제와 섞이는 것을 막기 위해서다.

## 6. 각 도구의 역할

| 작업 | Chat | Code CLI agent |
|---|---:|---:|
| MEKF convention 변경 판단 | 주도 | 변경 금지, 영향 보고 |
| source 문서 간 모순 검토 | 주도 | 발견·보고 |
| repository 구조 탐색 | 결과 검토 | 주도 |
| 코드 diff 작성 | 검토 | 주도 |
| unit/integration test 실행 | 결과 해석 | 주도 |
| Package B/C 합격 판단 | 주도 | evidence 생성 |
| neural 단계 진입 결정 | 주도 | 임의 진행 금지 |

## 7. 매번 Chat으로 가져올 최소 결과

```text
1. git commit hash와 branch
2. git diff --stat
3. 변경 파일 목록과 각 파일 책임
4. 실행한 command 전체
5. test pass/fail summary
6. 실패 test의 실제 출력
7. generated report/manifest 경로
8. locked convention과 다른 구현이 발견됐는지
9. legacy behavior가 변했는지
10. 다음 작업 제안
```
