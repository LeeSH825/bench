# P1 Exit CONDITIONAL_GO Approval and Condition-Closure Handoff

- 결정일: 2026-08-02
- Phase 1B Step 2: **PASS_P1B_STEP2_SENSOR_FUSION_C4**
- 현재 P1 Exit: **CONDITIONAL_GO**
- 다음 작업: **P1 Exit Condition Closure — Posterior Covariance Calibration and Initial-Transient Separation**
- Phase 2 구현: **아직 승인하지 않음**

## 1. 승인된 Phase 1 결과

다음을 최종 Phase 1 evidence로 동결한다.

- Phase 1A MEKF math, typed events, Basilisk truth, canonical metrics, runner integration
- UNIT-ST C1/C2/C3/C5 paired N=50
- MAIN-FUSION stationary paired N=50
- STRESS-MAG paired N=50
- C4 combined-event paired N=50
- sensor-specific mag/sun/ST NIS
- all-one oracle exact equivalence
- process-only / measurement-only / full oracle / wrong-side ablations
- same-realization and truth/oracle information boundaries
- all regressions and dirty-tree integrity

## 2. CONDITIONAL_GO의 정확한 조건

MAIN-FUSION의 settled sensor consistency는 다음과 같이 matched sanity를 통과했다.

```text
mag NIS/DOF = 1.023
sun NIS/DOF = 1.000
ST  NIS/DOF = 1.092
```

그러나 settled posterior state consistency는:

```text
NEES/DOF = 1.873
```

으로 남아 있다.

따라서 다음을 구분해야 한다.

1. 초기 attitude/bias uncertainty가 P0에 충분히 반영되지 않은 transient 문제인지
2. settled process model/Qg/Qb가 실제 error growth를 과소평가하는지
3. attitude block, bias block, cross-covariance 중 어느 부분이 주요 원인인지
4. 단순 공분산 팽창으로 오차를 숨기는 것이 아니라, 사전 선언된 classical P0/Q calibration으로 개선 가능한지

## 3. 동결 baseline

### Primary

```text
F-BASE
```

### Frozen sensitivity comparator

```text
F-TUNED:
s_Qg   = 0.125
s_Qb   = 0.125
s_R_ST = 8.0
```

F-TUNED은 재튜닝하거나 primary baseline으로 승격하지 않는다.

### Frozen Phase 1 ablations

```text
ORACLE-PROCESS
ORACLE-MEASUREMENT
ORACLE-FULL
WRONG-PROCESS
WRONG-MEASUREMENT
```

### Frozen datasets/results

기존 Phase 1 test trajectories, results, manifests를 calibration candidate 선택에 사용하지 않는다.
기존 결과는 문제를 발견한 evidence일 뿐 tuning data가 아니다.

## 4. Condition-closure 범위

수행:

- transient/settled time-resolved NEES decomposition
- attitude/bias marginal consistency
- full 6D whitened-error diagnostics
- new independent calibration train/validation/test streams
- P0 attitude/bias scale calibration
- fixed Qg/Qb scale calibration
- sensor R를 고정한 classical calibration
- candidate `F-CALIBRATED`의 independent N=50 confirmation
- MAIN-FUSION stationary와 C4 regression
- updated P1 Exit Review

금지:

- existing F-BASE/F-TUNED overwrite
- 기존 Phase 1 test 결과를 tuning에 사용
- sensor R retuning
- post-hoc reported-P scaling
- per-event oracle context
- covariance jitter/clipping/repair
- learned model
- Phase 2 implementation

## 5. 최종 판단 원칙

Condition closure는 새로운 연구 Gate가 아니다.
P1 Exit Review의 named condition을 닫는 단일 후속 연구 Step이다.

완료 후 P1 Exit decision을 다시 다음 중 하나로 갱신한다.

```text
GO
CONDITIONAL_GO
STOP
```

Phase 2는 별도 승인된 설계 Step에서만 시작한다.
