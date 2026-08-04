# Phase 1A Foundation Final Approval and Phase 1B Handoff

- 결정일: 2026-08-02
- Phase 1A 상태: **COMPLETE**
- 다음 단계: **Phase 1B Step 1 — UNIT-ST Classical Baseline and Minimum Problem-Existence Pilot**
- Neural/ANN/SNN 진행: **승인하지 않음**

## 1. Phase 1A 최종 승인 범위

다음 기반을 승인한다.

- scalar-first Hamilton, active body-to-navigation, right-error 6D Kinematic MEKF
- exact-pi q/-q determinism과 immutable state
- typed gyro/star-tracker event schema
- zero-latency deterministic ordering
- synthetic UNIT-ST generator
- Basilisk rigid-body UNIT-ST truth and project-owned gyro/ST sensor layer
- verified schema/generator/manifest/semantic hashes
- trajectory-level split
- direct replay, D1 bridge, suite-runner replay exact equivalence
- canonical attitude/bias/NIS/NEES/SPD metrics
- lossless q/b/P and compact ST residual/S artifacts
- strict cache verification and stale-cache rejection
- truth-free estimator boundary and post-estimation exact truth join

통합 ID:

```text
task_family = mekf_unit_st_v1
model_id    = mekf_event_replay_v1
```

## 2. Phase 1A가 의미하는 것

Phase 1A는 classical MEKF 연구 결과를 완성한 단계가 아니다.

완료된 것은 다음이다.

```text
검증된 필터 수학
+ 재현 가능한 sensor-level dataset
+ Basilisk truth
+ 공식 metric
+ benchmark runner integration
```

아직 남은 것은 다음이다.

```text
stationary matched baseline
fixed Q/R tuning
mismatched Q/R baseline
Monte Carlo
time-varying gyro/ST uncertainty
oracle Q/R usefulness
process/measurement identifiability
magnetometer/sun sensor expansion
Phase 1 exit review
```

## 3. Phase 1B Step 1 범위

이번 Step은 UNIT-ST만 사용한다.

수행:

- C1 stationary matched baseline
- fixed Q/R tuning
- under/overestimated Q/R variants
- C2 gyro process-uncertainty step
- C3 ST inlier measurement-reliability step
- C5 matched innovation-RMS A/B pair
- fixed/tuned/oracle/wrong-side comparison
- paired pilot Monte Carlo
- long-horizon stationary subset

수행하지 않음:

- magnetometer/sun sensor
- outage/false solution/outlier/robust gate
- C4 slow drift + fast multi-sensor event
- learned context
- KalmanNet/Split-KalmanNet
- ANN/SNN/FPGA
- closed-loop control

## 4. 실험 원칙

1. 동일 estimator 비교는 동일 raw gyro/ST measurement realization을 사용한다.
2. fixed/tuned estimator는 event time, oracle scale, hidden label을 받지 않는다.
3. oracle estimator만 current simulation-assigned Q/R scale을 사용한다.
4. oracle sidecar는 sensor dataset과 분리하고 deployable artifact에 포함하지 않는다.
5. tuning은 train/validation trajectory만 사용하고 test 결과를 보지 않는다.
6. 모든 비교는 paired seed와 trajectory-level split을 사용한다.
7. 실험은 analytic/debug/pilot Monte Carlo 순서로 진행한다.
8. representative normalized profile을 flight-grade sensor 성능으로 주장하지 않는다.
9. 실패한 가설을 더 복잡한 neural model로 숨기지 않는다.
10. Phase 1B Step 1 완료 후에도 neural 단계로 자동 진행하지 않는다.

## 5. Step 1 종료 후 판단할 가설

- H1: time-varying gyro/ST uncertainty가 fixed MEKF를 반복적으로 악화시키는가
- H2: correct-side oracle Q/R이 fixed/tuned baseline보다 유용한가
- H3: process-side와 measurement-side action을 분리할 필요가 있는가
- H4: scalar innovation RMS만으로 원인을 구분하기 어려운가

Step 1은 위 가설의 UNIT-ST pilot evidence를 제공한다.
최종 Phase 1 Exit는 MAIN-FUSION과 sensor-validity 범위까지 완료한 뒤 판단한다.
