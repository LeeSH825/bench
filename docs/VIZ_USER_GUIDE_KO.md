# Run Inspector 사용자 가이드 (한국어)

이 문서는 `docs/VIZ_USER_GUIDE.md`(영어, source of truth)의 한국어 번역이다. 기술적 의미가 달라지지 않도록 번역했으며, 내용이 달라지면 영어 문서를 먼저 수정한 뒤 이 문서를 갱신한다. 앱 내부의 "Help & guide" popover는 영어로만 제공된다(언어 정책은 보고서 참조).

## 1. 이 도구는 무엇인가

**이 도구는 offline artifact viewer다. 실시간 센서, 실시간 추론, training dashboard가 아니다.**

흐름:

```
모델 평가 실행
  → visualization artifact emit(opt-in)
  → Streamlit 실행
  → run과 trajectory 선택
  → 결과 확인·비교
```

Visualization artifact는 자동으로 생성되지 않는다. `bench/runners/run_suite.py`는 `--emit-viz-artifacts` 플래그를 명시적으로 줬을 때만(기본값: off) artifact를 쓴다. `meta.json`이 없는 `runs/` 디렉터리를 가리키면 보여줄 것이 없다.

## 2. Quick start

1. Data split과 suite를 선택한다.
2. Task, scenario, seed, track, primary run(Model / Init-checkpoint)을 선택한다.
3. 대표 trajectory를 선택한다(Trajectory view).
4. **Models to display**에서 겹쳐 볼 run을 선택한다.
5. A–F 패널을 읽는다.

실행 명령(artifact 경로와 Python 경로는 환경에 맞게 조정):

```bash
env VIZ_RUNS_ROOT=/path/to/runs \
MPLCONFIGDIR=/tmp/matplotlib \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
-m streamlit run viz/app/main.py \
--server.headless true \
--server.address 127.0.0.1 \
--server.port 8501
```

`VIZ_RUNS_ROOT`을 지정하지 않으면 현재 작업 디렉터리 기준 `runs`가 기본값이다.

## 3. Navigation 컨트롤

화면 상단의 선택 컨트롤들이 전체 run 중 하나의 **primary run**을 결정한다.

| 컨트롤 | 필터 기준 |
|---|---|
| Data split | artifact가 속한 평가 subset(예: `test`). 서로 다른 split의 run은 절대 겹쳐 보여주지 않는다. |
| Suite | run을 만든 실험 묶음. |
| Task | 벤치마크 task 정의. |
| Scenario | run을 만든 물리적/합성 테스트 조건. |
| Model | 모델 식별자(예: `kalmannet_tsp`, `oracle_kf`). |
| Seed | run의 seed 값(실제로 무엇을 통제하는지는 4절 참조). |
| Track | 벤치마크 실행 track(예: `frozen`, `budgeted` — 평가 중 온라인 적응 여부). |
| Init/checkpoint | 이 run의 초기화/학습 라벨(예: `trained`, `pretrained`, `untrained`). |

**선택한 Model이 primary navigation run이다.** "Models to display"(5절)에서 나중에 plot에서 꺼도 Dataset Summary와 artifact metadata 배지는 계속 이 run 기준이다.

## 4. Primary model과 Models to display의 차이

비슷해 보이지만 화면의 서로 다른 부분을 제어한다. 둘을 혼동하는 것이 "왜 안 바뀌지"류 질문의 가장 흔한 원인이다.

**Primary model**(3절 navigation으로 선택):
- Dataset Summary와 artifact metadata 배지(commit, run status, artifact version 등)의 기준.
- Models to display에서 자신의 checkbox로 A–F plot에서 꺼질 수 있다.

**Models to display**(5절, Trajectory view 바로 아래):
- A–F 패널에 실제로 trace로 나타날 run variant를 결정한다.
- toggle ON한 variant만 로드되고 plot된다. 그 외에는 아무 것도 로드되지 않는다.
- 항상 최소 하나의 run은 선택 상태를 유지해야 한다.

### 초기화/학습 라벨은 provenance이지 hard filter가 아니다

run의 `init_id`(toggle 라벨에 `init=...`로 표시됨)는 초기화/학습 **라벨**이지 물리적 호환성 조건이 아니다. 그 자체로는 후보 제외나 overlay 차단 사유가 되지 않는다. 예를 들어 다음이 동시에 나타날 수 있다.

```
oracle_kf · init=pretrained
kalmannet_tsp · init=trained
split_knet · init=trained
```

같은 evaluation context(suite/task/scenario/split/seed/track)를 공유하고 패널별 호환성 검사를 통과하면 `init_id`와 무관하게 overlay될 수 있다. 선택된 run들의 `init_id`가 서로 다르면, 학습 조건이 동일하다고 해석하지 말고 baseline/ablation/adaptation 비교로 해석하라는 비차단(non-blocking) 안내가 표시된다 — 이 안내는 그 자체로 overlay를 막지 않는다.

## 5. Dataset Summary와 Selected Trajectory의 차이

- **Dataset Summary**: 전체 evaluation split에 대한 aggregate metric, 항상 **primary navigation run** 기준. Models to display 토글이나 Trajectory view 변경에 반응하지 않는다 — 의도된 동작이다.
- **Selected Trajectory**: "Trajectory view"에서 고른 Source ID 하나에 대한 metric과 시계열 패널. A–F 패널이 실제로 그리는 대상이다.

## 6. Source ID와 Stored index

- **Stored index**: artifact 내부 저장 순번(내부 offset).
- **Source ID**: evaluation dataset 안에서의 trajectory identity.

Source ID provenance가 `test_split_row_index_fallback`이면, 그 ID는 **같은 dataset 파일 · 같은 row 순서**를 쓰는 다른 run과 비교할 때만 신뢰할 수 있다 — 임의의 artifact 사이에서 전역적으로 안정적인 trajectory identity가 아니다. Run Inspector는 Source ID가 없다고 해서 다른 stored index로 자동 대체하지 않는다; 후보 run에 선택한 Source ID가 없으면 그 trajectory에는 그냥 쓸 수 없다.

## 7. Models to display

- 후보는 primary run과 같은 suite/task/scenario/split/seed/track(같은 evaluation context)인 run으로 제한된다.
- 같은 모델의 여러 init/학습 variant(예: `kalmannet_tsp · init=untrained`, `· init=trained`, `· init=adapted`)가 독립적으로 toggle 가능한 별도 후보로 나타날 수 있다.
- toggle OFF한 run은 로드되지 않고 어떤 패널에도 trace가 없다.
- 선택한 run이 특정 패널과 호환되지 않으면(예: innovation residual 정의 불일치) **그 패널에서만** 제외된다 — global toggle은 ON을 유지하고, 호환되는 다른 모든 패널에는 계속 나타난다.
- 초기화/학습 provenance 차이는 해석 참고사항이지 자동 제외 사유가 아니다(4절).

## 8. Axis 및 display 컨트롤

- **3-axis split**: 각 축(x/y/z)이 별도 subplot을 가진다.
- **Combined axes**: 한 trace의 x/y/z 성분을 하나의 plot에 함께 그린다. "여러 모델을 합친다"는 뜻이 아니다.
- **Norm only**: 성분별 값 대신 벡터 norm을 그린다.
- **Transient window**: 표시되는 metric/plot에서 trajectory 초반 구간을 제외한다. 저장된 데이터 자체는 변경되지 않는다.
- **Gain source**: F 패널에 표시할 gain 종류(combined gain, 또는 선택 모델이 제공하는 Split-KalmanNet G1/G2 factor).
- **Gain display**: Frobenius norm(스텝당 스칼라 하나) 또는 특정 matrix element.
- **Matrix element row/col**: "Matrix element" 모드에서만 표시되며, gain matrix의 어느 원소를 그릴지 고른다.

## 9. A–F 패널 읽는 법

### A. Attitude RPY

- **무엇인가**: 각 모델의 canonical attitude representation에서 얻은 Roll/Pitch/Yaw와 Truth.
- **왜 보는가**: 자세 추적을 직관적으로 파악할 수 있다.
- **어떻게 쓰는가**: Models to display에서 모델을 고른다; Truth는 항상 중립 스타일로 그려진다.
- **주의**: RPY는 직관적 파악용이다. 정량 비교에는 B 패널의 geodesic attitude error를 쓴다. frame과 RPY/quaternion convention이 primary와 호환된다고 선언된 모델만 여기 겹쳐진다.

### B. Attitude Error + 3σ

- **무엇인가**: 추정치와 truth 사이의 실제 자세 오차(geodesic 또는 axis error).
- **왜 보는가**: RPY 패널과 달리 정량적 자세 정확도 지표다.
- **어떻게 쓰는가**: 선택되고 호환되는 모델들의 오차 크기·거동을 비교한다.
- **주의**: **physical ±3σ band**는 유효한 physical covariance(`P`)를 제공하는 모델에만 그려진다. physical `P`가 없는 learned model은 여기서 physical 3σ band를 절대 가지지 않는다 — 버그가 아니라 정상 동작이다. MRP와 rotation-vector covariance는 변환 계수가 다르므로 선언된 covariance space에 따라 band 계산이 달라진다.
- **Physical ±3σ vs. empirical spread —같은 것이 아니다**:
  - *Physical ±3σ*: 모델 자신이 예측한 covariance에서 유도.
  - *Empirical spread*: 여러 trajectory에 걸친 관측 오차의 sample spread. 필터가 예측한 covariance가 아니며, 시각적으로 구분되는(채우지 않은/점선) 스타일로 표시된다.

### C. Bias + 3σ

- **무엇인가**: gyro bias truth/estimate/error, 단위는 deg/h.
- **왜 보는가**: bias 추적은 자세 추적과 별개의 실패 모드다.
- **어떻게 쓰는가**: artifact가 호환되는 bias state를 선언한 모델만 여기 표시된다.
- **주의**: bias state가 아예 없는 모델(예: 순수 attitude filter)은 이 패널 자체가 비활성화될 수 있다; physical 3σ band는 추가로 physical bias covariance block이 필요하다.

### D. Innovation

- **무엇인가**: `innovation = measurement − predicted measurement`.
- **왜 보는가**: 크고 구조적인 innovation은 대개 모델 예측과 실제 측정값이 어긋난다는 뜻이다.
- **어떻게 쓰는가**: 선택 모델들의 innovation 크기·채널별 거동을 비교한다.
- **주의**: measurement 종류, residual 정의, frame, 단위, 채널 순서가 같은 모델끼리만 겹쳐진다 — gyro residual과 attitude-reference residual은 직접 비교하지 않는다. `innov_valid=false`인 구간(measurement update가 없는 시점)은 0으로 그려지지 않고 아예 표시되지 않는다.

### E. NEES / NIS

- **무엇인가**: **NEES**(Normalized Estimation Error Squared)는 state 추정 오차와 모델이 예측한 state covariance `P` 사이의 일관성을 확인한다. **NIS**(Normalized Innovation Squared)는 innovation과 모델이 예측한 innovation covariance `S` 사이의 일관성을 확인한다.
- **왜 보는가**: 표준적인 필터 일관성 진단이다.
- **어떻게 쓰는가**: physical `P`/`S`를 제공하는 모델에서만 의미가 있다.
- **주의**: 이 저장소의 KalmanNet/Split-KalmanNet artifact는 physical `P`/`S`를 제공하지 않으므로 NEES/NIS가 **unavailable**이다 — artifact 내용을 그대로 반영한 것이지 모델 실패가 아니다. χ² 범위는 통계적 일관성을 나타낼 뿐 모델 품질의 pass/fail 판정이 아니며, sample 수가 적으면 변동할 수 있다.

### F. Kalman Gain

- **무엇인가**: innovation을 state correction으로 변환하는 행렬. model-based gain, learned gain, Split-KalmanNet combined gain 등 여러 형태가 있다.
- **왜 보는가**: 필터가 새 측정값에 얼마나 큰 보정 가중치를 두는지 보여준다.
- **어떻게 쓰는가**: display 컨트롤(8절)에서 Frobenius norm(스텝당 스칼라) 또는 특정 matrix element를 고른다.
- **주의**: raw gain은 state row semantics, measurement column semantics, 단위, scaling, shape가 모두 일치할 때만("strict" 호환성 검사) 모델 간에 겹쳐진다 — A–C 패널의 물리량 검사보다 엄격하다.

### Split G1/G2

`Combined gain = G1 @ H.T @ G2`. G1과 G2는 **Split-KalmanNet 고유의 learned internal factor**다:
- physical state covariance `P`가 **아니다**,
- physical innovation covariance `S`(또는 그 역행렬)가 **아니다**,
- 실제 covariance 행렬처럼 대칭이거나 positive semi-definite임이 보장되지 **않는다**,
- 이 도구는 G1/G2로 NEES/NIS를 계산하지 **않는다**.

`gain_g1`/`gain_g2`를 제공하지 않는 모델은 G1/G2를 선택한 gain view에서 제외되며, 패널 caption에 사유가 표시된다 — 해당 모델의 global toggle 자체는 영향받지 않는다.

### Regime Timeline

패널들과 같은 시간축 위에 저장된 event/eclipse(또는 유사한 regime) flag를 보여준다. artifact에 그런 flag가 없으면 unavailable로 표시된다 — 이는 "event가 발생하지 않았다"(여전히 available이지만 timeline이 비어 있거나 평평하게 보임)와는 다르다.

## 10. 모델 capability 표

| Feature | MB-KF / EKF / MEKF | KalmanNet | Split-KalmanNet |
|---|---|---|---|
| State estimate | Yes | Yes | Yes |
| Innovation | Yes | Yes | Yes |
| Standard/combined gain | Yes | Yes | Yes |
| G1/G2 | No | No | Yes |
| Physical P | 대체로 Yes | No | No |
| Physical S | 대체로 Yes | No | No |
| Physical 3σ | P가 있으면 | No | No |
| NEES/NIS | P/S가 있으면 | No | No |
| Empirical uncertainty | trajectory가 여러 개면 | trajectory가 여러 개면 | trajectory가 여러 개면 |

"대체로"/"있으면"이라는 표현은 의도적이다 — capability는 모델 계열이 아니라 해당 artifact가 실제로 담고 있는 내용으로 결정된다. 모델 이름에서 나오는 가정보다 항상 패널 자체의 "not shown / unavailable" caption을 신뢰하라.

## 11. 모델 비교: 무엇을 겹쳐도 공정한가

**대체로 비교 가능한 물리량**(9절의 패널별 검사를 통과해야 함): RPY, attitude error, bias, 두 모델 모두 제공하는 physical uncertainty.

**더 엄격하게 검사되는 내부량**: innovation, raw gain, NEES/NIS, G1/G2.

**초기화/학습 provenance는 라벨이지 물리적 호환성 조건이 아니다**(4절). 이 도구가 지원하도록 만들어진 비교 예:
- trained learned model vs. model-based KF,
- 같은 모델의 trained vs. untrained ablation,
- adaptation 전 vs. adaptation 후 run.

**Overlay compatibility와 benchmark fairness는 다른 질문이다.** Overlay compatibility는 "이 run들이 같은 evaluation 데이터와 같은 물리/내부량을 설명하는가?"를 묻는다. Benchmark fairness는 "학습 데이터, oracle 정보, 튜닝 노력, adaptation budget이 공정하게 비교 가능한가?"를 묻는다. 이 도구는 (호환성 guard를 통해) 첫 번째 질문에만 답하고, 두 번째 질문에는 의도적으로 답하지 않는다 — 같은 데이터로 평가됐다면 많이 튜닝된 모델과 적게 튜닝된 모델을 겹쳐 보여줄 수 있고, 어느 쪽이 "더 낫다"고 판단하지 않는다.

## 12. Troubleshooting

### Models to display에 모델이 하나뿐이다
가능한 원인: 현재 suite/task/scenario/split/seed/track context에 artifact가 하나뿐이거나, 다른 run이 선택된 Source ID를 저장하고 있지 않거나, 다른 evaluation context에 속해 있다. metadata parser가 실패했다고 **단정하지 않는다** — "Why only one candidate?" expander에서 정확한 카운트를 확인한다.

### 선택한 모델이 특정 패널에만 안 보인다
global toggle은 여전히 ON이다. 대개 패널별 semantics/capability 불일치로 **그 패널에서만** 제외된 것이다. 해당 패널 바로 아래 caption이나 페이지 하단 "Advanced compatibility diagnostics" 표를 확인한다.

### G1/G2가 안 보인다
선택한 run의 artifact에 `gain_g1`/`gain_g2`가 없다. 이 component를 선언한 Split-KalmanNet 계열 run만 표시할 수 있으며, standard/combined gain은 다른 source라 영향받지 않는다.

### NEES/NIS가 unavailable이다
모델 artifact에 physical `P`/`S`가 없다. 이 저장소의 KalmanNet/Split-KalmanNet에서는 정상적인 상태다 — empirical uncertainty와 혼동하지 않는다.

### Physical 3σ가 unavailable이다
physical covariance가 없거나, 이를 해석하는 데 필요한 covariance block/space metadata가 없다. Run Inspector는 관련 없는 값으로 band를 만들어내지 않는다.

### legacy artifact에서 cross-model comparison이 안 된다
일부 artifact는 `comparison_spec` metadata가 도입되기 전에 만들어졌다. 해당 run 자체의 단일-run A–F 패널은 계속 동작하며, cross-model overlay만 제한되거나 불가능할 수 있다.

### Source ID mismatch
후보 run이 같은 trajectory를 저장하고 있지 않다. Run Inspector는 이를 다른 stored index로 대체해 넘어가지 않는다.

### artifact root가 비어 있다
`VIZ_RUNS_ROOT`가 `meta.json` 파일이 있는 디렉터리를 가리키는지, runner가 `--emit-viz-artifacts`로 실행됐는지(1절) 확인한다.

### 초기 scan이 느리다
run artifact가 많은 `runs/` 디렉터리는 처음 열 때 색인 시간이 걸린다. 이 스캔은 rerun당 한 번만 발생하며, 이후에는 현재 선택된 모델의 trajectory만 lazy load된다.

## 13. Known limitations

- offline viewer 전용 — 1절 참조.
- 일부 최신 기능(예: production-scale ADCS artifact에서의 cross-model overlay)의 실제 artifact 검증 범위는 아직 제한적이다.
- Source ID가 안정적인 dataset-native identity가 아니라 row-index fallback일 수 있다(6절).
- physical `P`/`S`가 없는 learned filter는 physical covariance 기반 NEES/NIS를 계산할 수 없다.
- G1/G2는 learned internal factor이지 covariance가 아니다(9절).
- time-varying `H`와 non-square Split-KalmanNet checkpoint에 대한 검증은 아직 충분하지 않다.
- GPU/production-scale ADCS 성능 범위는 완전히 특성화되지 않았다.
- 이 저장소에서 `reports/`는 `.gitignore`에 포함돼 있어 생성된 report가 일반 `git status`에 나타나지 않을 수 있다.

전체 근거가 명시된 제한사항 목록은 `docs/VIZ_KNOWN_LIMITATIONS.md`를 참조한다.
