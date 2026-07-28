# Visualization Tool Known Limitations

이 문서는 VIZ-R1 감사에서 확인한 제한을 기록한다. 각 항목은 현재 동작, 영향, 우회 방법, 후속 작업을 구분한다.

## Offline artifact viewer

- 제한: [확인] `viz/app/main.py:3-9`는 저장된 artifact를 읽는 Streamlit entry point이며 `run_inspector.py:571-577`에는 live sensor/training 경로가 없다.
- 영향: [확인] 실시간 센서, 실시간 추론, training dashboard로 사용할 수 없다.
- 우회 방법: [확인] runner에서 artifact를 생성한 뒤 `VIZ_RUNS_ROOT`로 해당 디렉터리를 지정한다.
- 후속 작업: [추정] live 기능이 필요하면 별도 보안·상태관리 설계가 필요하다.

## Split coverage

- 제한: [확인] 현재 실제 cross-model artifact의 `data_spec.split`은 `test`이고 legacy EKF artifact는 `unknown`이다. 감사 출력과 `tests/test_viz_release_readiness.py:47-63`이 근거다.
- 영향: [확인] validation artifact가 실제로 생성됐다고 주장할 수 없다.
- 우회 방법: [확인] UI는 explicit `validation` metadata를 표시하고 test와 overlay를 차단한다.
- 후속 작업: [추정] validation emit 경로와 fixture를 별도로 검증해야 한다.

## Source trajectory provenance

- 제한: [확인] 실제 Split artifact의 `source_trajectory_id_source`는 `test_split_row_index_fallback`이다. 실제 `meta.json`과 V-4c report가 근거다.
- 영향: [확인] Source ID는 원본 데이터베이스의 영구 identity가 아닐 수 있다.
- 우회 방법: [확인] UI는 provenance를 표시하며 stored index와 source ID를 분리한다.
- 후속 작업: [추정] dataset-native trajectory ID를 runner가 전달해야 한다.

## Learned-filter covariance

- 제한: [확인] KalmanNet/Split artifact에는 `P`, `S`가 없고 capability가 false다. `tests/test_viz_release_readiness.py:88-102`와 actual key audit가 근거다.
- 영향: [확인] learned filter에서 physical NEES/NIS와 model-predicted 3-sigma를 계산할 수 없다.
- 우회 방법: [확인] UI는 `Physical covariance unavailable`과 empirical ensemble uncertainty를 별도로 표시한다.
- 후속 작업: [추정] physical uncertainty를 제공하는 learned-filter 설계가 별도 정의되어야 한다.

## Split G1/G2 semantics

- 제한: [확인] `bench/models/split_knet.py:1095-1101`은 G1/G2를 learned split factors와 `gain_g1 @ H.T @ gain_g2`로 기술한다.
- 영향: [확인] G1/G2를 physical covariance 또는 uncertainty로 해석할 수 없다.
- 우회 방법: [확인] UI label은 learned factor이고, component capability는 실제 key에 따라 노출된다.
- 후속 작업: [추정] ME-Split 확장 시 동일 semantics contract를 재검증해야 한다.

## Determinism

- 제한: [확인] CPU에서만 audit했으며 GPU는 `torch.cuda.is_available() == False`였다.
- 영향: [불명] GPU에서의 device-specific exact equality는 이번 감사로 증명되지 않았다.
- 우회 방법: [확인] GPU 실행 시 `CUBLAS_WORKSPACE_CONFIG=:4096:8`와 deterministic PyTorch 설정을 사용해야 한다.
- 후속 작업: [추정] CUDA 환경에서 OFF/ON과 repeat 비교를 수행한다.

## Performance scope

- 제한: [확인] 대형 synthetic 검증은 `T=10000,N=32,K=8,n=m=3`이며 실제 model artifact는 tiny smoke이다.
- 영향: [확인] ADCS production checkpoint와 `T~36000` 성능을 보장하지 않는다.
- 우회 방법: [확인] deterministic downsampling과 lazy trajectory load를 사용한다.
- 후속 작업: [추정] production-scale ADCS artifact에서 RSS와 paint time을 측정한다.

## H and checkpoint coverage

- 제한: [확인] time-varying H와 non-square Split checkpoint는 이번 audit에서 검증하지 않았다.
- 영향: [불명] 해당 변형의 component reconstruction과 UI shape selector 동작은 미확인이다.
- 우회 방법: [확인] 현재 artifact의 fixed-H metadata만 사용한다.
- 후속 작업: [추정] V-4d 또는 별도 compatibility fixture에서 검증한다.

## Dependency packaging

- 제한: [확인] `pyproject.toml:13-22`에는 Streamlit, Plotly, Playwright가 선언되어 있지 않지만 audit environment에는 설치되어 있다.
- 영향: [조건부] clean package install만으로 Streamlit UI가 보장되지 않는다.
- 우회 방법: [확인] 현재 지정 Python 환경에서 실행한다.
- 후속 작업: [추정] VIZ-R1.1에서 optional `viz` dependency extra 또는 설치 문서를 결정한다.

## Reports and screenshots

- 제한: [확인] `.gitignore:7-10`은 `reports/`를 무시한다.
- 영향: [확인] audit report와 screenshot이 일반 git status에 나타나지 않을 수 있다.
- 우회 방법: [확인] 파일 존재와 크기를 별도로 확인한다.
- 후속 작업: [추정] Phase 0 정책에 따라 reports 보존 방식을 결정한다.
