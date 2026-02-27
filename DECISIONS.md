# DECISION LOG (Step 0-1)

본 문서는 Step 0에서 발생한 "결정/가정/검증방법"을 기록한다.

## D0. 수치 스펙(x_dim, y_dim, T, N_train/val/test, q2/r2) 부재
- DECISION: v0는 작동 가능한 기본값을 ASSUMPTION으로 잠금하고,
  Step 1 Audit에서 레포 기본값과 충돌 시 suite를 업데이트한다.
- HOW TO VERIFY:
  - KalmanNet_TSP: Simulations/config.py, Simulations/<model>/parameters.py
  - Adaptive-KNet: simulations/ 및 pipelines/ 설정 파일
  - MAML-KalmanNet: Simulations/<model>/*_syntheticNShot.py
  - Split-KalmanNet: config.ini

## D1. 공정성 핵심은 Frozen vs Budgeted 2트랙
- DECISION: 모든 shift 결과는 2트랙을 함께 보고한다.
- 근거: Adaptive/MAML은 적응이 강점이므로 단일 트랙은 불공정 논쟁을 유발.

## D2. 학습 예산은 max_updates(총 gradient update)로 통일
- DECISION: epoch 대신 max_updates를 1차 공정 기준으로 사용.
- 근거: 레포별 epoch 정의/데이터로더가 상이.

## D3. 데이터 모드: bench_generated를 공식 결과로 채택
- DECISION: 공식 리포트는 "벤치가 생성한 공통 데이터" 기반.
- NOTE: repo_default 재현 모드는 디버그/회귀 테스트용으로만 허용(공식 결과 제외).

## D4. NLL 정책
- DECISION: 공분산 미지원 모델의 NLL은 NA로 기록.
- 근거: 억지 가정으로 분산을 부여하면 비교가 왜곡됨.

## D5. MAML 데이터 생성 use_cuda=False 권고
- DECISION: MAML-KalmanNet 연동 시 데이터 생성은 CPU(use_cuda=False)를 우선.
- HOW TO VERIFY: MAML 레포 문서/코드에서 해당 플래그 확인.

## D6. 결정론(deterministic) 규칙
- DECISION: runner 기본 deterministic=true.
- 예외 발생 시(레포가 완전 결정론 미보장) summary에 분산/CI를 함께 기록하고 원인을 로그에 남긴다.

## D7. recovery_k 실패 정책
- DECISION: recovery_k_failure_policy = cap (recovery_k = T - t0)
- WHY: 회복 실패를 NA로 두면 집계/순위 비교에서 결측 처리가 복잡해지고, 실패 자체의 심각도를 반영하기 어렵다.
  cap(T-t0)은 "최대 지연"으로 해석 가능해 평균/분산 계산과 모델 간 비교가 단순해진다.
- HOW TO VERIFY: 프로젝트 파일의 메트릭/shift 회복 정의(섹션 1~2 및 5의 MVP 설명)에 실패 시 NA/cap 정책 언급이 없어 기본값(cap)을 적용한다.

## D8. MVP 혼선 방지(첫 주 범위 정렬)
- DECISION: suite_basic.yaml의 B_lorenz_dt_v0 태스크를 enabled:false로 비활성화한다.
- WHY: 프로젝트 파일의 "첫 주 MVP" 범위는 (A) Linear canonical + (C) 간단 shift 중심이며, Lorenz(B)는 2~4주차 확장으로 명시돼 있다.
- ASSUMPTION: 벤치 러너는 task.enabled=false인 태스크를 스킵한다.
- HOW TO VERIFY: 프로젝트 파일 섹션 5(MVP 범위) 문구 확인 + Step 1에서 runner 구현/검토 시 enabled 처리 여부를 확인한다.

## D9. suite_shift 엔트리포인트 매핑(가정)
- DECISION: suite_shift.yaml의 repo.entrypoints는 "파일명 기준"으로 잠금한다(스크립트 존재는 CONFIRMED). 인자/세부 실행은 Step 1 Audit에서 확정한다.
- NOTE: KalmanNet_TSP/Adaptive-KNet/MAML의 main 스크립트는 기본적으로 레포 내부 데이터 생성/로딩 파이프라인을 사용한다. v0 shift(t0 이후 파라미터 변화)는 D3(bench_generated)+adapter(import-mode)에서 주로 구현하며, subprocess 재현은 Step 2~에서 별도 래퍼가 필요할 수 있다.
  - KalmanNet_TSP: main_linear_canonical.py
  - Adaptive-KNet: main_cm_linear_noise.py
  - MAML-KalmanNet: main_linear.py
  - 내 모델: adapter_native(어댑터 내부 실행)
  - Split-KalmanNet: shift 직접 지원 불명확하여 suite_shift에서 제외(래퍼 필요 가능)
- HOW TO VERIFY: 각 레포에서 해당 스크립트가 noise shift/mismatch 실험을 재현 가능한지(또는 bench adapter로 연결 가능한지) Step 1에서 확인한다.


## D10. Split-KalmanNet의 shift suite 포함 여부
- DECISION: Split-KalmanNet은 shift suite에서 제외 유지한다.
- WHY: 레포는 (SyntheticNL)/(NCLT) 파이프라인 중심이며, v0의 "within-sequence noise shift(t0 이후 파라미터 변화)"를 직접 재현하는 엔트리포인트가 확인되지 않았다(래퍼/추가 구현 필요).
- HOW TO VERIFY: Split-KalmanNet README.md 및 config.ini에서 shift/mismatch(특히 noise 분포 변화) 실험 옵션이 있는지 확인.

## D11. runner enabled 규칙(O5)
- DECISION: runner는 task.enabled=false 및 model.enabled=false 항목을 스킵한다.
  - enabled 키가 없으면 enabled:true로 간주한다(기본값).
- HOW TO VERIFY: Step 2 구현에서 bench runner가 suite의 enabled_policy를 실제로 적용하는지 확인.

## D12. shift suite MVP 범위 정렬
- DECISION: suite_shift.yaml의 C_shift_dist_v0를 enabled:false로 비활성화한다(MVP는 R_scale shift 중심).
- WHY: 프로젝트 파일 섹션 5의 MVP 설명에서 첫 주 범위가 Q/R scale 2단계 중심으로 명시돼 있다.
- HOW TO VERIFY: 프로젝트 파일 섹션 5의 "MVP: 첫 주" 문구 확인.

## (패치 변경 요약)
- suite_shift.yaml: models repo.path/entrypoints 매핑 추가
- DECISIONS.md: D7/D8/D9 추가
- suite_basic.yaml: B_lorenz_dt_v0 enabled:false 추가
- STEP0_CHECKLIST.md: 패치 항목 추가

## (Step1 변경 요약)
- suite_shift.yaml: C_shift_dist_v0 enabled:false 추가(D12) + entrypoints 코멘트 CONFIRMED로 갱신
- suite_basic.yaml/suite_shift.yaml: runner.enabled_policy 추가(D11)
- DECISIONS.md: D10/D11/D12 추가 및 D9 NOTE 보강

## D13. 벤치 코드베이스 아키텍처 / 서브모듈 전략 (Step 2)

- DECISION: 벤치는 "모노레포 + third_party 서브모듈(또는 subtree) + 벤치 어댑터(wrapper)" 구조를 기본으로 채택한다.
  - `bench/` 아래에 벤치 코드(`bench/bench/*`)를 두고,
    서드파티 레포는 `third_party/*`로 **그대로** 가져온다(벤치가 직접 수정하지 않음).
- EXECUTION MODE:
  1) import-mode 우선: 벤치 어댑터가 서드파티 모듈을 import하여 `ModelAdapter` 인터페이스로 감싼다.
  2) fallback 확장: import가 어렵거나 의존성 충돌이 심하면, `bench/bench/runners/orchestrate.py`에서 subprocess/컨테이너 실행으로 확장한다(자리만 유지).
- WHY: "원본 레포 수정 최소화" 원칙을 지키면서도, 공통 데이터/공정성 규칙(D3, FAIRNESS)을 벤치에서 강제하기 위함.
- HOW TO VERIFY:
  - `/mnt/data/벤치마킹 프레임워크 설계.txt`의 Step 2/3 섹션(아키텍처 결정 및 orchestrate 언급) 확인.
  - third_party 레포는 submodule/subtree로 붙이고, 변경은 벤치 어댑터 레이어에서만 수행하는지 PR 리뷰 체크리스트로 확인.

## D14. 실행 환경/의존성 전략 (Step 3)

- DECISION: MVP 기본 전략은 "단일 env"를 우선한다.
  - `pyproject.toml`로 벤치 공통 의존성을 정의하고,
  - `requirements.lock`(pip freeze snapshot)을 유지한다.
- FALLBACK: 의존성 충돌(예: torch/torchvision/cuda 버전 충돌)이 발생하면
  모델별 Docker 이미지로 실행하고, 결과(run_dir)만 수집한다.
  - Docker 템플릿은 `bench/docker/<model>.Dockerfile`에 둔다(자리만 유지).
- HOW TO VERIFY:
  - 각 run_dir에 `env.json`, `pip_freeze.txt`, `git_versions.txt`가 항상 생성되는지 확인한다.
  - 실행 환경이 달라도 동일 seed에서 데이터 hash/메트릭 재현(허용 오차 내)이 되는지 smoke test로 확인한다.

  ## D15. bench_generated 데이터 포맷(v0) — NPZ + Canonical Tensor Layout

- DECISION:
  1) bench_generated 캐시 포맷은 `.npz`를 사용한다.
     - keys: `x`, `y`, `(optional) u`, `(optional) F`, `(optional) H`, `meta_json`
  2) Canonical tensor layout은 **[N,T,D]**(= NTD)로 고정한다.
     - `x.shape == [N, T, x_dim]`, `y.shape == [N, T, y_dim]`
     - 레포별 요구 레이아웃(예: [B,D,T])은 **어댑터에서 변환**한다.
  3) 캐시 경로 규칙(v0):
     - `bench_data_cache/{suite_name}/{task_id}/scenario_{scenario_id}/seed_{seed}/{split}.npz`
     - cache_root는 기본 `<bench_repo_root>/bench_data_cache/`, 필요 시 `BENCH_DATA_CACHE`로 override.

- WHY:
  - NTD는 PyTorch DataLoader/배치 처리에 자연스럽고, time-major/feature-major 변환이 명확하다.
  - 레포별 입력 텐서 관례 차이(예: KalmanNet 계열의 [B,dim,T])는 비교 공정성의 핵심이 아니므로, 공통 포맷을 하나로 잠그고 어댑터에서만 처리한다.
  - meta를 `meta_json`으로 고정해, seed/split/scenario/noise/shift 정보를 파일 자체에 포함하여 재현성을 높인다.

- HOW TO VERIFY:
  - `/mnt/data/벤치마킹 프레임워크 설계.txt`의 Step 4 체크리스트(포맷 `.npz`, meta 포함, bench_data_cache 디렉토리 규칙) 확인.
  - smoke test(`python -m bench.tasks.smoke_data ...`)로 생성→캐시→로딩→첫 배치 shape가 `[B,T,D]`로 일관되는지 확인.

## D16. Train budget semantics (train_max_updates 카운팅 단위)

- DECISION: train_max_updates는 기본적으로 "train loop의 공식 update 카운트"를 제한한다.
- MODEL-SPECIFIC RULE (MAML-KNet):
  - train_outer_updates_used(outer optimizer.step)만 train_max_updates로 강제한다.
  - train_inner_updates_used(inner optimizer.step)는 별도 추적/보고하며 train_max_updates로 직접 제한하지 않는다.
  - train_updates_used는 backward-compat alias로 train_outer_updates_used와 동일하게 기록한다.
- WHY: 메타러닝은 inner/outer loop가 분리되어 있어, 모든 optimizer.step을 단일 cap에 강제하면 기존 실험 의미와 크게 달라질 수 있다.
- HOW TO VERIFY:
  - bench/models/maml_knet.py의 ledger 필드(train_outer_updates_used, train_inner_updates_used) 확인.
  - bench/runners/run_suite.py의 train_max_updates 검증이 train_outer_updates_used 기준인지 확인.

## D17. Checkpoint + model_cache_dir 정책(표준 경로/캐시 키 포함)

- DECISION: 
  1) 모든 Route B run_dir은 아래 체크포인트 아티팩트를 표준 경로로 남긴다.
  2) checkpoints/model.pt (또는 model.safetensors — 포맷은 어댑터가 택하되 확장자는 고정)
  3) checkpoints/train_state.json (best_step, early_stop 상태, val_curve_digest 등)
  4) budget_ledger.json (train/adapt 업데이트 카운트)
  5) run_plan.json (init_id/track_id 포함)
- OPTION(권장): 중복 학습 방지를 위한 model_cache_dir를 지원한다(캐시 hit 시 train 생략, eval만 수행).
- CACHE KEY(권장): hash(model_id + adapter_version + task_id + scenario_id + seed + train_budget + data_hash + git_versions + env_digest)
- WHY:
  - run_dir ckpt는 “실험 감사/재현”을 위한 기록(항상 남김).
  - model_cache는 “운영 효율”을 위한 옵션(동일 조건 재실행 시 시간 절약).
- HOW TO VERIFY:
  - bench/bench/runners/run_suite.py에서 run_dir 생성부에 checkpoints/ 하위 파일 생성 확인.
  - bench/bench/utils/cache_key.py(신규) 또는 동등 위치에서 cache key 계산 함수 추가 후, 동일 입력에서 key 안정성 테스트.

## D18. Adapt overflow policy (선택) — 예산 초과 시 처리

- DECISION(권장): adapt 예산 초과 시 즉시 실패(status=failed) 처리하고 failure.json에 failure_type=budget_overflow로 기록한다.
- WHY:
  - Budgeted track의 공정성 위반은 결과를 “부분 저장”해도 해석이 혼선(어디까지가 허용 범위인지 불명확)이며,
  - fail-fast가 감사/재현성/자동 집계에 가장 단순하다.
- HOW TO VERIFY:
  - bench/bench/models/* 어댑터에서 overflow 감지 시 BudgetOverflowError 발생 또는 ledger에 violation 기록
  - runner가 이를 받아 failure.json 생성 후 다음 조합으로 계속 진행(FAIRNESS의 “실패 격리” 원칙)

## D18-1 Fail-Fast Overflow Verification

- Status: verified in practice.
- Evidence:
  - `suite_adapt_smoke_overflow.yaml` uses `track=budgeted`, `max_updates=0`.
  - Run produces `failure.json` with `failure_type=budget_overflow` (adapter fail-fast path).

## D19. Determinism policy (S7)

- DECISION: 재현성 검증은 "stability metrics + 허용오차" 규칙으로 강제한다.
- Stability metrics:
  - `mse_db` (필수)
  - `recovery_k` (shift 태스크에서 사용 가능할 때)
  - (보고 집계용) `failure_type`/`fail_rate`는 동일 조건에서 동일해야 함
- CPU policy:
  - 동일 `(suite, task, scenario_id, seed, model, init_id, track_id, data_hash)`에서
    `mse_db`는 abs tol `1e-9` 이내, `recovery_k`는 exact match(정수 동일)여야 한다.
- GPU/accelerator policy:
  - 동일 조건에서 작은 수치 오차 허용:
    - `mse_db`: `abs<=1e-5` 또는 `rel<=1e-5`
    - `recovery_k`: exact match 권장, 불가 시 `abs<=1`
- Cache invariance rule:
  - 동일 조건의 cache miss vs cache hit는 성능 메트릭(`mse`, `rmse`, `mse_db`, `recovery_k`)을 바꾸면 안 된다.
  - 차이가 허용되는 것은 ledger/cache 필드(`train_skipped`, `cache_hit`, `train_updates_used`)뿐이다.
- HOW TO VERIFY:
  - `bench/tests/test_determinism_cpu_seed_stable.py`
  - `bench/tests/test_cache_invariance.py`
  - `python -m bench.tests.run_all --device cpu`

## D19-1. GPU determinism enforcement status (S8)

- Status: enforced at smoke level with skip-aware policy.
- Evidence:
  - `bench/tests/test_determinism_gpu_seed_stable.py`
  - `bench/tests/run_gpu_checks.py`
  - GPU absent: clean SKIP behavior.
  - GPU present: two-run stability check enforces D19 tolerance for `mse_db` and `recovery_k` (if present).
