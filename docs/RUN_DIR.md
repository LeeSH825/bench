본 문서는 **run_dir 산출물 표준**을 고정합니다.  
SoT: `/mnt/data/suite_basic.yaml`, `/mnt/data/suite_shift.yaml`, `/mnt/data/METRICS.md`

> Step 2에서는 실제 파일 생성 로직을 구현하지 않습니다.
> 다만 Step 4~에서 runner가 이 문서/스펙을 준수하도록 구현됩니다.

---

## 1) Directory template

### suite_basic.yaml
- `reporting.output_dir_template`:
  - `runs/{suite.name}/{task_id}/{model_id}/{track_id}/seed_{seed}`

### suite_shift.yaml
- `reporting.output_dir_template`:
  - `runs/{suite.name}/{task_id}/{model_id}/{track_id}/seed_{seed}/scenario_{scenario_id}`

#### scenario_id 규칙(정의)
- sweep(그리드)이 없는 경우: `scenario_id = "default"`
- sweep이 있는 경우: runner가 sweep 조합별로 `scenario_id`를 생성(예: `s0`, `s1`, ...)
- scenario_id는 **summary/metrics.json/폴더명에서 일관**되어야 함

---

## 2) Required artifacts (per run_dir)

suite YAML의 `reporting.artifacts` 기준(현재 basic/shift 공통):

- `config_snapshot.yaml`
- `metrics.json`
- `metrics_step.csv`
- `timing.csv`
- `stdout.log`
- `stderr.log`
- `env.txt`
- `git_versions.txt`

> 실패 시에도 최소한 `stderr.log`, `stdout.log`, `config_snapshot.yaml`은 남기는 것을 권장합니다.
> (FAIRNESS.md “실패/예외 처리” 준수)

---

## 3) metrics.json schema (Reporting schema)

SoT: `/mnt/data/METRICS.md`의 “Reporting schema” + suite의 metric list

### 최소 필드(권장)
- `suite_name` (e.g., "basic" / "shift")
- `suite_version` (e.g., "0.1.0")
- `task_id`
- `model_id`
- `track_id` ("frozen" / "budgeted")
- `seed` (int)
- `scenario_id` (string)
- `status` ("ok" | "failed" | "skipped")
- `metrics` (dict)
  - `mse`
  - `rmse`
  - `mse_db`
  - `timing_ms_per_step_mean`
  - `timing_ms_per_step_std`
  - `nll`  (optional; 공분산 미지원이면 `"NA"`)
  - `shift_recovery_k` (shift suite에서만; 실패 정책은 DECISIONS D7)

### NLL=NA 규칙(고정)
- 공분산(또는 분산) 미지원 모델은 `nll = "NA"`
- 해당 규칙은 Step 6~에서 metrics 계산 구현 시 강제

---

## 4) metrics_step.csv schema (time-series)

목적: `MSE(t)` 등 **시간축 곡선**을 플롯/분석하기 위함.

### 권장 컬럼
- `t` (0..T-1)
- `mse_t`
- `rmse_t`
- `mse_db_t`
- `nll_t` (optional; 공분산 미지원이면 컬럼 자체를 생략하거나 NA)

---

## 5) timing.csv schema

목적: `timing_ms_per_step`의 근거 로그를 남기기 위함.

### 권장 컬럼
- `split` ("test")
- `warmup_steps` (int)
- `measured_steps` (int)
- `ms_per_step_mean`
- `ms_per_step_std`

---

## 6) config_snapshot.yaml

목적: 재현성(reproducibility). 해당 run에서 사용된 설정을 그대로 남김.

### 포함 권장
- suite 전체(또는 해당 task/model/track에 필요한 subset)
- 선택된 sweep 파라미터(= scenario)
- runner 예산/early_stopping/precision/deterministic
- env 요약(가능하면 git hash, pip freeze는 env.txt로)

---

## 7) Summary table (generated outside run_dir)

suite YAML에 따라:
- basic: `reports/summary_basic.csv`
- shift: `reports/summary_shift.csv`

권장 컬럼(SoT: METRICS.md)
- `model_id, task_id, scenario_id, seed, track_id, status, metric_* ...`

## bench_generated 데이터 캐시(v0)

- 공식 결과 데이터 모드: `bench_generated` (DECISIONS D3)
- 캐시 루트(기본): `<bench_repo_root>/bench_data_cache/`
  (환경변수 `BENCH_DATA_CACHE`로 override 가능)
- 디렉토리 규칙(v0): `bench_data_cache/{suite_name}/{task_id}/scenario_{scenario_id}/seed_{seed}/{split}.npz`
- split 파일은 **한 번 생성 후 고정**되며, 동일 (suite/task/scenario/seed)면 항상 동일한 train/val/test가 로딩된다.
- 저장 포맷: `.npz` with `x, y, (optional) u, (optional) F, (optional) H, meta_json`
- canonical layout: `x,y`는 `[N,T,D]` (D15). 레포별 입력 차원 차이는 어댑터에서 변환한다.
- 간단 검증: `python -m bench.tasks.smoke_data --suite-yaml /mnt/data/suite_shift.yaml --task C_shift_Rscale_v0 --seed 0`

