# Agent Lookup Recipes

모든 명령은 repository root `/home/dss-pc-05/bench`에서 read-only로 실행한다.
branch, HEAD, commit history, 과거 commit delta는 조회하지 않는다.

## Recipe 0: index 자체 검증

```bash
python3 tools/research/validate_phase0_1_evidence_index.py \
  --repo-root /home/dss-pc-05/bench
python3 -m pytest -q tests/test_phase0_1_evidence_index.py
```

첫 명령은 JSON canonical sorting, ID uniqueness, path/locator existence, 수치
source linkage, 80개 이상 question, current/historical exit 관계, 21개 representative
lookup을 검사한다.

## Recipe 1: 새 대화의 current status 확인

```bash
sed -n '1,220p' experiments/phase1b/reports/P1_EXIT_REVIEW_UPDATED.md
python3 - <<'PY'
import json
from pathlib import Path
p = Path('experiments/phase1b/results/p1_exit_covariance_closure_v1/updated_exit_review.json')
d = json.loads(p.read_text())
for key in ('decision', 'status', 'phase2_implemented', 'acceptance',
            'remaining_classical_limitation'):
    print(key, d[key])
PY
```

기대: `CONDITIONAL_GO`, stationary passed, C4 not passed,
`phase2_implemented=False`.

## Recipe 2: 질문을 machine index에서 찾기

```bash
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path('docs/research/index/phase0_1/phase0_1_evidence_index.json').read_text())
needle = 'NEES'.casefold()
for topic in d['topics']:
    hits = [q for q in topic['question_patterns'] if needle in q.casefold()]
    if hits:
        src = topic['canonical_sources'][0]
        print(topic['id'], hits, src['path'], src['locator'], topic['status'])
PY
```

`needle`만 바꾼다. 결과의 첫 source부터 읽고 topic의 implementation/test/result
paths를 그 다음에 연다.

## Recipe 3: MEKF convention과 source locator

```bash
rg -n 'D04|D05|D06|D07|D08' \
  docs/research/phase0a/decision_lock/P0_01_DECISION_LEDGER.md
rg -n '^(class MEKFState|def align_quaternion|def quat_log|def star_tracker_residual|def propagate_state|def star_tracker_update)' \
  bench/estimators/mekf.py
rg -n 'exact.pi|read.only|immutable|q_NB|body.to.navigation' \
  docs/research/phase1a/P1A_GATE_A_FINAL_APPROVAL.md \
  tests/test_mekf_core.py tests/test_mekf_conventions.py
```

## Recipe 4: B1 schema, identity, ordering, hash

```bash
sed -n '1,260p' docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
sed -n '1,240p' docs/research/phase1a/P1A_GATE_B1_AMENDMENT_A1_CONTRACT.md
rg -n '^(SCHEMA_VERSION|GENERATOR_ID|class MEKFEventTable|def save_event_dataset|def load_event_dataset|def split_trajectory_ids|def replay_trajectory)' \
  bench/tasks/generator/mekf_events.py
```

active/passive prose가 충돌하면 event contract의 Gate B2 erratum과 executable
frame proof가 우선한다.

## Recipe 5: Basilisk executable proof

```bash
sed -n '1,260p' docs/research/phase1a/P1A_GATE_B2_FINAL_APPROVAL.md
sed -n '1,300p' docs/research/phase1a/P1A_BASILISK_FRAME_CONVENTION_PROOF.md
rg -n '^(GENERATOR_ID|def basilisk_sigma_BN_to_q_NB|def run_static_frame_proof|def run_dynamic_rate_proof|def generate_basilisk_unit_st)' \
  bench/tasks/generator/basilisk_unit_st.py
```

필요하면 기존 test만 read-only 실행한다.

```bash
python3 -m pytest -q tests/test_basilisk_unit_st_generator.py
```

## Recipe 6: canonical metric source 찾기

```bash
rg -n '^(def right_local_state_error|def attitude_geodesic_error_rad|def bias_error_summary|def spd_diagnostics|def star_tracker_nis|def right_local_nees|def consistency_summary)' \
  bench/metrics/mekf.py
sed -n '1,240p' docs/research/phase1a/P1A_GATE_C_FINAL_APPROVAL.md
```

report의 metric 이름보다 source의 actual function/normalization을 우선한다.

## Recipe 7: adapter/runner exact path 확인

```bash
rg -n '^(class DatasetIdentity|class MEKFReplayArtifact|class MEKFEventReplayBridge)' \
  bench/models/mekf.py
rg -n 'mekf_unit_st_v1|mekf_event_replay_v1|def _is_p1a_mekf_event_replay_pair|def _p1a_exact_truth_join|def _run_p1a_mekf_event_replay|def _load_split_npz|class _SeqDataset|def _predict_batches' \
  bench/runners/run_suite.py
sed -n '115,150p' experiments/phase1a/reports/P1A_CP4_VALIDATION_REPORT.md
```

event replay 분기가 `_load_split_npz`, `_SeqDataset`, `_predict_batches`보다 먼저
실행되는지 source line order로 확인한다.

## Recipe 8: frozen Step 1 tuning만 출력

```bash
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path('experiments/phase1b/results/unit_st_classical_v1/tuning.json').read_text())
print('fixed_tuning=', d['fixed_tuning'])
print('c5_alpha=', d['frozen_c5_B_alpha_R'])
print('c5_matching=', d['c5_matching'])
print('test_split_accessed=', d['test_split_accessed'])
PY
```

policy 역할은 숫자 JSON만 보고 추정하지 말고 Step 1 final approval §2를 함께
읽는다.

## Recipe 9: C1/C2/C3/C5 group 조회

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('experiments/phase1b/results/unit_st_classical_v1/pilot_summary.json')
d = json.loads(p.read_text())
print('paired_N=', d['completed_paired_N_per_condition'])
for key, value in d['summary'].items():
    if any(code in key for code in ('C1-', 'C2-', 'C3-', 'C5-')):
        print(key, value)
print('c5_test=', d['c5_AB_independent_test'])
PY
```

해석은 `P1B_PROBLEM_EXISTENCE_REPORT.md`와
`P1B_IDENTIFIABILITY_PILOT_REPORT.md`의 limitation 문단을 함께 사용한다.

## Recipe 10: long-horizon 조회

```bash
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path('experiments/phase1b/results/unit_st_classical_v1/long_horizon.json').read_text())
print(d['status'], d['duration_s'], d['num_trajectories'])
for row in d['records']:
    print(row['trajectory_id'], row['policy_id'], row['attitude_event_rmse_rad'],
          row['nis_normalized_mean'], row['nees_normalized_mean'], row['diverged'])
PY
```

aggregate는 baseline report의 long-horizon table을 사용한다.

## Recipe 11: fusion schema와 same-time order

```bash
rg -n '^(SCHEMA_VERSION|ORACLE_SCHEMA_VERSION|SAME_TIME_ORDER_ID|class Fusion)' \
  bench/tasks/generator/mekf_fusion_events.py
rg -n 'gyro.*mag.*sun.*star|invalid|skip|inlier|WMM|flight' \
  docs/research/phase1b/P1B_STEP1_FINAL_APPROVAL_AND_STEP2_HANDOFF.md \
  experiments/phase1b/reports/P1B_STEP2_VALIDATION_REPORT.md
```

## Recipe 12: original MAIN settled consistency

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('experiments/phase1b/results/sensor_fusion_c4_v1/settled_consistency.json')
d = json.loads(p.read_text())
print('cut_fraction=', d['cut_fraction'])
print(d['main_fusion_stationary_F_BASE'])
PY
```

이 값은 original Step 2 MAIN-FUSION test N=50이다. closure validation 또는
confirmation 값과 섞지 않는다.

## Recipe 13: STRESS-MAG와 C4 paired comparisons

```bash
rg -n '0\.197681|0\.195676|0\.00133072|observab' \
  experiments/phase1b/reports/P1B_STRESS_MAG_REPORT.md
rg -n '100000|16|28\.56|32\.57|wrong|oracle' \
  experiments/phase1b/reports/P1B_C4_COMBINED_EVENT_REPORT.md
```

machine result 전체가 필요하면 다음 key만 출력한다.

```bash
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path('experiments/phase1b/results/sensor_fusion_c4_v1/pilot_summary.json').read_text())
for key in ('pilot_status', 'record_count', 'same_realization', 'summary'):
    print(key, d[key])
PY
```

## Recipe 14: closure data leakage와 frozen scales

```bash
python3 - <<'PY'
import json
from pathlib import Path
base = Path('experiments/phase1b/results/p1_exit_covariance_closure_v1')
diagnosis = json.loads((base/'diagnosis.json').read_text())
updated = json.loads((base/'updated_exit_review.json').read_text())
print('split=', diagnosis['split'])
print('sensor_R=', diagnosis['sensor_R_scales'])
print('F_CALIBRATED=', updated['F_CALIBRATED_status'])
PY
```

확인할 invariants: train 30, validation 20, frozen Phase 1 test overlap empty,
confirmation not accessed at freeze, sensor R all 1.

## Recipe 15: 네 NEES를 scope와 함께 출력

```bash
python3 - <<'PY'
import json
from pathlib import Path
step2 = json.loads(Path('experiments/phase1b/results/sensor_fusion_c4_v1/settled_consistency.json').read_text())
diagnosis = json.loads(Path('experiments/phase1b/results/p1_exit_covariance_closure_v1/diagnosis.json').read_text())
updated = json.loads(Path('experiments/phase1b/results/p1_exit_covariance_closure_v1/updated_exit_review.json').read_text())
print('original_step2_test_FBASE_settled=', step2['main_fusion_stationary_F_BASE']['nees']['normalized_mean'])
print('closure_validation_FBASE_settled=', diagnosis['groups']['validation']['aggregate']['partitions']['settled']['full_nees_normalized'])
print('closure_confirmation_FBASE_settled=', updated['confirmation_F_BASE_settled']['full_nees_normalized'])
print('closure_confirmation_FCAL_settled=', updated['confirmation_F_CALIBRATED_settled']['full_nees_normalized'])
PY
```

## Recipe 16: closure stationary/C4 acceptance

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('experiments/phase1b/results/p1_exit_covariance_closure_v1/updated_exit_review.json')
d = json.loads(p.read_text())
print('acceptance=', d['acceptance'])
print('FBASE=', d['confirmation_F_BASE_settled'])
print('FCAL=', d['confirmation_F_CALIBRATED_settled'])
print('remaining=', d['remaining_classical_limitation'])
PY
```

## Recipe 17: current/historical decision 차이

```bash
rg -n '^(#|##)|CONDITIONAL_GO|GO|STOP|NEES|Phase 2|Phase2' \
  experiments/phase1b/reports/P1_EXIT_REVIEW.md \
  experiments/phase1b/reports/P1_EXIT_REVIEW_UPDATED.md
```

인용 순서는 updated current → original historical이다.

## Recipe 18: test-count provenance

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('experiments/phase1b/results/p1_exit_covariance_closure_v1/regression_evidence.json')
d = json.loads(p.read_text())
print('status=', d.get('status'))
print('commands=', d.get('commands'))
PY
```

각 Gate의 의미는 해당 final approval/report를 함께 연다. count만으로 contract
내용을 추론하지 않는다.

## Recipe 19: manifest boundary 확인

```bash
find experiments/phase1b/manifests -type f -name '*.json' -print | sort | \
  rg 'sensor/|oracle_simulation_only/' | sed -n '1,120p'
```

`oracle_simulation_only` manifest를 deployable estimator artifact로 제시하지 않는다.

## Recipe 20: master summary provenance와 claim-audit 상태

```bash
master=docs/research/phase1b/AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md
test -r "$master"
sha256sum "$master"
sed -n '1,40p' "$master"
sed -n '1,260p' experiments/research_index/reports/PHASE0_1_MASTER_SUMMARY_AUDIT_REPORT.md
```

Expected input artifact digest는
`657b956362457472c25ba03177e521114d1d92082cc30e10cf5f4170f52b96a2`,
evidence correction 후 indexed final digest는
`8827f4a3996b0e1f6b736de13a33a738bab06bee30e789bc1e85876e8dd40526`다.

## Recipe 21: master claim에서 canonical evidence까지 추적

```bash
python3 - <<'PY'
import json
from pathlib import Path

index = json.loads(Path(
    'docs/research/index/phase0_1/phase0_1_evidence_index.json'
).read_text())
needle = 'C4'.casefold()
for topic in index['topics']:
    questions = [q for q in topic['question_patterns'] if needle in q.casefold()]
    if not questions:
        continue
    print('\n', topic['id'], questions)
    print('canonical=', topic['canonical_sources'])
    print('navigation=', topic.get('navigation_sources', []))
    print('results=', topic.get('result_paths', []))
PY
```

`needle`을 master claim family로 바꾼다. master는 `navigation_sources`에만
사용하고, 수치는 `numeric_facts[*].source`, 현재 판정은 updated review source까지
추적한다. Phase 2 Design Review는 explicit user request로만 시작하며 이 lookup은
그 요청이나 implementation authorization을 대신하지 않는다.

## 금지된 lookup 관행

- branch/HEAD/log/diff로 과거 commit을 재구성하지 않는다.
- scope 없는 “최종 NEES”, “최종 RMSE”를 만들지 않는다.
- report의 서술이 actual source와 충돌할 때 report를 우선하지 않는다.
- historical review를 current decision처럼 사용하지 않는다.
- oracle manifest/labels를 estimator-facing inputs로 재해석하지 않는다.
- 이 evidence lookup에서 Phase 2 code/design 작업으로 자동 진행하지 않는다.
