# 0. 감사 메타 정보

## 기본 정보

- [확인] 감사 일시: `2026-07-26T16:43:02+09:00` (`date -Is` 실행 결과).
- [확인] 저장소 경로: `/home/dss-pc-05/bench` (`pwd` 실행 결과).
- [확인] git branch: `main` (`git rev-parse --abbrev-ref HEAD` 실행 결과).
- [확인] 최신 commit hash: `3cab581ba4e2558974dc412a425c0ff2e1cb80d4` (`git rev-parse HEAD` 실행 결과).
- [확인] uncommitted change 있음. `git status --short --branch`에서 tracked modified/deleted 파일, untracked configs/tests/reports/third_party 디렉토리가 다수 확인되었다. 이 보고서는 clean commit이 아니라 현재 dirty worktree 상태를 감사한 것이다.
- [확인] 감사 시작 시점에 `AUDIT_CURRENT_STATE.md`는 없었다. `test -e AUDIT_CURRENT_STATE.md; echo $?` 출력이 `1`이었다.

## 실행한 명령 전체 목록

학습(training)은 실행하지 않았다. 아래 명령 중 `--help`, import 확인, `unittest`는 1분 이내 smoke 확인으로 실행했다.

```text
1. pwd
2. date -Is
3. git status --short --branch
4. git rev-parse --abbrev-ref HEAD
5. git rev-parse HEAD
6. rg --files
7. find . -maxdepth 1 -mindepth 1 -print
8. rg -n "handoff|HANDOFF|AI_ADCS|Research_Handoff|measurement enhancement|enhancement|개선" -S .
9. find . -maxdepth 3 -type f -iname '*handoff*' -o -iname '*Handoff*' -o -iname '*HANDOFF*'
10. find . -maxdepth 3 -type f -name '*.md'
11. find . -maxdepth 4 -type f -name 'model.pt' -o -name '*.pt' -o -name '*.pth' -o -name 'train_state.json'
12. wc -l HANDOFF_PACKET.v3 README.md AUDIT_Adaptive_KNet.md AUDIT_KalmanNet_TSP.md bench/models/*.py bench/tasks/generator/*.py bench/runners/run_suite.py bench/tasks/bench_generated.py bench/models/me_split_knet.py bench/models/measurement_enhancer.py
13. rg -n "class |def |register|ModelSpec|loss|optimizer|Adam|MSE|geodesic|quat|quaternion|gyro|mag|sun|star|bias|scale|misalignment|saturation|outlier|temperature|vibration|normalize|Q|R|Jacobian|jacobian|EKF|Basilisk|measurement|enhance" bench/models bench/tasks bench/runners scripts tests reports/HANDOFF_PACKET.v3 HANDOFF_PACKET.v3 -S
14. rg -n "measurement enhancement|enhancer|enhancement|ME-Split|me_split|improv|개선|no clear|negative|worse|best|summary|mse|rmse|attitude|geodesic" reports scripts bench/configs HANDOFF_PACKET.v3 -S
15. find bench -maxdepth 2 -type d -print
16. ls
17. ls bench/models
18. ls bench/tasks/generator
19. ls bench/tasks/generators
20. ls reports
21. ls runs
22. nl -ba bench/models/registry.py
23. nl -ba bench/models/base.py
24. nl -ba HANDOFF_PACKET.v3
25. nl -ba README.md
26. rg -n "^def |^class |task_family|x_dim|y_dim|state|omega|sigma|quat|MRP|gyro|delta|bias|measurement_mode|sparse|reference|profile|noise|Q|R|F|H|dt|Basilisk|generate_" bench/tasks/generator/basilisk_imu_adcs.py -S
27. rg -n "^def |^class |task_family|x_dim|y_dim|state|omega|sigma|quat|MRP|gyro|measurement|star|sun|mag|noise|Q|R|F|H|dt|Basilisk|generate_" bench/tasks/generator/basilisk_adcs.py -S
28. rg -n "^def |^class |mse|rmse|mse_db|geodesic|attitude|NIS|NEES|nll|cov|metric|quaternion|MRP" bench/metrics bench/runners/run_suite.py scripts/audit_basilisk_imu_sparse_ref_phase3.py scripts/audit_basilisk_imu_bias_phase2.py -S
29. rg -n "^def |^class |argparse|add_argument|if __name__|suite-yaml|main\(" bench/runners/run_suite.py bench/runners/smoke_run.py bench/models/smoke_model.py main.py scripts/*.py -S
30. rg -n "^class |def setup|def train|def eval|def predict|def adapt|def save|def load|MSELoss|Adam|optimizer|lr|epochs|train_max_updates|sequence|T|x_dim|y_dim|SystemModel|Build|InitSequence|NNBuild|NNTrain|NNTest|loss|state_estimation_loss|H|F|Q|R|jacobian|finite_difference" bench/models/kalmannet_tsp.py bench/models/split_knet.py bench/models/adaptive_knet.py bench/models/maml_knet.py bench/models/mb_kf.py bench/models/basilisk_mrp_ekf.py -S
31. rg -n "^class |def __init__|def forward|def enhance|def train|def setup|def transform_measurements|def _pretrain_enhancer|enhancer_pretrain_target|MSELoss|Adam|denoise|corr|regularization|delta|causal|Conv1d|kernel|dropout|LayerNorm|loss|freeze|end-to-end|separate" bench/models/measurement_enhancer.py bench/models/me_split_knet.py -S
32. rg -n "^class |def |Spike|SNN|LIF|loss|MSELoss|Adam|train|predict|adapter_only|event|reliability|gate|checkpoint|save|load" bench/models/spike_split_knet.py bench/models/spike_ra_knet.py bench/models/g1_snn_split_knet.py -S
33. rg -n "^class |^def |task_family|generate_|_GENERATOR|registry|train|val|test|normal|normalize|split|seed|cache|x0|q2|r2|F|H|Q|R|scenario|Dataset" bench/tasks/bench_generated.py bench/tasks/data_format.py bench/tasks/generator/schema.py bench/tasks/generator/contract.py bench/tasks/generator/linear_mismatch.py bench/tasks/generator/noise_schedule.py bench/tasks/generator/ucm.py bench/tasks/generator/sine_poly.py bench/tasks/generator/switching_dynamics.py bench/tasks/generator/lorenz.py -S
34. awk 'NR>=1 && NR<=220 {print NR ":" $0}' bench/models/split_knet.py
35. awk 'NR>=220 && NR<=560 {print NR ":" $0}' bench/models/split_knet.py
36. awk 'NR>=560 && NR<=970 {print NR ":" $0}' bench/models/split_knet.py
37. awk 'NR>=970 && NR<=1320 {print NR ":" $0}' bench/models/split_knet.py
38. awk 'NR>=1320 && NR<=1514 {print NR ":" $0}' bench/models/split_knet.py
39. awk 'NR>=1 && NR<=260 {print NR ":" $0}' third_party/Split_KalmanNet/GSSFiltering/filtering.py
40. awk 'NR>=1 && NR<=260 {print NR ":" $0}' third_party/Split_KalmanNet/GSSFiltering/dnn.py
41. awk 'NR>=1 && NR<=220 {print NR ":" $0}' third_party/KalmanNet_TSP/KNet/KalmanNet_nn.py
42. awk 'NR>=1 && NR<=220 {print NR ":" $0}' third_party/KalmanNet_TSP/Filters/EKF.py
43. awk 'NR>=258 && NR<=320 {print NR ":" $0}' third_party/Split_KalmanNet/GSSFiltering/filtering.py
44. awk 'NR>=188 && NR<=360 {print NR ":" $0}' bench/models/basilisk_mrp_ekf.py
45. awk 'NR>=599 && NR<=736 {print NR ":" $0}' bench/models/basilisk_mrp_ekf.py
46. awk 'NR>=736 && NR<=767 {print NR ":" $0}' bench/models/basilisk_mrp_ekf.py
47. awk 'NR>=364 && NR<=595 {print NR ":" $0}' bench/models/basilisk_mrp_ekf.py
48. awk 'NR>=1 && NR<=260 {print NR ":" $0}' bench/models/mb_kf.py
49. awk 'NR>=260 && NR<=620 {print NR ":" $0}' bench/models/mb_kf.py
50. awk 'NR>=620 && NR<=791 {print NR ":" $0}' bench/models/mb_kf.py
51. awk 'NR>=1 && NR<=120 {print NR ":" $0}' bench/tasks/generator/basilisk_imu_adcs.py
52. awk 'NR>=143 && NR<=240 {print NR ":" $0}' bench/tasks/generator/basilisk_imu_adcs.py
53. awk 'NR>=527 && NR<=603 {print NR ":" $0}' bench/tasks/generator/basilisk_imu_adcs.py
54. awk 'NR>=602 && NR<=727 {print NR ":" $0}' bench/tasks/generator/basilisk_imu_adcs.py
55. awk 'NR>=728 && NR<=840 {print NR ":" $0}' bench/tasks/generator/basilisk_imu_adcs.py
56. awk 'NR>=840 && NR<=930 {print NR ":" $0}' bench/tasks/generator/basilisk_imu_adcs.py
57. awk 'NR>=930 && NR<=1035 {print NR ":" $0}' bench/tasks/generator/basilisk_imu_adcs.py
58. awk 'NR>=1035 && NR<=1258 {print NR ":" $0}' bench/tasks/generator/basilisk_imu_adcs.py
59. awk 'NR>=1258 && NR<=1450 {print NR ":" $0}' bench/tasks/generator/basilisk_imu_adcs.py
60. awk 'NR>=1450 && NR<=1705 {print NR ":" $0}' bench/tasks/generator/basilisk_imu_adcs.py
61. awk 'NR>=400 && NR<=516 {print NR ":" $0}' bench/tasks/generator/basilisk_imu_adcs.py
62. awk 'NR>=271 && NR<=399 {print NR ":" $0}' bench/tasks/generator/basilisk_imu_adcs.py
63. awk 'NR>=151 && NR<=359 {print NR ":" $0}' bench/tasks/generator/basilisk_adcs.py
64. awk 'NR>=398 && NR<=635 {print NR ":" $0}' bench/tasks/generator/basilisk_adcs.py
65. rg -n "suite_name|task_id|task_family|x_dim|y_dim|sequence_length|n_train|n_val|n_test|simulation:|dt:|inertia|disturbance|noise:|Q:|R:|gyro_noise_std|bias_init_std|bias_rw_std|scale|misalignment|vibration|outlier|saturation|profile_id|measurement_mode|sparse_ref|ref_|models:|model_id|repo|train_max_updates|lr|batch_size|enhancer|event" bench/configs/gpu_basilisk_imu_pilot.yaml bench/configs/gpu_basilisk_imu_pilot_pretrained_enhancer.yaml bench/configs/gpu_basilisk_me_split_full.yaml bench/configs/gpu_basilisk_me_split_tuning.yaml bench/configs/gpu_basilisk_structured_corruption_full.yaml bench/configs/gpu_basilisk_imu_bias_pilot.yaml bench/configs/gpu_basilisk_imu_sparse_ref_pilot.yaml bench/configs/gpu_basilisk_imu_sparse_ref_sanity_500.yaml bench/configs/suite_basilisk_spike_split_pilot_cuda.yaml bench/configs/suite_basilisk_spike_ra_phase_a_event.yaml -S
66. awk 'NR>=1 && NR<=240 {print NR ":" $0}' bench/configs/gpu_basilisk_imu_pilot_pretrained_enhancer.yaml
67. awk 'NR>=1 && NR<=260 {print NR ":" $0}' bench/configs/gpu_basilisk_imu_bias_pilot.yaml
68. awk 'NR>=1 && NR<=320 {print NR ":" $0}' bench/configs/gpu_basilisk_imu_sparse_ref_pilot.yaml
69. awk 'NR>=1 && NR<=130 {print NR ":" $0}' bench/models/measurement_enhancer.py
70. awk 'NR>=131 && NR<=199 {print NR ":" $0}' bench/models/measurement_enhancer.py
71. awk 'NR>=1 && NR<=170 {print NR ":" $0}' bench/models/me_split_knet.py
72. awk 'NR>=170 && NR<=268 {print NR ":" $0}' bench/models/me_split_knet.py
73. awk 'NR>=268 && NR<=430 {print NR ":" $0}' bench/models/me_split_knet.py
74. awk 'NR>=429 && NR<=518 {print NR ":" $0}' bench/models/me_split_knet.py
75. awk 'NR>=520 && NR<=690 {print NR ":" $0}' bench/tasks/bench_generated.py
76. awk 'NR>=873 && NR<=1018 {print NR ":" $0}' bench/tasks/bench_generated.py
77. awk 'NR>=1034 && NR<=1113 {print NR ":" $0}' bench/tasks/bench_generated.py
78. awk 'NR>=1133 && NR<=1161 {print NR ":" $0}' bench/tasks/bench_generated.py
79. awk 'NR>=1 && NR<=160 {print NR ":" $0}' bench/tasks/generator/contract.py
80. awk 'NR>=1180 && NR<=1375 {print NR ":" $0}' bench/runners/run_suite.py
81. awk 'NR>=1376 && NR<=1625 {print NR ":" $0}' bench/runners/run_suite.py
82. awk 'NR>=2200 && NR<=2325 {print NR ":" $0}' bench/runners/run_suite.py
83. awk 'NR>=2520 && NR<=2595 {print NR ":" $0}' bench/runners/run_suite.py
84. awk 'NR>=1625 && NR<=1825 {print NR ":" $0}' bench/runners/run_suite.py
85. awk 'NR>=1825 && NR<=2015 {print NR ":" $0}' bench/runners/run_suite.py
86. awk 'NR>=2015 && NR<=2205 {print NR ":" $0}' bench/runners/run_suite.py
87. awk 'NR>=1 && NR<=180 {print NR ":" $0}' bench/metrics/adcs_event.py
88. awk 'NR>=1 && NR<=190 {print NR ":" $0}' bench/metrics/core.py
89. rg -n "NIS|NEES|geodesic|attitude_rmse|mse_db|nll|cov|metrics.json|adcs_event" bench scripts tests reports -S
90. find runs -type f \( -name 'model.pt' -o -name 'checkpoint.pt' -o -name '*.pth' -o -name 'train_state.json' \) -print
91. find reports -maxdepth 2 -type f \( -name '*summary*.csv' -o -name '*analysis*.md' -o -name '*notes*.md' \) -print
92. find runs -type f \( -name 'model.pt' -o -name 'checkpoint.pt' -o -name 'train_state.json' \) | wc -l
93. find runs -type f \( -name 'model.pt' -o -name 'checkpoint.pt' -o -name 'train_state.json' \) | rg 'basilisk|Basilisk|phase6|spike|me_split|structured|imu|adcs'
94. find . -maxdepth 2 -type f \( -name 'pyproject.toml' -o -name 'requirements*.txt' -o -name 'environment*.yml' -o -name 'environment*.yaml' -o -name 'uv.lock' -o -name 'poetry.lock' \) -print
95. rg -n "dependencies|requires-python|torch|numpy|Basilisk|pytest|pyyaml|matplotlib" pyproject.toml requirements*.txt environment*.yml environment*.yaml uv.lock -S
96. rg -n "dependencies|requires-python|torch|numpy|Basilisk|pytest|pyyaml|matplotlib" pyproject.toml uv.lock -S
97. awk 'NR>=1 && NR<=180 {print NR ":" $0}' pyproject.toml
98. rg -n "Basilisk|AVS|avslab|pip|install|uv|environment|requirements|CUDA|torch" README.md docs scripts bench/configs -S
99. rg -n "normaliz|Normalize|mean|std|train set|train_set|scaler|standard" bench scripts tests -S
100. awk 'NR>=1 && NR<=80 {print NR ":" $0}' reports/me_split_knet_full_summary.csv
101. awk 'NR>=1 && NR<=120 {print NR ":" $0}' reports/me_split_knet_full_analysis.md
102. awk 'NR>=1 && NR<=80 {print NR ":" $0}' reports/me_split_structured_corruption_full_summary.csv
103. awk 'NR>=1 && NR<=120 {print NR ":" $0}' reports/me_split_structured_corruption_full_analysis.md
104. awk 'NR>=1 && NR<=80 {print NR ":" $0}' reports/basilisk_imu_pretrained_enhancer_gpu_pilot_summary.csv
105. awk 'NR>=1 && NR<=100 {print NR ":" $0}' reports/basilisk_imu_bias_gpu_pilot_summary.csv
106. awk 'NR>=1 && NR<=120 {print NR ":" $0}' reports/basilisk_imu_sparse_ref_gpu_pilot_summary.csv
107. awk 'NR>=1 && NR<=120 {print NR ":" $0}' reports/basilisk_imu_sparse_ref_sanity_500_summary.csv
108. wc -c reports/basilisk_imu_sparse_ref_gpu_pilot_summary.csv reports/basilisk_imu_sparse_ref_gpu_pilot_failure_summary.csv reports/summary_gpu_basilisk_imu_sparse_ref_pilot.csv reports/summary_gpu_basilisk_imu_sparse_ref_sanity_500.csv
109. awk 'NR>=1 && NR<=80 {print NR ":" $0}' reports/basilisk_imu_sparse_ref_gpu_pilot_failure_summary.csv
110. awk 'NR>=1 && NR<=80 {print NR ":" $0}' reports/summary_gpu_basilisk_imu_sparse_ref_pilot.csv
111. awk 'NR>=1 && NR<=80 {print NR ":" $0}' reports/summary_gpu_basilisk_imu_sparse_ref_sanity_500.csv
112. awk 'NR>=1 && NR<=140 {print NR ":" $0}' reports/basilisk_imu_sparse_ref_sanity_500_analysis.md
113. awk 'NR>=1 && NR<=160 {print NR ":" $0}' reports/me_split_knet_ablation_notes.md
114. awk 'NR>=1 && NR<=80 {print NR ":" $0}' reports/me_split_knet_ablation_summary.csv
115. awk 'NR>=1 && NR<=80 {print NR ":" $0}' reports/me_split_knet_tuning_summary.csv
116. awk 'NR>=500 && NR<=930 {print NR ":" $0}' bench/models/adaptive_knet.py
117. awk 'NR>=1200 && NR<=1335 {print NR ":" $0}' bench/models/maml_knet.py
118. awk 'NR>=1 && NR<=220 {print NR ":" $0}' bench/models/spike_split_knet.py
119. awk 'NR>=1 && NR<=220 {print NR ":" $0}' bench/models/g1_snn_split_knet.py
120. awk 'NR>=220 && NR<=520 {print NR ":" $0}' bench/models/spike_split_knet.py
121. awk 'NR>=1 && NR<=260 {print NR ":" $0}' bench/models/spike_ra_knet.py
122. awk 'NR>=520 && NR<=920 {print NR ":" $0}' bench/models/spike_ra_knet.py
123. rg -n "class .*Adapter|def train|def eval|def predict|def adapt|get_adapter_meta|implementation_stage|event|reliability|MSELoss|Adam" bench/models/spike_split_knet.py bench/models/g1_snn_split_knet.py bench/models/spike_ra_knet.py -S
124. awk 'NR>=315 && NR<=430 {print NR ":" $0}' bench/models/kalmannet_tsp.py
125. awk 'NR>=430 && NR<=575 {print NR ":" $0}' bench/models/kalmannet_tsp.py
126. awk 'NR>=575 && NR<=740 {print NR ":" $0}' bench/models/kalmannet_tsp.py
127. awk 'NR>=740 && NR<=815 {print NR ":" $0}' bench/models/kalmannet_tsp.py
128. awk 'NR>=873 && NR<=1018 {print NR ":" $0}' bench/tasks/bench_generated.py
129. awk 'NR>=1133 && NR<=1161 {print NR ":" $0}' bench/tasks/bench_generated.py
130. awk 'NR>=1 && NR<=95 {print NR ":" $0}' bench/tasks/generator/contract.py
131. awk 'NR>=95 && NR<=180 {print NR ":" $0}' bench/tasks/generator/contract.py
132. rg -n "normaliz|Normalize|normalizer|scaler|standardize|mean_std|zscore|z-score|train_mean|train_std|input_mean|input_std" bench/models bench/tasks bench/runners scripts tests -S
133. rg -n "magnet|magnetometer|sun sensor|sun_sensor|star tracker|star_tracker|star|sun|mag" bench/tasks/generator bench/configs bench/models scripts tests -S
134. rg -n "temperature|temp|vibration|outlier|saturation|scale|misalignment|bias|random_walk|drift|white|gaussian|LSB" bench/tasks/generator bench/configs scripts -S
135. rg -n "real|NCLT|UZH|dataset|csv|rosbag|imu|groundtruth|ground_truth" bench/tasks/generator/datasets bench/tasks/generator scripts/datasets bench/configs -S
136. rg -n "magnetometer|\bmag\b|sun_sensor|sun sensor|star_tracker|star tracker" bench/tasks/generator bench/configs bench/models scripts tests -S
137. rg -n "temperature|temp" bench/tasks/generator bench/configs bench/models scripts tests -S
138. rg -n "saturation|set_oSatBounds|set_aSatBounds|setLSBs|LSB|scale factor|scale_factor|misalignment|axis_misalignment|outlier|vibration|random_walk|drift|bias|gaussian" bench/tasks/generator/basilisk_adcs.py bench/tasks/generator/basilisk_imu_adcs.py bench/configs/gpu_basilisk_structured_corruption_full.yaml bench/configs/gpu_basilisk_imu_pilot_pretrained_enhancer.yaml bench/configs/gpu_basilisk_imu_bias_pilot.yaml bench/configs/gpu_basilisk_imu_sparse_ref_pilot.yaml -S
139. rg -n "Basilisk|SimulationBaseClass|spacecraft|extForceTorque|sigma_BN|omega_BN_B|mrp|shadow|Normalize|normalize" bench/tasks/generator/basilisk_adcs.py bench/tasks/generator/basilisk_imu_adcs.py bench/models/basilisk_mrp_ekf.py bench/metrics/adcs_event.py -S
140. timeout 60 python -m bench.runners.run_suite --help
141. timeout 60 python -c "from bench.models.registry import available_models; print(sorted(available_models()))"
142. timeout 60 python -c "from Basilisk.simulation import spacecraft, imuSensor; from Basilisk.utilities import SimulationBaseClass; print('Basilisk import ok')"
143. timeout 60 python -m pytest tests/test_me_split_knet.py tests/test_basilisk_mrp_ekf.py tests/test_basilisk_imu_model_compat.py -q
144. timeout 60 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m bench.runners.run_suite --help
145. timeout 60 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -c "from bench.models.registry import available_models; print(sorted(available_models()))"
146. timeout 60 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -c "from Basilisk.simulation import spacecraft, imuSensor; from Basilisk.utilities import SimulationBaseClass; print('Basilisk import ok')"
147. timeout 60 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest tests/test_me_split_knet.py tests/test_basilisk_mrp_ekf.py tests/test_basilisk_imu_model_compat.py -q
148. awk 'NR>=1 && NR<=90 {print NR ":" $0}' bench/models/registry.py
149. timeout 60 MPLCONFIGDIR=/tmp/matplotlib /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -c "from bench.models.registry import list_model_ids; print(list_model_ids())"
150. timeout 60 MPLCONFIGDIR=/tmp/matplotlib /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -c "from bench.models.basilisk_mrp_ekf import BasiliskMRPEKFAdapter; from bench.models.measurement_enhancer import MeasurementEnhancer; print('imports ok')"
151. timeout 60 MPLCONFIGDIR=/tmp/matplotlib /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m unittest tests.test_me_split_knet tests.test_basilisk_mrp_ekf tests.test_basilisk_imu_model_compat
152. env MPLCONFIGDIR=/tmp/matplotlib timeout 60 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -c "from bench.models.registry import list_model_ids; print(list_model_ids())"
153. env MPLCONFIGDIR=/tmp/matplotlib timeout 60 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -c "from bench.models.basilisk_mrp_ekf import BasiliskMRPEKFAdapter; from bench.models.measurement_enhancer import MeasurementEnhancer; print('imports ok')"
154. env MPLCONFIGDIR=/tmp/matplotlib timeout 60 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m unittest tests.test_me_split_knet tests.test_basilisk_mrp_ekf tests.test_basilisk_imu_model_compat
155. env MPLCONFIGDIR=/tmp/matplotlib timeout 60 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -c "from Basilisk.simulation import spacecraft, imuSensor; from Basilisk.utilities import SimulationBaseClass; print('Basilisk import ok')"
156. test -e AUDIT_CURRENT_STATE.md; echo $?
157. git status --short --branch
158. find . -maxdepth 1 -mindepth 1 -printf '%f\n' | sort
159. find third_party -maxdepth 2 -type d -print | sort
160. find runs -type f \( -name 'model.pt' -o -name 'train_state.json' -o -name 'metrics.json' \) | rg 'basilisk_mrp_ekf|Basilisk_MRP|mrp_ekf|basilisk_gpu_with_ekf|gpu_basilisk_adcs_with_ekf'
161. awk 'NR>=1 && NR<=80 {print NR ":" $0}' reports/summary_basilisk_mrp_ekf_smoke.csv
162. awk 'NR>=1 && NR<=100 {print NR ":" $0}' reports/basilisk_gpu_with_ekf_summary.csv
163. awk 'NR>=1 && NR<=140 {print NR ":" $0}' bench/configs/gpu_basilisk_adcs_with_ekf.yaml
164. rg -n "^(Basilisk|basilisk|torch|numpy|pytest|PyYAML|pyyaml|matplotlib|pandas|scipy)==|^# requirements.lock" requirements.lock -S
165. awk 'NR>=1 && NR<=60 {print NR ":" $0}' requirements.lock
166. wc -l requirements.lock
167. rg -n "Basilisk|basilisk" requirements.lock -S
168. awk 'NR>=1 && NR<=120 {print NR ":" $0}' main.py
169. awk 'NR>=1 && NR<=160 {print NR ":" $0}' bench/runners/smoke_run.py
170. rg -n "def main|argparse|if __name__|prepare_bench_generated|smoke|run_suite" bench/tasks scripts bench/runners main.py -S
171. find scripts -maxdepth 2 -type f -print | sort
```

## 감사에서 다루지 못한 디렉토리/파일과 이유

- [확인] `bench/tasks/generator/datasets/` 및 실제 데이터셋 계열(`nclt`, `uzh_fpv`)은 `rg`로 존재와 키워드만 확인했고 세부 loader 로직은 정독하지 못했다. 이번 요청의 초점이 KalmanNet 계열 위성 자세 추정/ADCS였기 때문이다 (`bench/tasks/bench_generated.py:557-690`, 명령 135).
- [확인] `runs/`는 파일 수가 많아 `model.pt`, `train_state.json`, `metrics.json` 중심으로 확인했다. 전체 로그와 모든 체크포인트 텐서는 열지 않았다 (명령 90, 92, 93, 160).
- [확인] `plots/`, `logs/`, `bench_data_cache/`, `.venv/`, `.basilisk_deps/`, `docker/`, `.github/`는 구조 존재만 확인했거나 검색 대상에서 제외했다. 감사 목적상 모델/필터/데이터/평가 코드와 직접 연결되는 근거가 우선이었다 (명령 158).
- [확인] `third_party/Adaptive-KNet-ICASSP24`, `third_party/MAML_KalmanNet`은 wrapper가 참조하는 범위 중심으로 확인했고 원본 저장소 전체를 정독하지 않았다. 반면 `third_party/KalmanNet_TSP`와 `third_party/Split_KalmanNet`의 필터 핵심 파일은 직접 확인했다 (명령 39-43, 159).
- [불명: handoff 문서 없음] `AI_ADCS_KalmanNet_Research_Handoff.md` 또는 ADCS/enhancement 전용 handoff 문서는 발견하지 못했다. 발견된 handoff 유사 파일은 `HANDOFF_PACKET.v3` 하나이며 내용은 Adaptive-KNet S3 인수인계였다 (명령 8-10, `HANDOFF_PACKET.v3:1-13`, `HANDOFF_PACKET.v3:104-107`).

# 1. 저장소 인벤토리

## 최상위 구조

- [확인] `bench/`: 핵심 Python 패키지. 모델 adapter, runner, task generator, metrics가 포함된다 (`find bench -maxdepth 2 -type d -print`, `bench/models/registry.py:26-47`).
- [확인] `bench/models/`: KalmanNet/Split-KalmanNet/Adaptive/MAML/Spike/ME/EKF/KF adapter 구현 위치 (`ls bench/models`, `bench/models/registry.py:26-47`).
- [확인] `bench/tasks/`: synthetic/real/replay task generation 및 split/cache 로딩 구현 위치 (`bench/tasks/bench_generated.py:557-690`, `bench/tasks/bench_generated.py:873-1018`).
- [확인] `bench/tasks/generator/`: task generator별 구현 위치. ADCS 관련 `basilisk_adcs.py`, `basilisk_imu_adcs.py`가 있다 (`ls bench/tasks/generator`, `bench/tasks/bench_generated.py:557-690`).
- [확인] `bench/runners/`: suite 실행기와 smoke runner 위치. 주 진입점은 `bench/runners/run_suite.py`이다 (`bench/runners/run_suite.py:2686`, `bench/runners/smoke_run.py:51-70`).
- [확인] `bench/metrics/`: core MSE/RMSE/NLL 및 ADCS event/attitude metrics 구현 위치 (`bench/metrics/core.py:14-46`, `bench/metrics/adcs_event.py:65-108`).
- [확인] `bench/configs/`: YAML suite/task/model/runner 설정 파일 위치. Basilisk ADCS/IMU/ME/Spike 관련 config가 다수 존재한다 (명령 65, `bench/configs/gpu_basilisk_adcs_with_ekf.yaml:1-124`).
- [확인] `scripts/`: audit/report/run shell script 위치. GPU suite 실행 스크립트, 분석 스크립트가 있다 (`find scripts -maxdepth 2 -type f -print | sort`).
- [확인] `third_party/`: 외부 공개 코드 복사본 또는 fork로 보이는 디렉토리. `Adaptive-KNet-ICASSP24`, `KalmanNet_TSP`, `MAML_KalmanNet`, `Split_KalmanNet`가 확인되었다 (`find third_party -maxdepth 2 -type d -print | sort`).
- [확인] `reports/`: CSV/Markdown 분석 결과 저장 위치. ME, Basilisk, sparse-ref, EKF summary가 확인되었다 (`ls reports`, 명령 100-115, 161-162).
- [확인] `runs/`: 개별 실행 결과, 체크포인트, metrics 저장 위치. 다수의 `model.pt`, `train_state.json`, `metrics.json`가 확인되었다 (명령 90, 92, 93, 160).
- [확인] `tests/`: 단위 테스트 위치. `tests.test_me_split_knet`, `tests.test_basilisk_mrp_ekf`, `tests.test_basilisk_imu_model_compat`는 `unittest`로 통과했다 (명령 154).
- [확인] `docs/`: 문서성 자료 위치. 이번 감사에서는 키워드 검색 수준으로만 확인했다 (명령 98).
- [확인] `main.py`: 현재는 `print("Hello from bench!")`만 수행하는 scaffold성 entry point다 (`main.py:1-6`).

## 실행 진입점(entry point)

- [확인] `python -m bench.runners.run_suite --suite-yaml ...`가 핵심 suite runner 진입점이다. CLI help가 정상 출력되었다 (명령 144, `bench/runners/run_suite.py:2686`).
- [확인] `bench/runners/smoke_run.py`는 smoke 실행용 CLI를 가진다. `argparse` 인자와 `run_one` 호출이 있다 (`bench/runners/smoke_run.py:51-70`, `bench/runners/smoke_run.py:109-123`).
- [확인] `bench/tasks/smoke_data.py`는 smoke data 생성/검사용 진입점을 가진다 (`rg ... def main`, 명령 170).
- [확인] `main.py`는 실제 학습/평가 진입점이 아니라 hello 출력뿐이다 (`main.py:1-6`).
- [확인] `scripts/run_*.sh` 계열이 GPU/CPU suite 실행용 shell entry point로 존재한다 (`find scripts -maxdepth 2 -type f -print | sort`).
- [확인] `scripts/audit_*.py`, `scripts/report_*.py` 계열은 결과 요약/감사용 스크립트로 존재한다 (`find scripts -maxdepth 2 -type f -print | sort`).
- [확인] notebook은 `find . -maxdepth 3 -type f -name '*.md'`와 `rg --files` 범위에서 주요 진입점으로 확인하지 못했다. 별도 `.ipynb` 전수 명령은 실행하지 않았다.

## 주요 모듈 분류

### 동작 확인됨

- [확인] `bench.runners.run_suite`는 명시 Python에서 `--help`가 60초 내 정상 동작했다 (명령 144).
- [확인] `bench.models.registry.list_model_ids()`는 명시 Python에서 정상 import 및 model id 목록 출력이 가능했다 (명령 152, `bench/models/registry.py:64-65`).
- [확인] `BasiliskMRPEKFAdapter`, `MeasurementEnhancer` import가 정상 동작했다 (명령 153).
- [확인] Basilisk 핵심 모듈 `spacecraft`, `imuSensor`, `SimulationBaseClass` import가 정상 동작했다 (명령 155).
- [확인] `tests.test_me_split_knet`, `tests.test_basilisk_mrp_ekf`, `tests.test_basilisk_imu_model_compat`는 `unittest` 기준 14 tests, 0.227s, OK였다 (명령 154).

### 미완성(WIP)

- [확인] `adaptive_knet`는 adapter가 학습/평가/적응을 제공하지만 handoff에서 S3 known issue로 "unsupervised observation reconstruction MSE" 적응 목적이 원 논문 E2E와 다르다고 적혀 있다 (`HANDOFF_PACKET.v3:104-107`, `bench/models/adaptive_knet.py:743-876`).
- [확인] `me_split_knet_v0` 계열은 wrapper 문서상 two-stage 실험 구현이며, Stage A enhancer pretrain 후 freeze하고 Stage B Split-KNet을 학습한다 (`bench/models/me_split_knet.py:13-20`, `bench/models/me_split_knet.py:71-125`, `bench/models/me_split_knet.py:223-254`).
- [확인] `spike_split_knet`, `g1_snn_split_knet`, `spike_ra_knet`는 branch replacement / reliability adapter 실험 구현이다. meta에 `implementation_stage`가 있고 SpikeRA는 gate diagnostics를 추가한다 (`bench/models/spike_split_knet.py:1-6`, `bench/models/g1_snn_split_knet.py:1-19`, `bench/models/spike_ra_knet.py:1-26`, `bench/models/spike_ra_knet.py:895-920`).
- [확인] Basilisk sparse-ref original GPU pilot는 모든 run이 `train_nan`으로 실패한 결과 파일이 남아 있다 (`reports/basilisk_imu_sparse_ref_gpu_pilot_failure_summary.csv:2-9`, `reports/summary_gpu_basilisk_imu_sparse_ref_pilot.csv:2-9`).

### 사용되지 않음(dead code 의심)

- [확인] `main.py`는 hello 출력뿐이며 실제 benchmark entry point와 연결되지 않는다 (`main.py:1-6`).
- [추정] `third_party/KalmanNet_TSP/Filters/EKF.py`는 원본 EKF 구현으로 존재하지만 `bench/models/registry.py`의 등록 모델에는 직접 등록되어 있지 않다. bench의 EKF는 별도 `basilisk_mrp_ekf` adapter다 (`third_party/KalmanNet_TSP/Filters/EKF.py:1-70`, `bench/models/registry.py:26-47`).
- [추정] `README.md`는 현재 ADCS/ME/Spike 상태를 반영하지 못하는 scaffold 문서에 가깝다. README가 source-of-truth를 `/mnt/data`로 설명하고 third_party는 수정하지 않는다고 적지만, 현재 repo에는 ADCS/Basilisk/Spike/ME 코드와 dirty third_party 항목이 많다 (`README.md:9-22`, `README.md:72-78`, `git status --short --branch`).

## 외부 코드 출처 구분

- [확인] `third_party/KalmanNet_TSP`는 KalmanNet 원본 계열 코드로 보인다. `KalmanNetNN`과 `Filters/EKF.py`가 있으며 wrapper `bench/models/kalmannet_tsp.py`가 이를 import해 사용한다 (`third_party/KalmanNet_TSP/KNet/KalmanNet_nn.py:1-31`, `third_party/KalmanNet_TSP/Filters/EKF.py:1-70`, `bench/models/kalmannet_tsp.py:315-410`).
- [확인] `third_party/Split_KalmanNet`는 Split-KalmanNet 원본 계열 코드로 보인다. `GSSFiltering/filtering.py`와 `dnn.py`의 `split_KNet`를 wrapper `bench/models/split_knet.py`가 사용한다 (`third_party/Split_KalmanNet/GSSFiltering/filtering.py:214-260`, `third_party/Split_KalmanNet/GSSFiltering/dnn.py:14-101`, `bench/models/split_knet.py:441-562`).
- [확인] `third_party/Adaptive-KNet-ICASSP24`, `third_party/MAML_KalmanNet`도 외부 공개 코드 계열 디렉토리로 존재한다 (`find third_party -maxdepth 2 -type d -print | sort`).
- [확인] `bench/models/*Adapter`, `bench/tasks/generator/basilisk_*.py`, `bench/runners/run_suite.py`, `bench/metrics/*`는 이 benchmark에 맞춘 직접 작성 wrapper/adapter/runner/generator 코드로 분류된다 (`bench/models/registry.py:26-47`, `bench/tasks/bench_generated.py:557-690`, `bench/runners/run_suite.py:1872-1964`).

# 2. 구현된 모델 목록

## EKF / KF 계열

- [확인] `basilisk_mrp_ekf` 구현 위치는 `bench/models/basilisk_mrp_ekf.py`이며 registry에 등록되어 있다 (`bench/models/registry.py:43-47`).
- [확인] `basilisk_mrp_ekf`는 학습 모델이 아니라 deterministic EKF adapter다. `train()`은 checkpoint/metadata 저장만 수행하고 learnable parameter 학습은 하지 않는다 (`bench/models/basilisk_mrp_ekf.py:364-402`).
- [확인] `basilisk_mrp_ekf` 완성도는 full-state MRP ADCS task에 대한 추론/평가 가능으로 판단된다. setup에서 `x_dim==6`, `y_dim==6`을 요구하고 `eval()`/`predict()`가 구현되어 있다 (`bench/models/basilisk_mrp_ekf.py:268-271`, `bench/models/basilisk_mrp_ekf.py:404-494`).
- [확인] `basilisk_mrp_ekf`의 관련 실행 결과는 `reports/summary_basilisk_mrp_ekf_smoke.csv`와 `reports/basilisk_gpu_with_ekf_summary.csv`에 남아 있다. smoke summary는 `mse=6.763876e-06`, `mse_db=-51.698043`이다 (`reports/summary_basilisk_mrp_ekf_smoke.csv:2`).
- [확인] `basilisk_mrp_ekf` 관련 run 파일 검색에서는 `metrics.json`은 확인되었지만 `model.pt`는 핵심 결과 파일로 확인되지 않았다. 이는 train이 no-op인 adapter 특성과 일치한다 (명령 160, `bench/models/basilisk_mrp_ekf.py:364-402`).
- [확인] `mb_kf_oracle`, `mb_kf_nominal`, `oracle_kf`, `nominal_kf`, `oracle_shift_kf`는 linear KF baseline으로 registry에 등록되어 있다 (`bench/models/registry.py:36-42`).
- [확인] `mb_kf.py`는 F/H/Q/R를 system_info/config에서 받아 standard linear Kalman filter rollout을 수행하고, train은 no-op이다 (`bench/models/mb_kf.py:328-355`, `bench/models/mb_kf.py:373-417`, `bench/models/mb_kf.py:654-745`).

## KalmanNet

- [확인] `kalmannet_tsp` 구현 위치는 `bench/models/kalmannet_tsp.py`이고 registry에 등록되어 있다 (`bench/models/registry.py:26-27`).
- [확인] `kalmannet_tsp`는 학습 가능 adapter다. setup에서 third_party `KalmanNetNN`/`SystemModel`를 구성하고, train에서 Adam과 MSELoss로 sequence prediction loss를 최소화한다 (`bench/models/kalmannet_tsp.py:315-410`, `bench/models/kalmannet_tsp.py:430-573`).
- [확인] `kalmannet_tsp`는 추론/eval/save/load가 구현되어 있다 (`bench/models/kalmannet_tsp.py:575-719`).
- [확인] `kalmannet_tsp`의 `adapt()`는 no-op으로 frozen only다 (`bench/models/kalmannet_tsp.py:690-703`).
- [확인] 원 논문 구조 대비 bench wrapper는 third_party KalmanNet을 직접 수정하기보다 `F/H/Q/R`, dims, train budget, checkpoint, metrics contract를 adapter로 감싼 형태다 (`third_party/KalmanNet_TSP/KNet/KalmanNet_nn.py:15-31`, `bench/models/kalmannet_tsp.py:315-410`).
- [확인] 학습 checkpoint는 `runs/` 아래 다수 `model.pt`/`train_state.json` 검색 결과에 포함되어 있으나, 어떤 checkpoint가 최종 논문/보고서용인지 코드만으로는 특정할 수 없다 (명령 90, 92, 93).

## Split-KalmanNet

- [확인] `split_knet` 구현 위치는 `bench/models/split_knet.py`이고 registry에 등록되어 있다 (`bench/models/registry.py:30-31`).
- [확인] `split_knet`는 학습 가능 adapter다. setup에서 third_party Split-KalmanNet를 불러오고, train에서 Adam/MSE 기반 full sequence 학습을 수행한다 (`bench/models/split_knet.py:441-562`, `bench/models/split_knet.py:609-904`).
- [확인] `split_knet`는 `predict`, `eval`, `save`, `load`가 구현되어 있으며 `adapt()`는 no-op이다 (`bench/models/split_knet.py:915-1083`).
- [확인] 원 논문 구조 대비 bench wrapper는 외부 `split_KNet` 본체를 사용하되 `F/H` linear system adapter, seed, budget, dtype/device, checkpoint ledger를 추가한다 (`bench/models/split_knet.py:36-69`, `bench/models/split_knet.py:441-562`, `third_party/Split_KalmanNet/GSSFiltering/filtering.py:214-260`).
- [확인] 학습 checkpoint는 `runs/` 아래 다수 존재하지만 최종 채택 checkpoint 구분은 코드만으로 불명확하다 (명령 90, 92, 93).

## Adaptive KalmanNet

- [확인] `adaptive_knet` 구현 위치는 `bench/models/adaptive_knet.py`이고 registry에 등록되어 있다 (`bench/models/registry.py:28-29`).
- [확인] `adaptive_knet`는 train/eval/adapt를 구현한다. train은 supervised MSE(pred, x), adapt는 unsupervised observation reconstruction MSE를 사용한다 (`bench/models/adaptive_knet.py:500-649`, `bench/models/adaptive_knet.py:743-876`).
- [확인] `HANDOFF_PACKET.v3`는 이 구현이 원 Adaptive-KNet 논문 Pipeline_NE의 end-to-end adaptation이 아니라 observation reconstruction 목적이라는 known issue를 명시한다 (`HANDOFF_PACKET.v3:104-107`).
- [추정] 현재 `adaptive_knet`는 연구용 prototype 또는 S3 integration 상태에 가깝고, 원 논문 fidelity claim에는 주의가 필요하다 (`HANDOFF_PACKET.v3:5-13`, `HANDOFF_PACKET.v3:104-107`).

## MAML KalmanNet

- [확인] `maml_knet` 구현 위치는 `bench/models/maml_knet.py`이고 registry에 등록되어 있다 (`bench/models/registry.py:29-30`).
- [확인] `maml_knet`는 third_party import-model 기반 integration으로 보이며, `adapt()`는 unsupported/no-op이다 (`bench/models/maml_knet.py:1275-1297`, `bench/models/maml_knet.py:1312-1335`).
- [확인] evaluation rollout과 validation MSE loss가 구현되어 있다 (`bench/models/maml_knet.py:1200-1273`).
- [추정] 완성도는 "학습/평가 adapter 존재, meta-adaptation 기능은 bench 기준 미완성 또는 제한적"으로 분류된다 (`bench/models/maml_knet.py:1275-1335`).

## Measurement Enhancement 네트워크

- [확인] `MeasurementEnhancer` 구현 위치는 `bench/models/measurement_enhancer.py`이며 causal TCN residual enhancer다 (`bench/models/measurement_enhancer.py:11-104`).
- [확인] `me_split_knet_v0` 구현 위치는 `bench/models/me_split_knet.py`이고 registry에 여러 variant로 등록되어 있다 (`bench/models/registry.py:32-35`).
- [확인] `me_split_knet_v0`는 enhancer 단독 pretrain 후 freeze하고 Split-KalmanNet을 학습하는 two-stage wrapper다 (`bench/models/me_split_knet.py:13-20`, `bench/models/me_split_knet.py:71-125`, `bench/models/me_split_knet.py:223-254`).
- [확인] enhancement checkpoint는 combined checkpoint로 저장된다. train_state에는 enhancer pretrain ledger와 split train_state가 들어간다 (`bench/models/me_split_knet.py:460-518`).

## Spike / SNN 계열

- [확인] `spike_split_knet`, `g1_snn_split_knet`, `spike_ra_knet`는 registry에 등록되어 있다 (`bench/models/registry.py:31-36`).
- [확인] `spike_split_knet`는 Split-KNet의 G2 branch를 SNN으로 교체하는 실험 구현이다 (`bench/models/spike_split_knet.py:1-6`, `bench/models/spike_split_knet.py:347-388`).
- [확인] `g1_snn_split_knet`는 G1 branch SNN ablation 구현이다 (`bench/models/g1_snn_split_knet.py:1-19`, `bench/models/g1_snn_split_knet.py:121-159`).
- [확인] `spike_ra_knet`는 reliability adapter/gate를 추가하며 event-weighted loss와 gate diagnostics를 구현한다 (`bench/models/spike_ra_knet.py:1-26`, `bench/models/spike_ra_knet.py:520-564`, `bench/models/spike_ra_knet.py:702-861`).
- [추정] Spike 계열은 ADCS event 실험용 WIP branch로 보이며, canonical KalmanNet/Split-KNet 대비 원 논문 대응 구조가 아닌 자체 확장 실험이다 (`bench/models/spike_ra_knet.py:895-920`, `bench/configs/suite_basilisk_spike_ra_phase_a_event.yaml`, 명령 65).

# 3. 필터 정식화 — 이 섹션이 가장 중요하다

## 상태 벡터 x 정의

- [확인] full-state Basilisk ADCS task의 상태는 `x = [sigma_BN(3), omega_BN_B(3)]`, 차원 6이다. 여기서 `sigma_BN`은 quaternion이 아니라 MRP(Modified Rodrigues Parameters)다 (`bench/tasks/generator/basilisk_adcs.py:408-449`, `bench/configs/gpu_basilisk_adcs_with_ekf.yaml:11-41`).
- [확인] `basilisk_mrp_ekf`도 동일하게 `x=[MRP sigma(3), angular velocity omega(3)]`, `state_dim=6`을 사용한다 (`bench/models/basilisk_mrp_ekf.py:188-195`, `bench/models/basilisk_mrp_ekf.py:268-271`).
- [확인] 기본 IMU ADCS task `basilisk_imu_adcs_v0`의 상태도 `x=[sigma_BN(3), omega_BN_B(3)]`, 차원 6이다 (`bench/tasks/generator/basilisk_imu_adcs.py:527-599`, `bench/tasks/generator/basilisk_imu_adcs.py:602-915`).
- [확인] gyro bias 포함 IMU task `basilisk_imu_bias_adcs_v0`의 상태는 `x=[sigma(3), omega(3), bias(3)]`, 차원 9이다 (`bench/tasks/generator/basilisk_imu_adcs.py:936-939`, `bench/tasks/generator/basilisk_imu_adcs.py:1044-1048`, `bench/tasks/generator/basilisk_imu_adcs.py:1080-1228`).
- [확인] sparse reference task `basilisk_imu_sparse_ref_adcs_v0`의 상태는 `x=[sigma(3), omega(3), bias(3)]`, 차원 9이고 measurement는 IMU 6차원 + sparse sigma reference 3차원으로 `y_dim=9`이다 (`bench/tasks/generator/basilisk_imu_adcs.py:1272-1279`, `bench/tasks/generator/basilisk_imu_adcs.py:1490-1705`).
- [확인] quaternion은 filter state로 쓰이지 않는다. quaternion은 ADCS attitude metric 계산에서 MRP를 변환하는 용도로 사용된다 (`bench/metrics/adcs_event.py:65-96`).

## gyro 데이터의 역할: measurement인가, propagation/process 입력인가

- [확인] KalmanNet_TSP wrapper에서 `y`는 third_party KalmanNet의 observation 입력이다. third_party KalmanNet은 `self.m1y = self.h(self.m1x_prior)`로 predicted observation을 만들고 innovation `dy = y - self.m1y`를 사용한다 (`third_party/KalmanNet_TSP/KNet/KalmanNet_nn.py:143-149`, `third_party/KalmanNet_TSP/KNet/KalmanNet_nn.py:175-189`).
- [확인] Split-KalmanNet wrapper도 `y`를 observation으로 넣는다. third_party Split filter는 `x_predict=f(x_last)`, `y_predict=g(x_predict)`, `resid = observation - y_predict`, `gain = ...`, posterior update 순서를 사용한다 (`third_party/Split_KalmanNet/GSSFiltering/filtering.py:214-260`).
- [확인] IMU generator의 assumed model은 gyro rows를 `omega` measurement로, delta-angle rows를 `dt*omega` measurement로 H에 매핑한다. 따라서 현재 KalmanNet/Split/KF 계열에서 gyro는 propagation input이 아니라 measurement `y`다 (`bench/tasks/generator/basilisk_imu_adcs.py:219-239`, `bench/tasks/generator/basilisk_imu_adcs.py:199-216`).
- [확인] bias task에서도 measurement model은 `y=[omega+bias, dt*(omega+bias)]` 형태이며 bias는 상태에 포함된다 (`bench/tasks/generator/basilisk_imu_adcs.py:427-454`, `bench/tasks/generator/basilisk_imu_adcs.py:936-1048`).
- [확인] sparse-ref task도 IMU part는 `omega+bias` 및 `dt*(omega+bias)`이고, reference part는 `sigma` identity row로 들어간다 (`bench/tasks/generator/basilisk_imu_adcs.py:490-513`, `bench/tasks/generator/basilisk_imu_adcs.py:1407-1444`).
- [확인] `basilisk_mrp_ekf`는 full-state measurement `y=[sigma, omega]`를 가정하며 innovation은 `y_t - x_pred`다. gyro-only propagation input 형태가 아니다 (`bench/models/basilisk_mrp_ekf.py:666-699`, `bench/models/basilisk_mrp_ekf.py:567-595`).
- [추정] 실제 ADCS 관점에서 gyro를 propagation input으로 쓰는 error-state EKF 구조와는 다르다. 현재 구현은 neural/KF 모두 gyro를 observation channel로 모델링한다는 점이 가장 큰 정식화 차이다 (`bench/tasks/generator/basilisk_imu_adcs.py:219-239`, `third_party/Split_KalmanNet/GSSFiltering/filtering.py:214-260`).

## measurement model h(x)와 센서 목록

- [확인] full-state `basilisk_adcs_v0`의 measurement model은 `h(x)=x`, 즉 `sigma`와 `omega`를 직접 관측하는 6차원 identity observation이다. config도 `observation: full_state_adcs`, `measurement_model: direct_sigma_omega`, `H: identity_6`라고 명시한다 (`bench/configs/gpu_basilisk_adcs_with_ekf.yaml:11-41`, `bench/tasks/generator/basilisk_adcs.py:453-622`).
- [확인] `basilisk_mrp_ekf`의 meta도 `h(x)=x`, `H=I_6`을 명시한다 (`bench/models/basilisk_mrp_ekf.py:567-595`).
- [확인] basic IMU task의 measurement channel은 `gyro`, `accel`, `delta_theta`, `delta_velocity` 조합이며 mode는 `gyro_only`, `gyro_accel`, `gyro_delta_angle`, `full_imu`다 (`bench/tasks/generator/basilisk_imu_adcs.py:23-41`, `bench/tasks/generator/basilisk_imu_adcs.py:199-216`).
- [확인] bias/sparse-ref config에서 사용한 mode는 `gyro_delta_angle`이고, measurement는 gyro angular velocity와 delta angle 중심이다 (`bench/configs/gpu_basilisk_imu_bias_pilot.yaml:1-260`, `bench/configs/gpu_basilisk_imu_sparse_ref_pilot.yaml:1-320`).
- [확인] sparse-ref task는 IMU measurement 외에 sparse attitude reference `sparse_sigma_ref ~= sigma_BN when ref_mask=1`를 y 뒤쪽 3차원에 붙인다 (`bench/tasks/generator/basilisk_imu_adcs.py:1407-1444`, `bench/tasks/generator/basilisk_imu_adcs.py:1490-1705`).
- [확인] magnetometer, sun sensor, star tracker라는 명시적 센서 구현은 검색되지 않았다. exact pattern 검색 `rg -n "magnetometer|\bmag\b|sun_sensor|sun sensor|star_tracker|star tracker" ...`는 출력 없이 종료했다 (명령 136).
- [추정] sparse sigma reference가 연구 의도상 star tracker에 해당할 가능성은 있지만, 코드 식별자와 metadata에는 `star_tracker`가 아니라 `sparse_sigma_ref`/`ref_mask`로만 표현되어 있어 동일하다고 단정할 수 없다 (`bench/tasks/generator/basilisk_imu_adcs.py:457-513`, `bench/tasks/generator/basilisk_imu_adcs.py:1490-1705`).

## 센서 sampling rate / multirate 처리

- [확인] Basilisk full-state generator는 sequence `dt` 단일 주기를 사용한다. config 기본 `dt=0.1`이고 trajectory와 measurement가 같은 `T` 길이로 생성된다 (`bench/configs/gpu_basilisk_adcs_with_ekf.yaml:11-41`, `bench/tasks/generator/basilisk_adcs.py:453-622`).
- [확인] IMU generator는 clean/measured IMU recorder 모두 같은 sampling period로 구성한다. state와 y가 같은 sequence length로 저장된다 (`bench/tasks/generator/basilisk_imu_adcs.py:527-599`).
- [확인] sparse-ref는 실제로 별도 rate stream을 만들지 않고, 같은 `T` timeline에 reference row를 두되 `ref_mask`로 update timestep만 유효하게 표시한다 (`bench/tasks/generator/basilisk_imu_adcs.py:457-487`, `bench/tasks/generator/basilisk_imu_adcs.py:1407-1444`, `bench/tasks/generator/basilisk_imu_adcs.py:1490-1705`).
- [추정] 따라서 현재 ADCS 구현은 true multirate sensor fusion이라기보다 single-rate tensor + optional mask 방식이다.

## quaternion / MRP 처리 방식

- [확인] filter state는 quaternion이 아니라 MRP `sigma`다. full-state, IMU, EKF 모두 `sigma_BN`과 `omega_BN_B`를 사용한다 (`bench/tasks/generator/basilisk_adcs.py:408-449`, `bench/tasks/generator/basilisk_imu_adcs.py:527-599`, `bench/models/basilisk_mrp_ekf.py:188-195`).
- [확인] MRP shadow set 처리가 있다. generator와 EKF dynamics/propagation에서 `_shadow_mrp` 또는 shadow 관련 처리를 수행한다 (`bench/tasks/generator/basilisk_adcs.py:408-449`, `bench/tasks/generator/basilisk_imu_adcs.py:527-599`, `bench/models/basilisk_mrp_ekf.py:611-626`).
- [확인] `basilisk_mrp_ekf`는 MRP dynamics를 additive state로 propagate/update하고, update 후 shadow set 처리 및 covariance symmetry 처리를 한다 (`bench/models/basilisk_mrp_ekf.py:599-626`, `bench/models/basilisk_mrp_ekf.py:666-699`).
- [확인] quaternion sign ambiguity는 metric 변환에서 `abs(dot)` 기반 shortest geodesic으로 처리된다 (`bench/metrics/adcs_event.py:65-96`).
- [확인] quaternion normalization이 filter 내부에 있는 것은 확인되지 않았다. quaternion이 state가 아니기 때문에 정규화 대상은 MRP shadow mapping이며, quaternion은 metric 계산용 변환 결과다 (`bench/metrics/adcs_event.py:65-96`).
- [확인] error-state/multiplicative quaternion EKF 구조는 구현되어 있지 않다. 현재 EKF는 6D MRP+omega additive state EKF다 (`bench/models/basilisk_mrp_ekf.py:188-195`, `bench/models/basilisk_mrp_ekf.py:599-699`).

## Jacobian(F, H) 계산 방식

- [확인] KalmanNet_TSP/Split-KalmanNet용 linear system wrapper는 `F`와 `H`를 그대로 반환하는 해석적 linear Jacobian 방식을 사용한다 (`bench/models/split_knet.py:36-69`, `bench/models/kalmannet_tsp.py:315-410`).
- [확인] Split third_party filter는 `Jacobian_g(x_predict)`를 호출하지만 bench `_LinearSystemModel`에서는 constant `H`가 반환된다 (`third_party/Split_KalmanNet/GSSFiltering/filtering.py:214-260`, `bench/models/split_knet.py:36-69`).
- [확인] `basilisk_mrp_ekf`는 nonlinear dynamics F Jacobian을 central finite difference로 계산하고, H는 identity로 둔다 (`bench/models/basilisk_mrp_ekf.py:628-647`, `bench/models/basilisk_mrp_ekf.py:567-595`).
- [확인] third_party KalmanNet_TSP EKF는 `getJacobian` 기반 Extended Kalman Filter 구현을 포함하지만 bench ADCS EKF adapter는 이 파일을 직접 등록 모델로 사용하지 않는다 (`third_party/KalmanNet_TSP/Filters/EKF.py:1-70`, `bench/models/registry.py:26-47`).
- [확인] ADCS filter 코드에서 autograd Jacobian 사용은 확인되지 않았다. 검색 범위에서 ADCS EKF는 finite difference, KNet/Split은 linear F/H 경로였다 (명령 30, 139).

## Q, R 정의와 시간 변화

- [확인] full-state ADCS config는 `Q.q2=1e-8`, `R.r2=1e-4`를 사용한다 (`bench/configs/gpu_basilisk_adcs_with_ekf.yaml:31-40`).
- [확인] full-state generator의 small-angle assumed model은 `F[0:3,3:6]=0.25*dt*I`, `H=I_6`, `Q=q2*I_6`, `R=r2*I_6`로 구성된다 (`bench/tasks/generator/basilisk_adcs.py:76-81`, `bench/tasks/generator/basilisk_adcs.py:453-622`).
- [확인] IMU assumed model은 H와 R을 measurement mode와 gyro/accel/bias std에 따라 구성한다. gyro rows는 omega, delta_theta rows는 `dt*omega`에 매핑된다 (`bench/tasks/generator/basilisk_imu_adcs.py:219-239`).
- [확인] bias model의 H/R/Q는 bias state와 random walk를 포함한다. H는 `omega + bias`와 `dt*(omega+bias)`를 관측하도록 구성된다 (`bench/tasks/generator/basilisk_imu_adcs.py:427-454`).
- [확인] sparse-ref model의 H/R은 IMU 6차원 + sigma reference 3차원을 포함한다 (`bench/tasks/generator/basilisk_imu_adcs.py:490-513`).
- [확인] `basilisk_mrp_ekf` setup은 `Q`, `R`, `P0`, `dt`, `inertia`, `torque`를 system_info/config/meta에서 읽어 device tensor로 보관한다 (`bench/models/basilisk_mrp_ekf.py:275-340`).
- [확인] `basilisk_mrp_ekf` rollout에서는 `Q`, `R`이 timestep마다 같은 행렬로 사용된다 (`bench/models/basilisk_mrp_ekf.py:666-699`).
- [추정] ADCS Basilisk 계열에서는 Q/R이 fixed assumed covariance로 쓰인다. event나 sparse mask는 measurement contents/extras를 바꾸지만 filter 내부에서 time-varying R로 바꿔 쓰는 경로는 확인하지 못했다 (`bench/tasks/generator/basilisk_imu_adcs.py:285-397`, `bench/models/basilisk_mrp_ekf.py:666-699`).

# 4. 데이터 파이프라인

## trajectory 생성 방식

- [확인] full-state Basilisk generator는 `SimulationBaseClass`, `spacecraft.Spacecraft`, optional `extForceTorque`를 사용해 rigid-body spacecraft trajectory를 생성한다. 상태로 `sigma_BN`, `omega_BN_B`를 기록한다 (`bench/tasks/generator/basilisk_adcs.py:398-450`).
- [확인] IMU generator도 Basilisk spacecraft와 `imuSensor.ImuSensor`를 사용하며 clean/measured IMU sensor recorder를 동시에 둔다 (`bench/tasks/generator/basilisk_imu_adcs.py:44-58`, `bench/tasks/generator/basilisk_imu_adcs.py:527-599`).
- [확인] full-state config의 기본 inertia는 `[10.0, 8.0, 6.0]`, torque는 `[0.0, 0.0, 0.0]`, `dt=0.1`, `T=100`이다 (`bench/configs/gpu_basilisk_adcs_with_ekf.yaml:11-31`).
- [확인] IMU pilot config도 `dt=0.1`, `T=100`, `n_train=256`, `n_val=64`, `n_test=64`를 사용한다 (`bench/configs/gpu_basilisk_imu_pilot_pretrained_enhancer.yaml:1-36`).
- [확인] measurement event disturbance는 truth dynamics를 바꾸지 않고 event window에서 measured IMU y에 bias/noise 등을 추가하는 방식이다 (`bench/tasks/generator/basilisk_imu_adcs.py:271-397`, `bench/tasks/generator/basilisk_imu_adcs.py:708-727`).
- [불명] 실제 위성 maneuver torque 시나리오의 물리적 출처나 목적은 config/코드만으로 확인되지 않았다. 기본 ADCS configs에서는 zero torque가 확인된다 (`bench/configs/gpu_basilisk_adcs_with_ekf.yaml:24-31`).

## 구현된 잡음/오차 모델

- [확인] full-state `basilisk_adcs.py`에는 Gaussian white noise, constant bias, random walk/drift, scale, axis misalignment, outlier, vibration corruption이 구현되어 있다 (`bench/tasks/generator/basilisk_adcs.py:151-359`, 명령 138).
- [확인] IMU sensor configuration에는 gyro/accel noise covariance, bias, walk/error bounds, saturation bounds, LSB quantization 설정이 있다 (`bench/tasks/generator/basilisk_imu_adcs.py:143-189`).
- [확인] IMU bias task는 bias random walk와 measurement noise를 별도 component로 생성하고 extras에 `bias_component_seq`, `noise_component_seq`, `imu_error_seq` 등을 저장한다 (`bench/tasks/generator/basilisk_imu_adcs.py:1029-1048`, `bench/tasks/generator/basilisk_imu_adcs.py:1231-1255`).
- [확인] sparse-ref task는 reference update period, reference noise, dropout/mask를 구현한다 (`bench/tasks/generator/basilisk_imu_adcs.py:457-487`, `bench/tasks/generator/basilisk_imu_adcs.py:1407-1444`).
- [확인] temperature-dependent sensor model은 검색되지 않았다. `temperature|temp` 검색은 `tempfile` 등 비센서 용례 외에는 관련 구현을 찾지 못했다 (명령 137).
- [확인] magnetometer/sun sensor/star tracker 명시 구현은 검색되지 않았다 (명령 136).

## 잡음 파라미터 값과 출처

- [확인] full-state ADCS baseline config: `q2=1e-8`, `r2=1e-4`, sensor noise sweep `[-10, 0, 10, 20, 30] dB`가 설정되어 있다 (`bench/configs/gpu_basilisk_adcs_with_ekf.yaml:31-41`, `bench/configs/gpu_basilisk_adcs_with_ekf.yaml:92-124`).
- [확인] IMU pretrained enhancer pilot config에는 profiles `clean`, `noisy`, `biased`, `low_cost`가 있고 gyro/accel noise std, bias std, walk bounds, saturation, LSB 값을 profile별로 둔다 (`bench/configs/gpu_basilisk_imu_pilot_pretrained_enhancer.yaml:37-116`).
- [확인] IMU bias pilot config에는 `bias_init_std`가 `0`, `5e-4`, `2e-3`, `5e-3`, `bias_rw_std`가 `0`, `1e-5`, `5e-5`, `1e-4`인 profiles가 있다 (`bench/configs/gpu_basilisk_imu_bias_pilot.yaml:1-260`).
- [확인] sparse-ref pilot config에는 reference noise/dropout/update period 및 IMU bias/noise profiles가 있다 (`bench/configs/gpu_basilisk_imu_sparse_ref_pilot.yaml:1-320`).
- [불명] 위 잡음/오차 파라미터가 특정 IMU datasheet, 실측, 또는 임의 설정 중 어디에서 유래했는지는 코드와 configs에서 확인되지 않았다. README/docs 검색에서도 Basilisk 설치와 실행 언급은 있었지만 datasheet 근거는 발견하지 못했다 (명령 98, 134, 138).

## 실측 데이터

- [확인] benchmark registry에는 `nclt`, `uzh_fpv`, `adcs_replay` family가 존재한다 (`bench/tasks/bench_generated.py:557-690`).
- [추정] NCLT/UZH는 실제 데이터 계열로 보이지만, 이번 감사에서는 ADCS/KalmanNet 위성 자세 추정과 직접 관련된 실측 ADCS sensor dataset 사용을 확인하지 못했다 (`bench/tasks/bench_generated.py:557-690`, 명령 135).
- [불명] 실제 위성 센서 데이터가 어디에 저장되어 있고 ADCS generator에서 쓰이는지는 확인되지 않았다. `basilisk_adcs`와 `basilisk_imu_adcs`는 simulation generator 경로로 확인된다 (`bench/tasks/generator/basilisk_adcs.py:398-622`, `bench/tasks/generator/basilisk_imu_adcs.py:527-599`).

## train/val/test 분리와 데이터 누수 위험

- [확인] generator contract는 동일 입력 config와 seed에 대해 deterministic output을 요구하고, x/y shape을 NTD로 정의한다 (`bench/tasks/generator/contract.py:6-24`, `bench/tasks/generator/contract.py:101-123`).
- [확인] generated task loader는 cache가 없으면 generator output을 만들고, pre-split payload가 없을 경우 deterministic permutation으로 train/val/test split을 만든다 (`bench/tasks/bench_generated.py:873-1018`).
- [확인] runner는 same suite/task/scenario/seed cache path를 모델 간 공유하고, 동일 split npz를 train/val/test loader로 읽는다 (`bench/runners/run_suite.py:1544-1580`, `bench/runners/run_suite.py:1872-1947`).
- [추정] synthetic ADCS generator는 각 trajectory를 N 차원 sample로 독립 생성한 뒤 split하므로 같은 trajectory의 overlapping window가 train/test에 동시에 들어갈 가능성은 낮아 보인다. 근거는 generator output이 N개의 sequence로 생성되고 split permutation이 sample index 기준으로 일어난다는 점이다 (`bench/tasks/generator/contract.py:17-20`, `bench/tasks/bench_generated.py:873-1018`).
- [불명] NCLT/UZH 같은 real dataset/window loader에서 같은 physical trajectory의 overlapping window가 train/test에 동시에 들어갈 가능성은 이번 감사에서 확정하지 못했다.
- [확인] DataLoader shuffle은 deterministic torch generator seed를 받는다 (`bench/runners/run_suite.py:1287-1335`).

## normalization 통계

- [확인] train set에서 mean/std를 계산해 input/output normalization에 쓰는 scaler 경로는 검색되지 않았다. `normalizer`, `train_mean`, `train_std`, `zscore` 등 검색 결과에서 ADCS/KNet 학습용 normalization 구현을 확인하지 못했다 (명령 132).
- [추정] 현재 ADCS 계열 실험은 raw state/measurement scale 그대로 MSE/Kalman update에 들어가는 구조다.

# 5. 학습 구성

## loss 함수

- [확인] `kalmannet_tsp` train loss는 `MSELoss(pred, x)` 형태의 state component MSE다. quaternion geodesic loss가 아니다 (`bench/models/kalmannet_tsp.py:430-573`).
- [확인] `split_knet` train loss는 state estimation MSE다. 구현상 `state_estimation_loss`가 있고 T>1인 경우 첫 timestep을 skip하는 경로가 있다 (`bench/models/split_knet.py:609-904`).
- [확인] `adaptive_knet` supervised train loss는 `MSELoss(pred, x)`다 (`bench/models/adaptive_knet.py:500-649`).
- [확인] `adaptive_knet` adapt loss는 `MSE(H x_hat_step, y_step)` 형태의 unsupervised observation reconstruction MSE다 (`bench/models/adaptive_knet.py:743-876`).
- [확인] `maml_knet` validation/eval loss는 prediction과 target state 사이 MSE다 (`bench/models/maml_knet.py:1243-1273`).
- [확인] `spike_split_knet`와 `g1_snn_split_knet`는 Split-KNet branch replacement adapter라서 기본 학습 loss는 Split-KNet 경로를 상속한다 (`bench/models/spike_split_knet.py:347-388`, `bench/models/g1_snn_split_knet.py:121-159`, `bench/models/split_knet.py:609-904`).
- [확인] `spike_ra_knet`는 event flag가 있으면 event-weighted state MSE를 적용한다. weight는 `1 + event_loss_lambda * event_flag` 형태다 (`bench/models/spike_ra_knet.py:520-564`).
- [확인] `me_split_knet_v0` enhancer pretrain loss는 measurement-space loss다. `denoise_loss = MSE(y_enh, y_clean)`, optional correction loss `MSE(applied_delta, -imu_error)`, regularization/identity loss가 더해진다 (`bench/models/me_split_knet.py:268-401`).
- [확인] `basilisk_mrp_ekf`와 `mb_kf` train은 no-op이므로 optimizer 기반 학습 loss가 없다 (`bench/models/basilisk_mrp_ekf.py:364-402`, `bench/models/mb_kf.py:373-417`).

## optimizer / learning rate / sequence / batch

- [확인] KNet/Split/Adaptive/ME/SpikeRA 학습 코드는 Adam optimizer를 사용한다 (`bench/models/kalmannet_tsp.py:430-573`, `bench/models/split_knet.py:609-904`, `bench/models/adaptive_knet.py:500-649`, `bench/models/me_split_knet.py:344-401`, `bench/models/spike_ra_knet.py:566-675`).
- [확인] full-state ADCS+EKF config는 KalmanNet_TSP lr `1e-4`, Split-KNet lr `1e-4`, train batch `8`, eval batch `16`, `train_max_updates=500`을 설정한다 (`bench/configs/gpu_basilisk_adcs_with_ekf.yaml:43-124`).
- [확인] IMU pretrained enhancer pilot config는 Split/ME lr `1e-4`, ME enhancer lr `1e-4`, `enhancer_pretrain_updates=100`, `train_max_updates=500`, batch size `8`을 사용한다 (`bench/configs/gpu_basilisk_imu_pilot_pretrained_enhancer.yaml:117-240`).
- [확인] sparse-ref sanity500 config는 lower lr/gradient clipping 계열로 보이며, `train_max_updates=500`, `gradient_clip_norm=1.0`이 검색되었다 (명령 65, `reports/basilisk_imu_sparse_ref_sanity_500_analysis.md:31-38`).
- [확인] sequence length는 config의 `sequence_length`/`T`가 그대로 full sequence rollout에 사용된다. 별도 truncated BPTT 길이는 확인되지 않았다 (`bench/configs/gpu_basilisk_adcs_with_ekf.yaml:11-17`, `bench/runners/run_suite.py:1872-1947`, `bench/models/split_knet.py:609-904`).
- [추정] BPTT는 full sequence 기준이다. `train_max_updates`는 update count를 제한하지만 sequence 내부 truncation은 확인되지 않았다.

## config 관리 방식

- [확인] 주 실행 config는 YAML suite 파일이다. runner는 `--suite-yaml` CLI 인자를 받는다 (명령 144, `bench/runners/run_suite.py:2686`).
- [확인] task/model/runner parameter는 YAML과 adapter default가 섞여 있다. 예를 들어 ME target default는 code에 있고 config에서 override 가능하다 (`bench/models/me_split_knet.py:42-70`, `bench/configs/gpu_basilisk_imu_pilot_pretrained_enhancer.yaml:117-240`).
- [확인] 일부 물리/잡음 default는 generator 코드에 hardcoded fallback으로 존재한다. 예: IMU bias default profile, sparse reference default config (`bench/tasks/generator/basilisk_imu_adcs.py:400-487`).

# 6. Enhancement 실험의 실체

## handoff 문서와 "개선 없음" claim

- [불명: handoff 문서 없음] 요청에서 언급한 "handoff 문서상 measurement enhancement 실험에서 뚜렷한 개선이 없었다"는 문장을 담은 ADCS/enhancement handoff 문서는 저장소에서 발견하지 못했다. 발견된 `HANDOFF_PACKET.v3`는 Adaptive-KNet S3 handoff이며 enhancement claim을 담은 문서가 아니다 (명령 8-10, `HANDOFF_PACKET.v3:1-13`, `HANDOFF_PACKET.v3:104-107`).
- [확인] 다만 `reports/`의 ME 결과 파일들은 "뚜렷한 개선 없음 / marginal / inconclusive / negative" 판정을 뒷받침한다 (`reports/me_split_knet_full_analysis.md:3-19`, `reports/me_split_structured_corruption_full_analysis.md:3-15`, `reports/basilisk_imu_pretrained_enhancer_gpu_pilot_summary.csv:2-9`, `reports/basilisk_imu_bias_gpu_pilot_summary.csv:2-5`).

## enhancement 네트워크 구조, 입력, 출력

- [확인] `MeasurementEnhancer`는 causal 1D convolution residual network다. 입력 shape은 `[B,T,y_dim]`, 내부에서 `[B,y_dim,T]`로 transpose하여 Conv1d stack을 통과하고 residual delta를 y에 더한다 (`bench/models/measurement_enhancer.py:11-104`).
- [확인] final projection layer는 zero initialization된다. 초기 enhancer는 identity에 가깝게 시작하도록 설계된 것으로 보인다 (`bench/models/measurement_enhancer.py:11-104`).
- [확인] 출력은 enhanced measurement `y_enh = y + delta`이며 optional delta clipping/safety가 있다 (`bench/models/measurement_enhancer.py:11-104`, `bench/models/me_split_knet.py:412-440`).
- [확인] enhancer diagnostics는 delta norm, raw/enhanced y norm, innovation norm, IMU MSE reduction, correction alignment 등을 계산한다 (`bench/models/measurement_enhancer.py:131-199`).

## enhancement 학습 loss의 정확한 실체

- [확인] ME pretrain target은 config/adapter에서 `x`, `imu_clean_y_seq`, `measurement_clean_y_seq` 중 하나를 선택한다. IMU 계열에서는 clean measurement extras를 target으로 쓰는 경로가 있다 (`bench/models/me_split_knet.py:42-70`, `bench/models/me_split_knet.py:268-342`).
- [확인] IMU pretrained enhancer pilot config의 target은 `imu_clean_y_seq`다 (`bench/configs/gpu_basilisk_imu_pilot_pretrained_enhancer.yaml:117-240`).
- [확인] IMU bias pilot config의 target도 `imu_clean_y_seq`다 (`bench/configs/gpu_basilisk_imu_bias_pilot.yaml:1-260`).
- [확인] sparse-ref pilot/sanity config의 target은 `measurement_clean_y_seq`다 (`bench/configs/gpu_basilisk_imu_sparse_ref_pilot.yaml:1-320`, 명령 65).
- [확인] loss 본체는 measurement denoising MSE다: enhanced y와 clean y 사이 MSE, optional correction delta와 `-imu_error` 사이 MSE, delta/smooth/identity regularization을 더한다 (`bench/models/me_split_knet.py:344-401`).
- [확인] filter를 통과한 최종 state/attitude error를 enhancer pretrain loss로 쓰는 end-to-end 경로는 없다. `measurement_extra_loss()`도 현재 zero를 반환한다 (`bench/models/me_split_knet.py:168-178`, `bench/models/me_split_knet.py:223-254`).
- [확인] full-state direct observation task에서 target `x`를 쓴다면 measurement dimension과 state dimension이 같은 경우라 `MSE(y_enh, x)`가 사실상 noisy full-state measurement denoising이 되지만, quaternion/geodesic attitude loss는 아니다 (`bench/models/me_split_knet.py:268-342`, `bench/configs/gpu_basilisk_me_split_full.yaml`, 명령 65).

## 별도 학습인가, end-to-end인가

- [확인] ME-Split wrapper 문서와 meta는 two-stage 구조를 명시한다. Stage A enhancer pretrain, Stage B frozen enhancer + unchanged Split-KNet 학습이다 (`bench/models/me_split_knet.py:13-20`, `bench/models/me_split_knet.py:71-125`, `bench/models/me_split_knet.py:223-254`).
- [확인] `stage_b_freeze_enhancer`와 `joint_finetune: False`가 adapter meta에 기록된다 (`bench/models/me_split_knet.py:223-254`).
- [확인] enhancer와 후단 filter는 end-to-end로 함께 최종 state loss를 역전파하지 않는다 (`bench/models/me_split_knet.py:168-178`, `bench/models/me_split_knet.py:223-254`, `bench/models/me_split_knet.py:344-401`).

## 실험 데이터에 포함된 오차 유형

- [확인] full-state ME full 결과는 `sensor_noise_scale_db` sweep을 기준으로 한 direct full-state Gaussian/noise scale 실험이다 (`reports/me_split_knet_full_summary.csv:2-6`, `bench/configs/gpu_basilisk_me_split_full.yaml`, 명령 65).
- [확인] structured corruption full 결과는 clean/mild/moderate/severe severity별 Gaussian+bias/random-walk/scale/misalignment/outlier/vibration 계열 corruption이 포함된 실험이다 (`reports/me_split_structured_corruption_full_summary.csv:2-13`, `bench/tasks/generator/basilisk_adcs.py:151-359`).
- [확인] IMU pretrained enhancer pilot는 IMU noise/bias/walk/saturation/LSB profiles `clean/noisy/biased/low_cost`를 포함한다 (`bench/configs/gpu_basilisk_imu_pilot_pretrained_enhancer.yaml:37-116`, `reports/basilisk_imu_pretrained_enhancer_gpu_pilot_summary.csv:2-9`).
- [확인] IMU bias pilot는 explicit gyro bias state/random walk profiles를 포함한다 (`bench/configs/gpu_basilisk_imu_bias_pilot.yaml:1-260`, `reports/basilisk_imu_bias_gpu_pilot_summary.csv:2-5`).
- [확인] sparse-ref sanity는 reference availability/noise/update period 변형을 포함한다 (`reports/basilisk_imu_sparse_ref_sanity_500_summary.csv:2-5`, `reports/basilisk_imu_sparse_ref_sanity_500_analysis.md:31-38`).

## "개선 없음" 근거 수치

- [확인] full-state ME full summary의 improvement dB는 `-10 dB: +0.0785`, `0 dB: -0.7114`, `10 dB: +0.0354`, `20 dB: -0.0342`, `30 dB: +0.0139`다. analysis는 `decision_category: B`와 marginal/inconclusive를 명시한다 (`reports/me_split_knet_full_summary.csv:2-6`, `reports/me_split_knet_full_analysis.md:3-19`).
- [확인] structured corruption full summary의 improvement dB는 `clean_gaussian: -0.4205`, `mild_structured: +0.3290`, `moderate_structured: -0.0155`, `severe_structured: -0.0406`이다. analysis는 `ME safe but inconclusive`로 적는다 (`reports/me_split_structured_corruption_full_summary.csv:2-13`, `reports/me_split_structured_corruption_full_analysis.md:3-15`).
- [확인] IMU pretrained enhancer GPU pilot는 모든 severity에서 negative improvement다: `clean -0.0516`, `noisy -0.0358`, `biased -0.0348`, `low_cost -0.0219` dB (`reports/basilisk_imu_pretrained_enhancer_gpu_pilot_summary.csv:2-9`).
- [확인] IMU bias GPU pilot도 모두 negative다: `clean -0.0660`, `mild_bias -0.0247`, `moderate_bias -0.0102`, `low_cost_bias -0.0960` dB (`reports/basilisk_imu_bias_gpu_pilot_summary.csv:2-5`).
- [확인] original sparse-ref GPU pilot는 summary CSV가 비어 있고 failure summary/runner summary에서 모든 split/me runs가 `train_nan`으로 실패했다 (`wc -c ...`, `reports/basilisk_imu_sparse_ref_gpu_pilot_failure_summary.csv:2-9`, `reports/summary_gpu_basilisk_imu_sparse_ref_pilot.csv:2-9`).
- [확인] sparse-ref sanity500 summary는 `ref_disabled +0.0008`, `sparse_ref_period_20 +0.0275`, `sparse_ref_period_5 -0.0033`, `dense_ref -0.0889` dB다. analysis는 reference-rich case에서 negative/diagnostic이라는 해석을 남긴다 (`reports/basilisk_imu_sparse_ref_sanity_500_summary.csv:2-5`, `reports/basilisk_imu_sparse_ref_sanity_500_analysis.md:31-38`).
- [확인] ME ablation notes는 v0가 stable하지만 seed-0 benefit이 없고 0/20/30 dB에서 negative라고 적는다. summary도 `0 dB -0.3133`, `20 dB -0.0711`, `30 dB -0.0020` improvement dB를 기록한다 (`reports/me_split_knet_ablation_notes.md:8-13`, `reports/me_split_knet_ablation_summary.csv:2-7`).

# 7. 평가 및 비교

## 구현된 평가 지표

- [확인] core metrics는 per-step MSE, scalar MSE, RMSE, MSE dB를 구현한다 (`bench/metrics/core.py:14-46`).
- [확인] Gaussian NLL metric은 covariance가 있을 때 optional로 계산하도록 구현되어 있다 (`bench/metrics/core.py:124-162`, `bench/runners/run_suite.py:2537-2572`).
- [확인] ADCS event metrics는 MRP를 quaternion으로 변환한 뒤 attitude error degree, attitude RMSE, angular velocity RMSE, event/non-event summary를 계산한다 (`bench/metrics/adcs_event.py:65-151`).
- [확인] runner는 `mse_t_mean`, `mse`, `rmse`, `mse_db`, optional ADCS event metrics를 `metrics.json`/summary에 쓴다 (`bench/runners/run_suite.py:2257-2300`, `bench/runners/run_suite.py:2537-2572`).
- [확인] NIS/NEES 구현은 검색되지 않았다. `rg -n "NIS|NEES|..."`에서 NIS/NEES metric 구현 파일을 찾지 못했다 (명령 89).
- [확인] bias RMSE 전용 metric은 검색 결과에서 별도 core metric으로 확인되지 않았다. bias 상태가 x에 들어간 경우 전체 state MSE에는 포함될 수 있지만, bias-only RMSE summary는 확인하지 못했다 (명령 28, 89).

## 모델 간 비교의 동일 데이터 realization 보장

- [확인] runner는 scenario canonical basis로 `scenario_id`를 만들고, suite/task/scenario/seed 기반 cache path를 구성한다 (`bench/runners/run_suite.py:1544-1550`).
- [확인] 같은 scenario run 안에서 model id별 run_dir는 달라지지만 train/val/test split npz 경로는 같은 task cache에서 로드된다 (`bench/runners/run_suite.py:1561-1580`, `bench/runners/run_suite.py:1872-1947`).
- [확인] split generation은 deterministic stable seed와 permutation을 사용한다 (`bench/tasks/bench_generated.py:873-1018`).
- [추정] 동일 suite/scenario/seed 내 model 비교는 같은 generated data realization을 쓰도록 설계되어 있다. 단, 이미 존재하는 cache가 dirty worktree/다른 코드 버전에서 생성되었는지 여부는 run artifact metadata를 추가로 대조해야 한다.

## 비교 결과 저장 위치

- [확인] ME full 결과는 `reports/me_split_knet_full_summary.csv`와 `reports/me_split_knet_full_analysis.md`에 있다 (`reports/me_split_knet_full_summary.csv:2-6`, `reports/me_split_knet_full_analysis.md:3-19`).
- [확인] structured corruption ME 결과는 `reports/me_split_structured_corruption_full_summary.csv`와 `reports/me_split_structured_corruption_full_analysis.md`에 있다 (`reports/me_split_structured_corruption_full_summary.csv:2-13`, `reports/me_split_structured_corruption_full_analysis.md:3-15`).
- [확인] IMU pretrained enhancer 결과는 `reports/basilisk_imu_pretrained_enhancer_gpu_pilot_summary.csv`에 있다 (`reports/basilisk_imu_pretrained_enhancer_gpu_pilot_summary.csv:2-9`).
- [확인] IMU bias 결과는 `reports/basilisk_imu_bias_gpu_pilot_summary.csv`에 있다 (`reports/basilisk_imu_bias_gpu_pilot_summary.csv:2-5`).
- [확인] sparse-ref original failure와 sanity500 결과는 `reports/basilisk_imu_sparse_ref_gpu_pilot_failure_summary.csv`, `reports/summary_gpu_basilisk_imu_sparse_ref_pilot.csv`, `reports/basilisk_imu_sparse_ref_sanity_500_summary.csv`, `reports/basilisk_imu_sparse_ref_sanity_500_analysis.md`에 있다 (`reports/basilisk_imu_sparse_ref_gpu_pilot_failure_summary.csv:2-9`, `reports/basilisk_imu_sparse_ref_sanity_500_summary.csv:2-5`).
- [확인] EKF 비교 결과는 `reports/basilisk_gpu_with_ekf_summary.csv`에 있다. 여기서 EKF는 sensor noise scale -10/0/10/20 dB에서 Split/KalmanNet보다 낮은 MSE dB를 보이나, 30 dB에서는 Split-KNet이 더 낮은 MSE dB를 보인다 (`reports/basilisk_gpu_with_ekf_summary.csv:2-16`).

# 8. 재현성

## 지금 당장 실행 가능한 것

- [확인] runner help 확인: `timeout 60 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m bench.runners.run_suite --help`는 정상 동작했다 (명령 144).
- [확인] model registry 확인: `env MPLCONFIGDIR=/tmp/matplotlib timeout 60 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -c "from bench.models.registry import list_model_ids; print(list_model_ids())"`는 정상 동작했고 registry model id 목록을 출력했다 (명령 152, `bench/models/registry.py:26-47`, `bench/models/registry.py:64-65`).
- [확인] 핵심 ADCS/ME imports 확인: `env MPLCONFIGDIR=/tmp/matplotlib timeout 60 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -c "from bench.models.basilisk_mrp_ekf import BasiliskMRPEKFAdapter; from bench.models.measurement_enhancer import MeasurementEnhancer; print('imports ok')"`는 정상 동작했다 (명령 153).
- [확인] Basilisk import 확인: `env MPLCONFIGDIR=/tmp/matplotlib timeout 60 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -c "from Basilisk.simulation import spacecraft, imuSensor; from Basilisk.utilities import SimulationBaseClass; print('Basilisk import ok')"`는 정상 동작했다 (명령 155).
- [확인] 단위 테스트: `env MPLCONFIGDIR=/tmp/matplotlib timeout 60 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m unittest tests.test_me_split_knet tests.test_basilisk_mrp_ekf tests.test_basilisk_imu_model_compat`는 `Ran 14 tests in 0.227s`, `OK`였다 (명령 154).

## 환경 정의

- [확인] `pyproject.toml`이 존재하고 Python `>=3.9`, dependencies `numpy`, `scipy`, `pandas`, `matplotlib`, `pyyaml`, `tqdm`, `torch`를 선언한다 (`pyproject.toml:10-22`).
- [확인] optional dev dependency로 `pytest`가 선언되어 있다 (`pyproject.toml:30-34`).
- [확인] `uv.lock`가 존재하고 torch/numpy/matplotlib/pyyaml 등 dependency pin을 포함한다 (명령 94, 96).
- [확인] `requirements.lock`는 16줄짜리 placeholder이며 실제 pip freeze snapshot으로 보기 어렵다. 파일 안에도 "initial placeholder"라고 적혀 있다 (`requirements.lock:1-16`, `wc -l requirements.lock`).
- [확인] `requirements.lock`에는 Basilisk entry가 없다. `rg -n "Basilisk|basilisk" requirements.lock -S`는 출력 없이 종료했다 (명령 167).
- [확인] 현재 명시 Python 환경에는 Basilisk가 설치되어 import 가능하지만, 이 의존성은 `pyproject.toml`/`requirements.lock`에서 재현 가능하게 선언되어 있지 않다 (명령 155, `pyproject.toml:13-22`, `requirements.lock:1-16`).

## 실행이 깨지는 경로

- [확인] 기본 `python` 명령은 pyenv shim 문제로 실패했다. `timeout 60 python -m bench.runners.run_suite --help`, `timeout 60 python -c ...`, `timeout 60 python -m pytest ...` 모두 실패했다 (명령 140-143).
- [확인] 명시 Python에서도 `python -m pytest ...`는 `No module named pytest`로 실패했다. `unittest`는 통과했다 (명령 147, 154).
- [확인] `timeout 60 MPLCONFIGDIR=/tmp/matplotlib ...` 형태는 shell env assignment가 아니라 실행 파일명으로 해석되어 실패했다. 올바른 형태는 `env MPLCONFIGDIR=/tmp/matplotlib timeout 60 ...`였다 (명령 149-151, 152-155).
- [확인] Matplotlib는 기본 config path `/home/dss-pc-05/.config/matplotlib`가 writable이 아니라 임시 cache 경고를 냈다. `MPLCONFIGDIR=/tmp/matplotlib`를 쓰면 import/test 명령은 정상 실행되었다 (명령 146, 152-155).

# 9. 발견된 문제와 위험

- 심각도: 치명 / 확신도: [확인] / 위치: `bench/tasks/generator/basilisk_imu_adcs.py:219-239`, `bench/tasks/generator/basilisk_imu_adcs.py:602-915`, `third_party/Split_KalmanNet/GSSFiltering/filtering.py:214-260`
  - IMU-only ADCS task에서 gyro/delta-angle은 propagation input이 아니라 measurement로 들어간다. 상태에는 attitude `sigma`가 포함되지만 basic IMU measurement에는 absolute attitude reference가 없다. 이 구조는 attitude drift correction 관점에서 근본적인 observability/정식화 위험이 있다. 다만 연구 의도가 "partial observation mismatch stress test"라면 버그가 아니라 실험 설계일 수 있다.

- 심각도: 치명 / 확신도: [확인] / 위치: `reports/basilisk_imu_sparse_ref_gpu_pilot_failure_summary.csv:2-9`, `reports/summary_gpu_basilisk_imu_sparse_ref_pilot.csv:2-9`
  - original sparse-ref GPU pilot 결과는 모든 split/me runs가 `train_nan`으로 실패했다. 후속 sanity500 결과가 따로 있으나, original pilot 결과를 성공 실험으로 인용하면 안 된다.

- 심각도: 주의 / 확신도: [확인] / 위치: `bench/models/me_split_knet.py:13-20`, `bench/models/me_split_knet.py:168-178`, `bench/models/me_split_knet.py:344-401`
  - ME-v0는 downstream attitude/state/filter loss로 end-to-end 학습되지 않는다. loss는 clean measurement denoising/correction MSE다. 따라서 "measurement enhancement가 자세 추정을 개선한다"는 claim은 결과 파일 수치로만 검증해야 하며, 학습 목적 자체가 attitude geodesic error를 직접 최적화하지 않는다.

- 심각도: 주의 / 확신도: [확인] / 위치: `reports/me_split_knet_full_analysis.md:3-19`, `reports/basilisk_imu_pretrained_enhancer_gpu_pilot_summary.csv:2-9`, `reports/basilisk_imu_bias_gpu_pilot_summary.csv:2-5`
  - ME 결과는 전반적으로 marginal/inconclusive/negative다. full-state 일부 severity에서 작은 양수 improvement가 있으나 크기가 작고 일관되지 않다. IMU pretrained/bias pilot는 모두 negative improvement다.

- 심각도: 주의 / 확신도: [확인] / 위치: `bench/models/split_knet.py:609-904`, `bench/models/split_knet.py:915-1018`, `bench/models/split_knet.py:1085-1161`
  - Split-KNet 학습/eval 초기화 정책이 결과에 영향을 줄 수 있다. train은 `train_init_from_gt` 기본값 경로가 있고, eval은 `eval_init_from_gt`가 기본 false인 경로가 확인된다. 첫 timestep 처리도 loss/eval에서 다르게 취급될 수 있어 모델 간 t=0 MSE 비교 해석에 주의가 필요하다.

- 심각도: 주의 / 확신도: [확인] / 위치: `pyproject.toml:13-22`, `requirements.lock:1-16`, 명령 155
  - Basilisk는 현재 환경에서 import 가능하지만 dependency manifest에는 없다. `requirements.lock`도 placeholder다. 새 환경 재현성은 불완전하다.

- 심각도: 주의 / 확신도: [확인] / 위치: 명령 140-147, 명령 152-155
  - 기본 `python` 경로는 실패하고 명시 pyenv Python만 동작했다. README/스크립트가 기본 `python`을 가정하면 재현 실패 가능성이 있다.

- 심각도: 주의 / 확신도: [확인] / 위치: `git status --short --branch`
  - worktree가 매우 dirty하다. 수정/삭제/untracked 파일이 많아 현재 결과가 commit `3cab581...`만으로 재현된다고 말할 수 없다.

- 심각도: 주의 / 확신도: [추정] / 위치: `bench/configs/gpu_basilisk_adcs_with_ekf.yaml:11-41`, `bench/tasks/generator/basilisk_adcs.py:453-622`, 명령 136
  - full-state ADCS task는 `sigma/omega` 직접 관측을 사용하므로 실제 센서 fusion 문제보다 쉬운 observation model일 수 있다. magnetometer/sun sensor/star tracker 구현은 확인되지 않았다.

- 심각도: 주의 / 확신도: [확인] / 위치: `bench/models/basilisk_mrp_ekf.py:268-271`, `bench/models/basilisk_mrp_ekf.py:567-595`
  - `basilisk_mrp_ekf`는 `x_dim=y_dim=6`이면 `h(x)=x`로 해석한다. 만약 IMU `gyro_delta_angle` y_dim=6 데이터를 잘못 연결하면 차원은 맞지만 의미가 다른 measurement를 full-state로 업데이트할 위험이 있다. 현재 확인한 EKF config는 full-state ADCS와 함께 사용한다 (`bench/configs/gpu_basilisk_adcs_with_ekf.yaml:43-90`).

- 심각도: 주의 / 확신도: [확인] / 위치: 명령 132
  - train-set 기반 normalization/scaler가 확인되지 않았다. severity나 센서 단위가 다른 실험에서 raw scale 차이가 loss와 optimization에 직접 영향을 준다.

- 심각도: 사소 / 확신도: [확인] / 위치: `README.md:29`, `requirements.lock:1-16`
  - README는 `requirements.lock`를 pip freeze snapshot처럼 설명하지만 실제 파일은 placeholder다.

- 심각도: 사소 / 확신도: [확인] / 위치: 명령 146, 명령 152-155
  - Matplotlib config dir warning이 발생한다. audit/test에는 `MPLCONFIGDIR=/tmp/matplotlib`로 회피 가능했지만 자동화 환경에서는 설정 필요하다.

- 심각도: 주의 / 확신도: [확인] / 위치: `HANDOFF_PACKET.v3:1-13`, `HANDOFF_PACKET.v3:104-107`, 명령 8-10
  - 요청에서 기대한 ADCS/enhancement handoff와 실제 발견된 handoff 파일 내용이 다르다. 현재 저장소에는 Adaptive-KNet S3 handoff만 확인된다. 따라서 "handoff가 ME 개선 없음 판정을 내렸다"는 문서 근거는 현재 repo에서 확인되지 않는다.

# 10. 사람 확인이 필요한 미해결 질문

- [불명] ADCS/enhancement 전용 handoff 문서가 원래 있었는가? 있다면 파일명과 위치는 무엇인가?
- [불명] `reports/`의 ME summary 중 최종 연구 결론으로 채택해야 하는 파일은 어느 것인가? `me_split_knet_full`, `structured_corruption_full`, `imu_pretrained`, `imu_bias`, `sparse_ref_sanity500` 중 우선순위가 코드만으로는 불명확하다.
- [불명] `runs/`의 어떤 `model.pt`/`train_state.json`가 최종 checkpoint인가? dirty worktree와 다수 run artifact 때문에 코드만으로 특정할 수 없다.
- [불명] IMU noise/bias/saturation/LSB/scale/misalignment 값은 특정 IMU datasheet 기반인가, 실측 기반인가, 임의 stress-test 값인가?
- [불명] 현재 ADCS 정식화에서 gyro를 measurement로 둔 것이 의도인가? 실제 attitude filter처럼 gyro를 propagation/process input으로 쓰는 구조를 배제한 이유가 있는가?
- [불명] sparse sigma reference는 star tracker를 의도한 proxy인가, 아니면 단순 synthetic reference인가?
- [불명] full-state direct `sigma/omega` observation task를 실제 센서 비교 claim에 포함해도 되는가, 아니면 algorithm sanity benchmark로만 보아야 하는가?
- [불명] quaternion 기반 state/filter가 연구 범위에 포함될 예정인가? 현재 구현은 MRP additive state이며 quaternion은 metric 변환에만 쓰인다.
- [불명] original sparse-ref pilot의 `train_nan` 실패 원인은 무엇이며, `sparse_ref_sanity_500`이 공식 replacement인가?
- [불명] real dataset 계열 `nclt`, `uzh_fpv`, `adcs_replay`가 ADCS 논문/실험에 실제로 사용되는가? 사용된다면 train/val/test trajectory separation 정책 확인이 필요하다.
- [불명] `requirements.lock` placeholder 대신 실제 재현 환경 snapshot은 어디에 있는가?
- [불명] `HANDOFF_PACKET.v3`의 Adaptive-KNet known issue가 후속 연구에서 반드시 해결해야 할 항목인지, 아니면 benchmark scope 밖인지 결정이 필요하다.
