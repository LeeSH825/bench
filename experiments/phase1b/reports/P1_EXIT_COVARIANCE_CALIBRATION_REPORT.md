# P1 Exit Covariance Calibration Report

Search status: `COMPLETE`; logical candidates: `101`.
Selected `F-CALIBRATED-v1`: `{'s_P0_att': 2.0, 's_P0_bias': 4.0, 's_Qb': 8.0, 's_Qg': 2.0}`.
Candidate selection used only the listed validation IDs; confirmation was inaccessible before freeze.

| Candidate | Stage | P0-att | P0-bias | Qg | Qb | Guard | Settled full | Att | Bias |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| stage1_p0_att_000_0.5_1_1_1 | stage1_p0_att | 0.5 | 1.0 | 1.0 | 1.0 | True | 1.9063673 | 1.4350111 | 2.745104 |
| stage1_p0_att_001_1_1_1_1 | stage1_p0_att | 1.0 | 1.0 | 1.0 | 1.0 | True | 1.9062451 | 1.4348135 | 2.7448526 |
| stage1_p0_att_002_2_1_1_1 | stage1_p0_att | 2.0 | 1.0 | 1.0 | 1.0 | True | 1.9061832 | 1.4347156 | 2.7447258 |
| stage1_p0_att_003_4_1_1_1 | stage1_p0_att | 4.0 | 1.0 | 1.0 | 1.0 | True | 1.906152 | 1.434667 | 2.7446621 |
| stage1_p0_att_004_8_1_1_1 | stage1_p0_att | 8.0 | 1.0 | 1.0 | 1.0 | False | failure | failure | failure |
| stage1_p0_bias_005_4_0.5_1_1 | stage1_p0_bias | 4.0 | 0.5 | 1.0 | 1.0 | True | 1.9104198 | 1.436634 | 2.7531984 |
| stage1_p0_bias_006_4_1_1_1 | stage1_p0_bias | 4.0 | 1.0 | 1.0 | 1.0 | True | 1.906152 | 1.434667 | 2.7446621 |
| stage1_p0_bias_007_4_2_1_1 | stage1_p0_bias | 4.0 | 2.0 | 1.0 | 1.0 | True | 1.9019768 | 1.4325584 | 2.7362997 |
| stage1_p0_bias_008_4_4_1_1 | stage1_p0_bias | 4.0 | 4.0 | 1.0 | 1.0 | True | 1.89848 | 1.4305085 | 2.7292851 |
| stage1_p0_bias_009_4_8_1_1 | stage1_p0_bias | 4.0 | 8.0 | 1.0 | 1.0 | True | 1.8955945 | 1.4285401 | 2.7235092 |
| stage2_qg_010_4_2_0.5_1 | stage2_qg | 4.0 | 2.0 | 0.5 | 1.0 | True | 2.6110795 | 1.8455085 | 3.5486691 |
| stage2_qg_011_4_2_1_1 | stage2_qg | 4.0 | 2.0 | 1.0 | 1.0 | True | 1.9019768 | 1.4325584 | 2.7362997 |
| stage2_qg_012_4_2_2_1 | stage2_qg | 4.0 | 2.0 | 2.0 | 1.0 | True | 1.4116004 | 1.0833637 | 2.0627161 |
| stage2_qg_013_4_2_4_1 | stage2_qg | 4.0 | 2.0 | 4.0 | 1.0 | False | 1.090153 | 0.83842302 | 1.568371 |
| stage2_qg_014_4_2_8_1 | stage2_qg | 4.0 | 2.0 | 8.0 | 1.0 | False | 0.8866062 | 0.68875017 | 1.2317542 |
| stage2_qb_015_4_2_2_0.5 | stage2_qb | 4.0 | 2.0 | 2.0 | 0.5 | True | 1.4134372 | 1.0836292 | 2.0663389 |
| stage2_qb_016_4_2_2_1 | stage2_qb | 4.0 | 2.0 | 2.0 | 1.0 | True | 1.4116004 | 1.0833637 | 2.0627161 |
| stage2_qb_017_4_2_2_2 | stage2_qb | 4.0 | 2.0 | 2.0 | 2.0 | True | 1.4079462 | 1.0828336 | 2.0555077 |
| stage2_qb_018_4_2_2_4 | stage2_qb | 4.0 | 2.0 | 2.0 | 4.0 | True | 1.4007145 | 1.081777 | 2.0412388 |
| stage2_qb_019_4_2_2_8 | stage2_qb | 4.0 | 2.0 | 2.0 | 8.0 | True | 1.3865499 | 1.079678 | 2.0132776 |
| stage3_local_020_2_1_1_2 | stage3_local | 2.0 | 1.0 | 1.0 | 2.0 | True | 1.8983416 | 1.4333547 | 2.7299698 |
| stage3_local_021_2_1_1_4 | stage3_local | 2.0 | 1.0 | 1.0 | 4.0 | True | 1.8829291 | 1.4306466 | 2.7009455 |
| stage3_local_022_2_1_1_8 | stage3_local | 2.0 | 1.0 | 1.0 | 8.0 | True | 1.8531409 | 1.4252846 | 2.6447691 |
| stage3_local_023_2_1_2_2 | stage3_local | 2.0 | 1.0 | 2.0 | 2.0 | True | 1.4118665 | 1.0843968 | 2.0633512 |
| stage3_local_024_2_1_2_4 | stage3_local | 2.0 | 1.0 | 2.0 | 4.0 | True | 1.4046029 | 1.0833353 | 2.0490183 |
| stage3_local_025_2_1_2_8 | stage3_local | 2.0 | 1.0 | 2.0 | 8.0 | True | 1.390376 | 1.0812265 | 2.0209318 |
| stage3_local_026_2_1_4_2 | stage3_local | 2.0 | 1.0 | 4.0 | 2.0 | False | 1.092118 | 0.83935104 | 1.5723035 |
| stage3_local_027_2_1_4_4 | stage3_local | 2.0 | 1.0 | 4.0 | 4.0 | False | 1.0887538 | 0.83900198 | 1.5655796 |
| stage3_local_028_2_1_4_8 | stage3_local | 2.0 | 1.0 | 4.0 | 8.0 | False | 1.0821067 | 0.8383069 | 1.552292 |
| stage3_local_029_2_2_1_2 | stage3_local | 2.0 | 2.0 | 1.0 | 2.0 | True | 1.894143 | 1.4312211 | 2.721559 |
| stage3_local_030_2_2_1_4 | stage3_local | 2.0 | 2.0 | 1.0 | 4.0 | True | 1.8787839 | 1.4285231 | 2.6926411 |
| stage3_local_031_2_2_1_8 | stage3_local | 2.0 | 2.0 | 1.0 | 8.0 | True | 1.8490991 | 1.4231809 | 2.6366706 |
| stage3_local_032_2_2_2_2 | stage3_local | 2.0 | 2.0 | 2.0 | 2.0 | True | 1.407934 | 1.0828441 | 2.0554829 |
| stage3_local_033_2_2_2_4 | stage3_local | 2.0 | 2.0 | 2.0 | 4.0 | True | 1.4007024 | 1.0817875 | 2.0412143 |
| stage3_local_034_2_2_2_8 | stage3_local | 2.0 | 2.0 | 2.0 | 8.0 | True | 1.3865381 | 1.0796885 | 2.0132537 |
| stage3_local_035_2_2_4_2 | stage3_local | 2.0 | 2.0 | 4.0 | 2.0 | False | 1.0884624 | 0.83825358 | 1.5649922 |
| stage3_local_036_2_2_4_4 | stage3_local | 2.0 | 2.0 | 4.0 | 4.0 | False | 1.0851162 | 0.83790664 | 1.5583041 |
| stage3_local_037_2_2_4_8 | stage3_local | 2.0 | 2.0 | 4.0 | 8.0 | False | 1.0785044 | 0.83721581 | 1.5450875 |
| stage3_local_038_2_4_1_2 | stage3_local | 2.0 | 4.0 | 1.0 | 2.0 | True | 1.8905811 | 1.429124 | 2.7144123 |
| stage3_local_039_2_4_1_4 | stage3_local | 2.0 | 4.0 | 1.0 | 4.0 | True | 1.8752672 | 1.4264356 | 2.6855851 |
| stage3_local_040_2_4_1_8 | stage3_local | 2.0 | 4.0 | 1.0 | 8.0 | True | 1.8456697 | 1.4211125 | 2.6297901 |
| stage3_local_041_2_4_2_2 | stage3_local | 2.0 | 4.0 | 2.0 | 2.0 | True | 1.4044904 | 1.0813364 | 2.0485907 |
| stage3_local_042_2_4_2_4 | stage3_local | 2.0 | 4.0 | 2.0 | 4.0 | True | 1.3972869 | 1.0802844 | 2.0343789 |
| stage3_local_043_2_4_2_8 | stage3_local | 2.0 | 4.0 | 2.0 | 8.0 | True | 1.3831779 | 1.0781948 | 2.0065296 |
| stage3_local_044_2_4_4_2 | stage3_local | 2.0 | 4.0 | 4.0 | 2.0 | False | 1.0851725 | 0.83720519 | 1.5584119 |
| stage3_local_045_2_4_4_4 | stage3_local | 2.0 | 4.0 | 4.0 | 4.0 | False | 1.0818424 | 0.8368603 | 1.5517565 |
| stage3_local_046_2_4_4_8 | stage3_local | 2.0 | 4.0 | 4.0 | 8.0 | False | 1.0752627 | 0.83617356 | 1.5386043 |
| stage3_local_047_4_1_1_2 | stage3_local | 4.0 | 1.0 | 1.0 | 2.0 | True | 1.8983105 | 1.4333061 | 2.7299064 |
| stage3_local_048_4_1_1_4 | stage3_local | 4.0 | 1.0 | 1.0 | 4.0 | True | 1.8828981 | 1.4305982 | 2.7008825 |
| stage3_local_049_4_1_1_8 | stage3_local | 4.0 | 1.0 | 1.0 | 8.0 | True | 1.8531104 | 1.4252364 | 2.6447073 |
| stage3_local_050_4_1_2_2 | stage3_local | 4.0 | 1.0 | 2.0 | 2.0 | True | 1.4118346 | 1.0843665 | 2.063287 |
| stage3_local_051_4_1_2_4 | stage3_local | 4.0 | 1.0 | 2.0 | 4.0 | True | 1.4045711 | 1.0833051 | 2.0489546 |
| stage3_local_052_4_1_2_8 | stage3_local | 4.0 | 1.0 | 2.0 | 8.0 | True | 1.3903446 | 1.0811965 | 2.0208689 |
| stage3_local_053_4_1_4_2 | stage3_local | 4.0 | 1.0 | 4.0 | 2.0 | False | 1.0920861 | 0.83933438 | 1.5722397 |
| stage3_local_054_4_1_4_4 | stage3_local | 4.0 | 1.0 | 4.0 | 4.0 | False | 1.088722 | 0.83898533 | 1.565516 |
| stage3_local_055_4_1_4_8 | stage3_local | 4.0 | 1.0 | 4.0 | 8.0 | False | 1.0820752 | 0.83829031 | 1.552229 |
| stage3_local_056_4_2_1_2 | stage3_local | 4.0 | 2.0 | 1.0 | 2.0 | True | 1.8941622 | 1.4312025 | 2.7215975 |
| stage3_local_057_4_2_1_4 | stage3_local | 4.0 | 2.0 | 1.0 | 4.0 | True | 1.8788028 | 1.4285045 | 2.6926788 |
| stage3_local_058_4_2_1_8 | stage3_local | 4.0 | 2.0 | 1.0 | 8.0 | True | 1.8491171 | 1.4231624 | 2.6367069 |
| stage3_local_059_4_2_2_2 | stage3_local | 4.0 | 2.0 | 2.0 | 2.0 | True | 1.4079462 | 1.0828336 | 2.0555077 |
| stage3_local_060_4_2_2_4 | stage3_local | 4.0 | 2.0 | 2.0 | 4.0 | True | 1.4007145 | 1.081777 | 2.0412388 |
| stage3_local_061_4_2_2_8 | stage3_local | 4.0 | 2.0 | 2.0 | 8.0 | True | 1.3865499 | 1.079678 | 2.0132776 |
| stage3_local_062_4_2_4_2 | stage3_local | 4.0 | 2.0 | 4.0 | 2.0 | False | 1.0884696 | 0.83824917 | 1.5650067 |
| stage3_local_063_4_2_4_4 | stage3_local | 4.0 | 2.0 | 4.0 | 4.0 | False | 1.0851233 | 0.83790223 | 1.5583185 |
| stage3_local_064_4_2_4_8 | stage3_local | 4.0 | 2.0 | 4.0 | 8.0 | False | 1.0785113 | 0.83721141 | 1.5451017 |
| stage3_local_065_4_4_1_2 | stage3_local | 4.0 | 4.0 | 1.0 | 2.0 | True | 1.8906878 | 1.4291574 | 2.7146278 |
| stage3_local_066_4_4_1_4 | stage3_local | 4.0 | 4.0 | 1.0 | 4.0 | True | 1.8753724 | 1.4264688 | 2.6857976 |
| stage3_local_067_4_4_1_8 | stage3_local | 4.0 | 4.0 | 1.0 | 8.0 | True | 1.8457718 | 1.4211453 | 2.6299969 |
| stage3_local_068_4_4_2_2 | stage3_local | 4.0 | 4.0 | 2.0 | 2.0 | True | 1.4045788 | 1.0813598 | 2.0487689 |
| stage3_local_069_4_4_2_4 | stage3_local | 4.0 | 4.0 | 2.0 | 4.0 | True | 1.3973746 | 1.0803078 | 2.0345555 |
| stage3_local_070_4_4_2_8 | stage3_local | 4.0 | 4.0 | 2.0 | 8.0 | True | 1.3832639 | 1.0782179 | 2.0067031 |
| stage3_local_071_4_4_4_2 | stage3_local | 4.0 | 4.0 | 4.0 | 2.0 | False | 1.0852467 | 0.83722158 | 1.5585606 |
| stage3_local_072_4_4_4_4 | stage3_local | 4.0 | 4.0 | 4.0 | 4.0 | False | 1.0819162 | 0.83687666 | 1.5519045 |
| stage3_local_073_4_4_4_8 | stage3_local | 4.0 | 4.0 | 4.0 | 8.0 | False | 1.0753357 | 0.83618984 | 1.5387508 |
| stage3_local_074_8_1_1_2 | stage3_local | 8.0 | 1.0 | 1.0 | 2.0 | False | failure | failure | failure |
| stage3_local_075_8_1_1_4 | stage3_local | 8.0 | 1.0 | 1.0 | 4.0 | False | failure | failure | failure |
| stage3_local_076_8_1_1_8 | stage3_local | 8.0 | 1.0 | 1.0 | 8.0 | True | 1.853095 | 1.4252124 | 2.6446763 |
| stage3_local_077_8_1_2_2 | stage3_local | 8.0 | 1.0 | 2.0 | 2.0 | True | 1.4118185 | 1.0843515 | 2.0632549 |
| stage3_local_078_8_1_2_4 | stage3_local | 8.0 | 1.0 | 2.0 | 4.0 | True | 1.4045552 | 1.08329 | 2.0489226 |
| stage3_local_079_8_1_2_8 | stage3_local | 8.0 | 1.0 | 2.0 | 8.0 | True | 1.3903288 | 1.0811815 | 2.0208374 |
| stage3_local_080_8_1_4_2 | stage3_local | 8.0 | 1.0 | 4.0 | 2.0 | False | failure | failure | failure |
| stage3_local_081_8_1_4_4 | stage3_local | 8.0 | 1.0 | 4.0 | 4.0 | False | failure | failure | failure |
| stage3_local_082_8_1_4_8 | stage3_local | 8.0 | 1.0 | 4.0 | 8.0 | False | failure | failure | failure |
| stage3_local_083_8_2_1_2 | stage3_local | 8.0 | 2.0 | 1.0 | 2.0 | True | 1.8941718 | 1.4311934 | 2.7216165 |
| stage3_local_084_8_2_1_4 | stage3_local | 8.0 | 2.0 | 1.0 | 4.0 | False | failure | failure | failure |
| stage3_local_085_8_2_1_8 | stage3_local | 8.0 | 2.0 | 1.0 | 8.0 | True | 1.849126 | 1.4231533 | 2.6367249 |
| stage3_local_086_8_2_2_2 | stage3_local | 8.0 | 2.0 | 2.0 | 2.0 | True | 1.4079523 | 1.0828285 | 2.05552 |
| stage3_local_087_8_2_2_4 | stage3_local | 8.0 | 2.0 | 2.0 | 4.0 | True | 1.4007204 | 1.0817718 | 2.0412509 |
| stage3_local_088_8_2_2_8 | stage3_local | 8.0 | 2.0 | 2.0 | 8.0 | True | 1.3865557 | 1.0796729 | 2.0132894 |
| stage3_local_089_8_2_4_2 | stage3_local | 8.0 | 2.0 | 4.0 | 2.0 | False | 1.0884732 | 0.83824701 | 1.5650138 |
| stage3_local_090_8_2_4_4 | stage3_local | 8.0 | 2.0 | 4.0 | 4.0 | False | 1.0851268 | 0.83790007 | 1.5583256 |
| stage3_local_091_8_2_4_8 | stage3_local | 8.0 | 2.0 | 4.0 | 8.0 | False | 1.0785148 | 0.83720925 | 1.5451087 |
| stage3_local_092_8_4_1_2 | stage3_local | 8.0 | 4.0 | 1.0 | 2.0 | True | 1.8907408 | 1.4291743 | 2.7147351 |
| stage3_local_093_8_4_1_4 | stage3_local | 8.0 | 4.0 | 1.0 | 4.0 | True | 1.8754246 | 1.4264856 | 2.6859034 |
| stage3_local_094_8_4_1_8 | stage3_local | 8.0 | 4.0 | 1.0 | 8.0 | True | 1.8458226 | 1.4211619 | 2.6300998 |
| stage3_local_095_8_4_2_2 | stage3_local | 8.0 | 4.0 | 2.0 | 2.0 | True | 1.4046228 | 1.0813716 | 2.0488576 |
| stage3_local_096_8_4_2_4 | stage3_local | 8.0 | 4.0 | 2.0 | 4.0 | True | 1.3974182 | 1.0803196 | 2.0346435 |
| stage3_local_097_8_4_2_8 | stage3_local | 8.0 | 4.0 | 2.0 | 8.0 | True | 1.3833067 | 1.0782297 | 2.0067895 |
| stage3_local_098_8_4_4_2 | stage3_local | 8.0 | 4.0 | 4.0 | 2.0 | False | 1.0852837 | 0.83722988 | 1.5586348 |
| stage3_local_099_8_4_4_4 | stage3_local | 8.0 | 4.0 | 4.0 | 4.0 | False | 1.081953 | 0.83688493 | 1.5519782 |
| stage3_local_100_8_4_4_8 | stage3_local | 8.0 | 4.0 | 4.0 | 8.0 | False | 1.0753721 | 0.83619807 | 1.5388237 |

## Independent confirmation

- Status/N: `COMPLETE` / `50`.
- Stationary paired NEES-distance evidence: `-0.2080373`, 95% CI `[-0.30500210173950504, -0.11436263824653703]`.
- Acceptance: `{'c4': {'N50_all': True, 'calibrated_accuracy_not_seriously_worse': False, 'calibrated_settled_sensor_nis': False, 'full_beats_wrong_measurement_state': True, 'full_oracle_cause_advantage': True, 'full_oracle_settled_sensor_nis': True, 'process_beats_wrong_process_slow_bias': True, 'strict_P_S_SPD': True, 'wrong_measurement_state_worse_than_base': True, 'zero_divergence': True}, 'c4_passed': False, 'full_oracle_fast_peak_improvement_fraction': 0.4132085082893989, 'full_oracle_slow_bias_improvement_fraction': 0.3208727762378273, 'stationary': {'N50': True, 'attitude_accuracy': True, 'bias_accuracy': True, 'mag_nis': True, 'materially_closer': True, 'st_nis': True, 'strict_P_S_SPD': True, 'sun_nis': True, 'target_full_nees': True, 'zero_divergence': True}, 'stationary_passed': True}`.
- Stationary/C4 same realization: `True`.
- Calibration/confirmation and frozen-test disjointness: `True` / `True`.

No sensor R tuning, test-set candidate selection, reported-P scaling, event-wise inflation, oracle label input, or numerical covariance fallback was used.
