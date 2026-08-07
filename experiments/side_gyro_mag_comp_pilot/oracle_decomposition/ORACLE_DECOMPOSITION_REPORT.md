# Oracle Compensation Decomposition Report

Diagnostic conclusion: `MAG_DOMINANT_HEADROOM`

This report separates committed four-arm evidence, new classical sensor interventions, fixed-N0-checkpoint neural interventions, and separately trained N1 performance. It does not authorize Step 2.

## Checkpoint provenance

N0 and N1 use separate checkpoints trained on different sensor inputs for every seed. They share one normalization digest and the same normalization source IDs. N1 is not a fixed-N0 intervention.

## Existing committed C0/C1/N0/N1 comparison

The following values were reconstructed from committed records; no old arm was rerun. Negative contrast means lower candidate error. C0/C1 and cross-backend comparisons were not previous pilot gates.

| Comparison | Regime | Metric | Reference | Candidate | Contrast | 95% CI | Seed mean directions |
|---|---|---|---:|---:|---:|---|---|
| C1_minus_C0 | R0_NOMINAL | attitude_geodesic_rmse_rad | 0.0566571507 | 0.0566571507 | +0 | [+0, +0] | deterministic |
| C1_minus_C0 | R0_NOMINAL | residual_gyro_bias_rmse | 4.52379242e-05 | 4.52379242e-05 | +0 | [+0, +0] | deterministic |
| C1_minus_C0 | R0_NOMINAL | corrected_gyro_rate_rmse_rad_s | 0 | 0 | +0 | [+0, +0] | deterministic |
| C1_minus_C0 | R0_NOMINAL | integrated_gyro_increment_rmse_rad | 0 | 0 | +0 | [+0, +0] | deterministic |
| C1_minus_C0 | R0_NOMINAL | corrected_magnetometer_angular_error_rad | 0 | 0 | +0 | [+0, +0] | deterministic |
| C1_minus_C0 | R0_NOMINAL | weak_axis_rmse | 0.0495739188 | 0.0495739188 | +0 | [+0, +0] | deterministic |
| C1_minus_C0 | R0_NOMINAL | observable_plane_rmse | 0.022362206 | 0.022362206 | +0 | [+0, +0] | deterministic |
| C1_minus_C0 | R0_NOMINAL | divergence_count | 0 | 0 | +0 | [+0, +0] | deterministic |
| C1_minus_C0 | R1_GYRO_BIAS_SCALE | attitude_geodesic_rmse_rad | 0.0550054592 | 0.0554315143 | +0.000426055117 | [+8.61224405e-07, +0.000831102773] | deterministic |
| C1_minus_C0 | R1_GYRO_BIAS_SCALE | residual_gyro_bias_rmse | 0.0001108469 | 4.70937181e-05 | -6.37531822e-05 | [-7.45676275e-05, -5.1418334e-05] | deterministic |
| C1_minus_C0 | R1_GYRO_BIAS_SCALE | corrected_gyro_rate_rmse_rad_s | 0.00102576778 | 0 | -0.00102576778 | [-0.00102583548, -0.00102569845] | deterministic |
| C1_minus_C0 | R1_GYRO_BIAS_SCALE | integrated_gyro_increment_rmse_rad | 0.000891389555 | 0 | -0.000891389555 | [-0.000909489205, -0.000873620502] | deterministic |
| C1_minus_C0 | R1_GYRO_BIAS_SCALE | corrected_magnetometer_angular_error_rad | 0 | 0 | +0 | [+0, +0] | deterministic |
| C1_minus_C0 | R1_GYRO_BIAS_SCALE | weak_axis_rmse | 0.0481015384 | 0.0486507459 | +0.000549207489 | [+6.68874556e-05, +0.00101269573] | deterministic |
| C1_minus_C0 | R1_GYRO_BIAS_SCALE | observable_plane_rmse | 0.0224599242 | 0.0223834063 | -7.65179664e-05 | [-8.82419636e-05, -6.48679158e-05] | deterministic |
| C1_minus_C0 | R1_GYRO_BIAS_SCALE | divergence_count | 0 | 0 | +0 | [+0, +0] | deterministic |
| C1_minus_C0 | R2_MAG_HARD_SOFT_IRON | attitude_geodesic_rmse_rad | 0.0797424007 | 0.0451687729 | -0.0345736279 | [-0.0373770084, -0.0318360783] | deterministic |
| C1_minus_C0 | R2_MAG_HARD_SOFT_IRON | residual_gyro_bias_rmse | 6.5605449e-05 | 4.52243457e-05 | -2.03811032e-05 | [-2.58247214e-05, -1.51350826e-05] | deterministic |
| C1_minus_C0 | R2_MAG_HARD_SOFT_IRON | corrected_gyro_rate_rmse_rad_s | 0 | 0 | +0 | [+0, +0] | deterministic |
| C1_minus_C0 | R2_MAG_HARD_SOFT_IRON | integrated_gyro_increment_rmse_rad | 0 | 0 | +0 | [+0, +0] | deterministic |
| C1_minus_C0 | R2_MAG_HARD_SOFT_IRON | corrected_magnetometer_angular_error_rad | 0.0501633917 | 0 | -0.0501633917 | [-0.0510487063, -0.0493223473] | deterministic |
| C1_minus_C0 | R2_MAG_HARD_SOFT_IRON | weak_axis_rmse | 0.0369953131 | 0.0366425934 | -0.000352719731 | [-0.000894013557, +0.000179101693] | deterministic |
| C1_minus_C0 | R2_MAG_HARD_SOFT_IRON | observable_plane_rmse | 0.067515099 | 0.0223611103 | -0.0451539886 | [-0.045968227, -0.0443732285] | deterministic |
| C1_minus_C0 | R2_MAG_HARD_SOFT_IRON | divergence_count | 0 | 0 | +0 | [+0, +0] | deterministic |
| C1_minus_C0 | R3_COMBINED | attitude_geodesic_rmse_rad | 0.106510527 | 0.062410845 | -0.0440996825 | [-0.0489572665, -0.0393007303] | deterministic |
| C1_minus_C0 | R3_COMBINED | residual_gyro_bias_rmse | 8.6361752e-05 | 4.56798424e-05 | -4.06819096e-05 | [-5.4072815e-05, -2.79378715e-05] | deterministic |
| C1_minus_C0 | R3_COMBINED | corrected_gyro_rate_rmse_rad_s | 0.00129151495 | 0 | -0.00129151495 | [-0.00129158914, -0.00129143781] | deterministic |
| C1_minus_C0 | R3_COMBINED | integrated_gyro_increment_rmse_rad | 0.00109958451 | 0 | -0.00109958451 | [-0.00111448055, -0.00108552263] | deterministic |
| C1_minus_C0 | R3_COMBINED | corrected_magnetometer_angular_error_rad | 0.0627639692 | 0 | -0.0627639692 | [-0.0638537207, -0.0616280931] | deterministic |
| C1_minus_C0 | R3_COMBINED | weak_axis_rmse | 0.0553293847 | 0.0555868133 | +0.000257428606 | [-0.00061595151, +0.00113012246] | deterministic |
| C1_minus_C0 | R3_COMBINED | observable_plane_rmse | 0.0851404612 | 0.0223812218 | -0.0627592394 | [-0.063850854, -0.0616130268] | deterministic |
| C1_minus_C0 | R3_COMBINED | divergence_count | 0 | 0 | +0 | [+0, +0] | deterministic |
| N1_minus_N0 | R0_NOMINAL | attitude_geodesic_rmse_rad | 0.0637176409 | 0.0548044571 | -0.00891318387 | [-0.0135803514, -0.00499445709] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R0_NOMINAL | residual_gyro_bias_rmse | 0.00807484659 | 0.00499374197 | -0.00308110462 | [-0.00453120365, -0.00207646442] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R0_NOMINAL | corrected_gyro_rate_rmse_rad_s | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_N0 | R0_NOMINAL | integrated_gyro_increment_rmse_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_N0 | R0_NOMINAL | corrected_magnetometer_angular_error_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_N0 | R0_NOMINAL | weak_axis_rmse | 0.0483748671 | 0.048195867 | -0.000179000081 | [-0.000947765746, +0.000653947816] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R0_NOMINAL | observable_plane_rmse | 0.0315036416 | 0.0208120156 | -0.010691626 | [-0.0172888088, -0.00426160368] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R0_NOMINAL | divergence_count | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_N0 | R1_GYRO_BIAS_SCALE | attitude_geodesic_rmse_rad | 0.0571574101 | 0.0515914857 | -0.00556592431 | [-0.00857695261, -0.00273646536] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R1_GYRO_BIAS_SCALE | residual_gyro_bias_rmse | 0.00630259449 | 0.00513227986 | -0.00117031463 | [-0.00228993195, -8.19953319e-05] | 31001:negative, 31002:negative, 31003:positive |
| N1_minus_N0 | R1_GYRO_BIAS_SCALE | corrected_gyro_rate_rmse_rad_s | 0.00102576778 | 0 | -0.00102576778 | [-0.00102583548, -0.00102569845] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R1_GYRO_BIAS_SCALE | integrated_gyro_increment_rmse_rad | 0.000891389555 | 0 | -0.000891389555 | [-0.000909489205, -0.000873620502] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R1_GYRO_BIAS_SCALE | corrected_magnetometer_angular_error_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_N0 | R1_GYRO_BIAS_SCALE | weak_axis_rmse | 0.0463470743 | 0.0452630836 | -0.0010839907 | [-0.00190507138, -0.0002156753] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R1_GYRO_BIAS_SCALE | observable_plane_rmse | 0.026554162 | 0.0205825478 | -0.00597161425 | [-0.0116702551, -0.000520441224] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R1_GYRO_BIAS_SCALE | divergence_count | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_N0 | R2_MAG_HARD_SOFT_IRON | attitude_geodesic_rmse_rad | 0.0564970904 | 0.0434851225 | -0.013011968 | [-0.0189281618, -0.00776236967] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R2_MAG_HARD_SOFT_IRON | residual_gyro_bias_rmse | 0.00836088324 | 0.00426927997 | -0.00409160327 | [-0.00514249104, -0.00304087462] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R2_MAG_HARD_SOFT_IRON | corrected_gyro_rate_rmse_rad_s | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_N0 | R2_MAG_HARD_SOFT_IRON | integrated_gyro_increment_rmse_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_N0 | R2_MAG_HARD_SOFT_IRON | corrected_magnetometer_angular_error_rad | 0.0501633917 | 0 | -0.0501633917 | [-0.0510487063, -0.0493223473] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R2_MAG_HARD_SOFT_IRON | weak_axis_rmse | 0.0360103482 | 0.0356804031 | -0.000329945057 | [-0.00171216733, +0.00106360836] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R2_MAG_HARD_SOFT_IRON | observable_plane_rmse | 0.0371499717 | 0.0206631342 | -0.0164868376 | [-0.0243007857, -0.00924618494] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R2_MAG_HARD_SOFT_IRON | divergence_count | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_N0 | R3_COMBINED | attitude_geodesic_rmse_rad | 0.0796075044 | 0.0623590856 | -0.0172484188 | [-0.0240839091, -0.0108135666] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R3_COMBINED | residual_gyro_bias_rmse | 0.0110993344 | 0.00491080425 | -0.00618853013 | [-0.00760896983, -0.00465371249] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R3_COMBINED | corrected_gyro_rate_rmse_rad_s | 0.00129151495 | 0 | -0.00129151495 | [-0.00129158914, -0.00129143781] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R3_COMBINED | integrated_gyro_increment_rmse_rad | 0.00109958451 | 0 | -0.00109958451 | [-0.00111448055, -0.00108552263] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R3_COMBINED | corrected_magnetometer_angular_error_rad | 0.0627639692 | 0 | -0.0627639692 | [-0.0638537207, -0.0616280931] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R3_COMBINED | weak_axis_rmse | 0.0563207662 | 0.0566817308 | +0.000360964559 | [-0.00179699899, +0.00257138169] | 31001:positive, 31002:positive, 31003:positive |
| N1_minus_N0 | R3_COMBINED | observable_plane_rmse | 0.0485843348 | 0.0206340883 | -0.0279502465 | [-0.0379511146, -0.0184937219] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_N0 | R3_COMBINED | divergence_count | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R0_NOMINAL | attitude_geodesic_rmse_rad | 0.0566571507 | 0.0637176409 | +0.00706049023 | [+0.00215491774, +0.0131592939] | 31001:positive, 31002:positive, 31003:positive |
| N0_minus_C0 | R0_NOMINAL | residual_gyro_bias_rmse | 4.52379242e-05 | 0.00807484659 | +0.00802960867 | [+0.00629364174, +0.0102851721] | 31001:positive, 31002:positive, 31003:positive |
| N0_minus_C0 | R0_NOMINAL | corrected_gyro_rate_rmse_rad_s | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R0_NOMINAL | integrated_gyro_increment_rmse_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R0_NOMINAL | corrected_magnetometer_angular_error_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R0_NOMINAL | weak_axis_rmse | 0.0495739188 | 0.0483748671 | -0.00119905175 | [-0.00373098221, +0.00162042218] | 31001:negative, 31002:negative, 31003:negative |
| N0_minus_C0 | R0_NOMINAL | observable_plane_rmse | 0.022362206 | 0.0315036416 | +0.00914143566 | [+0.00240599192, +0.0161777117] | 31001:positive, 31002:positive, 31003:positive |
| N0_minus_C0 | R0_NOMINAL | divergence_count | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R1_GYRO_BIAS_SCALE | attitude_geodesic_rmse_rad | 0.0550054592 | 0.0571574101 | +0.00215195086 | [-0.000829941685, +0.00521320758] | 31001:positive, 31002:positive, 31003:positive |
| N0_minus_C0 | R1_GYRO_BIAS_SCALE | residual_gyro_bias_rmse | 0.0001108469 | 0.00630259449 | +0.00619174759 | [+0.00460377457, +0.0079020963] | 31001:positive, 31002:positive, 31003:positive |
| N0_minus_C0 | R1_GYRO_BIAS_SCALE | corrected_gyro_rate_rmse_rad_s | 0.00102576778 | 0.00102576778 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R1_GYRO_BIAS_SCALE | integrated_gyro_increment_rmse_rad | 0.000891389555 | 0.000891389555 | -3.61400724e-21 | [-1.80700362e-20, +1.08420217e-20] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R1_GYRO_BIAS_SCALE | corrected_magnetometer_angular_error_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R1_GYRO_BIAS_SCALE | weak_axis_rmse | 0.0481015384 | 0.0463470743 | -0.00175446408 | [-0.00407836447, +0.000380199457] | 31001:negative, 31002:negative, 31003:negative |
| N0_minus_C0 | R1_GYRO_BIAS_SCALE | observable_plane_rmse | 0.0224599242 | 0.026554162 | +0.0040942378 | [-0.00159565332, +0.0101801249] | 31001:positive, 31002:positive, 31003:positive |
| N0_minus_C0 | R1_GYRO_BIAS_SCALE | divergence_count | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R2_MAG_HARD_SOFT_IRON | attitude_geodesic_rmse_rad | 0.0797424007 | 0.0564970904 | -0.0232453103 | [-0.028856369, -0.017021708] | 31001:negative, 31002:negative, 31003:negative |
| N0_minus_C0 | R2_MAG_HARD_SOFT_IRON | residual_gyro_bias_rmse | 6.5605449e-05 | 0.00836088324 | +0.00829527779 | [+0.00702522122, +0.00961430143] | 31001:positive, 31002:positive, 31003:positive |
| N0_minus_C0 | R2_MAG_HARD_SOFT_IRON | corrected_gyro_rate_rmse_rad_s | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R2_MAG_HARD_SOFT_IRON | integrated_gyro_increment_rmse_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R2_MAG_HARD_SOFT_IRON | corrected_magnetometer_angular_error_rad | 0.0501633917 | 0.0501633917 | +4.62592927e-19 | [-6.9388939e-19, +1.85037171e-18] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R2_MAG_HARD_SOFT_IRON | weak_axis_rmse | 0.0369953131 | 0.0360103482 | -0.000984964911 | [-0.00253520429, +0.00054994681] | 31001:negative, 31002:negative, 31003:negative |
| N0_minus_C0 | R2_MAG_HARD_SOFT_IRON | observable_plane_rmse | 0.067515099 | 0.0371499717 | -0.0303651272 | [-0.0375150441, -0.0226806327] | 31001:negative, 31002:negative, 31003:negative |
| N0_minus_C0 | R2_MAG_HARD_SOFT_IRON | divergence_count | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R3_COMBINED | attitude_geodesic_rmse_rad | 0.106510527 | 0.0796075044 | -0.0269030231 | [-0.0361638372, -0.0172478428] | 31001:negative, 31002:negative, 31003:negative |
| N0_minus_C0 | R3_COMBINED | residual_gyro_bias_rmse | 8.6361752e-05 | 0.0110993344 | +0.0110129726 | [+0.00928084947, +0.0129353376] | 31001:positive, 31002:positive, 31003:positive |
| N0_minus_C0 | R3_COMBINED | corrected_gyro_rate_rmse_rad_s | 0.00129151495 | 0.00129151495 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R3_COMBINED | integrated_gyro_increment_rmse_rad | 0.00109958451 | 0.00109958451 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R3_COMBINED | corrected_magnetometer_angular_error_rad | 0.0627639692 | 0.0627639692 | -2.31296463e-19 | [-9.25185854e-19, +4.62592927e-19] | 31001:zero, 31002:zero, 31003:zero |
| N0_minus_C0 | R3_COMBINED | weak_axis_rmse | 0.0553293847 | 0.0563207662 | +0.000991381534 | [-0.00242672512, +0.00445499449] | 31001:positive, 31002:positive, 31003:positive |
| N0_minus_C0 | R3_COMBINED | observable_plane_rmse | 0.0851404612 | 0.0485843348 | -0.0365561265 | [-0.0478290875, -0.0246741922] | 31001:negative, 31002:negative, 31003:negative |
| N0_minus_C0 | R3_COMBINED | divergence_count | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R0_NOMINAL | attitude_geodesic_rmse_rad | 0.0566571507 | 0.0548044571 | -0.00185269364 | [-0.0045397426, +0.00117738953] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_C1 | R0_NOMINAL | residual_gyro_bias_rmse | 4.52379242e-05 | 0.00499374197 | +0.00494850405 | [+0.00407637245, +0.00594321043] | 31001:positive, 31002:positive, 31003:positive |
| N1_minus_C1 | R0_NOMINAL | corrected_gyro_rate_rmse_rad_s | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R0_NOMINAL | integrated_gyro_increment_rmse_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R0_NOMINAL | corrected_magnetometer_angular_error_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R0_NOMINAL | weak_axis_rmse | 0.0495739188 | 0.048195867 | -0.00137805183 | [-0.00444838615, +0.00211327052] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_C1 | R0_NOMINAL | observable_plane_rmse | 0.022362206 | 0.0208120156 | -0.00155019035 | [-0.00245896618, -0.000671655584] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_C1 | R0_NOMINAL | divergence_count | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R1_GYRO_BIAS_SCALE | attitude_geodesic_rmse_rad | 0.0554315143 | 0.0515914857 | -0.00384002857 | [-0.0064211567, -0.00136165747] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_C1 | R1_GYRO_BIAS_SCALE | residual_gyro_bias_rmse | 4.70937181e-05 | 0.00513227986 | +0.00508518614 | [+0.00429139671, +0.00593121707] | 31001:positive, 31002:positive, 31003:positive |
| N1_minus_C1 | R1_GYRO_BIAS_SCALE | corrected_gyro_rate_rmse_rad_s | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R1_GYRO_BIAS_SCALE | integrated_gyro_increment_rmse_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R1_GYRO_BIAS_SCALE | corrected_magnetometer_angular_error_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R1_GYRO_BIAS_SCALE | weak_axis_rmse | 0.0486507459 | 0.0452630836 | -0.00338766227 | [-0.00613430772, -0.00078134472] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_C1 | R1_GYRO_BIAS_SCALE | observable_plane_rmse | 0.0223834063 | 0.0205825478 | -0.00180085848 | [-0.0026257769, -0.00100282213] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_C1 | R1_GYRO_BIAS_SCALE | divergence_count | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R2_MAG_HARD_SOFT_IRON | attitude_geodesic_rmse_rad | 0.0451687729 | 0.0434851225 | -0.00168365038 | [-0.00393807243, +0.000642279169] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_C1 | R2_MAG_HARD_SOFT_IRON | residual_gyro_bias_rmse | 4.52243457e-05 | 0.00426927997 | +0.00422405563 | [+0.00343221992, +0.00502963366] | 31001:positive, 31002:positive, 31003:positive |
| N1_minus_C1 | R2_MAG_HARD_SOFT_IRON | corrected_gyro_rate_rmse_rad_s | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R2_MAG_HARD_SOFT_IRON | integrated_gyro_increment_rmse_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R2_MAG_HARD_SOFT_IRON | corrected_magnetometer_angular_error_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R2_MAG_HARD_SOFT_IRON | weak_axis_rmse | 0.0366425934 | 0.0356804031 | -0.000962190237 | [-0.00362612098, +0.00177789319] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_C1 | R2_MAG_HARD_SOFT_IRON | observable_plane_rmse | 0.0223611103 | 0.0206631342 | -0.00169797614 | [-0.00241477914, -0.000981924773] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_C1 | R2_MAG_HARD_SOFT_IRON | divergence_count | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R3_COMBINED | attitude_geodesic_rmse_rad | 0.062410845 | 0.0623590856 | -5.17594138e-05 | [-0.00297564915, +0.00304243149] | 31001:negative, 31002:positive, 31003:negative |
| N1_minus_C1 | R3_COMBINED | residual_gyro_bias_rmse | 4.56798424e-05 | 0.00491080425 | +0.00486512441 | [+0.00380247488, +0.00601181396] | 31001:positive, 31002:positive, 31003:positive |
| N1_minus_C1 | R3_COMBINED | corrected_gyro_rate_rmse_rad_s | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R3_COMBINED | integrated_gyro_increment_rmse_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R3_COMBINED | corrected_magnetometer_angular_error_rad | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |
| N1_minus_C1 | R3_COMBINED | weak_axis_rmse | 0.0555868133 | 0.0566817308 | +0.00109491749 | [-0.00232597621, +0.00475712276] | 31001:positive, 31002:positive, 31003:positive |
| N1_minus_C1 | R3_COMBINED | observable_plane_rmse | 0.0223812218 | 0.0206340883 | -0.00174713354 | [-0.00255909317, -0.000939297579] | 31001:negative, 31002:negative, 31003:negative |
| N1_minus_C1 | R3_COMBINED | divergence_count | 0 | 0 | +0 | [+0, +0] | 31001:zero, 31002:zero, 31003:zero |

## Oracle sensor decomposition

Positive E values mean lower error after intervention. Positive I means combined improvement exceeds the sum of isolated improvements; negative I indicates overlap or redundancy.

| Backend | Regime | Metric | E_G | E_M | E_GM | I |
|---|---|---|---:|---:|---:|---:|
| classical_mekf | R0_NOMINAL | attitude_geodesic_rmse_rad | +0 | +0 | +0 | +0 |
| classical_mekf | R0_NOMINAL | residual_gyro_bias_rmse | +0 | +0 | +0 | +0 |
| classical_mekf | R0_NOMINAL | corrected_gyro_rate_rmse_rad_s | +0 | +0 | +0 | +0 |
| classical_mekf | R0_NOMINAL | integrated_gyro_increment_rmse_rad | +0 | +0 | +0 | +0 |
| classical_mekf | R0_NOMINAL | corrected_magnetometer_angular_error_rad | +0 | +0 | +0 | +0 |
| classical_mekf | R0_NOMINAL | weak_axis_rmse | +0 | +0 | +0 | +0 |
| classical_mekf | R0_NOMINAL | observable_plane_rmse | +0 | +0 | +0 | +0 |
| classical_mekf | R0_NOMINAL | divergence_count | +0 | +0 | +0 | +0 |
| classical_mekf | R1_GYRO_BIAS_SCALE | attitude_geodesic_rmse_rad | -0.000426055117 | +0 | -0.000426055117 | +0 |
| classical_mekf | R1_GYRO_BIAS_SCALE | residual_gyro_bias_rmse | +6.37531822e-05 | +0 | +6.37531822e-05 | +0 |
| classical_mekf | R1_GYRO_BIAS_SCALE | corrected_gyro_rate_rmse_rad_s | +0.00102576778 | +0 | +0.00102576778 | +0 |
| classical_mekf | R1_GYRO_BIAS_SCALE | integrated_gyro_increment_rmse_rad | +0.000891389555 | +0 | +0.000891389555 | +0 |
| classical_mekf | R1_GYRO_BIAS_SCALE | corrected_magnetometer_angular_error_rad | +0 | +0 | +0 | +0 |
| classical_mekf | R1_GYRO_BIAS_SCALE | weak_axis_rmse | -0.000549207489 | +0 | -0.000549207489 | +0 |
| classical_mekf | R1_GYRO_BIAS_SCALE | observable_plane_rmse | +7.65179664e-05 | +0 | +7.65179664e-05 | +0 |
| classical_mekf | R1_GYRO_BIAS_SCALE | divergence_count | +0 | +0 | +0 | +0 |
| classical_mekf | R2_MAG_HARD_SOFT_IRON | attitude_geodesic_rmse_rad | +0 | +0.0345736279 | +0.0345736279 | +0 |
| classical_mekf | R2_MAG_HARD_SOFT_IRON | residual_gyro_bias_rmse | +0 | +2.03811032e-05 | +2.03811032e-05 | +0 |
| classical_mekf | R2_MAG_HARD_SOFT_IRON | corrected_gyro_rate_rmse_rad_s | +0 | +0 | +0 | +0 |
| classical_mekf | R2_MAG_HARD_SOFT_IRON | integrated_gyro_increment_rmse_rad | +0 | +0 | +0 | +0 |
| classical_mekf | R2_MAG_HARD_SOFT_IRON | corrected_magnetometer_angular_error_rad | +0 | +0.0501633917 | +0.0501633917 | +0 |
| classical_mekf | R2_MAG_HARD_SOFT_IRON | weak_axis_rmse | +0 | +0.000352719731 | +0.000352719731 | +0 |
| classical_mekf | R2_MAG_HARD_SOFT_IRON | observable_plane_rmse | +0 | +0.0451539886 | +0.0451539886 | +0 |
| classical_mekf | R2_MAG_HARD_SOFT_IRON | divergence_count | +0 | +0 | +0 | +0 |
| classical_mekf | R3_COMBINED | attitude_geodesic_rmse_rad | +0.000256637032 | +0.0439310941 | +0.0440996825 | -8.80486877e-05 |
| classical_mekf | R3_COMBINED | residual_gyro_bias_rmse | +8.04660829e-06 | +1.00557757e-05 | +4.06819096e-05 | +2.25795256e-05 |
| classical_mekf | R3_COMBINED | corrected_gyro_rate_rmse_rad_s | +0.00129151495 | +0 | +0.00129151495 | +0 |
| classical_mekf | R3_COMBINED | integrated_gyro_increment_rmse_rad | +0.00109958451 | +0 | +0.00109958451 | +0 |
| classical_mekf | R3_COMBINED | corrected_magnetometer_angular_error_rad | +0 | +0.0627639692 | +0.0627639692 | +0 |
| classical_mekf | R3_COMBINED | weak_axis_rmse | +0.000128041526 | -0.000308379451 | -0.000257428606 | -7.70906818e-05 |
| classical_mekf | R3_COMBINED | observable_plane_rmse | +0.000151052286 | +0.0626041902 | +0.0627592394 | +3.99696519e-06 |
| classical_mekf | R3_COMBINED | divergence_count | +0 | +0 | +0 | +0 |
| fixed_n0_split_knet | R0_NOMINAL | attitude_geodesic_rmse_rad | +0 | +0 | +0 | +0 |
| fixed_n0_split_knet | R0_NOMINAL | residual_gyro_bias_rmse | +0 | +0 | +0 | +0 |
| fixed_n0_split_knet | R0_NOMINAL | corrected_gyro_rate_rmse_rad_s | +0 | +0 | +0 | +0 |
| fixed_n0_split_knet | R0_NOMINAL | integrated_gyro_increment_rmse_rad | +0 | +0 | +0 | +0 |
| fixed_n0_split_knet | R0_NOMINAL | corrected_magnetometer_angular_error_rad | +0 | +0 | +0 | +0 |
| fixed_n0_split_knet | R0_NOMINAL | weak_axis_rmse | +0 | +0 | +0 | +0 |
| fixed_n0_split_knet | R0_NOMINAL | observable_plane_rmse | +0 | +0 | +0 | +0 |
| fixed_n0_split_knet | R0_NOMINAL | divergence_count | +0 | +0 | +0 | +0 |
| fixed_n0_split_knet | R1_GYRO_BIAS_SCALE | attitude_geodesic_rmse_rad | +1.28678644e-05 | +0 | +1.28678644e-05 | +0 |
| fixed_n0_split_knet | R1_GYRO_BIAS_SCALE | residual_gyro_bias_rmse | +6.16580087e-06 | +0 | +6.16580087e-06 | +0 |
| fixed_n0_split_knet | R1_GYRO_BIAS_SCALE | corrected_gyro_rate_rmse_rad_s | +0.00102576778 | +0 | +0.00102576778 | +0 |
| fixed_n0_split_knet | R1_GYRO_BIAS_SCALE | integrated_gyro_increment_rmse_rad | +0.000891389555 | +0 | +0.000891389555 | +0 |
| fixed_n0_split_knet | R1_GYRO_BIAS_SCALE | corrected_magnetometer_angular_error_rad | +0 | +0 | +0 | +0 |
| fixed_n0_split_knet | R1_GYRO_BIAS_SCALE | weak_axis_rmse | -3.83720973e-05 | +0 | -3.83720973e-05 | +0 |
| fixed_n0_split_knet | R1_GYRO_BIAS_SCALE | observable_plane_rmse | +9.18050714e-05 | +0 | +9.18050714e-05 | +0 |
| fixed_n0_split_knet | R1_GYRO_BIAS_SCALE | divergence_count | +0 | +0 | +0 | +0 |
| fixed_n0_split_knet | R2_MAG_HARD_SOFT_IRON | attitude_geodesic_rmse_rad | +0 | +0.010351644 | +0.010351644 | +0 |
| fixed_n0_split_knet | R2_MAG_HARD_SOFT_IRON | residual_gyro_bias_rmse | +0 | +0.00202088955 | +0.00202088955 | +0 |
| fixed_n0_split_knet | R2_MAG_HARD_SOFT_IRON | corrected_gyro_rate_rmse_rad_s | +0 | +0 | +0 | +0 |
| fixed_n0_split_knet | R2_MAG_HARD_SOFT_IRON | integrated_gyro_increment_rmse_rad | +0 | +0 | +0 | +0 |
| fixed_n0_split_knet | R2_MAG_HARD_SOFT_IRON | corrected_magnetometer_angular_error_rad | +0 | +0.0501633917 | +0.0501633917 | +0 |
| fixed_n0_split_knet | R2_MAG_HARD_SOFT_IRON | weak_axis_rmse | +0 | +0.000286301551 | +0.000286301551 | +0 |
| fixed_n0_split_knet | R2_MAG_HARD_SOFT_IRON | observable_plane_rmse | +0 | +0.0145572361 | +0.0145572361 | +0 |
| fixed_n0_split_knet | R2_MAG_HARD_SOFT_IRON | divergence_count | +0 | +0 | +0 | +0 |
| fixed_n0_split_knet | R3_COMBINED | attitude_geodesic_rmse_rad | +0.000202704774 | +0.0114127598 | +0.0115063702 | -0.000109094421 |
| fixed_n0_split_knet | R3_COMBINED | residual_gyro_bias_rmse | -1.77566273e-06 | +0.00332009871 | +0.00332501254 | +6.68948971e-06 |
| fixed_n0_split_knet | R3_COMBINED | corrected_gyro_rate_rmse_rad_s | +0.00129151495 | +0 | +0.00129151495 | +0 |
| fixed_n0_split_knet | R3_COMBINED | integrated_gyro_increment_rmse_rad | +0.00109958451 | +0 | +0.00109958451 | +0 |
| fixed_n0_split_knet | R3_COMBINED | corrected_magnetometer_angular_error_rad | +0 | +0.0627639692 | +0.0627639692 | +0 |
| fixed_n0_split_knet | R3_COMBINED | weak_axis_rmse | +1.65087096e-06 | -0.000678089236 | -0.0006338675 | +4.25708644e-05 |
| fixed_n0_split_knet | R3_COMBINED | observable_plane_rmse | +0.000227014288 | +0.0192370338 | +0.0192934186 | -0.000170629526 |
| fixed_n0_split_knet | R3_COMBINED | divergence_count | +0 | +0 | +0 | +0 |

Full paired confidence intervals and seed directions are in `ORACLE_DECOMPOSITION_SUMMARY.json`.

## Scope

Classical CG/CM keep the MEKF fixed. NG/NM/NGM use the exact frozen N0 checkpoint for each seed and change only selected sensor values. Existing N1 instead uses separately trained oracle-input checkpoints. No interaction is called causal synergy without an interval excluding zero and these fixed-intervention assumptions.
