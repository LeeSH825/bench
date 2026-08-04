# P1 Exit Transient Diagnostic Report

Status: `COMPLETE`. Only independent calibration train/validation data are used.

| Split/partition | Full NEES/DOF | Attitude marginal | Bias marginal | Mag NIS | Sun NIS | ST NIS |
|---|---:|---:|---:|---:|---:|---:|
| train/whole | 4.9527258 | 3.2971971 | 6.1939313 | 4.4022856 | 1.9759765 | 1.3108978 |
| train/initial | 15.593452 | 9.0103784 | 17.098607 | 18.248866 | 6.2905356 | 2.6311616 |
| train/middle | 3.2212925 | 2.4266739 | 4.984381 | 1.1123751 | 1.0224531 | 1.1312626 |
| train/settled | 1.6254608 | 1.4516882 | 2.2320441 | 0.99968243 | 1.0207627 | 0.94042306 |
| validation/whole | 5.0995461 | 3.3200384 | 6.4880154 | 2.49117 | 2.0799836 | 1.4097917 |
| validation/initial | 15.558045 | 9.2018933 | 17.058789 | 8.6099599 | 6.8034536 | 2.610339 |
| validation/middle | 3.324323 | 2.4153759 | 5.2234901 | 1.0467622 | 1.0150724 | 1.2649146 |
| validation/settled | 1.9062451 | 1.4348135 | 2.7448526 | 0.97816274 | 1.0550646 | 1.0544408 |

## Whitened and cross-covariance evidence

- Settled whitened coordinate energy: `[1.8120471653201071, 1.1576736797520144, 1.3347195461425456, 2.4093306107175647, 2.5772939491941744, 2.146405929513212]`.
- Settled whitened attitude/bias grouped energy: `4.3044404` / `7.1330305`.
- Settled mean relative attitude-bias P cross norm: `0.55955034`.
- Settled correlation-normalized P cross block: `[[-0.4913718242966231, -0.014319455913444685, 0.017662641567675552], [0.04426487515969785, -0.49718950896941305, -0.03581105162295347], [-0.04705352195304158, 0.03171902551845661, -0.5149375927939868]]`.
- Settled whitened attitude-bias cross-correlation: `[[0.2544054039183805, 0.008528800338887014, 0.0674861142123178], [0.025697889981751532, 0.3679267729488537, -0.052599595135859364], [0.17490783879524266, -0.06259128919915768, 0.34572349930265117]]`.
- Predeclared validation settling-bin result: `0.4` horizon fraction.

The likely-source ordering is based on train/validation marginal diagnostics only: `['bias_marginal', 'attitude_marginal']`. Sensor R remains fixed, so matched sensor NIS and posterior inconsistency are reported separately.

Limitations: the decomposition is for the representative normalized MAIN-FUSION benchmark; it is not an orbit, WMM, eclipse, flight-sensor, or universal calibration claim.
