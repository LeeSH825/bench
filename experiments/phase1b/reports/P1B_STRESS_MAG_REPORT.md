# Phase 1B STRESS-MAG Report

Status: `PASS_CHARACTERIZED`, paired held-out `N=50`, gyro+mag only.

F-BASE had zero numerical divergence. Mean attitude RMSE was `0.197681 rad`,
mean bias vector RMSE was `0.00302585 rad/s`, mag NIS normalized mean was
`2.85737`, and NEES normalized mean was `33.2369`. The all-one oracle was
exactly equal to F-BASE.

The required weak-direction evidence is decisive: mean attitude error RMS
parallel to the predicted magnetic axis was `0.195676 rad`, whereas the
observable-plane RMS was `0.00133072 rad`. The large overall/NEES values are
therefore reported as the expected single-vector weak/unobservable rotation
direction, not a numerical failure and not evidence of full attitude
observability. No sun or star-tracker evidence was synthesized in this case.

This result characterizes the controlled normalized-reference benchmark only.
It does not assert that arbitrary orbital motion, a real geomagnetic profile,
or a learned model can recover the missing instantaneous degree of freedom.
