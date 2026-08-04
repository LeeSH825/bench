# Phase 1B Sensor Fusion Contract

The Step 2 schema is `p1b-mekf-fusion-events-v1`; its generator identity is
`basilisk-sensor-fusion-regimes-v1`, and it retains the active scalar-first
Hamilton B-to-N/right-error convention. Sensor codes are gyro `1`, star tracker
`2`, magnetometer `3`, and sun `4`; simultaneous processing order is strictly
gyro, magnetometer, sun, star tracker.

The physical artifact contains typed zero-latency events and separate truth.
Gyro, star-tracker, magnetometer, and sun payload tables are disjoint float64
struct-of-arrays. Magnetometer payloads use representative-normalized 3D body
vectors and a full-rank inlier Gaussian covariance. Sun payloads use unit body
vectors and a 2D tangent covariance. Invalid sun events remain nonzero unit
measurements with `valid=false` and are skipped. Object arrays, pickle, dense
float32 coercion, zero-filled pseudo-measurements, covariance repair, inverse,
pseudo-inverse, jitter, and clipping are forbidden.

Basilisk supplies only spherical-inertia, zero-torque rigid-body attitude/rate
truth. `normalized-magnetic-reference-v1`, `normalized-sun-reference-v1`, and
`deterministic-fov-eclipselike-v1` are deterministic parameterized benchmark
providers, not WMM, orbit, eclipse, or flight-environment facts. The model and
true reference fields remain separate even in the primary matched case.

The simulation-only oracle sidecar has a distinct file, manifest, and semantic
hash. It carries only current-event `alpha_b`, `alpha_R_mag`, window, and
scenario labels through a forward-only cursor. Fixed and tuned replay APIs do
not accept it. Truth is joined only after estimation by exact trajectory ID and
float64 timestamp.

C4 locks slow bias-random-walk intensity `alpha_b=100000` on `[0.2T,0.8T)` and
fast magnetometer inlier covariance `alpha_R_mag=16` on `[0.45T,0.6T)`. The
large dimensionless bias multiplier is a normalized stress setting relative to
the deliberately small `1e-12 rad^2/s^3` base PSD, not a hardware claim.
