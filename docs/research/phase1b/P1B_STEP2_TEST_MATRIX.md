# Phase 1B Step 2 Test Matrix

The new tests cover exact dtype/shape/immutability, payload ownership,
zero-latency and four-sensor ordering, deterministic serialization and semantic
hashes, strict generator identity, corruption rejection, whole-trajectory
splits, reference geometry, sensor equations, magnetometer and sun finite-
difference Jacobians, sun-noise statistics, validity skip behavior, strict SPD
failures, q/-q invariance, gyro+ST reduction, all-one equality, truth-free fixed
APIs, forward-only oracle use, sensor-specific NIS/counts, paired bootstrap,
N>=50 guards, scenario stream pairing, intended-intervention isolation, and the
absence of any neural path.

The exit run additionally executes every frozen Phase 1A group, the Phase 1A
fresh/cache smoke, all Phase 1B Step 1 tests and read-only validate/report exact
checks, and the legacy regression. Expected values and tolerances are not
changed to obtain a pass.
