# SC-DR0 Design Review

Decision: `GO_TO_INDEPENDENT_DR0_AUDIT`.

The live repository establishes the required right-local MEKF convention, gyro propagation role, magnetometer innovation, and 6x3 Split gain construction. SC-00 states the deterministic-calibration versus residual-bias role and freezes the causal information boundary; SC-01 makes that split identifiable by a population gauge anchor, scale separation, and excitation certificate. SC-01 numerically freezes R0-R4 supports, C0/C1/N0-N3S identities, paired populations, statistics, and G0-G4 predicates before test access. SC-02 prevents reuse of incompatible Euclidean or frozen main-Phase-2 checkpoints.

The method is coherent and minimally falsifiable. This is design authorization only: no neural code or numeric performance claim exists yet. Independent audit must verify the sealed manifest before implementation.
