"""
bench.metrics

Step 6 MVP에서 필요한 최소 메트릭 구현:
- MSE / RMSE / MSE(dB)
- shift recovery_k
- (optional) Gaussian NLL (cov 제공 시)
"""

from .core import (
    mse_per_step,
    mse_scalar,
    rmse_scalar,
    mse_db_scalar,
    compute_shift_recovery_k,
    gaussian_nll_per_step,
)

