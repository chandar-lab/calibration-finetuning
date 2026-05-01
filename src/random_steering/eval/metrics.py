from __future__ import annotations

import math

import numpy as np
from scipy import stats

from random_steering.types import DistributionSpec


def frozen_distribution(spec: DistributionSpec):
    constructor = getattr(stats, spec.scipy_name)
    return constructor(**spec.scipy_params)


def wasserstein_1_to_target(values: list[float], spec: DistributionSpec) -> float:
    if not values:
        return math.inf
    dist = frozen_distribution(spec)
    sample = np.sort(np.asarray(values, dtype=float))
    n = sample.shape[0]
    quantiles = (np.arange(n, dtype=float) + 0.5) / float(n)
    target = dist.ppf(quantiles)
    finite_mask = np.isfinite(sample) & np.isfinite(target)
    if not np.any(finite_mask):
        return math.inf
    return float(np.mean(np.abs(sample[finite_mask] - target[finite_mask])))


def moment_errors(values: list[float], spec: DistributionSpec) -> dict[str, float]:
    if not values:
        return {
            "sample_mean": math.nan,
            "sample_variance": math.nan,
            "mean_error": math.nan,
            "variance_error": math.nan,
            "wasserstein_1": math.inf,
        }

    dist = frozen_distribution(spec)
    arr = np.asarray(values, dtype=float)
    sample_mean = float(np.mean(arr))
    sample_var = float(np.var(arr))

    target_mean, target_var = dist.stats(moments="mv")
    target_mean_f = float(target_mean) if np.isfinite(target_mean) else math.nan
    target_var_f = float(target_var) if np.isfinite(target_var) else math.nan

    mean_error = abs(sample_mean - target_mean_f) if np.isfinite(target_mean_f) else math.nan
    variance_error = abs(sample_var - target_var_f) if np.isfinite(target_var_f) else math.nan

    return {
        "sample_mean": sample_mean,
        "sample_variance": sample_var,
        "mean_error": float(mean_error),
        "variance_error": float(variance_error),
        "wasserstein_1": wasserstein_1_to_target(values, spec),
    }
