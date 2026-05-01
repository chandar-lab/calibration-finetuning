from __future__ import annotations

import math

import numpy as np
from scipy import stats

from random_steering.eval.metrics import frozen_distribution
from random_steering.types import DistributionSpec


_ANDERSON_DIST_MAP = {
    "norm": "norm",
    "expon": "expon",
    "logistic": "logistic",
    "weibull_min": "weibull_min",
}


def _chi_square_discrete(values: list[float], spec: DistributionSpec) -> dict[str, float | str]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"gof_test": "chi_square", "gof_statistic": math.nan, "gof_pvalue": math.nan}

    dist = frozen_distribution(spec)

    rounded = np.round(arr).astype(int)
    raw_lo = float(dist.ppf(0.001))
    raw_hi = float(dist.ppf(0.999))
    quantile_lo = int(math.floor(raw_lo)) if np.isfinite(raw_lo) else int(np.min(rounded))
    quantile_hi = int(math.ceil(raw_hi)) if np.isfinite(raw_hi) else int(np.max(rounded))
    lo = int(spec.support_min) if spec.support_min is not None else min(int(np.min(rounded)), quantile_lo)
    hi = int(spec.support_max) if spec.support_max is not None else max(int(np.max(rounded)), quantile_hi)

    _MAX_SUPPORT_SIZE = 10_000_000
    if hi - lo + 1 > _MAX_SUPPORT_SIZE:
        return {"gof_test": "chi_square", "gof_statistic": math.nan, "gof_pvalue": math.nan}

    support = np.arange(lo, hi + 1)

    observed = np.array([np.sum(rounded == k) for k in support], dtype=float)
    expected_probs = dist.pmf(support)

    if spec.support_min is None:
        lower_tail_obs = np.sum(rounded < support[0])
        lower_tail_prob = max(0.0, float(dist.cdf(support[0] - 1)))
        observed = np.insert(observed, 0, lower_tail_obs)
        expected_probs = np.insert(expected_probs, 0, lower_tail_prob)

    if spec.support_max is None:
        upper_tail_obs = np.sum(rounded > support[-1])
        upper_tail_prob = max(0.0, 1.0 - float(dist.cdf(support[-1])))
        observed = np.append(observed, upper_tail_obs)
        expected_probs = np.append(expected_probs, upper_tail_prob)

    expected = expected_probs * observed.sum()
    expected = np.clip(expected, 1e-12, None)
    expected *= observed.sum() / expected.sum()

    chi = stats.chisquare(f_obs=observed, f_exp=expected)
    return {
        "gof_test": "chi_square",
        "gof_statistic": float(chi.statistic),
        "gof_pvalue": float(chi.pvalue),
    }


def _ks_continuous(values: list[float], spec: DistributionSpec) -> dict[str, float | str]:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return {"gof_test": "ks", "gof_statistic": math.nan, "gof_pvalue": math.nan}
    dist = frozen_distribution(spec)
    ks = stats.kstest(arr, dist.cdf)
    return {
        "gof_test": "ks",
        "gof_statistic": float(ks.statistic),
        "gof_pvalue": float(ks.pvalue),
    }


def _anderson_if_available(values: list[float], spec: DistributionSpec) -> dict[str, float]:
    if spec.scipy_name not in _ANDERSON_DIST_MAP:
        return {}
    arr = np.asarray(values, dtype=float)
    if arr.size < 5:
        return {}
    try:
        anderson = stats.anderson(arr, dist=_ANDERSON_DIST_MAP[spec.scipy_name])
    except (FloatingPointError, ValueError, RuntimeError):
        return {}
    result = {
        "anderson_statistic": float(anderson.statistic),
        "anderson_critical_5pct": math.nan,
    }
    for level, crit in zip(anderson.significance_level, anderson.critical_values):
        if int(level) == 5:
            result["anderson_critical_5pct"] = float(crit)
            break
    return result


def run_gof_test(values: list[float], spec: DistributionSpec) -> dict[str, float | str]:
    if spec.family == "discrete":
        return _chi_square_discrete(values, spec)
    result = _ks_continuous(values, spec)
    result.update(_anderson_if_available(values, spec))
    return result
