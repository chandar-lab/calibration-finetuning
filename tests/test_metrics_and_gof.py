from __future__ import annotations

import importlib.util
import unittest

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None

if SCIPY_AVAILABLE:
    import numpy as np

    from random_steering.distributions.registry import get_distribution
    from random_steering.eval.metrics import moment_errors, wasserstein_1_to_target
    from random_steering.eval.tests import run_gof_test


@unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for metrics/goF tests")
class MetricsAndGofTests(unittest.TestCase):
    def test_gaussian_metrics_contract(self) -> None:
        spec = get_distribution("gaussian_0_1")
        rng = np.random.default_rng(123)
        values = rng.normal(size=400).tolist()

        metrics = moment_errors(values, spec)
        self.assertIn("mean_error", metrics)
        self.assertIn("variance_error", metrics)
        self.assertGreaterEqual(wasserstein_1_to_target(values, spec), 0.0)

    def test_discrete_gof_contract(self) -> None:
        spec = get_distribution("binomial_10_0_5")
        rng = np.random.default_rng(123)
        values = rng.binomial(n=10, p=0.5, size=500).astype(float).tolist()

        result = run_gof_test(values, spec)
        self.assertEqual(result["gof_test"], "chi_square")
        pvalue = result["gof_pvalue"]
        self.assertTrue(0.0 <= float(pvalue) <= 1.0)

    def test_weibull_anderson_failure_does_not_crash(self) -> None:
        spec = get_distribution("weibull_1_5_1")
        values = [1.0] * 64

        result = run_gof_test(values, spec)
        self.assertEqual(result["gof_test"], "ks")
        self.assertIn("gof_pvalue", result)
        self.assertNotIn("anderson_statistic", result)


if __name__ == "__main__":
    unittest.main()
