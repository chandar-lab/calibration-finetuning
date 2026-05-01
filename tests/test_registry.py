from __future__ import annotations

import unittest

from random_steering.distributions.registry import all_distribution_ids, get_distribution, resolve_distributions


class RegistryTests(unittest.TestCase):
    def test_registry_has_15_distributions(self) -> None:
        self.assertEqual(len(all_distribution_ids()), 15)

    def test_resolve_all(self) -> None:
        specs = resolve_distributions("all")
        self.assertEqual(len(specs), 15)

    def test_get_distribution(self) -> None:
        spec = get_distribution("gaussian_0_1")
        self.assertEqual(spec.scipy_name, "norm")


if __name__ == "__main__":
    unittest.main()
