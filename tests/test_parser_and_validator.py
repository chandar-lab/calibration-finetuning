from __future__ import annotations

import unittest

from random_steering.distributions.parser import parse_batch_numbers, parse_single_number
from random_steering.distributions.registry import get_distribution
from random_steering.distributions.validators import validate_value


class ParserValidatorTests(unittest.TestCase):
    def test_parse_single_strict_success(self) -> None:
        parsed = parse_single_number(" 1.23e-2 ")
        self.assertTrue(parsed.valid)
        self.assertAlmostEqual(parsed.value or 0.0, 0.0123)

    def test_parse_single_strict_failure(self) -> None:
        parsed = parse_single_number("1.0 and 2.0")
        self.assertFalse(parsed.valid)
        self.assertEqual(parsed.error_code, "malformed")

    def test_parse_batch(self) -> None:
        parsed = parse_batch_numbers("1, 2, 3")
        self.assertEqual(len(parsed), 3)
        self.assertTrue(all(p.valid for p in parsed))

    def test_validator_support_and_integer(self) -> None:
        bern = get_distribution("bernoulli_0_7")
        ok, err = validate_value(bern, 1.0)
        self.assertTrue(ok)
        self.assertIsNone(err)

        ok2, err2 = validate_value(bern, 0.5)
        self.assertFalse(ok2)
        self.assertEqual(err2, "non_integer")


if __name__ == "__main__":
    unittest.main()
