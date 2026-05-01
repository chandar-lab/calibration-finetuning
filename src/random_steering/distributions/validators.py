from __future__ import annotations

import math

from random_steering.types import DistributionSpec


def is_integer_like(value: float, tol: float = 1e-9) -> bool:
    return abs(value - round(value)) <= tol


def validate_value(spec: DistributionSpec, value: float) -> tuple[bool, str | None]:
    if math.isnan(value) or math.isinf(value):
        return False, "non_finite"
    if spec.support_min is not None and value < spec.support_min:
        return False, "below_support"
    if spec.support_max is not None and value > spec.support_max:
        return False, "above_support"
    if spec.integer_only and not is_integer_like(value):
        return False, "non_integer"
    return True, None
