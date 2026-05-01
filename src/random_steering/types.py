from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DistributionSpec:
    distribution_id: str
    display_name: str
    parameters: dict[str, float]
    family: str
    support_min: float | None
    support_max: float | None
    integer_only: bool
    scipy_name: str
    scipy_params: dict[str, float]


@dataclass
class SampleRecord:
    protocol: str
    distribution_id: str
    request_index: int
    seed: int
    prompt: str
    raw_text: str
    parsed_value: float | None
    is_valid: bool
    support_ok: bool
    error_code: str | None
    steering_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricRecord:
    distribution_id: str
    protocol: str
    steering_name: str
    seed: int
    metric_name: str
    value: float
