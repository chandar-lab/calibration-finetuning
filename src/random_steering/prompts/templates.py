from __future__ import annotations

from random_steering.types import DistributionSpec


def _format_params(parameters: dict[str, float]) -> str:
    return ", ".join(f"{k}={v}" for k, v in parameters.items())


def build_batch_prompt(spec: DistributionSpec, num_samples: int) -> str:
    return (
        "You are a random number generator. "
        f"Generate exactly {num_samples} independent samples from a "
        f"{spec.display_name} distribution with parameters {_format_params(spec.parameters)}. "
        "Output ONLY the numbers, separated by commas."
    )


def build_independent_prompt(spec: DistributionSpec) -> str:
    return (
        "Generate exactly ONE random number from a "
        f"{spec.display_name} distribution with parameters {_format_params(spec.parameters)}. "
        "Output ONLY the number."
    )
