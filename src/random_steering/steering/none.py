from __future__ import annotations

from random_steering.steering.base import SteeringPolicy
from random_steering.types import DistributionSpec


class NoneSteeringPolicy(SteeringPolicy):
    def setup(self, model, tokenizer, cfg) -> None:
        _ = (model, tokenizer, cfg)

    def on_request_start(self, target_distribution: DistributionSpec, request_seed: int) -> None:
        _ = (target_distribution, request_seed)

    def install_hooks(self) -> None:
        return

    def remove_hooks(self) -> None:
        return
