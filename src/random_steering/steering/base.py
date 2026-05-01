from __future__ import annotations

from abc import ABC, abstractmethod

from random_steering.types import DistributionSpec


class SteeringPolicy(ABC):
    @abstractmethod
    def setup(self, model, tokenizer, cfg) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_request_start(self, target_distribution: DistributionSpec, request_seed: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def install_hooks(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove_hooks(self) -> None:
        raise NotImplementedError
