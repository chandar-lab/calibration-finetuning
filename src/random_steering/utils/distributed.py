from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch


@dataclass(frozen=True)
class DistributedRuntime:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def is_distributed_enabled(cfg: Any | None) -> bool:
    return bool(getattr(cfg, "enabled", False))


def is_distributed_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    if is_distributed_initialized():
        return int(torch.distributed.get_rank())
    return 0


def get_world_size() -> int:
    if is_distributed_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed_initialized():
        torch.distributed.barrier()


def reduce_mean(value: float | torch.Tensor, *, device: torch.device) -> float:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    if is_distributed_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor /= float(get_world_size())
    return float(tensor.item())


def initialize_distributed(distributed_cfg: Any | None, *, device_name: str) -> DistributedRuntime:
    if not is_distributed_enabled(distributed_cfg):
        return DistributedRuntime(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device(device_name),
        )

    if device_name != "cuda":
        raise ValueError("Distributed training currently requires model.device=cuda.")

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Distributed training requires torchrun to set LOCAL_RANK.")

    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)

    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")

    if not torch.distributed.is_initialized():
        backend = str(getattr(distributed_cfg, "backend", "nccl"))
        torch.distributed.init_process_group(backend=backend)

    return DistributedRuntime(
        enabled=True,
        rank=get_rank(),
        local_rank=local_rank,
        world_size=get_world_size(),
        device=torch.device("cuda", local_rank),
    )


def cleanup_distributed() -> None:
    if is_distributed_initialized():
        torch.distributed.destroy_process_group()
