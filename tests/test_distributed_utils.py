from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from random_steering.utils.distributed import (
    DistributedRuntime,
    get_local_rank,
    initialize_distributed,
    is_distributed_enabled,
    is_main_process,
)


class DistributedUtilsTests(unittest.TestCase):
    def test_distributed_disabled_uses_single_process_defaults(self) -> None:
        runtime = initialize_distributed(type("Cfg", (), {"enabled": False})(), device_name="cpu")
        self.assertEqual(
            runtime,
            DistributedRuntime(enabled=False, rank=0, local_rank=0, world_size=1, device=torch.device("cpu")),
        )

    def test_get_local_rank_reads_env(self) -> None:
        with patch.dict(os.environ, {"LOCAL_RANK": "3"}, clear=True):
            self.assertEqual(get_local_rank(), 3)

    def test_is_distributed_enabled_reads_cfg(self) -> None:
        self.assertTrue(is_distributed_enabled(type("Cfg", (), {"enabled": True})()))
        self.assertFalse(is_distributed_enabled(type("Cfg", (), {"enabled": False})()))

    def test_main_process_default_without_dist(self) -> None:
        self.assertTrue(is_main_process())


if __name__ == "__main__":
    unittest.main()
