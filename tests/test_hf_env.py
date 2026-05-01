from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from random_steering.utils.hf import ensure_hf_home


class HfEnvTests(unittest.TestCase):
    def test_sets_hf_home_from_scratch_when_unset(self) -> None:
        with patch.dict(os.environ, {"SCRATCH": "/tmp/scratch-root"}, clear=True):
            ensure_hf_home()
            self.assertEqual(os.environ.get("HF_HOME"), "/tmp/scratch-root")

    def test_preserves_existing_hf_home(self) -> None:
        with patch.dict(os.environ, {"SCRATCH": "/tmp/scratch-root", "HF_HOME": "/tmp/custom-hf"}, clear=True):
            ensure_hf_home()
            self.assertEqual(os.environ.get("HF_HOME"), "/tmp/custom-hf")

    def test_leaves_hf_home_unset_without_scratch(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            ensure_hf_home()
            self.assertNotIn("HF_HOME", os.environ)


if __name__ == "__main__":
    unittest.main()
