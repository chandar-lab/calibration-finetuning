from __future__ import annotations

import os


def ensure_hf_home() -> None:
    if os.environ.get("HF_HOME"):
        return

    scratch_dir = os.environ.get("SCRATCH")
    if scratch_dir:
        os.environ["HF_HOME"] = scratch_dir
