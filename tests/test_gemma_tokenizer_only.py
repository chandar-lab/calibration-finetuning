from __future__ import annotations

import os
from pathlib import Path
import unittest

from random_steering.inference.chat_format import format_prompt
from random_steering.utils.hf import ensure_hf_home

try:
    from huggingface_hub import login
except ImportError:
    login = None

try:
    from transformers import AutoProcessor, AutoTokenizer
except ImportError:
    AutoProcessor = None
    AutoTokenizer = None


def _resolve_hf_token() -> str | None:
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token.strip()

    candidate = Path(__file__).resolve().parents[2] / "hf_token.txt"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip()

    return None


class GemmaTokenizerOnlyTests(unittest.TestCase):
    def _assert_text_only_prompt_matches_processor(self, model_id: str) -> None:
        if login is None or AutoTokenizer is None or AutoProcessor is None:
            self.skipTest("huggingface_hub/transformers are not installed in this environment")

        token = _resolve_hf_token()
        if not token:
            self.skipTest("HF token is not available")

        ensure_hf_home()
        login(token=token, add_to_git_credential=False)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)

        prompt = "Say hello."
        tokenizer_rendered = format_prompt(tokenizer, prompt, enable_thinking=None)
        processor_rendered = processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        self.assertEqual(tokenizer_rendered, processor_rendered)

        tokenized = tokenizer(tokenizer_rendered, return_tensors="pt")
        self.assertIn("input_ids", tokenized)
        self.assertGreater(int(tokenized["input_ids"].numel()), 0)

    def test_gemma_3_4b_text_prompt_matches_processor_rendering(self) -> None:
        self._assert_text_only_prompt_matches_processor("google/gemma-3-4b-it")

    def test_gemma_3_12b_text_prompt_matches_processor_rendering(self) -> None:
        self._assert_text_only_prompt_matches_processor("google/gemma-3-12b-it")


if __name__ == "__main__":
    unittest.main()
