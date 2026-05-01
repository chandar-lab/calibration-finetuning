from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    from hydra import compose, initialize_config_dir
except ImportError:
    compose = None
    initialize_config_dir = None
try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None
import torch
import torch.nn as nn

from random_steering.open_random_gen.data import load_open_random_gen_prompts
from random_steering.open_random_gen.metrics import (
    count_unique_outputs,
    has_thinking_tags,
    normalize_open_random_output,
    top_p_support_size,
)

try:
    from random_steering.open_random_gen.eval import main_impl
except ImportError:
    main_impl = None


class _TinyTokenizer:
    chat_template = None

    def __init__(self) -> None:
        self.eos_token_id = 0
        self.pad_token_id = 0
        self._vocab: dict[str, int] = {}
        self._inverse_vocab: dict[int, str] = {}
        self._next_id = 1

    @property
    def vocab_size(self) -> int:
        return self._next_id + 8

    def _token_for_char(self, char: str) -> int:
        if char not in self._vocab:
            token_id = self._next_id
            self._vocab[char] = token_id
            self._inverse_vocab[token_id] = char
            self._next_id += 1
        return self._vocab[char]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [self._token_for_char(char) for char in text]

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(self._inverse_vocab.get(int(token_id), "") for token_id in token_ids if int(token_id) != 0)

    def __call__(self, text, return_tensors: str | None = None, padding: bool = False):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        encoded = [self.encode(item) for item in texts]
        max_length = max(len(item) for item in encoded)
        if padding:
            padded = [item + ([self.pad_token_id] * (max_length - len(item))) for item in encoded]
            masks = [([1] * len(item)) + ([0] * (max_length - len(item))) for item in encoded]
        else:
            padded = encoded
            masks = [[1] * len(item) for item in encoded]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        if isinstance(text, str):
            return {"input_ids": encoded[0]}
        return {"input_ids": encoded}


class _TinyOpenRandomModel(nn.Module):
    def __init__(self, tokenizer: _TinyTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(use_cache=False)
        self.generation_config = SimpleNamespace()
        self.vocab_size = tokenizer.vocab_size
        self.a_id = tokenizer._token_for_char("A")
        self.b_id = tokenizer._token_for_char("B")
        self.c_id = tokenizer._token_for_char("C")
        self.d_id = tokenizer._token_for_char("D")
        self.e_id = tokenizer._token_for_char("E")
        self.output_options = [
            ' "Alpha" ',
            "\n\nBeta City\nsecond line",
            "'Gamma'",
            "Delta   Name",
            "</think>\nEpsilon",
            "<think>reasoning</think>\nZeta",
        ]
        for option in self.output_options:
            tokenizer.encode(option)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        _ = attention_mask
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, self.vocab_size), -20.0, dtype=torch.float32, device=input_ids.device)
        logits[:, :, self.a_id] = torch.log(torch.tensor(0.6, device=input_ids.device))
        logits[:, :, self.b_id] = torch.log(torch.tensor(0.2, device=input_ids.device))
        logits[:, :, self.c_id] = torch.log(torch.tensor(0.1, device=input_ids.device))
        logits[:, :, self.d_id] = torch.log(torch.tensor(0.05, device=input_ids.device))
        logits[:, :, self.e_id] = torch.log(torch.tensor(0.05, device=input_ids.device))
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, generation_config=None):
        _ = attention_mask, generation_config
        batch_outputs: list[list[int]] = []
        max_output_length = 0
        for _row in range(input_ids.shape[0]):
            option_index = int(torch.randint(0, len(self.output_options), (1,)).item())
            generated = [int(token_id) for token_id in self.tokenizer.encode(self.output_options[option_index])]
            batch_outputs.append(generated)
            max_output_length = max(max_output_length, len(generated))

        generated_tensor = torch.zeros((input_ids.shape[0], input_ids.shape[1] + max_output_length), dtype=torch.long, device=input_ids.device)
        generated_tensor[:, : input_ids.shape[1]] = input_ids
        for row_index, generated in enumerate(batch_outputs):
            generated_tensor[row_index, input_ids.shape[1] : input_ids.shape[1] + len(generated)] = torch.tensor(
                generated,
                dtype=torch.long,
                device=input_ids.device,
            )
        return generated_tensor


class OpenRandomGenTests(unittest.TestCase):
    def test_prompt_loader_accepts_valid_json(self) -> None:
        with TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "prompts.json"
            payload = ["Prompt one", "Prompt two", "Prompt three"]
            prompt_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_open_random_gen_prompts(prompt_path), payload)

    def test_prompt_loader_rejects_invalid_entries(self) -> None:
        with TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "prompts.json"
            prompt_path.write_text(json.dumps(["ok", " "]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                load_open_random_gen_prompts(prompt_path)

            prompt_path.write_text(json.dumps(["ok", 3]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be a string"):
                load_open_random_gen_prompts(prompt_path)

    def test_top_p_support_size(self) -> None:
        probs = torch.tensor([0.6, 0.2, 0.1, 0.1], dtype=torch.float32)
        self.assertEqual(top_p_support_size(probs, 0.9), 3)

    def test_normalize_open_random_output(self) -> None:
        text = ' \n  "New   York"  \nsecond line\n'
        self.assertEqual(normalize_open_random_output(text), "New York")

    def test_normalize_open_random_output_strips_thinking_tags_and_orphans(self) -> None:
        self.assertEqual(normalize_open_random_output("<think>reasoning</think>\nParis"), "Paris")
        self.assertEqual(normalize_open_random_output("</think>\nParis"), "Paris")
        self.assertEqual(normalize_open_random_output("Paris </think>"), "Paris")
        self.assertEqual(normalize_open_random_output("<think>reasoning only"), "")
        self.assertEqual(normalize_open_random_output("</think>"), "")
        self.assertTrue(has_thinking_tags("</think>\nParis"))
        self.assertTrue(has_thinking_tags("<think>reasoning</think>\nParis"))
        self.assertFalse(has_thinking_tags("Paris"))

    def test_count_unique_outputs(self) -> None:
        outputs = ["alpha", "", "beta", "alpha", "gamma"]
        num_unique, unique_fraction, counts = count_unique_outputs(outputs)
        self.assertEqual(num_unique, 3)
        self.assertAlmostEqual(unique_fraction, 0.6)
        self.assertEqual(counts, {"alpha": 2, "beta": 1, "gamma": 1})

    def test_open_random_gen_config_composes(self) -> None:
        if initialize_config_dir is None or compose is None:
            self.skipTest("hydra is not installed in this environment")
        repo_root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base=None, config_dir=str(repo_root / "conf")):
            cfg = compose(config_name="open_random_gen_eval_config")
        self.assertEqual(cfg.eval_target.name, "open_random_gen_baseline")
        self.assertEqual(cfg.open_random_gen.top_p_threshold, 0.9)

    def test_open_random_gen_eval_writes_outputs(self) -> None:
        if OmegaConf is None or main_impl is None:
            self.skipTest("omegaconf/hydra stack is not installed in this environment")
        tokenizer = _TinyTokenizer()
        model = _TinyOpenRandomModel(tokenizer)

        with TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "prompts.json"
            prompt_path.write_text(
                json.dumps(
                    [
                        "Think of a random word. Output ONLY the answer.",
                        "Choose a random city. Output ONLY the answer.",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = OmegaConf.create(
                {
                    "seed": 11,
                    "output_root": tmpdir,
                    "model": {
                        "device": "cpu",
                        "dtype": "float32",
                        "enable_thinking": False,
                    },
                    "open_random_gen": {
                        "prompts_path": str(prompt_path),
                        "top_p_threshold": 0.9,
                        "num_samples_per_prompt": 4,
                        "batch_size": 2,
                        "generation": {
                            "max_new_tokens": 16,
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "do_sample": True,
                        },
                        "output_root": tmpdir,
                    },
                    "eval_target": {
                        "name": "open_random_gen_test",
                        "base_checkpoint": "base",
                        "adapter_checkpoint": None,
                        "tokenizer_checkpoint": "tokenizer",
                        "local_files_only": True,
                    },
                }
            )

            bundle = SimpleNamespace(model=model, tokenizer=tokenizer)
            with patch("random_steering.open_random_gen.eval.load_evaluation_bundle", return_value=bundle):
                run_dir = main_impl(cfg)

            self.assertTrue((run_dir / "config_resolved.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "metrics" / "summary.json").exists())
            self.assertTrue((run_dir / "metrics" / "per_prompt.csv").exists())
            samples_path = run_dir / "samples" / "per_prompt.jsonl"
            self.assertTrue(samples_path.exists())

            with (run_dir / "metrics" / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["num_prompts"], 2)
            self.assertEqual(summary["num_samples_per_prompt"], 4)

            with samples_path.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(rows[0]["normalized_outputs"]), 4)
            self.assertIn("output_counts", rows[0])
            self.assertIn("raw_outputs", rows[0])
            self.assertIn("empty_output_count", rows[0])
            self.assertIn("had_thinking_tags_count", rows[0])
            self.assertIn("total_outputs_with_thinking_tags", summary)


if __name__ == "__main__":
    unittest.main()
