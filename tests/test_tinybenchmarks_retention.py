from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
import torch.nn as nn

from random_steering.retention import tinybenchmarks_data as data
from random_steering.retention.tinybenchmarks_eval import run_retention_eval
from random_steering.retention.tinybenchmarks_metrics import evaluate_tinybenchmarks
from random_steering.retention.tinybenchmarks_tasks import (
    extract_last_numeric_answer,
    extract_strict_gsm8k_answer,
    run_multiple_choice_task,
    score_continuation,
)


class _TinyTokenizer:
    chat_template = None

    def __init__(self) -> None:
        self.eos_token_id = 0
        self.pad_token_id = 0
        self._vocab: dict[str, int] = {}
        self._inverse_vocab: dict[int, str] = {}
        self._next_id = 1

    def _token_for_char(self, char: str) -> int:
        if char not in self._vocab:
            token_id = self._next_id
            self._vocab[char] = token_id
            self._inverse_vocab[token_id] = char
            self._next_id += 1
        return self._vocab[char]

    @property
    def vocab_size(self) -> int:
        return self._next_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [self._token_for_char(char) for char in text]

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(self._inverse_vocab.get(int(token_id), "") for token_id in token_ids if int(token_id) != 0)

    def __call__(self, text: str, return_tensors: str | None = None):
        ids = self.encode(text)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
        return {"input_ids": ids}


class _TransitionLM(nn.Module):
    def __init__(self, vocab_size: int, *, space_id: int, a_id: int, b_id: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(use_cache=False)
        transition = torch.zeros(vocab_size, vocab_size, dtype=torch.float32)
        transition[space_id, a_id] = 5.0
        transition[space_id, b_id] = 1.0
        self.register_buffer("transition", transition)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        _ = attention_mask
        logits = self.transition[input_ids]
        return SimpleNamespace(logits=logits)


def _repeat(row: dict, count: int = 100) -> list[dict]:
    return [dict(row) for _ in range(count)]


class TinyBenchmarksRetentionTests(unittest.TestCase):
    def test_schema_snapshot_fixture_matches_code(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        fixture_path = repo_root / "tests" / "fixtures" / "tinybenchmarks_schema_snapshots.json"
        with fixture_path.open("r", encoding="utf-8") as handle:
            fixture = json.load(handle)
        self.assertEqual(fixture, data.SCHEMA_SNAPSHOTS)

    def test_loaders_use_exact_dataset_calls_and_normalize(self) -> None:
        cases = {
            "tiny_mmlu": (
                data.load_tiny_mmlu,
                {
                    "question": "Q",
                    "subject": "math",
                    "choices": ["x", "y", "z", "w"],
                    "answer": 2,
                    "input_formatted": "Prompt",
                },
            ),
            "tiny_hellaswag": (
                data.load_tiny_hellaswag,
                {
                    "ind": 7,
                    "activity_label": "label",
                    "ctx_a": "a",
                    "ctx_b": "b",
                    "ctx": "ctx",
                    "endings": ["e0", "e1", "e2", "e3"],
                    "source_id": "src",
                    "split": "val",
                    "split_type": "indomain",
                    "label": "1",
                    "input_formatted": "Prompt",
                },
            ),
            "tiny_truthfulqa": (
                data.load_tiny_truthfulqa,
                {
                    "question": "Q",
                    "mc1_targets": {"choices": ["true", "false"], "labels": [1, 0]},
                    "mc2_targets": {"choices": ["true", "false"], "labels": [1, 0]},
                    "input_formatted": "Prompt",
                },
            ),
            "tiny_winogrande": (
                data.load_tiny_winogrande,
                {
                    "sentence": "A _ B",
                    "option1": "Alice",
                    "option2": "Bob",
                    "answer": "2",
                    "input_formatted": "Prompt",
                },
            ),
            "tiny_gsm8k": (
                data.load_tiny_gsm8k,
                {
                    "question": "How many?",
                    "answer": "work\n#### 42",
                    "input_formatted": "unused",
                },
            ),
        }

        for task_name, (loader_fn, row) in cases.items():
            calls: list[tuple[tuple, dict]] = []

            def loader(*args, **kwargs):
                calls.append((args, kwargs))
                return _repeat(row)

            with self.subTest(task_name=task_name):
                examples = loader_fn(loader=loader)
                snapshot = data.SCHEMA_SNAPSHOTS[task_name]
                self.assertEqual(calls, [(tuple(snapshot["dataset_args"]), dict(snapshot["dataset_kwargs"]))])
                self.assertEqual(len(examples), 100)
                self.assertEqual(examples[0].metadata["row_index"], 0)
                self.assertEqual(examples[-1].metadata["row_index"], 99)

    def test_loader_enforces_exact_dataset_size(self) -> None:
        row = {
            "question": "Q",
            "subject": "math",
            "choices": ["x", "y", "z", "w"],
            "answer": 2,
            "input_formatted": "Prompt",
        }

        def loader(*args, **kwargs):
            _ = args, kwargs
            return _repeat(row, count=99)

        with self.assertRaisesRegex(ValueError, "expected 100 examples"):
            data.load_tiny_mmlu(loader=loader)

    def test_choice_scoring_prefers_higher_logprob_label(self) -> None:
        tokenizer = _TinyTokenizer()
        prompt = "Prompt?"
        tokenizer.encode(prompt + " A")
        tokenizer.encode(prompt + " B")
        model = _TransitionLM(
            tokenizer.vocab_size,
            space_id=tokenizer._vocab[" "],
            a_id=tokenizer._vocab["A"],
            b_id=tokenizer._vocab["B"],
        )
        score_a = score_continuation(model, tokenizer, prompt, " A")
        score_b = score_continuation(model, tokenizer, prompt, " B")
        self.assertGreater(score_a, score_b)

    def test_gsm8k_extractors(self) -> None:
        self.assertEqual(extract_strict_gsm8k_answer("work\n#### 1,234.00"), "1234")
        self.assertIsNone(extract_strict_gsm8k_answer("No marker here"))
        self.assertEqual(extract_last_numeric_answer("steps 10 then final 12.50"), "12.5")

    def test_vendored_estimator_matches_reference_values(self) -> None:
        vector = [1 if index % 3 == 0 else 0 for index in range(100)]
        metrics = evaluate_tinybenchmarks(vector, "mmlu")
        self.assertAlmostEqual(metrics["irt"], 0.3231872855436154, places=9)
        self.assertAlmostEqual(metrics["pirt"], 0.3388445659317948, places=9)
        self.assertAlmostEqual(metrics["gpirt"], 0.3371074242933746, places=9)

    def test_vendored_estimator_requires_length_100(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 100"):
            evaluate_tinybenchmarks([1, 0, 1], "mmlu")

    def test_retention_eval_writes_outputs(self) -> None:
        tokenizer = _TinyTokenizer()
        prompt = "Prompt?"
        tokenizer.encode(prompt + " A")
        tokenizer.encode(prompt + " B")
        model = _TransitionLM(
            tokenizer.vocab_size,
            space_id=tokenizer._vocab[" "],
            a_id=tokenizer._vocab["A"],
            b_id=tokenizer._vocab["B"],
        )

        examples = [
            data.MultipleChoiceExample(
                task_name="tiny_mmlu",
                example_id=f"tiny_mmlu_{index:03d}",
                prompt=prompt,
                choices=("choice_a", "choice_b"),
                correct_choice_index=0,
                metadata={"row_index": index, "row": {"question": "Q"}},
            )
            for index in range(100)
        ]
        retention_cfg = OmegaConf.create(
            {
                "enabled_tasks": ["tiny_mmlu"],
                "max_examples_per_task": None,
                "use_chat_template": False,
                "generation": {"max_new_tokens": 8, "temperature": 0.0, "top_p": 1.0},
            }
        )
        full_cfg = OmegaConf.create(
            {
                "seed": 11,
                "model": {"enable_thinking": False},
                "retention": OmegaConf.to_container(retention_cfg, resolve=True),
                "eval_target": {"name": "tinybenchmarks_baseline"},
            }
        )

        with TemporaryDirectory() as tmpdir:
            with patch(
                "random_steering.retention.tinybenchmarks_eval.load_enabled_tasks",
                return_value=[("tiny_mmlu", "mmlu", examples)],
            ):
                summary = run_retention_eval(
                    model=model,
                    tokenizer=tokenizer,
                    model_cfg=SimpleNamespace(enable_thinking=False),
                    retention_cfg=retention_cfg,
                    eval_target_cfg=SimpleNamespace(name="tinybenchmarks_baseline"),
                    output_dir=Path(tmpdir),
                    full_cfg=full_cfg,
                )

            self.assertEqual(summary["num_tasks"], 1)
            self.assertTrue((Path(tmpdir) / "config_resolved.json").exists())
            self.assertTrue((Path(tmpdir) / "summary.json").exists())
            self.assertTrue((Path(tmpdir) / "task_summary.csv").exists())
            per_example_path = Path(tmpdir) / "per_example" / "tiny_mmlu.jsonl"
            self.assertTrue(per_example_path.exists())
            with per_example_path.open("r", encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in handle), 100)

    def test_retention_config_composes(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base=None, config_dir=str(repo_root / "conf")):
            cfg = compose(config_name="retention_eval_config")
        self.assertEqual(cfg.eval_target.name, "tinybenchmarks_baseline")
        self.assertIn("tiny_mmlu", list(cfg.retention.enabled_tasks))


if __name__ == "__main__":
    unittest.main()
