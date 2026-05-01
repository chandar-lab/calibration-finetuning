from __future__ import annotations

import gzip
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

from random_steering.perplexity.data import iter_paloma_texts, resolve_configured_slices, resolve_paloma_files
from random_steering.perplexity.evaluate import run_perplexity_eval
from random_steering.perplexity.eval import main_impl


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

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [self._token_for_char(char) for char in text]


class _TinyTransitionLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(use_cache=False)
        self.vocab_size = max(vocab_size, 32)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        _ = attention_mask
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, self.vocab_size), -8.0, dtype=torch.float32, device=input_ids.device)
        for batch_index in range(batch_size):
            for time_index in range(seq_len):
                target_id = int(input_ids[batch_index, time_index].item())
                logits[batch_index, time_index, target_id] = 4.0
        return SimpleNamespace(logits=logits)


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


class PerplexityEvalTests(unittest.TestCase):
    def test_paloma_loader_resolves_and_streams_texts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "paloma"
            test_path = dataset_root / "slice_a" / "test" / "test.jsonl.gz"
            _write_jsonl_gz(
                test_path,
                [
                    {"id": "0", "text": "alpha beta"},
                    {"id": "1", "text": "  "},
                    {"id": "2", "text": "gamma"},
                ],
            )

            files = resolve_paloma_files(dataset_root, "slice_a", "test")
            self.assertEqual(files, [test_path])
            resolved = resolve_configured_slices(dataset_root, ["slice_a"], "test")
            self.assertEqual(resolved["slice_a"], [test_path])
            texts = list(iter_paloma_texts(dataset_root, "slice_a", "test"))
            self.assertEqual(texts, ["alpha beta", "gamma"])

    def test_perplexity_eval_writes_outputs(self) -> None:
        tokenizer = _TinyTokenizer()
        _ = tokenizer.encode("alpha beta gamma delta")
        model = _TinyTransitionLM(vocab_size=tokenizer._next_id + 4)

        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "paloma"
            _write_jsonl_gz(
                dataset_root / "slice_a" / "test" / "test.jsonl.gz",
                [
                    {"id": "0", "text": "alpha beta"},
                    {"id": "1", "text": "gamma delta"},
                ],
            )
            _write_jsonl_gz(
                dataset_root / "slice_b" / "test" / "test.jsonl.gz",
                [
                    {"id": "0", "text": "epsilon zeta"},
                ],
            )

            cfg = OmegaConf.create(
                {
                    "seed": 11,
                    "perplexity": {
                        "dataset_root": str(dataset_root),
                        "slices": ["slice_a", "slice_b"],
                        "split": "test",
                        "max_documents": None,
                        "max_tokens": None,
                        "context_length": 16,
                        "stride": 16,
                        "batch_size": 2,
                        "text_field": "text",
                    },
                    "eval_target": {"name": "perplexity_test"},
                }
            )

            summary = run_perplexity_eval(
                model=model,
                tokenizer=tokenizer,
                perplexity_cfg=cfg.perplexity,
                eval_target_cfg=cfg.eval_target,
                output_dir=Path(tmpdir) / "run",
                full_cfg=cfg,
            )

            self.assertTrue((Path(tmpdir) / "run" / "config_resolved.yaml").exists())
            self.assertTrue((Path(tmpdir) / "run" / "summary.json").exists())
            self.assertTrue((Path(tmpdir) / "run" / "metrics" / "summary.json").exists())
            self.assertTrue((Path(tmpdir) / "run" / "metrics" / "per_slice_summary.csv").exists())
            self.assertTrue((Path(tmpdir) / "run" / "metrics" / "per_file_metrics.csv").exists())
            self.assertEqual(summary["num_slices"], 2)
            self.assertEqual(summary["num_files"], 2)
            self.assertGreater(summary["num_tokens"], 0)
            self.assertIsNotNone(summary["perplexity"])
            self.assertIsNotNone(summary["bits_per_byte"])

    def test_perplexity_eval_main_impl_writes_outputs(self) -> None:
        tokenizer = _TinyTokenizer()
        _ = tokenizer.encode("alpha beta gamma delta")
        model = _TinyTransitionLM(vocab_size=tokenizer._next_id + 4)

        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "paloma"
            _write_jsonl_gz(
                dataset_root / "slice_a" / "test" / "test.jsonl.gz",
                [
                    {"id": "0", "text": "alpha beta"},
                    {"id": "1", "text": "gamma delta"},
                ],
            )

            cfg = OmegaConf.create(
                {
                    "seed": 11,
                    "output_root": tmpdir,
                    "model": {"device": "cpu", "dtype": "float32"},
                    "perplexity": {
                        "dataset_root": str(dataset_root),
                        "slices": ["slice_a"],
                        "split": "test",
                        "max_documents": None,
                        "max_tokens": None,
                        "context_length": 16,
                        "stride": 16,
                        "batch_size": 2,
                        "text_field": "text",
                    },
                    "eval_target": {
                        "name": "perplexity_eval_test",
                        "base_checkpoint": "base",
                        "adapter_checkpoint": None,
                        "tokenizer_checkpoint": "tokenizer",
                        "local_files_only": True,
                    },
                    "inference": {"backend": "hf"},
                }
            )

            assets = SimpleNamespace(hf_model=model, tokenizer=tokenizer, generation_backend=None)
            with patch("random_steering.perplexity.eval.load_evaluation_assets", return_value=assets):
                run_dir = main_impl(cfg)

            self.assertTrue((run_dir / "config_resolved.yaml").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "metrics" / "summary.json").exists())

    def test_perplexity_config_composes(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base=None, config_dir=str(repo_root / "conf")):
            cfg = compose(config_name="perplexity_eval_config")
        self.assertEqual(cfg.inference.backend, "hf")
        self.assertEqual(cfg.perplexity.split, "test")
        self.assertIn("wikitext_103", list(cfg.perplexity.slices))


if __name__ == "__main__":
    unittest.main()
