from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from omegaconf import OmegaConf
import torch
import torch.nn as nn

from random_steering.calibrate_sft.data import OutputSpace, PromptSpec, prepare_prompt_spec, sample_training_example
from random_steering.hard_label_sft.data import (
    HardLabelExample,
    expand_train_prompts_for_epoch,
    sample_hard_label_example,
    write_hard_label_dataset_artifacts,
)
from random_steering.hard_label_sft.train_loop import collate_hard_label_examples, compute_batch_loss, run_train_step
from random_steering.train import run_training


class _TinyTokenizer:
    chat_template = None

    def __init__(self) -> None:
        self.eos_token_id = 0
        self.pad_token_id = 0
        self._vocab: dict[str, int] = {}
        self._next_id = 1

    def _token_for_char(self, char: str) -> int:
        if char not in self._vocab:
            self._vocab[char] = self._next_id
            self._next_id += 1
        return self._vocab[char]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [self._token_for_char(char) for char in text]

    def __call__(self, text: str, add_special_tokens: bool = True, return_tensors: str | None = None):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
        return {"input_ids": ids}


class _TinyLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 24) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = SimpleNamespace(use_cache=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        _ = attention_mask
        hidden = torch.tanh(self.proj(self.embed(input_ids)))
        return SimpleNamespace(logits=self.lm_head(hidden))


class HardLabelSftTests(unittest.TestCase):
    def _make_prompt(self, family_id: str, distribution_id: str | None = None):
        return SimpleNamespace(
            prompt_spec=PromptSpec(
                split="train",
                family_id=family_id,
                display_name=family_id.title(),
                tier="II",
                distribution_id=distribution_id or f"{family_id}_test",
                parameters={"p": 0.5},
                kind="discrete" if family_id in {"bernoulli", "binomial", "geometric"} else "continuous",
                integer_only=family_id in {"bernoulli", "binomial", "geometric"},
            )
        )

    def test_sample_hard_label_example_matches_calibration_sampling(self) -> None:
        tokenizer = _TinyTokenizer()
        data_cfg = OmegaConf.create(
            {
                "quantile_low": 0.001,
                "quantile_high": 0.999,
                "num_decimals": 2,
                "max_bins": 1001,
                "tail_policy": "clip_to_edge_bin",
            }
        )
        prompt_spec = PromptSpec(
            split="train",
            family_id="binomial",
            display_name="Binomial",
            tier="II",
            distribution_id="binomial_test",
            parameters={"n": 10.0, "p": 0.5},
            kind="discrete",
            integer_only=True,
        )
        prepared_prompt = prepare_prompt_spec(prompt_spec, tokenizer, SimpleNamespace(name="tiny"), data_cfg)

        hard_label_example = sample_hard_label_example(prepared_prompt, sample_seed=17, eos_token_id=tokenizer.eos_token_id)
        calibration_example = sample_training_example(prepared_prompt, sample_seed=17)

        self.assertEqual(hard_label_example.candidate_value, calibration_example.candidate_value)
        self.assertEqual(
            hard_label_example.candidate_token_ids,
            calibration_example.candidate_token_ids + (int(tokenizer.eos_token_id),),
        )

    def test_expand_train_prompts_for_epoch_repeats_each_prompt(self) -> None:
        prompts = [
            self._make_prompt("gaussian", "g1"),
            self._make_prompt("gaussian", "g2"),
            self._make_prompt("binomial", "b1"),
        ]
        expanded = expand_train_prompts_for_epoch(prompts, samples_per_prompt_per_epoch=32, seed=11)

        counts: dict[str, int] = {}
        for prompt in expanded:
            distribution_id = prompt.prompt_spec.distribution_id
            counts[distribution_id] = counts.get(distribution_id, 0) + 1

        self.assertEqual(len(expanded), 96)
        self.assertEqual(counts, {"g1": 32, "g2": 32, "b1": 32})

    def test_collate_masks_prompt_tokens(self) -> None:
        prepared_prompt = SimpleNamespace(prompt_token_ids=(10, 11, 12))
        example = HardLabelExample(
            prepared_prompt=SimpleNamespace(prompt_token_ids=prepared_prompt.prompt_token_ids),
            candidate_value="42",
            candidate_token_ids=(21, 22, 0),
        )

        batch = collate_hard_label_examples(
            [example],
            pad_token_id=0,
            device=torch.device("cpu"),
            model_cfg=SimpleNamespace(name="tiny"),
        )

        self.assertTrue(torch.equal(batch["input_ids"], torch.tensor([[10, 11, 12, 21, 22, 0]])))
        self.assertTrue(torch.equal(batch["labels"], torch.tensor([[-100, -100, -100, 21, 22, 0]])))

    def test_run_train_step_reduces_loss_on_toy_example(self) -> None:
        tokenizer = _TinyTokenizer()
        tokenizer.encode("prompt42")
        model = _TinyLM(vocab_size=tokenizer._next_id + 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        example = HardLabelExample(
            prepared_prompt=SimpleNamespace(prompt_token_ids=tuple(tokenizer.encode("prompt"))),
            candidate_value="42",
            candidate_token_ids=tuple(tokenizer.encode("42")) + (tokenizer.eos_token_id,),
        )
        batch = collate_hard_label_examples([example], pad_token_id=0, device=torch.device("cpu"))

        before = float(compute_batch_loss(model, batch).item())
        _ = run_train_step(model=model, optimizer=optimizer, batch=batch)
        after = float(compute_batch_loss(model, batch).item())

        self.assertTrue(torch.isfinite(torch.tensor(before)))
        self.assertLess(after, before)

    def test_write_hard_label_dataset_artifacts_includes_manifest_fields(self) -> None:
        prompt = self._make_prompt("gaussian", "g1")
        prompt.prompt_text = "dummy"
        prompt.output_space = OutputSpace(
            values=("0.0", "1.0"),
            probabilities=(0.5, 0.5),
            lower_bound=0.0,
            upper_bound=1.0,
            integer_only=False,
            num_decimals=1,
        )
        prepared_splits = {"train": [prompt], "test_unseen_params": [], "test_ood_family": []}

        with TemporaryDirectory() as tmpdir:
            write_hard_label_dataset_artifacts(
                tmpdir,
                prepared_splits,
                samples_per_prompt_per_epoch=32,
                data_cfg=OmegaConf.create({"foo": "bar"}),
            )
            manifest = json.loads((Path(tmpdir) / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["training_mode"], "hard_label_sft")
        self.assertEqual(manifest["samples_per_prompt_per_epoch"], 32)
        self.assertEqual(manifest["splits"]["train"], 1)
        self.assertEqual(manifest["data_config"]["foo"], "bar")

    def test_run_training_dispatches_hard_label(self) -> None:
        cfg = OmegaConf.create({"train": {"name": "hard_label_sft"}})
        with mock.patch("random_steering.train.run_hard_label_sft") as run_hard_label, mock.patch(
            "random_steering.train.run_calibrate_sft"
        ) as run_calibrate:
            run_training(cfg)

        run_hard_label.assert_called_once_with(cfg)
        run_calibrate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
