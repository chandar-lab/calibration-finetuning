from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from omegaconf import OmegaConf
import torch
import torch.nn as nn

from random_steering.calibrate_sft.data import (
    OutputSpace,
    PrefixTarget,
    PromptSpec,
    build_distribution_spec,
    build_dataset_splits,
    build_output_space,
    build_tokenized_output_space,
    prepare_prompt_spec,
    sample_training_example,
)
from random_steering.calibrate_sft.evaluate import _select_prompts, monte_carlo_logit_kl
from random_steering.calibrate_sft.losses import sequence_calibration_loss
from random_steering.calibrate_sft.modeling import _infer_transformer_layer_classes
from random_steering.calibrate_sft.train_loop import (
    _balanced_prompt_order,
    _iter_balanced_batches,
    _shard_training_prompts,
    _requires_token_type_ids,
    collate_training_examples,
    compute_batch_loss,
    run_train_step,
)
from random_steering.eval.protocol_runner import run_protocol


class _TinyTokenizer:
    chat_template = None

    def __init__(self) -> None:
        self.eos_token_id = 0
        self.pad_token_id = 0
        self._vocab: dict[str, int] = {}
        self._next_id = 1

    @property
    def vocab_size(self) -> int:
        return self._next_id

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


class _TinyGemmaLikeLM(_TinyLM):
    def __init__(self, vocab_size: int, hidden_size: int = 24) -> None:
        super().__init__(vocab_size=vocab_size, hidden_size=hidden_size)
        self.seen_token_type_ids: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ):
        self.seen_token_type_ids = token_type_ids
        if token_type_ids is None:
            raise ValueError("token_type_ids required")
        return super().forward(input_ids=input_ids, attention_mask=attention_mask)


class _GemmaLikeDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _GemmaLikeTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_GemmaLikeDecoderLayer()])


class _GemmaLikeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _GemmaLikeTextModel()


class _GemmaLikeForConditionalGeneration(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _GemmaLikeModel()


class _BatchOnlyEngine:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.batch_calls = 0

    def generate_text(self, prompt: str, seed: int) -> str:
        raise AssertionError("generate_text should not be called when batched generation is available")

    def generate_text_batch(self, prompts: list[str], seeds: list[int]) -> list[str]:
        _ = prompts, seeds
        self.batch_calls += 1
        return list(self.outputs)


class _NoOpSteeringPolicy:
    def on_request_start(self, spec, seed) -> None:
        _ = spec, seed

    def install_hooks(self) -> None:
        return None

    def remove_hooks(self) -> None:
        return None


class CalibrateSftTests(unittest.TestCase):
    def _make_prompt(self, family_id: str, distribution_id: str | None = None) -> PromptSpec:
        return PromptSpec(
            split="train",
            family_id=family_id,
            display_name=family_id.title(),
            tier="II",
            distribution_id=distribution_id or f"{family_id}_test",
            parameters={"p": 0.5},
            kind="discrete" if family_id in {"bernoulli", "binomial", "geometric"} else "continuous",
            integer_only=family_id in {"bernoulli", "binomial", "geometric"},
        )

    def _make_prompt_like(self, family_id: str, distribution_id: str | None = None):
        return SimpleNamespace(prompt_spec=self._make_prompt(family_id, distribution_id))

    def test_dataset_splits_keep_unseen_params_disjoint(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(repo_root / "conf" / "data" / "calibrate_sft.yaml")
        splits = build_dataset_splits(cfg)

        train_gaussian = {
            (spec.parameters["mu"], spec.parameters["sigma"])
            for spec in splits["train"]
            if spec.family_id == "gaussian"
        }
        unseen_gaussian = {
            (spec.parameters["mu"], spec.parameters["sigma"])
            for spec in splits["test_unseen_params"]
            if spec.family_id == "gaussian"
        }

        self.assertTrue(train_gaussian)
        self.assertTrue(unseen_gaussian)
        self.assertTrue(train_gaussian.isdisjoint(unseen_gaussian))
        self.assertTrue(all(spec.family_id not in {"bernoulli", "poisson", "weibull"} for spec in splits["train"]))

    def test_dataset_splits_include_new_families_and_keep_one_heldout_per_tier(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(repo_root / "conf" / "data" / "calibrate_sft.yaml")
        splits = build_dataset_splits(cfg)

        train_families = {spec.family_id for spec in splits["train"]}
        heldout_families = {spec.family_id for spec in splits["test_ood_family"]}
        heldout_tiers = {spec.tier for spec in splits["test_ood_family"]}

        expected_new_train_families = {
            "geometric",
            "negative_binomial",
            "lognormal",
            "triangular",
            "rayleigh",
            "pareto",
            "hypergeometric",
            "gumbel",
            "skellam",
            "beta_binomial",
            "lomax",
            "inverse_gaussian",
        }

        self.assertTrue(expected_new_train_families.issubset(train_families))
        self.assertTrue({"bernoulli", "poisson", "weibull", "maxwell", "chi", "truncnorm"}.issubset(heldout_families))
        self.assertEqual(heldout_tiers, {"I", "II", "III"})

    def test_discrete_output_space_matches_binomial_support(self) -> None:
        cfg = OmegaConf.create({"quantile_low": 0.001, "quantile_high": 0.999, "num_decimals": 2, "max_bins": 1001, "tail_policy": "clip_to_edge_bin"})
        prompt = PromptSpec(
            split="train",
            family_id="binomial",
            display_name="Binomial",
            tier="II",
            distribution_id="binomial_test",
            parameters={"n": 10.0, "p": 0.5},
            kind="discrete",
            integer_only=True,
        )
        output_space = build_output_space(prompt, cfg)
        self.assertEqual(output_space.values[0], "0")
        self.assertEqual(output_space.values[-1], "10")
        self.assertEqual(len(output_space.values), 11)
        self.assertAlmostEqual(sum(output_space.probabilities), 1.0, places=6)

    def test_discrete_output_space_handles_signed_skellam_support(self) -> None:
        cfg = OmegaConf.create({"quantile_low": 0.001, "quantile_high": 0.999, "num_decimals": 2, "max_bins": 1001, "tail_policy": "clip_to_edge_bin"})
        prompt = PromptSpec(
            split="train",
            family_id="skellam",
            display_name="Skellam",
            tier="III",
            distribution_id="skellam_test",
            parameters={"mu1": 4.0, "mu2": 2.0},
            kind="discrete",
            integer_only=True,
        )
        output_space = build_output_space(prompt, cfg)
        numeric_values = [int(value) for value in output_space.values]
        self.assertLess(min(numeric_values), 0)
        self.assertGreater(max(numeric_values), 0)
        self.assertAlmostEqual(sum(output_space.probabilities), 1.0, places=6)

    def test_dataset_splits_support_per_parameter_num_points_and_logspace(self) -> None:
        cfg = OmegaConf.create(
            {
                "linspace_points_1d": 9,
                "linspace_points_2d_per_axis": 9,
                "train_max_specs_per_family": 90,
                "test_unseen_max_specs_per_family": 18,
                "ood_max_specs_per_family": 18,
                "train_families": ["exponential"],
                "heldout_families": [],
                "families": {
                    "exponential": {
                        "train": {
                            "max_specs": 5,
                            "lambda": {
                                "min": 1.0,
                                "max": 16.0,
                                "num_points": 5,
                                "spacing": "logspace",
                            },
                        },
                        "test_unseen": {
                            "max_specs": 3,
                            "lambda": {
                                "min": 32.0,
                                "max": 128.0,
                                "num_points": 3,
                                "spacing": "logspace",
                            },
                        },
                    }
                },
            }
        )

        splits = build_dataset_splits(cfg)
        train_values = [spec.parameters["lambda"] for spec in splits["train"]]
        unseen_values = [spec.parameters["lambda"] for spec in splits["test_unseen_params"]]

        self.assertEqual([round(value, 6) for value in train_values], [1.0, 2.0, 4.0, 8.0, 16.0])
        self.assertEqual([round(value, 6) for value in unseen_values], [32.0, 64.0, 128.0])

    def test_iter_balanced_batches_mixes_families(self) -> None:
        prompts = [self._make_prompt_like("uniform", f"uniform_{idx}") for idx in range(6)]
        prompts += [self._make_prompt_like("gaussian", f"gaussian_{idx}") for idx in range(2)]
        batches = _iter_balanced_batches(prompts, batch_size=4, seed=11)
        self.assertEqual(sum(len(batch) for batch in batches), len(prompts))
        first_batch_families = {prompt.prompt_spec.family_id for prompt in batches[0]}
        self.assertEqual(first_batch_families, {"uniform", "gaussian"})

    def test_shard_training_prompts_evenly_pads_to_world_size(self) -> None:
        prompts = [self._make_prompt_like("uniform", f"uniform_{idx}") for idx in range(5)]
        ordered = _balanced_prompt_order(prompts, seed=11)
        shard0 = _shard_training_prompts(ordered, rank=0, world_size=2)
        shard1 = _shard_training_prompts(ordered, rank=1, world_size=2)
        self.assertEqual(len(shard0), len(shard1))
        self.assertEqual(len(shard0), 3)
        self.assertEqual(len(shard0) + len(shard1), 6)

    def test_select_prompts_is_family_stratified(self) -> None:
        prompts = [self._make_prompt_like("uniform", f"uniform_{idx}") for idx in range(3)]
        prompts += [self._make_prompt_like("gaussian", f"gaussian_{idx}") for idx in range(3)]
        prompts += [self._make_prompt_like("beta", f"beta_{idx}") for idx in range(3)]
        selected = _select_prompts(prompts, max_prompts=3, selection_seed=0)
        self.assertEqual({prompt.prompt_spec.family_id for prompt in selected}, {"beta", "gaussian", "uniform"})

    def test_infer_transformer_layer_classes_supports_gemma_language_model_stack(self) -> None:
        model = _GemmaLikeForConditionalGeneration()
        self.assertEqual(_infer_transformer_layer_classes(model), {_GemmaLikeDecoderLayer})

    def test_requires_token_type_ids_for_gemma_configs(self) -> None:
        self.assertTrue(_requires_token_type_ids(SimpleNamespace(name="gemma_3_4b_it", checkpoint="google/gemma-3-4b-it")))
        self.assertFalse(_requires_token_type_ids(SimpleNamespace(name="qwen3_1p7b", checkpoint="Qwen/Qwen3-1.7B")))

    def test_collate_training_examples_adds_zero_token_type_ids_for_gemma(self) -> None:
        tokenizer = _TinyTokenizer()
        data_cfg = OmegaConf.create(
            {"quantile_low": 0.001, "quantile_high": 0.999, "num_decimals": 2, "max_bins": 1001, "tail_policy": "clip_to_edge_bin"}
        )
        prepared_prompt = prepare_prompt_spec(
            self._make_prompt("bernoulli"),
            tokenizer,
            SimpleNamespace(enable_thinking=False),
            data_cfg,
        )
        example = sample_training_example(prepared_prompt, sample_seed=7)
        batch = collate_training_examples(
            [example],
            pad_token_id=tokenizer.pad_token_id,
            device=torch.device("cpu"),
            model_cfg=SimpleNamespace(name="gemma_3_4b_it", checkpoint="google/gemma-3-4b-it"),
        )
        self.assertIsNotNone(batch["token_type_ids"])
        self.assertTrue(torch.equal(batch["token_type_ids"], torch.zeros_like(batch["input_ids"])))

    def test_compute_batch_loss_passes_token_type_ids_when_model_supports_them(self) -> None:
        tokenizer = _TinyTokenizer()
        data_cfg = OmegaConf.create(
            {"quantile_low": 0.001, "quantile_high": 0.999, "num_decimals": 2, "max_bins": 1001, "tail_policy": "clip_to_edge_bin"}
        )
        prepared_prompt = prepare_prompt_spec(
            self._make_prompt("bernoulli"),
            tokenizer,
            SimpleNamespace(enable_thinking=False),
            data_cfg,
        )
        example = sample_training_example(prepared_prompt, sample_seed=7)
        batch = collate_training_examples(
            [example],
            pad_token_id=tokenizer.pad_token_id,
            device=torch.device("cpu"),
            model_cfg=SimpleNamespace(name="gemma_3_4b_it", checkpoint="google/gemma-3-4b-it"),
        )
        model = _TinyGemmaLikeLM(vocab_size=max(tokenizer.vocab_size, 32))
        loss = compute_batch_loss(model, batch, OmegaConf.create({"tau": 1.0, "epsilon_smoothing": 0.0}))
        self.assertGreaterEqual(float(loss.item()), 0.0)
        self.assertIsNotNone(model.seen_token_type_ids)
        self.assertTrue(torch.equal(model.seen_token_type_ids, batch["token_type_ids"]))

    def test_protocol_runner_uses_batched_engine_for_independent_sampling(self) -> None:
        spec = build_distribution_spec(self._make_prompt("bernoulli"))
        engine = _BatchOnlyEngine(outputs=["0", "1", "0"])
        records = run_protocol(
            engine=engine,
            steering_policy=_NoOpSteeringPolicy(),
            steering_name="calibrate_sft",
            spec=spec,
            protocol="independent",
            num_samples=3,
            base_seed=11,
        )
        self.assertEqual(len(records), 3)
        self.assertEqual(engine.batch_calls, 1)

    def test_monte_carlo_logit_kl_matches_between_scalar_and_batched_modes(self) -> None:
        tokenizer = _TinyTokenizer()
        data_cfg = OmegaConf.create(
            {"quantile_low": 0.001, "quantile_high": 0.999, "num_decimals": 2, "max_bins": 1001, "tail_policy": "clip_to_edge_bin"}
        )
        prepared_prompt = prepare_prompt_spec(
            self._make_prompt("bernoulli"),
            tokenizer,
            SimpleNamespace(enable_thinking=False),
            data_cfg,
        )
        model = _TinyLM(vocab_size=max(tokenizer.vocab_size, 32))
        scalar = monte_carlo_logit_kl(
            model,
            prepared_prompt,
            tau=1.0,
            epsilon=0.0,
            num_samples=6,
            base_seed=11,
            batch_size=1,
        )
        batched = monte_carlo_logit_kl(
            model,
            prepared_prompt,
            tau=1.0,
            epsilon=0.0,
            num_samples=6,
            base_seed=11,
            batch_size=3,
        )
        self.assertAlmostEqual(scalar, batched, places=6)

    def test_trie_targets_share_prefixes_and_end_in_eos(self) -> None:
        tokenizer = _TinyTokenizer()
        output_space = OutputSpace(
            values=("0.1", "0.2"),
            probabilities=(0.4, 0.6),
            lower_bound=0.1,
            upper_bound=0.2,
            integer_only=False,
            num_decimals=1,
        )
        tokenized = build_tokenized_output_space(output_space, tokenizer)
        empty_target = tokenized.prefix_targets[()]
        self.assertEqual(len(empty_target.next_token_ids), 1)

        shared_prefix = tuple(tokenizer.encode("0.", add_special_tokens=False))
        shared_target = tokenized.prefix_targets[shared_prefix]
        self.assertEqual(len(shared_target.next_token_ids), 2)
        self.assertAlmostEqual(shared_target.next_token_probs[0] + shared_target.next_token_probs[1], 1.0, places=6)

        eos_prefix = tuple(tokenizer.encode("0.1", add_special_tokens=False))
        self.assertEqual(tokenized.prefix_targets[eos_prefix].next_token_ids, (tokenizer.eos_token_id,))

    def test_sequence_calibration_loss_prefers_target_aligned_logits(self) -> None:
        prefix_target = PrefixTarget(prefix_token_ids=(), next_token_ids=(2, 4), next_token_probs=(0.75, 0.25))
        aligned_logits = [torch.tensor([0.0, 0.0, 3.0, 0.0, 1.0], dtype=torch.float32)]
        misaligned_logits = [torch.tensor([0.0, 0.0, 1.0, 0.0, 3.0], dtype=torch.float32)]

        aligned = sequence_calibration_loss(aligned_logits, [prefix_target])
        misaligned = sequence_calibration_loss(misaligned_logits, [prefix_target])
        self.assertLess(float(aligned.item()), float(misaligned.item()))

    def test_sequence_calibration_loss_penalizes_invalid_vocab_mass(self) -> None:
        prefix_target = PrefixTarget(prefix_token_ids=(), next_token_ids=(2, 4), next_token_probs=(0.5, 0.5))
        valid_only_logits = [torch.tensor([0.0, 0.0, 3.0, 0.0, 3.0], dtype=torch.float32)]
        invalid_dominant_logits = [torch.tensor([0.0, 50.0, 3.0, 0.0, 3.0], dtype=torch.float32)]

        valid_loss = sequence_calibration_loss(valid_only_logits, [prefix_target])
        invalid_loss = sequence_calibration_loss(invalid_dominant_logits, [prefix_target])
        self.assertLess(float(valid_loss.item()), float(invalid_loss.item()))

    def test_one_train_step_smoke_without_model_download(self) -> None:
        tokenizer = _TinyTokenizer()
        data_cfg = OmegaConf.create({"quantile_low": 0.001, "quantile_high": 0.999, "num_decimals": 2, "max_bins": 101, "tail_policy": "clip_to_edge_bin"})
        model_cfg = SimpleNamespace(enable_thinking=False)
        train_cfg = SimpleNamespace(tau=1.0, epsilon_smoothing=1e-8)
        prompt = PromptSpec(
            split="train",
            family_id="bernoulli",
            display_name="Bernoulli",
            tier="I",
            distribution_id="bernoulli_test",
            parameters={"p": 0.7},
            kind="discrete",
            integer_only=True,
        )

        prepared = prepare_prompt_spec(prompt, tokenizer, model_cfg, data_cfg)
        example = sample_training_example(prepared, sample_seed=13)
        model = _TinyLM(vocab_size=tokenizer.vocab_size + 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        batch = collate_training_examples([example], pad_token_id=tokenizer.pad_token_id, device=torch.device("cpu"))

        before = model.embed.weight.detach().clone()
        loss = run_train_step(model=model, optimizer=optimizer, batch=batch, train_cfg=train_cfg)
        after = model.embed.weight.detach()

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
