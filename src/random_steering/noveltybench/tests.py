from __future__ import annotations

import json
import os
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

from random_steering.calibrate_sft.modeling import load_evaluation_assets
from random_steering.noveltybench.data import (
    SPLIT_OUTPUT_NAMES,
    artifact_is_complete,
    load_prompt_rows_from_path,
    split_output_dir,
)
from random_steering.noveltybench.eval import main_impl as eval_main_impl
from random_steering.noveltybench.evaluate import (
    generation_stage_is_complete,
    run_generation_stage,
    run_noveltybench_eval,
    run_partition_stage,
    run_score_stage,
)
from random_steering.noveltybench.generate import main_impl as generate_main_impl
from random_steering.noveltybench.partitioning import maybe_test_equality, partition_responses
from random_steering.noveltybench.scoring import score_partition, summarize_scores, transform_raw_reward


class _ToyBackend:
    def __init__(self) -> None:
        self._calls = 0
        self.calls: list[tuple[list[str], list[int]]] = []

    def format_prompt(self, prompt):
        return str(prompt)

    def strip_response(self, text: str) -> str:
        return text.replace("<think>hidden</think>", "").strip()

    def generate_text(self, prompt, seed: int) -> str:
        return self.generate_text_batch([prompt], [seed])[0]

    def generate_text_batch(self, prompts, seeds, *, stop_strings=None):
        _ = stop_strings
        self.calls.append((list(prompts), list(seeds)))
        outputs = []
        for prompt, seed in zip(prompts, seeds, strict=True):
            label = "alpha" if (seed + self._calls) % 2 == 0 else "beta"
            outputs.append(f"<think>hidden</think> {prompt.split()[0]}-{label}")
        self._calls += 1
        return outputs

    def score_prompt_continuation_pairs_batch(self, prompt_groups, continuation_groups):
        _ = (prompt_groups, continuation_groups)
        raise NotImplementedError


class _FakeClassifier:
    threshold = 0.5

    def score_pairs(self, response_pairs):
        scores = []
        for response_0, response_1 in response_pairs:
            scores.append(0.9 if response_0.split()[-1] == response_1.split()[-1] else 0.1)
        return scores


class _FakeScorer:
    def score_generations(self, prompt: str, generations: list[str]) -> list[float]:
        _ = prompt
        mapping = {"alpha": -1.0, "beta": -5.2}
        return [mapping[generation.split("-")[-1]] for generation in generations]


class NoveltyBenchTests(unittest.TestCase):
    def _tiny_prompts(self) -> dict[str, list[SimpleNamespace]]:
        return {
            "curated": [
                SimpleNamespace(split="curated", prompt_id="curated-0", prompt="Prompt zero", metadata={"id": "curated-0"}),
                SimpleNamespace(split="curated", prompt_id="curated-1", prompt="Prompt one", metadata={"id": "curated-1"}),
            ],
        }

    def _tiny_cfg(self, tmpdir: str):
        return OmegaConf.create(
            {
                "seed": 11,
                "output_root": tmpdir,
                "model": {
                    "device": "cpu",
                    "dtype": "float32",
                    "enable_thinking": False,
                },
                "noveltybench": {
                    "assets_root": tmpdir,
                    "enabled_splits": ["curated"],
                    "max_prompts_per_split": None,
                    "resume": True,
                    "output_root": tmpdir,
                    "generation": {
                        "num_generations": 3,
                        "batch_size": 2,
                        "max_new_tokens": 32,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "do_sample": True,
                    },
                    "classifier": {
                        "tokenizer_checkpoint": "classifier-tokenizer",
                        "checkpoint": "classifier-model",
                        "threshold": 0.102,
                        "max_length": 128,
                        "batch_size": 4,
                        "device": "cpu",
                        "local_files_only": True,
                    },
                    "reward_model": {
                        "checkpoint": "reward-model",
                        "batch_size": 4,
                        "dtype": "float32",
                        "device_map": None,
                        "attn_implementation": "eager",
                        "local_files_only": True,
                    },
                    "patience": 0.8,
                },
                "eval_target": {
                    "name": "noveltybench_test",
                    "base_checkpoint": "base",
                    "adapter_checkpoint": None,
                    "tokenizer_checkpoint": "tokenizer",
                    "local_files_only": True,
                },
                "inference": {
                    "backend": "hf",
                },
            }
        )

    def test_loader_validates_expected_count(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curated.jsonl"
            path.write_text(json.dumps({"id": "x", "prompt": "hello"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 2 prompts, found 1"):
                load_prompt_rows_from_path(path, split_name="curated", expected_count=2)

    def test_partition_short_answer_lexical_rule(self) -> None:
        self.assertTrue(maybe_test_equality("New York", "York New"))
        self.assertFalse(maybe_test_equality("red apple", "blue car"))
        self.assertIsNone(maybe_test_equality("This is a much longer answer", "This is a different longer answer"))

    def test_partition_responses_greedy(self) -> None:
        classifier = _FakeClassifier()
        partition = partition_responses(
            prompt="unused",
            responses=[
                "This response really ends with alpha",
                "Another fairly long answer ending alpha",
                "This response really ends with beta",
                "Yet another fairly long answer ending alpha",
            ],
            classifier=classifier,
        )
        self.assertEqual(partition, [0, 0, 1, 0])

    def test_reward_transform_and_utility(self) -> None:
        scorer = _FakeScorer()
        generation_scores, partition_scores, raw_rewards, utility = score_partition(
            prompt="Prompt",
            generations=["x-alpha", "x-beta", "y-alpha"],
            partition=[0, 1, 0],
            patience=0.8,
            scorer=scorer,
        )
        self.assertEqual(transform_raw_reward(-1.0), 10)
        self.assertEqual(transform_raw_reward(-5.2), 6)
        self.assertEqual(raw_rewards, [-1.0, -5.2, -1.0])
        self.assertEqual(generation_scores, [10, 6, 0])
        self.assertEqual(partition_scores, [10, 6])
        expected = (10.0 + (6.0 * 0.8)) / (1.0 + 0.8 + 0.64)
        self.assertAlmostEqual(utility, expected)

    def test_stage_pipeline_writes_expected_artifacts(self) -> None:
        if OmegaConf is None:
            self.skipTest("omegaconf is not installed in this environment")
        with TemporaryDirectory() as tmpdir:
            cfg = self._tiny_cfg(tmpdir)
            prompts = self._tiny_prompts()
            backend = _ToyBackend()
            run_dir = Path(tmpdir) / "run"
            run_generation_stage(
                generation_backend=backend,
                noveltybench_cfg=cfg.noveltybench,
                output_dir=run_dir,
                full_cfg=cfg,
                eval_target_cfg=cfg.eval_target,
                base_seed=int(cfg.seed),
                prompts_by_split=prompts,
            )
            with patch("random_steering.noveltybench.evaluate._load_prompts_by_split", return_value=prompts):
                run_partition_stage(
                    noveltybench_cfg=cfg.noveltybench,
                    run_dir=run_dir,
                    classifier=_FakeClassifier(),
                )
                summary = run_score_stage(
                    noveltybench_cfg=cfg.noveltybench,
                    run_dir=run_dir,
                    scorer=_FakeScorer(),
                )

            split_dir = split_output_dir(run_dir, "curated")
            self.assertTrue((run_dir / "config_resolved.json").exists())
            self.assertTrue((split_dir / "generations.jsonl").exists())
            self.assertTrue((split_dir / "partitions.jsonl").exists())
            self.assertTrue((split_dir / "scores.jsonl").exists())
            self.assertTrue((split_dir / "summary.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "metrics" / "summary.json").exists())
            self.assertEqual(summary["metrics"]["num_prompts"], 2)
            self.assertIn("curated", summary["splits"])
            generation_rows = [json.loads(line) for line in (split_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(generation_rows), 2)
            self.assertTrue(all(len(row["generations"]) == 3 for row in generation_rows))
            self.assertTrue(all(all("<think>" not in generation for generation in row["generations"]) for row in generation_rows))

    def test_resume_checks_completed_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.jsonl"
            rows = [{"id": "a"}, {"id": "b"}]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertTrue(artifact_is_complete(path, ["a", "b"]))
            self.assertFalse(artifact_is_complete(path, ["a", "c"]))

    def test_generation_stage_resumes_from_partial_split(self) -> None:
        if OmegaConf is None:
            self.skipTest("omegaconf is not installed in this environment")
        with TemporaryDirectory() as tmpdir:
            cfg = self._tiny_cfg(tmpdir)
            prompts = self._tiny_prompts()
            run_dir = Path(tmpdir) / "run"
            split_dir = split_output_dir(run_dir, "curated")
            split_dir.mkdir(parents=True, exist_ok=True)
            generations_path = split_dir / "generations.jsonl"
            completed_row = {
                "id": "curated-0",
                "split": "curated",
                "prompt": "Prompt zero",
                "metadata": {"id": "curated-0"},
                "generations": ["Prompt-alpha", "Prompt-beta", "Prompt-alpha"],
                "raw_generations": ["<think>hidden</think> Prompt-alpha"] * 3,
                "had_thinking_tags": [True, True, True],
                "seeds": [11, 12, 13],
            }
            generations_path.write_text(json.dumps(completed_row) + "\n", encoding="utf-8")

            backend = _ToyBackend()
            run_generation_stage(
                generation_backend=backend,
                noveltybench_cfg=cfg.noveltybench,
                output_dir=run_dir,
                full_cfg=cfg,
                eval_target_cfg=cfg.eval_target,
                base_seed=int(cfg.seed),
                prompts_by_split=prompts,
            )

            rows = [json.loads(line) for line in generations_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in rows], ["curated-0", "curated-1"])
            self.assertEqual(rows[0]["generations"], completed_row["generations"])
            self.assertEqual(len(rows[1]["generations"]), 3)
            self.assertEqual(len(backend.calls), 3)
            self.assertTrue(all(prompts == ["Prompt one"] for prompts, _ in backend.calls))

    def test_generation_stage_rejects_resume_config_mismatch(self) -> None:
        if OmegaConf is None:
            self.skipTest("omegaconf is not installed in this environment")
        with TemporaryDirectory() as tmpdir:
            cfg = self._tiny_cfg(tmpdir)
            prompts = self._tiny_prompts()
            run_dir = Path(tmpdir) / "run"

            run_generation_stage(
                generation_backend=_ToyBackend(),
                noveltybench_cfg=cfg.noveltybench,
                output_dir=run_dir,
                full_cfg=cfg,
                eval_target_cfg=cfg.eval_target,
                base_seed=int(cfg.seed),
                prompts_by_split=prompts,
            )

            cfg.noveltybench.generation.max_new_tokens = 64
            with self.assertRaisesRegex(ValueError, "generation resume config mismatch"):
                run_generation_stage(
                    generation_backend=_ToyBackend(),
                    noveltybench_cfg=cfg.noveltybench,
                    output_dir=run_dir,
                    full_cfg=cfg,
                    eval_target_cfg=cfg.eval_target,
                    base_seed=int(cfg.seed),
                    prompts_by_split=prompts,
                )

    def test_generate_entrypoint_skips_completed_run_before_model_load(self) -> None:
        if OmegaConf is None:
            self.skipTest("omegaconf is not installed in this environment")
        with TemporaryDirectory() as tmpdir:
            cfg = self._tiny_cfg(tmpdir)
            cfg.run_dir = str(Path(tmpdir) / "run")
            prompts = self._tiny_prompts()
            run_dir = Path(cfg.run_dir)
            run_generation_stage(
                generation_backend=_ToyBackend(),
                noveltybench_cfg=cfg.noveltybench,
                output_dir=run_dir,
                full_cfg=cfg,
                eval_target_cfg=cfg.eval_target,
                base_seed=int(cfg.seed),
                prompts_by_split=prompts,
            )
            self.assertTrue(
                generation_stage_is_complete(
                    noveltybench_cfg=cfg.noveltybench,
                    run_dir=run_dir,
                    model_cfg=cfg.model,
                    eval_target_cfg=cfg.eval_target,
                    base_seed=int(cfg.seed),
                    prompts_by_split=prompts,
                )
            )
            with patch(
                "random_steering.noveltybench.generate.load_split_prompts",
                return_value=prompts["curated"],
            ), patch(
                "random_steering.noveltybench.generate.load_evaluation_assets",
                side_effect=AssertionError("model load should not happen for a completed generation run"),
            ):
                actual_run_dir = generate_main_impl(cfg)
            self.assertEqual(actual_run_dir, run_dir)

    def test_config_composes(self) -> None:
        if initialize_config_dir is None or compose is None:
            self.skipTest("hydra is not installed in this environment")
        repo_root = Path(__file__).resolve().parents[3]
        with initialize_config_dir(version_base=None, config_dir=str(repo_root / "conf")):
            cfg = compose(config_name="noveltybench_eval_config")
        self.assertEqual(cfg.eval_target.name, "noveltybench_baseline")
        self.assertEqual(list(cfg.noveltybench.enabled_splits), ["curated", "wildchat"])

    def test_eval_entrypoint_runs_with_patched_assets(self) -> None:
        if OmegaConf is None:
            self.skipTest("omegaconf is not installed in this environment")
        with TemporaryDirectory() as tmpdir:
            cfg = self._tiny_cfg(tmpdir)
            cfg.inference.backend = "vllm"
            cfg.output_root = tmpdir
            run_dir = Path(tmpdir) / "eval-run"
            prompts = self._tiny_prompts()
            assets = SimpleNamespace(generation_backend=_ToyBackend())
            with patch("random_steering.noveltybench.eval.load_evaluation_assets", return_value=assets), patch(
                "random_steering.noveltybench.evaluate._load_prompts_by_split",
                return_value=prompts,
            ), patch(
                "random_steering.noveltybench.evaluate.GenerationSimilarityClassifier.from_config",
                return_value=_FakeClassifier(),
            ), patch(
                "random_steering.noveltybench.evaluate.RewardModelScorer.from_config",
                return_value=_FakeScorer(),
            ), patch(
                "random_steering.noveltybench.eval._resolve_run_dir",
                return_value=run_dir,
            ):
                actual_run_dir = eval_main_impl(cfg)
            self.assertEqual(actual_run_dir, run_dir)
            self.assertTrue((run_dir / SPLIT_OUTPUT_NAMES["curated"] / "scores.jsonl").exists())

    def test_qwen3_vllm_disable_thinking_smoke(self) -> None:
        if OmegaConf is None:
            self.skipTest("omegaconf is not installed in this environment")
        if not torch.cuda.is_available():
            self.skipTest("Qwen vLLM smoke requires CUDA")
        if "SCRATCH" not in os.environ:
            self.skipTest("Qwen vLLM smoke requires SCRATCH to be set")
        os.environ["HF_HOME"] = os.environ["SCRATCH"]
        cfg = OmegaConf.create(
            {
                "model": {
                    "checkpoint": "Qwen/Qwen3-1.7B",
                    "device": "cuda",
                    "dtype": "bfloat16",
                    "trust_remote_code": True,
                    "enable_thinking": False,
                },
                "eval_target": {
                    "name": "noveltybench_qwen3_1p7b_baseline",
                    "base_checkpoint": "Qwen/Qwen3-1.7B",
                    "adapter_checkpoint": None,
                    "tokenizer_checkpoint": "Qwen/Qwen3-1.7B",
                    "local_files_only": False,
                },
                "inference": {
                    "backend": "vllm",
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.9,
                    "max_num_seqs": 16,
                    "enforce_eager": False,
                    "trust_remote_code": True,
                },
                "noveltybench": {
                    "generation": {
                        "batch_size": 1,
                        "max_new_tokens": 64,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "do_sample": True,
                    },
                },
            }
        )
        model_cfg = OmegaConf.merge(cfg.model, cfg.noveltybench.generation, {"use_chat_template": True})
        assets = load_evaluation_assets(
            model_cfg,
            cfg.eval_target,
            cfg.inference,
            require_generation_backend=True,
            require_hf_model=False,
        )
        prompt = "Reply with exactly one animal name."
        formatted_prompt = assets.generation_backend.format_prompt(prompt)
        self.assertIsInstance(formatted_prompt, str)
        raw_output = assets.generation_backend.generate_text(prompt, seed=11)
        self.assertNotIn("<think>", raw_output.lower())

    def test_gpt_oss_vllm_disable_thinking_smoke(self) -> None:
        if OmegaConf is None:
            self.skipTest("omegaconf is not installed in this environment")
        if not torch.cuda.is_available():
            self.skipTest("GPT-OSS vLLM smoke requires CUDA")
        if "SCRATCH" not in os.environ:
            self.skipTest("GPT-OSS vLLM smoke requires SCRATCH to be set")
        os.environ["HF_HOME"] = os.environ["SCRATCH"]
        cfg = OmegaConf.create(
            {
                "model": {
                    "checkpoint": "openai/gpt-oss-20b",
                    "device": "cuda",
                    "dtype": "bfloat16",
                    "trust_remote_code": True,
                    "enable_thinking": False,
                    "reasoning_effort": "low",
                    "generation_prefix": "<|channel|>final<|message|>",
                },
                "eval_target": {
                    "name": "noveltybench_gpt_oss_20b_baseline",
                    "base_checkpoint": "openai/gpt-oss-20b",
                    "adapter_checkpoint": None,
                    "tokenizer_checkpoint": "openai/gpt-oss-20b",
                    "local_files_only": False,
                },
                "inference": {
                    "backend": "vllm",
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.9,
                    "max_num_seqs": 16,
                    "enforce_eager": False,
                    "trust_remote_code": True,
                },
                "noveltybench": {
                    "generation": {
                        "batch_size": 1,
                        "max_new_tokens": 64,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "do_sample": True,
                    },
                },
            }
        )
        model_cfg = OmegaConf.merge(cfg.model, cfg.noveltybench.generation, {"use_chat_template": True})
        assets = load_evaluation_assets(
            model_cfg,
            cfg.eval_target,
            cfg.inference,
            require_generation_backend=True,
            require_hf_model=False,
        )
        prompt = "Reply with exactly one city name."
        formatted_prompt = assets.generation_backend.format_prompt(prompt)
        self.assertIn("<|channel|>final<|message|>", formatted_prompt)
        raw_output = assets.generation_backend.generate_text(prompt, seed=11)
        self.assertNotIn("<think>", raw_output.lower())
        self.assertNotIn("<|channel|>analysis", raw_output.lower())

    def test_qwen3_small_scale_full_pipeline(self) -> None:
        if OmegaConf is None:
            self.skipTest("omegaconf is not installed in this environment")
        if not torch.cuda.is_available():
            self.skipTest("NoveltyBench Qwen integration requires CUDA")
        if "SCRATCH" not in os.environ:
            self.skipTest("NoveltyBench Qwen integration requires SCRATCH to be set")
        with TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create(
                {
                    "seed": 11,
                    "run_dir": str(Path(tmpdir) / "run"),
                    "output_root": tmpdir,
                    "model": {
                        "name": "qwen3_1p7b",
                        "checkpoint": "Qwen/Qwen3-1.7B",
                        "device": "cuda",
                        "dtype": "bfloat16",
                        "trust_remote_code": True,
                        "enable_thinking": False,
                    },
                    "eval_target": {
                        "name": "noveltybench_qwen3_1p7b_baseline",
                        "base_checkpoint": "Qwen/Qwen3-1.7B",
                        "adapter_checkpoint": None,
                        "tokenizer_checkpoint": "Qwen/Qwen3-1.7B",
                        "local_files_only": False,
                    },
                    "inference": {
                        "backend": "vllm",
                        "tensor_parallel_size": 1,
                        "gpu_memory_utilization": 0.9,
                        "max_num_seqs": 64,
                        "enforce_eager": False,
                        "trust_remote_code": True,
                    },
                    "noveltybench": {
                        "assets_root": str(Path(__file__).resolve().parents[3] / "benchmarks" / "noveltybench"),
                        "enabled_splits": ["curated"],
                        "max_prompts_per_split": 10,
                        "resume": False,
                        "output_root": tmpdir,
                        "generation": {
                            "num_generations": 5,
                            "batch_size": 4,
                            "max_new_tokens": 512,
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "do_sample": True,
                        },
                        "classifier": {
                            "tokenizer_checkpoint": "microsoft/deberta-v3-large",
                            "checkpoint": "yimingzhang/deberta-v3-large-generation-similarity",
                            "threshold": 0.102,
                            "max_length": 128,
                            "batch_size": 16,
                            "device": "cuda",
                            "local_files_only": False,
                        },
                        "reward_model": {
                            "checkpoint": "Skywork/Skywork-Reward-Gemma-2-27B-v0.2",
                            "batch_size": 4,
                            "dtype": "bfloat16",
                            "device_map": "auto",
                            "attn_implementation": "eager",
                            "local_files_only": False,
                        },
                        "patience": 0.8,
                    },
                }
            )
            os.environ["HF_HOME"] = os.environ["SCRATCH"]
            run_dir = eval_main_impl(cfg)
            self.assertTrue((run_dir / "summary.json").exists())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["metrics"]["num_prompts"], 10)


if __name__ == "__main__":
    unittest.main()
