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
from omegaconf import OmegaConf

from random_steering.calibrate_sft.data import OutputSpace, PreparedPrompt, PromptSpec, TokenizedOutputSpace
from random_steering.calibrate_sft.eval import run_calibrate_eval
from random_steering.inference.ssot import SSOTMode, build_ssot_prompt, extract_ssot_answer, maybe_wrap_generation_backend
from random_steering.open_random_gen.eval import main_impl
from random_steering.types import DistributionSpec


class _DummyTokenizer:
    chat_template = None


class _DummyGenerationBackend:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.tokenizer = _DummyTokenizer()
        self.model_cfg = SimpleNamespace(checkpoint="Qwen/Qwen3-1.7B", enable_thinking=False)
        self.seen_prompts: list[object] = []

    def format_prompt(self, prompt):
        return str(prompt)

    def strip_response(self, text: str) -> str:
        return text.strip()

    def generate_text(self, prompt, seed: int) -> str:
        return self.generate_text_batch([prompt], [seed])[0]

    def generate_text_batch(self, prompts, seeds, *, stop_strings=None) -> list[str]:
        _ = seeds, stop_strings
        self.seen_prompts.extend(prompts)
        return [self.outputs[index % len(self.outputs)] for index in range(len(prompts))]

    def score_prompt_continuation_pairs_batch(self, prompt_groups, continuation_groups):
        _ = prompt_groups, continuation_groups
        raise NotImplementedError


def _make_prepared_prompt() -> PreparedPrompt:
    prompt_spec = PromptSpec(
        split="test",
        family_id="uniform",
        display_name="Uniform",
        tier="I",
        distribution_id="uniform_test",
        parameters={"a": 0.0, "b": 1.0},
        kind="continuous",
        integer_only=False,
    )
    distribution_spec = DistributionSpec(
        distribution_id="uniform_test",
        display_name="Uniform",
        parameters={"a": 0.0, "b": 1.0},
        family="uniform",
        support_min=0.0,
        support_max=1.0,
        integer_only=False,
        scipy_name="uniform",
        scipy_params={"loc": 0.0, "scale": 1.0},
    )
    output_space = OutputSpace(
        values=("0.25", "0.75"),
        probabilities=(0.5, 0.5),
        lower_bound=0.0,
        upper_bound=1.0,
        integer_only=False,
        num_decimals=2,
    )
    tokenized_output_space = TokenizedOutputSpace(
        output_space=output_space,
        candidate_token_ids=((1,), (2,)),
        prefix_targets={},
    )
    return PreparedPrompt(
        prompt_spec=prompt_spec,
        distribution_spec=distribution_spec,
        prompt_text="Generate exactly ONE random number sampled from Uniform(a=0,b=1). Output ONLY the number.",
        formatted_prompt="Generate exactly ONE random number sampled from Uniform(a=0,b=1). Output ONLY the number.",
        prompt_token_ids=(1, 2, 3),
        output_space=output_space,
        tokenized_output_space=tokenized_output_space,
    )


class SSOTTests(unittest.TestCase):
    def test_extract_ssot_answer_prefers_answer_tags(self) -> None:
        answer, degraded = extract_ssot_answer(
            "<random_string>abc</random_string><thinking>foo</thinking><answer>0.42</answer>"
        )
        self.assertEqual(answer, "0.42")
        self.assertFalse(degraded)

    def test_extract_ssot_answer_requires_tags_in_strict_mode(self) -> None:
        answer, degraded = extract_ssot_answer("<random_string>abc</random_string>\nFinal choice")
        self.assertEqual(answer, "")
        self.assertTrue(degraded)

    def test_extract_ssot_answer_falls_back_in_non_strict_mode(self) -> None:
        answer, degraded = extract_ssot_answer("<random_string>abc</random_string>\nFinal choice", strict=False)
        self.assertEqual(answer, "Final choice")
        self.assertTrue(degraded)

    def test_build_ssot_prompt_uses_gpt_oss_developer_role(self) -> None:
        prompt = build_ssot_prompt(
            "Choose a random city.",
            mode=SSOTMode.DAG_OPEN_RANDOM,
            model_name="openai/gpt-oss-20b",
        )
        self.assertEqual(prompt[0]["role"], "developer")
        self.assertEqual(prompt[1]["role"], "user")

    def test_extract_ssot_answer_handles_gpt_oss_inline_answer(self) -> None:
        answer, degraded = extract_ssot_answer(
            "analysisNeed seed.assistantfinal<random_string>abc</random_string>"
            "<thinking>...</thinking>assistantanswer>Raccoon</answer>",
            model_name="openai/gpt-oss-20b",
        )
        self.assertEqual(answer, "Raccoon")
        self.assertFalse(degraded)

    def test_extract_ssot_answer_handles_gpt_oss_bare_final_answer(self) -> None:
        answer, degraded = extract_ssot_answer(
            "0.5436",
            model_name="openai/gpt-oss-20b",
        )
        self.assertEqual(answer, "0.5436")
        self.assertTrue(degraded)

    def test_config_defaults_include_inference_method(self) -> None:
        if initialize_config_dir is None or compose is None:
            self.skipTest("hydra is not installed in this environment")
        repo_root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base=None, config_dir=str(repo_root / "conf")):
            calibrate_cfg = compose(config_name="calibrate_eval_config")
            open_cfg = compose(config_name="open_random_gen_eval_config")
        self.assertEqual(calibrate_cfg.inference_method.name, "none")
        self.assertFalse(calibrate_cfg.inference_method.enabled)
        self.assertEqual(open_cfg.inference_method.name, "none")
        self.assertFalse(open_cfg.inference_method.enabled)

    def test_maybe_wrap_generation_backend_enables_reasoning_for_qwen3(self) -> None:
        backend = _DummyGenerationBackend(outputs=["<answer>0.5</answer>"])
        wrapped = maybe_wrap_generation_backend(
            backend,
            OmegaConf.create({"name": "ssot", "enabled": True, "force_enable_thinking": "auto"}),
            mode=SSOTMode.PIF_NUMERIC,
        )
        self.assertIsNotNone(wrapped)
        self.assertTrue(getattr(backend.model_cfg, "enable_thinking"))

    def test_maybe_wrap_generation_backend_preserves_gpt_oss_reasoning_toggle(self) -> None:
        backend = _DummyGenerationBackend(outputs=["<answer>0.5</answer>"])
        backend.model_cfg = SimpleNamespace(
            checkpoint="openai/gpt-oss-20b",
            enable_thinking=False,
            generation_prefix="<|channel|>final<|message|>",
        )
        wrapped = maybe_wrap_generation_backend(
            backend,
            OmegaConf.create({"name": "ssot", "enabled": True, "force_enable_thinking": "auto"}),
            mode=SSOTMode.PIF_NUMERIC,
        )
        self.assertIsNotNone(wrapped)
        self.assertFalse(getattr(backend.model_cfg, "enable_thinking"))
        self.assertIsNone(getattr(backend.model_cfg, "generation_prefix"))

    def test_calibrate_eval_rejects_ssot_logit_eval(self) -> None:
        cfg = OmegaConf.create(
            {
                "seed": 11,
                "output_root": "unused",
                "model": {"checkpoint": "Qwen/Qwen3-1.7B"},
                "data": {},
                "train": {"tau": 1.0, "epsilon_smoothing": 0.0},
                "experiment": {
                    "sample_eval": {"enabled": True, "protocol": "independent", "num_samples": 4, "seeds": [11], "batch_size": 2},
                    "logit_eval": {"enabled": True},
                },
                "eval_target": {"name": "calibrate_ssot_test", "splits": ["test"]},
                "inference": {"name": "hf"},
                "inference_method": {"name": "ssot", "enabled": True},
            }
        )
        with self.assertRaisesRegex(ValueError, "logit_eval"):
            run_calibrate_eval(cfg)

    def test_open_random_gen_rejects_ssot_support_metrics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "prompts.json"
            prompt_path.write_text(json.dumps(["Choose a random city."]), encoding="utf-8")
            cfg = OmegaConf.create(
                {
                    "seed": 11,
                    "output_root": tmpdir,
                    "model": {"device": "cpu", "dtype": "float32"},
                    "open_random_gen": {
                        "prompts_path": str(prompt_path),
                        "compute_support_metrics": True,
                        "compute_sampling_metrics": True,
                        "top_p_threshold": 0.9,
                        "num_samples_per_prompt": 4,
                        "batch_size": 2,
                        "generation": {"max_new_tokens": 16, "temperature": 1.0, "top_p": 1.0, "do_sample": True},
                        "output_root": tmpdir,
                    },
                    "eval_target": {"name": "open_random_ssot_test"},
                    "inference": {"name": "hf"},
                    "inference_method": {"name": "ssot", "enabled": True},
                }
            )
            with self.assertRaisesRegex(ValueError, "compute_support_metrics"):
                main_impl(cfg)

    def test_calibrate_eval_smoketest_with_ssot_wrapper(self) -> None:
        backend = _DummyGenerationBackend(
            outputs=[
                "<random_string>abc123</random_string><thinking>Use it</thinking><answer>0.25</answer>",
                "<random_string>xyz999</random_string><thinking>Use it</thinking><answer>0.75</answer>",
            ]
        )
        assets = SimpleNamespace(tokenizer=backend.tokenizer, hf_model=None, generation_backend=backend)
        prepared_splits = {"test": [_make_prepared_prompt()]}
        cfg = OmegaConf.create(
            {
                "seed": 11,
                "output_root": None,
                "model": {
                    "checkpoint": "Qwen/Qwen3-1.7B",
                    "device": "cpu",
                    "dtype": "float32",
                    "enable_thinking": False,
                },
                "data": {},
                "train": {"tau": 1.0, "epsilon_smoothing": 0.0},
                "experiment": {
                    "sample_eval": {
                        "enabled": True,
                        "protocol": "independent",
                        "num_samples": 8,
                        "seeds": [11],
                        "batch_size": 4,
                        "max_prompts_per_split": 1,
                    },
                    "logit_eval": {"enabled": False},
                },
                "eval_target": {"name": "calibrate_ssot_smoke", "splits": ["test"]},
                "inference": {"name": "hf"},
                "inference_method": {"name": "ssot", "enabled": True, "debug_log_examples": 1},
            }
        )

        with TemporaryDirectory() as tmpdir:
            cfg.output_root = tmpdir
            with patch("random_steering.calibrate_sft.eval.load_evaluation_assets", return_value=assets):
                with patch("random_steering.calibrate_sft.eval.build_prepared_splits", return_value=prepared_splits):
                    run_calibrate_eval(cfg)

            run_dir = Path(tmpdir) / "calibrate_ssot_smoke"
            self.assertTrue((run_dir / "config_resolved.json").exists())
            self.assertTrue((run_dir / "metrics" / "summary.json").exists())
            self.assertTrue((run_dir / "metrics" / "per_distribution.csv").exists())
            self.assertTrue(any(isinstance(prompt, list) for prompt in backend.seen_prompts))
            sample_file = run_dir / "samples" / "test_uniform_test_seed11.jsonl"
            self.assertTrue(sample_file.exists())
            with sample_file.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["raw_text"] in {"0.25", "0.75"} for row in rows))

    def test_open_random_gen_smoketest_with_ssot_wrapper(self) -> None:
        backend = _DummyGenerationBackend(
            outputs=[
                "<random_string>abc123</random_string><thinking>Use it</thinking><answer>Alpha City</answer>",
                "<random_string>xyz999</random_string><thinking>Use it</thinking><answer>Beta Town</answer>",
            ]
        )
        assets = SimpleNamespace(tokenizer=backend.tokenizer, hf_model=None, generation_backend=backend)

        with TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "prompts.json"
            prompt_path.write_text(json.dumps(["Choose a random city.", "Choose a random animal."]), encoding="utf-8")
            cfg = OmegaConf.create(
                {
                    "seed": 11,
                    "output_root": tmpdir,
                    "model": {
                        "checkpoint": "Qwen/Qwen3-1.7B",
                        "device": "cpu",
                        "dtype": "float32",
                        "enable_thinking": False,
                    },
                    "open_random_gen": {
                        "prompts_path": str(prompt_path),
                        "compute_support_metrics": False,
                        "compute_sampling_metrics": True,
                        "top_p_threshold": 0.9,
                        "num_samples_per_prompt": 8,
                        "batch_size": 4,
                        "generation": {"max_new_tokens": 16, "temperature": 1.0, "top_p": 1.0, "do_sample": True},
                        "output_root": tmpdir,
                    },
                    "eval_target": {
                        "name": "open_random_ssot_smoke",
                        "base_checkpoint": "base",
                        "adapter_checkpoint": None,
                        "tokenizer_checkpoint": "tokenizer",
                        "local_files_only": True,
                    },
                    "inference": {"name": "hf", "backend": "hf"},
                    "inference_method": {"name": "ssot", "enabled": True, "debug_log_examples": 1},
                }
            )

            with patch("random_steering.open_random_gen.eval.load_evaluation_assets", return_value=assets):
                run_dir = main_impl(cfg)

            self.assertTrue((run_dir / "config_resolved.json").exists())
            self.assertTrue((run_dir / "metrics" / "summary.json").exists())
            self.assertTrue((run_dir / "samples" / "per_prompt.jsonl").exists())
            self.assertTrue(any(isinstance(prompt, list) for prompt in backend.seen_prompts))
            with (run_dir / "samples" / "per_prompt.jsonl").open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["normalized_outputs"][0], "Alpha City")
            self.assertEqual(rows[1]["normalized_outputs"][0], "Beta Town")


if __name__ == "__main__":
    unittest.main()
