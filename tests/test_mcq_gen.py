from __future__ import annotations

import importlib.util
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

from random_steering.mcq_gen.metrics import compute_mcq_metrics
from random_steering.mcq_gen.parser import parse_mcq_generation
from random_steering.mcq_gen.prompt import MEDICAL_MCQ_PROMPT, get_medical_mcq_prompt
from random_steering.mcq_gen.evaluate import generate_mcq_rows

try:
    from random_steering.mcq_gen.eval import main_impl
except ImportError:
    main_impl = None


SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


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


class _TinyMcqModel(nn.Module):
    def __init__(self, tokenizer: _TinyTokenizer) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.tokenizer = tokenizer
        self.config = SimpleNamespace(use_cache=False)
        self.generation_config = SimpleNamespace()
        self.outputs = [
            "\nQuestion: Which medication is first-line for anaphylaxis?\nA. Oral diphenhydramine\nB. Intramuscular epinephrine\nC. Inhaled albuterol\nD. Oral prednisone\nCorrect Answer: B\nExplanation: Intramuscular epinephrine is the first-line treatment for anaphylaxis.\n",
            "Question: Which laboratory study best reflects long-term glycemic control?\nA. Fasting insulin\nB. Random glucose\nC. Hemoglobin A1c\nD. Serum ketones\nCorrect Answer: C\nExplanation: Hemoglobin A1c reflects average glycemia over roughly three months.\n",
            "Question: A patient with atrial fibrillation is most at risk of which complication?\nA. Embolic stroke\nB. Tension pneumothorax\nC. Acute pancreatitis\nD. Nephrolithiasis\nCorrect Answer: A\nExplanation: Atrial fibrillation increases risk of left atrial thrombus and embolic stroke.\n",
            "Question: Which imaging modality is preferred first for suspected gallstones?\nA. MRI brain\nB. CT angiography\nC. Chest radiograph\nD. Right upper quadrant ultrasound\nCorrect Answer: D\nExplanation: Right upper quadrant ultrasound is the usual first-line imaging study for gallstones.\n",
        ]
        for output in self.outputs:
            tokenizer.encode(output)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, generation_config=None, generator=None):
        _ = attention_mask, generation_config
        batch_outputs: list[list[int]] = []
        max_output_length = 0
        if isinstance(generator, list):
            generators = generator
        elif generator is None:
            generators = [None] * input_ids.shape[0]
        else:
            generators = [generator] * input_ids.shape[0]
        for row_index in range(input_ids.shape[0]):
            current_generator = generators[row_index]
            if current_generator is None:
                choice = int(torch.randint(0, len(self.outputs), (1,)).item())
            else:
                choice = int(torch.randint(0, len(self.outputs), (1,), generator=current_generator).item())
            generated = [int(token_id) for token_id in self.tokenizer.encode(self.outputs[choice])]
            batch_outputs.append(generated)
            max_output_length = max(max_output_length, len(generated))

        generated_tensor = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1] + max_output_length),
            dtype=torch.long,
            device=input_ids.device,
        )
        generated_tensor[:, : input_ids.shape[1]] = input_ids
        for row_index, generated in enumerate(batch_outputs):
            generated_tensor[row_index, input_ids.shape[1] : input_ids.shape[1] + len(generated)] = torch.tensor(
                generated,
                dtype=torch.long,
                device=input_ids.device,
            )
        return generated_tensor


class _TinyMcqModelNoGenerator(_TinyMcqModel):
    def __init__(self, tokenizer: _TinyTokenizer) -> None:
        super().__init__(tokenizer)
        self.batch_sizes: list[int] = []

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, generation_config=None, generator=None):
        if generator is not None:
            raise ValueError("generator is not supported")
        self.batch_sizes.append(int(input_ids.shape[0]))
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generator=None,
        )


class McqGenTests(unittest.TestCase):
    def test_prompt_matches_expected_text(self) -> None:
        self.assertEqual(get_medical_mcq_prompt(), MEDICAL_MCQ_PROMPT)
        self.assertIn("Please strictly follow this format:", MEDICAL_MCQ_PROMPT)
        self.assertIn("Correct Answer: [A/B/C/D]", MEDICAL_MCQ_PROMPT)
        self.assertIn("evenly distributed among A, B, C, D", MEDICAL_MCQ_PROMPT)

    def test_parser_accepts_valid_generation(self) -> None:
        parsed = parse_mcq_generation(
            """
            Question: Which organism commonly causes community-acquired pneumonia?
            A. Streptococcus pneumoniae
            B. Candida albicans
            C. Clostridioides difficile
            D. Plasmodium falciparum
            Correct Answer: A
            Explanation: Streptococcus pneumoniae is a common cause of community-acquired pneumonia.
            """
        )
        self.assertTrue(parsed.is_parseable)
        self.assertEqual(parsed.correct_answer, "A")
        self.assertTrue(parsed.unique_options)
        self.assertTrue(parsed.correct_option_nonempty)

    def test_parser_rejects_missing_correct_answer(self) -> None:
        parsed = parse_mcq_generation(
            """
            Question: Which medication is used for hypothyroidism?
            A. Levothyroxine
            B. Insulin
            C. Warfarin
            D. Heparin
            Explanation: Levothyroxine replaces deficient thyroid hormone.
            """
        )
        self.assertFalse(parsed.is_parseable)
        self.assertIn("missing_correct_answer", parsed.format_errors)

    def test_parser_rejects_invalid_answer_label(self) -> None:
        parsed = parse_mcq_generation(
            """
            Question: Which chamber ejects blood into the aorta?
            A. Left ventricle
            B. Right ventricle
            C. Left atrium
            D. Right atrium
            Correct Answer: E
            Explanation: The left ventricle ejects blood into the aorta.
            """
        )
        self.assertFalse(parsed.is_parseable)
        self.assertIn("invalid_correct_answer", parsed.format_errors)

    def test_parser_rejects_duplicated_option_prefixes(self) -> None:
        parsed = parse_mcq_generation(
            """
            Question: Which electrolyte abnormality causes peaked T waves?
            A. Hyperkalemia
            A. Hypokalemia
            C. Hyponatremia
            D. Hypercalcemia
            Correct Answer: A
            Explanation: Hyperkalemia is associated with peaked T waves.
            """
        )
        self.assertFalse(parsed.is_parseable)
        self.assertIn("duplicate_option_a", parsed.format_errors)
        self.assertIn("missing_option_b", parsed.format_errors)

    def test_parser_normalizes_recoverable_whitespace(self) -> None:
        parsed = parse_mcq_generation(
            "  Question:   Which test helps diagnose diabetes?  \n"
            " A.   Hemoglobin   A1c \n"
            "B.  Troponin   I\n"
            "C.  D-dimer\n"
            "D.  Lactate\n"
            "Correct Answer:   A \n"
            "Explanation:  Hemoglobin A1c is commonly used to diagnose diabetes.  \n"
        )
        self.assertTrue(parsed.is_parseable)
        self.assertEqual(parsed.question_text, "Which test helps diagnose diabetes?")
        self.assertEqual(parsed.option_a, "Hemoglobin A1c")

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for MCQ metrics tests")
    def test_metrics_uniform_counts(self) -> None:
        rows = [
            {"correct_answer": "A", "is_parseable": True, "has_valid_correct_answer": True, "unique_options": True},
            {"correct_answer": "B", "is_parseable": True, "has_valid_correct_answer": True, "unique_options": True},
            {"correct_answer": "C", "is_parseable": True, "has_valid_correct_answer": True, "unique_options": True},
            {"correct_answer": "D", "is_parseable": True, "has_valid_correct_answer": True, "unique_options": True},
        ]
        summary, _rows = compute_mcq_metrics(rows)
        self.assertEqual(summary["tv_distance_from_uniform"], 0.0)
        self.assertGreaterEqual(summary["chi_square_pvalue"], 0.99)

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for MCQ metrics tests")
    def test_metrics_biased_counts(self) -> None:
        rows = [
            {"correct_answer": "A", "is_parseable": True, "has_valid_correct_answer": True, "unique_options": True}
            for _ in range(8)
        ]
        rows.extend(
            {"correct_answer": "B", "is_parseable": True, "has_valid_correct_answer": True, "unique_options": False}
            for _ in range(2)
        )
        summary, _rows = compute_mcq_metrics(rows)
        self.assertGreater(summary["tv_distance_from_uniform"], 0.0)
        self.assertLess(summary["chi_square_pvalue"], 0.05)

    def test_mcq_gen_config_composes(self) -> None:
        if initialize_config_dir is None or compose is None:
            self.skipTest("hydra is not installed in this environment")
        repo_root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base=None, config_dir=str(repo_root / "conf")):
            cfg = compose(config_name="mcq_gen_eval_config")
        self.assertEqual(cfg.eval_target.name, "mcq_gen_baseline")
        self.assertEqual(cfg.mcq_gen.num_samples, 1000)

    def test_mcq_generation_keeps_batched_generation_when_generator_kwarg_is_rejected(self) -> None:
        tokenizer = _TinyTokenizer()
        model = _TinyMcqModelNoGenerator(tokenizer)
        rows = generate_mcq_rows(
            model=model,
            tokenizer=tokenizer,
            model_cfg=SimpleNamespace(enable_thinking=False),
            mcq_gen_cfg=SimpleNamespace(
                num_samples=4,
                batch_size=2,
                generation=SimpleNamespace(
                    max_new_tokens=256,
                    temperature=1.0,
                    top_p=1.0,
                    do_sample=True,
                ),
            ),
            prompt=get_medical_mcq_prompt(),
            base_seed=11,
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(all("raw_response" in row for row in rows))
        self.assertEqual(model.batch_sizes, [2, 2])

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for MCQ smoke evaluation")
    def test_mcq_gen_eval_writes_outputs(self) -> None:
        if OmegaConf is None or main_impl is None:
            self.skipTest("omegaconf/hydra stack is not installed in this environment")
        tokenizer = _TinyTokenizer()
        model = _TinyMcqModel(tokenizer)

        with TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create(
                {
                    "seed": 11,
                    "output_root": tmpdir,
                    "model": {
                        "device": "cpu",
                        "dtype": "float32",
                        "enable_thinking": False,
                    },
                    "mcq_gen": {
                        "num_samples": 6,
                        "batch_size": 2,
                        "generation": {
                            "max_new_tokens": 256,
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "do_sample": True,
                        },
                        "output_root": tmpdir,
                    },
                    "eval_target": {
                        "name": "mcq_gen_test",
                        "base_checkpoint": "base",
                        "adapter_checkpoint": None,
                        "tokenizer_checkpoint": "tokenizer",
                        "local_files_only": True,
                    },
                }
            )

            bundle = SimpleNamespace(model=model, tokenizer=tokenizer)
            with patch("random_steering.mcq_gen.eval.load_evaluation_bundle", return_value=bundle):
                run_dir = main_impl(cfg)

            self.assertTrue((run_dir / "config_resolved.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "metrics" / "summary.json").exists())
            self.assertTrue((run_dir / "metrics" / "answer_frequencies.csv").exists())
            samples_path = run_dir / "samples" / "generated_mcqs.jsonl"
            self.assertTrue(samples_path.exists())
            rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["is_parseable"] for row in rows))


if __name__ == "__main__":
    unittest.main()
