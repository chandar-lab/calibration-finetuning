from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn

from random_steering.inference.chat_format import strip_model_response
from random_steering.inference.hf_backend import HFGenerationBackend
from random_steering.retention.tinybenchmarks_data import (
    GenerationExample,
    MultipleChoiceExample,
    load_tiny_gsm8k,
    load_tiny_hellaswag,
    load_tiny_mmlu,
    load_tiny_truthfulqa,
    load_tiny_winogrande,
)
from random_steering.retention.tinybenchmarks_tasks import run_multiple_choice_task
from random_steering.utils.hf import ensure_hf_home


TASK_LOADERS = {
    "tiny_mmlu": load_tiny_mmlu,
    "tiny_hellaswag": load_tiny_hellaswag,
    "tiny_truthfulqa": load_tiny_truthfulqa,
    "tiny_winogrande": load_tiny_winogrande,
}


class _TinyTokenizer:
    chat_template = None

    def __init__(self) -> None:
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.padding_side = "left"
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

    def __call__(self, texts, return_tensors: str | None = None, padding: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(text) for text in texts]
        if not padding:
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(encoded, dtype=torch.long),
                    "attention_mask": torch.ones((len(encoded), len(encoded[0])), dtype=torch.long),
                }
            return {"input_ids": encoded[0]}
        max_length = max(len(ids) for ids in encoded)
        input_ids = []
        attention_mask = []
        for ids in encoded:
            pad_length = max_length - len(ids)
            if self.padding_side == "left":
                input_ids.append(([self.pad_token_id] * pad_length) + ids)
                attention_mask.append(([0] * pad_length) + ([1] * len(ids)))
            else:
                input_ids.append(ids + ([self.pad_token_id] * pad_length))
                attention_mask.append(([1] * len(ids)) + ([0] * pad_length))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class _TransitionLM(nn.Module):
    def __init__(self, vocab_size: int, preferred_chars: dict[str, float]) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self._transition = torch.zeros(vocab_size + 32, vocab_size + 32, dtype=torch.float32)
        self._preferred_chars = preferred_chars

    def set_tokenizer(self, tokenizer: _TinyTokenizer) -> None:
        self._transition.zero_()
        for previous_id in range(self._transition.shape[0]):
            for char, logit in self._preferred_chars.items():
                token_id = tokenizer._token_for_char(char)
                self._transition[previous_id, token_id] = float(logit)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        _ = attention_mask
        logits = self._transition[input_ids]
        return SimpleNamespace(logits=logits)


@dataclass
class _ToyBackend:
    backend: HFGenerationBackend

    @classmethod
    def build(cls) -> "_ToyBackend":
        tokenizer = _TinyTokenizer()
        tokenizer.encode("Prompt alpha beta")
        model = _TransitionLM(tokenizer._next_id + 16, {"a": 8.0, "b": 1.0, "A": 8.0, "B": 1.0})
        model.set_tokenizer(tokenizer)
        backend = HFGenerationBackend(
            model=model,
            tokenizer=tokenizer,
            model_cfg=SimpleNamespace(
                checkpoint="toy",
                use_chat_template=False,
                enable_thinking=False,
                reasoning_effort=None,
                generation_prefix=None,
                batch_size=4,
                do_sample=False,
                max_new_tokens=8,
            ),
            model_name="toy",
        )
        return cls(backend=backend)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect TinyBenchmarks retention prompts and scoring")
    parser.add_argument("--mode", choices=["static", "inspect", "smoke-model"], default="static")
    parser.add_argument("--task", choices=sorted(TASK_LOADERS), default="tiny_truthfulqa")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--use-chat-template", action="store_true", default=False)
    return parser.parse_args()


def _load_examples(task_name: str) -> list[MultipleChoiceExample]:
    return TASK_LOADERS[task_name]()


def _load_gsm8k_examples() -> list[GenerationExample]:
    return load_tiny_gsm8k()


def run_static_checks() -> None:
    examples_by_task = {task_name: _load_examples(task_name) for task_name in TASK_LOADERS}
    for task_name, examples in examples_by_task.items():
        example = examples[0]
        assert len(example.choices) == len(example.continuations) >= 2, task_name
        assert all(continuation for continuation in example.continuations), task_name

    mmlu = examples_by_task["tiny_mmlu"][0]
    assert "Answer:" in mmlu.prompt
    assert mmlu.choices[0] in mmlu.prompt
    assert mmlu.continuations == (" A", " B", " C", " D")
    assert mmlu.metric_name == "acc"

    hellaswag = examples_by_task["tiny_hellaswag"][0]
    assert hellaswag.prompt.endswith(".")
    assert hellaswag.continuations[0].strip() == hellaswag.choices[0]
    assert "A." not in hellaswag.prompt
    assert hellaswag.metric_name == "acc_norm"
    assert all("[" not in choice and "]" not in choice for choice in hellaswag.choices)

    truthfulqa = examples_by_task["tiny_truthfulqa"][0]
    assert truthfulqa.prompt.rstrip().endswith("A:")
    assert truthfulqa.continuations[0].strip() == truthfulqa.choices[0]
    assert truthfulqa.metric_name == "mc2"
    assert truthfulqa.target_scores is not None
    assert sum(truthfulqa.target_scores) >= 1.0

    winogrande = examples_by_task["tiny_winogrande"][0]
    assert winogrande.choice_contexts is not None
    assert winogrande.prompt.endswith(winogrande.choices[winogrande.correct_choice_index])
    assert all(
        context.endswith(choice) for context, choice in zip(winogrande.choice_contexts, winogrande.choices, strict=True)
    )
    assert winogrande.continuations[0] == winogrande.continuations[1]
    assert winogrande.continuations[0].startswith(" ")

    gsm8k = _load_gsm8k_examples()[0]
    assert gsm8k.prompt.rstrip().endswith("Answer:")
    assert "Question:" in gsm8k.prompt

    toy = _ToyBackend.build().backend
    scores = toy.score_prompt_continuation_pairs_batch([["Prompt", "Prompt"]], [[" a", " b"]])[0]
    assert scores[0] > scores[1]
    assert strip_model_response("<think>hidden</think> visible", model_name="Qwen/Qwen3-1.7B") == "visible"

    truth_result = run_multiple_choice_task(
        backend=toy,
        retention_cfg=SimpleNamespace(batch_size=1),
        task_name="tiny_truthfulqa",
        benchmark_name="truthfulqa",
        examples=[
            MultipleChoiceExample(
                task_name="tiny_truthfulqa",
                example_id="toy_truth_000",
                prompt="Prompt",
                choices=("alpha", "beta", "gamma"),
                continuations=(" a", " b", " b"),
                correct_choice_index=0,
                metric_name="mc2",
                metadata={},
                target_scores=(1.0, 1.0, 0.0),
            )
        ],
    )
    assert abs(truth_result.records[0]["example_score"] - 0.999) < 1e-3

    norm_result = run_multiple_choice_task(
        backend=toy,
        retention_cfg=SimpleNamespace(batch_size=1),
        task_name="tiny_hellaswag",
        benchmark_name="hellaswag",
        examples=[
            MultipleChoiceExample(
                task_name="tiny_hellaswag",
                example_id="toy_hsw_000",
                prompt="Prompt",
                choices=("a", "bbbbbbbb"),
                continuations=(" a", " bbbbbbbb"),
                correct_choice_index=0,
                metric_name="acc_norm",
                metadata={},
            )
        ],
    )
    assert norm_result.summary_row["metric_name"] == "acc_norm"
    assert "choice_scores_norm" in norm_result.records[0]
    print("Static retention checks passed.")


def inspect_example(task_name: str, index: int, checkpoint: str | None = None, *, use_chat_template: bool) -> None:
    examples = _load_examples(task_name)
    example = examples[index]
    print(f"task={task_name} index={index} example_id={example.example_id}")
    print("\nRAW PROMPT\n")
    print(example.prompt)
    print(f"\nMETRIC: {example.metric_name}\n")
    print("\nCANDIDATES\n")
    for label, (choice, continuation) in enumerate(zip(example.choices, example.continuations, strict=True)):
        letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[label]
        print(f"{letter}: choice={choice}")
        if example.choice_contexts is not None:
            print(f"{letter}: context={example.choice_contexts[label]!r}")
        print(f"{letter}: continuation={continuation!r}")
    if not checkpoint:
        return

    ensure_hf_home()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()
    backend = HFGenerationBackend(
        model=model,
        tokenizer=tokenizer,
        model_cfg=SimpleNamespace(
            checkpoint=checkpoint,
            use_chat_template=use_chat_template,
            enable_thinking=False,
            reasoning_effort=None,
            generation_prefix=None,
            batch_size=2,
            do_sample=False,
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        ),
        model_name=checkpoint,
    )
    formatted_prompt = backend.format_prompt(example.prompt)
    print("\nFORMATTED PROMPT\n")
    print(formatted_prompt)
    scores = backend.score_prompt_continuation_pairs_batch(
        [list(example.choice_contexts) if example.choice_contexts is not None else [example.prompt] * len(example.continuations)],
        [list(example.continuations)],
    )[0]
    print("\nSCORES\n")
    for label, score in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", scores, strict=False):
        print(f"{label}: {score:.4f}")


def run_model_smoke(task_name: str, checkpoint: str, max_examples: int, batch_size: int, *, use_chat_template: bool) -> None:
    ensure_hf_home()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()
    backend = HFGenerationBackend(
        model=model,
        tokenizer=tokenizer,
        model_cfg=SimpleNamespace(
            checkpoint=checkpoint,
            use_chat_template=use_chat_template,
            enable_thinking=False,
            reasoning_effort=None,
            generation_prefix=None,
            batch_size=batch_size,
            do_sample=False,
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        ),
        model_name=checkpoint,
    )
    examples = _load_examples(task_name)[:max_examples]
    result = run_multiple_choice_task(
        backend=backend,
        retention_cfg=SimpleNamespace(batch_size=batch_size),
        task_name=task_name,
        benchmark_name=task_name.replace("tiny_", ""),
        examples=examples,
    )
    predictions = Counter(record["prediction"] for record in result.records)
    print(f"task={task_name} checkpoint={checkpoint}")
    print(f"accuracy={result.summary_row['accuracy']:.4f}")
    print(f"prediction_counts={dict(predictions)}")
    for record in result.records[: min(3, len(result.records))]:
        print(
            f"{record['example_id']} pred={record['prediction']} gold={record['gold']} "
            f"pred_text={record['prediction_text']!r}"
        )


def main() -> None:
    args = parse_args()
    if args.mode == "static":
        run_static_checks()
        return
    if args.mode == "inspect":
        inspect_example(args.task, args.index, args.checkpoint, use_chat_template=args.use_chat_template)
        return
    run_model_smoke(
        args.task,
        args.checkpoint,
        args.max_examples,
        args.batch_size,
        use_chat_template=args.use_chat_template,
    )


if __name__ == "__main__":
    main()
