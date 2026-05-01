from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from datasets import DownloadConfig, load_dataset


DATASET_SIZE = 100


def _hellaswag_preprocess(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace(" [title]", ". ")
    while "[" in cleaned and "]" in cleaned:
        start = cleaned.find("[")
        end = cleaned.find("]", start)
        if end < 0:
            break
        cleaned = cleaned[:start] + cleaned[end + 1 :]
    return cleaned.replace("  ", " ")

SCHEMA_SNAPSHOTS: dict[str, dict[str, Any]] = {
    "tiny_mmlu": {
        "dataset_args": ["tinyBenchmarks/tinyMMLU"],
        "dataset_kwargs": {"split": "test"},
        "keys": ["answer", "choices", "input_formatted", "question", "subject"],
    },
    "tiny_hellaswag": {
        "dataset_args": ["tinyBenchmarks/tinyHellaswag"],
        "dataset_kwargs": {"split": "validation"},
        "keys": [
            "activity_label",
            "ctx",
            "ctx_a",
            "ctx_b",
            "endings",
            "ind",
            "input_formatted",
            "label",
            "source_id",
            "split",
            "split_type",
        ],
    },
    "tiny_truthfulqa": {
        "dataset_args": ["tinyBenchmarks/tinyTruthfulQA", "multiple_choice"],
        "dataset_kwargs": {"split": "validation"},
        "keys": ["input_formatted", "mc1_targets", "mc2_targets", "question"],
        "nested_keys": {
            "mc1_targets": ["choices", "labels"],
            "mc2_targets": ["choices", "labels"],
        },
    },
    "tiny_winogrande": {
        "dataset_args": ["tinyBenchmarks/tinyWinogrande", "winogrande_xl"],
        "dataset_kwargs": {"split": "validation"},
        "keys": ["answer", "input_formatted", "option1", "option2", "sentence"],
    },
    "tiny_gsm8k": {
        "dataset_args": ["tinyBenchmarks/tinyGSM8k", "main"],
        "dataset_kwargs": {"split": "test"},
        "keys": ["answer", "input_formatted", "question"],
    },
}


@dataclass(frozen=True)
class MultipleChoiceExample:
    task_name: str
    example_id: str
    prompt: str
    choices: tuple[str, ...]
    continuations: tuple[str, ...]
    correct_choice_index: int
    metric_name: str
    metadata: dict[str, Any]
    target_scores: tuple[float, ...] | None = None
    choice_contexts: tuple[str, ...] | None = None


@dataclass(frozen=True)
class GenerationExample:
    task_name: str
    example_id: str
    prompt: str
    target_answer: str
    metadata: dict[str, Any]


DatasetLoader = Callable[..., Any]


def _assert_size(task_name: str, dataset: Any) -> None:
    if len(dataset) != DATASET_SIZE:
        raise ValueError(f"{task_name} expected {DATASET_SIZE} examples but found {len(dataset)}")


def _assert_schema(task_name: str, row: dict[str, Any]) -> None:
    snapshot = SCHEMA_SNAPSHOTS[task_name]
    keys = sorted(row.keys())
    expected_keys = sorted(snapshot["keys"])
    if keys != expected_keys:
        raise KeyError(f"{task_name} schema mismatch. Expected {expected_keys}, found {keys}")
    for nested_key, expected_nested_keys in snapshot.get("nested_keys", {}).items():
        nested_value = row.get(nested_key)
        if not isinstance(nested_value, dict):
            raise KeyError(f"{task_name} expected dict at {nested_key}, found {type(nested_value).__name__}")
        nested_keys = sorted(nested_value.keys())
        if nested_keys != sorted(expected_nested_keys):
            raise KeyError(
                f"{task_name} nested schema mismatch for {nested_key}. "
                f"Expected {sorted(expected_nested_keys)}, found {nested_keys}"
            )


def _load_rows(task_name: str, loader: DatasetLoader | None = None) -> list[dict[str, Any]]:
    snapshot = SCHEMA_SNAPSHOTS[task_name]
    dataset_loader = loader or load_dataset
    try:
        dataset = dataset_loader(*snapshot["dataset_args"], **snapshot["dataset_kwargs"])
    except Exception:
        if loader is not None:
            raise
        dataset = dataset_loader(
            *snapshot["dataset_args"],
            download_config=DownloadConfig(local_files_only=True),
            **snapshot["dataset_kwargs"],
        )
    _assert_size(task_name, dataset)
    rows = [dict(dataset[index]) for index in range(len(dataset))]
    if rows:
        _assert_schema(task_name, rows[0])
    return rows


def _jsonable_row(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, tuple):
            payload[key] = list(value)
        else:
            payload[key] = value
    return payload


def _metadata(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "row_index": index,
        "row": _jsonable_row(row),
    }


def _label_index(labels: list[int], *, task_name: str, example_id: str) -> int:
    positive = [index for index, label in enumerate(labels) if int(label) == 1]
    if len(positive) != 1:
        raise ValueError(f"{task_name}:{example_id} expected exactly one positive label, found {positive}")
    return positive[0]


def _first_positive_index(labels: list[int], *, task_name: str, example_id: str) -> int:
    positive = [index for index, label in enumerate(labels) if int(label) == 1]
    if not positive:
        raise ValueError(f"{task_name}:{example_id} expected at least one positive label")
    return positive[0]


def _extract_gsm8k_target(answer: str, *, example_id: str) -> str:
    marker = "####"
    if marker not in answer:
        raise ValueError(f"tiny_gsm8k:{example_id} gold answer missing '{marker}' marker")
    target = answer.split(marker)[-1].strip()
    if not target:
        raise ValueError(f"tiny_gsm8k:{example_id} gold answer missing final value after '{marker}'")
    return target


def _prepend_space_if_needed(prompt: str, continuation: str) -> str:
    if not continuation:
        raise ValueError("Continuation text must be non-empty")
    if prompt.endswith((" ", "\n", "\t")):
        return continuation
    if continuation.startswith((" ", "\n", "\t")):
        return continuation
    return f" {continuation}"


def _label_continuations(prompt: str, num_choices: int) -> tuple[str, ...]:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if num_choices > len(labels):
        raise ValueError(f"Unsupported number of choices: {num_choices}")
    return tuple(_prepend_space_if_needed(prompt, label) for label in labels[:num_choices])


def _winogrande_prompt_prefix(input_formatted: str, option1: str, option2: str, sentence: str) -> str:
    for option in sorted((option1, option2), key=len, reverse=True):
        if input_formatted.endswith(option):
            return input_formatted[: -len(option)]
    if "_" not in sentence:
        raise ValueError("tiny_winogrande sentence is missing '_' placeholder")
    prefix, _suffix = sentence.split("_", 1)
    return prefix


def load_tiny_mmlu(loader: DatasetLoader | None = None) -> list[MultipleChoiceExample]:
    rows = _load_rows("tiny_mmlu", loader)
    examples: list[MultipleChoiceExample] = []
    for index, row in enumerate(rows):
        correct_choice_index = int(row["answer"])
        choices = tuple(str(choice) for choice in row["choices"])
        if not 0 <= correct_choice_index < len(choices):
            raise ValueError(f"tiny_mmlu:{index} answer index {correct_choice_index} out of bounds")
        examples.append(
            MultipleChoiceExample(
                task_name="tiny_mmlu",
                example_id=f"tiny_mmlu_{index:03d}",
                prompt=str(row["input_formatted"]),
                choices=choices,
                continuations=_label_continuations(str(row["input_formatted"]), len(choices)),
                correct_choice_index=correct_choice_index,
                metric_name="acc",
                metadata=_metadata(row, index),
            )
        )
    return examples


def load_tiny_hellaswag(loader: DatasetLoader | None = None) -> list[MultipleChoiceExample]:
    rows = _load_rows("tiny_hellaswag", loader)
    examples: list[MultipleChoiceExample] = []
    for index, row in enumerate(rows):
        correct_choice_index = int(row["label"])
        choices = tuple(_hellaswag_preprocess(str(choice)) for choice in row["endings"])
        if not 0 <= correct_choice_index < len(choices):
            raise ValueError(f"tiny_hellaswag:{index} label index {correct_choice_index} out of bounds")
        examples.append(
            MultipleChoiceExample(
                task_name="tiny_hellaswag",
                example_id=f"tiny_hellaswag_{index:03d}",
                prompt=str(row["input_formatted"]),
                choices=choices,
                continuations=tuple(_prepend_space_if_needed(str(row["input_formatted"]), choice) for choice in choices),
                correct_choice_index=correct_choice_index,
                metric_name="acc_norm",
                metadata=_metadata(row, index),
            )
        )
    return examples


def load_tiny_truthfulqa(loader: DatasetLoader | None = None) -> list[MultipleChoiceExample]:
    rows = _load_rows("tiny_truthfulqa", loader)
    examples: list[MultipleChoiceExample] = []
    for index, row in enumerate(rows):
        mc2_targets = dict(row["mc2_targets"])
        choices = tuple(str(choice) for choice in mc2_targets["choices"])
        labels = tuple(float(label) for label in mc2_targets["labels"])
        correct_choice_index = _first_positive_index(
            list(mc2_targets["labels"]),
            task_name="tiny_truthfulqa",
            example_id=str(index),
        )
        examples.append(
            MultipleChoiceExample(
                task_name="tiny_truthfulqa",
                example_id=f"tiny_truthfulqa_{index:03d}",
                prompt=str(row["input_formatted"]),
                choices=choices,
                continuations=tuple(_prepend_space_if_needed(str(row["input_formatted"]), choice) for choice in choices),
                correct_choice_index=correct_choice_index,
                metric_name="mc2",
                target_scores=labels,
                metadata=_metadata(row, index),
            )
        )
    return examples


def load_tiny_winogrande(loader: DatasetLoader | None = None) -> list[MultipleChoiceExample]:
    rows = _load_rows("tiny_winogrande", loader)
    examples: list[MultipleChoiceExample] = []
    for index, row in enumerate(rows):
        correct_choice_index = int(row["answer"]) - 1
        choices = (str(row["option1"]), str(row["option2"]))
        sentence = str(row["sentence"])
        prompt = str(row["input_formatted"])
        prompt_prefix = _winogrande_prompt_prefix(prompt, choices[0], choices[1], sentence)
        suffix = sentence.split("_", 1)[1].strip()
        correct_choice = choices[correct_choice_index]
        if not prompt.endswith(correct_choice):
            raise ValueError(f"tiny_winogrande:{index} input_formatted does not end with gold option")
        choice_contexts = tuple(f"{prompt_prefix}{choice}" for choice in choices)
        continuations = tuple(_prepend_space_if_needed(choice_context, suffix) for choice_context in choice_contexts)
        if not 0 <= correct_choice_index < len(choices):
            raise ValueError(f"tiny_winogrande:{index} answer index {correct_choice_index} out of bounds")
        examples.append(
            MultipleChoiceExample(
                task_name="tiny_winogrande",
                example_id=f"tiny_winogrande_{index:03d}",
                prompt=prompt,
                choices=choices,
                continuations=continuations,
                correct_choice_index=correct_choice_index,
                metric_name="acc",
                choice_contexts=choice_contexts,
                metadata=_metadata(row, index),
            )
        )
    return examples


def load_tiny_gsm8k(loader: DatasetLoader | None = None) -> list[GenerationExample]:
    rows = _load_rows("tiny_gsm8k", loader)
    examples: list[GenerationExample] = []
    for index, row in enumerate(rows):
        examples.append(
            GenerationExample(
                task_name="tiny_gsm8k",
                example_id=f"tiny_gsm8k_{index:03d}",
                prompt=str(row["input_formatted"]),
                target_answer=_extract_gsm8k_target(str(row["answer"]), example_id=str(index)),
                metadata=_metadata(row, index),
            )
        )
    return examples


def schema_snapshots_as_rows() -> list[dict[str, Any]]:
    return [
        {
            "task_name": task_name,
            **asdict(
                _SchemaSnapshot(
                    task_name=task_name,
                    dataset_args=tuple(str(arg) for arg in snapshot["dataset_args"]),
                    dataset_kwargs=dict(snapshot["dataset_kwargs"]),
                    keys=tuple(str(key) for key in snapshot["keys"]),
                    nested_keys={key: tuple(value) for key, value in snapshot.get("nested_keys", {}).items()},
                )
            ),
        }
        for task_name, snapshot in sorted(SCHEMA_SNAPSHOTS.items())
    ]


@dataclass(frozen=True)
class _SchemaSnapshot:
    task_name: str
    dataset_args: tuple[str, ...]
    dataset_kwargs: dict[str, Any]
    keys: tuple[str, ...]
    nested_keys: dict[str, tuple[str, ...]]
