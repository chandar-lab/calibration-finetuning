from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

from random_steering.inference.chat_format import strip_thinking_trace


_WHITESPACE_PATTERN = re.compile(r"\s+")
_THINK_TAG_PATTERN = re.compile(r"</?think>", re.IGNORECASE)
_OPTION_LABELS = ("A", "B", "C", "D")


def _normalize_inline(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text.strip())


def _clean_generation_text(text: str) -> str:
    cleaned = strip_thinking_trace(text)
    cleaned = _THINK_TAG_PATTERN.sub("", cleaned)
    return cleaned.strip()


@dataclass
class ParsedMcqGeneration:
    question_text: str = ""
    option_a: str = ""
    option_b: str = ""
    option_c: str = ""
    option_d: str = ""
    correct_answer: str = ""
    explanation: str = ""
    is_parseable: bool = False
    format_errors: list[str] = field(default_factory=list)
    unique_options: bool = False
    correct_option_nonempty: bool = False
    option_prefixes_valid: bool = False
    has_valid_correct_answer: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_mcq_generation(text: str) -> ParsedMcqGeneration:
    parsed = ParsedMcqGeneration()
    cleaned = _clean_generation_text(text)
    if not cleaned:
        parsed.format_errors.append("empty_response")
        return parsed

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    option_values: dict[str, str] = {}
    option_counts = {label: 0 for label in _OPTION_LABELS}
    question_count = 0
    explanation_count = 0
    correct_answer_count = 0

    for line in lines:
        if line.startswith("Question:"):
            question_count += 1
            if question_count > 1:
                parsed.format_errors.append("duplicate_question")
                continue
            parsed.question_text = _normalize_inline(line.partition(":")[2])
            continue

        matched_option = False
        for label in _OPTION_LABELS:
            prefix = f"{label}."
            if line.startswith(prefix):
                matched_option = True
                option_counts[label] += 1
                if option_counts[label] > 1:
                    parsed.format_errors.append(f"duplicate_option_{label.lower()}")
                    break
                option_values[label] = _normalize_inline(line[len(prefix) :])
                break
        if matched_option:
            continue

        if line.startswith("Correct Answer:"):
            correct_answer_count += 1
            if correct_answer_count > 1:
                parsed.format_errors.append("duplicate_correct_answer")
                continue
            parsed.correct_answer = _normalize_inline(line.partition(":")[2])
            continue

        if line.startswith("Explanation:"):
            explanation_count += 1
            if explanation_count > 1:
                parsed.format_errors.append("duplicate_explanation")
                continue
            parsed.explanation = _normalize_inline(line.partition(":")[2])
            continue

        parsed.format_errors.append("unexpected_line")

    for label in _OPTION_LABELS:
        if option_counts[label] == 0:
            parsed.format_errors.append(f"missing_option_{label.lower()}")

    if question_count == 0:
        parsed.format_errors.append("missing_question")
    if explanation_count == 0:
        parsed.format_errors.append("missing_explanation")
    if correct_answer_count == 0:
        parsed.format_errors.append("missing_correct_answer")

    parsed.option_a = option_values.get("A", "")
    parsed.option_b = option_values.get("B", "")
    parsed.option_c = option_values.get("C", "")
    parsed.option_d = option_values.get("D", "")

    if not parsed.question_text:
        parsed.format_errors.append("empty_question")
    if not parsed.explanation:
        parsed.format_errors.append("empty_explanation")

    option_texts = [
        ("a", parsed.option_a),
        ("b", parsed.option_b),
        ("c", parsed.option_c),
        ("d", parsed.option_d),
    ]
    for label, value in option_texts:
        if not value:
            parsed.format_errors.append(f"empty_option_{label}")

    parsed.has_valid_correct_answer = parsed.correct_answer in _OPTION_LABELS
    if not parsed.has_valid_correct_answer:
        parsed.format_errors.append("invalid_correct_answer")

    parsed.option_prefixes_valid = all(option_counts[label] == 1 for label in _OPTION_LABELS)
    normalized_options = [value for _label, value in option_texts if value]
    parsed.unique_options = len(set(normalized_options)) == 4

    answer_to_option = {
        "A": parsed.option_a,
        "B": parsed.option_b,
        "C": parsed.option_c,
        "D": parsed.option_d,
    }
    if parsed.has_valid_correct_answer:
        parsed.correct_option_nonempty = bool(answer_to_option[parsed.correct_answer])

    parsed.is_parseable = not parsed.format_errors
    return parsed
