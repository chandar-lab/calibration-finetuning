from __future__ import annotations

import re

from transformers import PreTrainedTokenizerBase


_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_PATTERN = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)


def infer_model_family(model_name: str | None) -> str:
    normalized = str(model_name or "").strip().lower()
    if "qwen3" in normalized:
        return "qwen3"
    if "gpt-oss" in normalized or "gpt_oss" in normalized:
        return "gpt_oss"
    if "llama" in normalized:
        return "llama"
    if "gemma" in normalized:
        return "gemma"
    return "default"


def strip_thinking_trace(text: str) -> str:
    cleaned = _THINK_BLOCK_PATTERN.sub("", text)
    cleaned = _UNCLOSED_THINK_PATTERN.sub("", cleaned)
    return cleaned.strip()


def strip_model_response(text: str, *, model_name: str | None = None) -> str:
    family = infer_model_family(model_name)
    if family == "qwen3":
        return strip_thinking_trace(text)
    return text.strip()


def _normalize_messages(prompt: str | list[dict[str, str]]) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [
        {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }
        for message in prompt
    ]


def _flatten_messages(messages: list[dict[str, str]], *, generation_prefix: str | None = None) -> str:
    sections: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().capitalize() or "User"
        content = str(message.get("content", "")).strip()
        sections.append(f"{role}:\n{content}")
    sections.append("Assistant:")
    flattened = "\n\n".join(sections)
    if generation_prefix:
        flattened += str(generation_prefix)
    return flattened


def format_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str | list[dict[str, str]],
    *,
    model_name: str | None = None,
    use_chat_template: bool = True,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    generation_prefix: str | None = None,
) -> str:
    messages = _normalize_messages(prompt)
    if use_chat_template and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        _ = infer_model_family(model_name)
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = str(reasoning_effort)
        try:
            formatted = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            kwargs.pop("reasoning_effort", None)
            formatted = tokenizer.apply_chat_template(messages, **kwargs)
        if generation_prefix:
            formatted += generation_prefix
        return formatted
    if isinstance(prompt, str):
        return prompt
    return _flatten_messages(messages, generation_prefix=generation_prefix)
