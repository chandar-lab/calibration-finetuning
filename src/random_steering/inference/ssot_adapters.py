from __future__ import annotations

from dataclasses import dataclass, field
import re

from random_steering.inference.chat_format import infer_model_family


@dataclass(frozen=True)
class SSOTModelAdapter:
    family: str
    system_role: str = "system"
    user_role: str = "user"
    auto_enable_thinking: bool | None = None
    backend_overrides: dict[str, object] = field(default_factory=dict)
    extra_answer_patterns: tuple[re.Pattern[str], ...] = ()

    def normalize_output(self, text: str) -> str:
        return text


_GPT_OSS_INLINE_TAG_PATTERN = re.compile(r"assistant(?:final|commentary)\s*(?=<)", re.IGNORECASE)
_GPT_OSS_ASSISTANT_ANSWER_OPEN_PATTERN = re.compile(r"assistantanswer>", re.IGNORECASE)
_GPT_OSS_ASSISTANT_ANSWER_INLINE_PATTERN = re.compile(
    r"assistantanswer\s*([^\n<][^\n<]*)",
    re.IGNORECASE,
)
_GPT_OSS_ASSISTANT_FINAL_INLINE_PATTERN = re.compile(
    r"assistantfinal\s*([^\n<][^\n<]*)",
    re.IGNORECASE,
)


class GPTOSSAdapter(SSOTModelAdapter):
    def normalize_output(self, text: str) -> str:
        normalized = _GPT_OSS_INLINE_TAG_PATTERN.sub("", text)
        normalized = _GPT_OSS_ASSISTANT_ANSWER_OPEN_PATTERN.sub("<answer>", normalized)
        return normalized


_DEFAULT_ADAPTER = SSOTModelAdapter(family="default")
_QWEN3_ADAPTER = SSOTModelAdapter(family="qwen3", auto_enable_thinking=True)
_LLAMA_ADAPTER = SSOTModelAdapter(family="llama")
_GEMMA_ADAPTER = SSOTModelAdapter(family="gemma")
_GPT_OSS_ADAPTER = GPTOSSAdapter(
    family="gpt_oss",
    system_role="developer",
    backend_overrides={"generation_prefix": None},
    extra_answer_patterns=(
        _GPT_OSS_ASSISTANT_ANSWER_INLINE_PATTERN,
        _GPT_OSS_ASSISTANT_FINAL_INLINE_PATTERN,
    ),
)


def get_ssot_model_adapter(model_name: str | None) -> SSOTModelAdapter:
    family = infer_model_family(model_name)
    if family == "qwen3":
        return _QWEN3_ADAPTER
    if family == "gpt_oss":
        return _GPT_OSS_ADAPTER
    if family == "llama":
        return _LLAMA_ADAPTER
    if family == "gemma":
        return _GEMMA_ADAPTER
    return _DEFAULT_ADAPTER
