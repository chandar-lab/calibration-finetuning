from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from types import SimpleNamespace
from typing import Any

from random_steering.inference.base import GenerationBackend, PromptInput
from random_steering.inference.ssot_adapters import SSOTModelAdapter, get_ssot_model_adapter
from random_steering.inference.ssot_prompts import build_system_prompt


_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*)$", re.DOTALL | re.IGNORECASE)
_RANDOM_STRING_PATTERN = re.compile(r"<random_string>.*?</random_string>", re.DOTALL | re.IGNORECASE)
_THINKING_PATTERN = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
_ANY_TAG_PATTERN = re.compile(r"</?[a-z_]+>", re.IGNORECASE)


class SSOTMode(str, Enum):
    PIF_NUMERIC = "pif_numeric"
    DAG_OPEN_RANDOM = "dag_open_random"


@dataclass(frozen=True)
class SSOTConfig:
    name: str = "ssot"
    enabled: bool = False
    mode: str = "auto"
    strict_answer_extraction: bool = True
    debug_log_examples: int = 3
    force_enable_thinking: str = "auto"
    prompt_variant: str = "faithful_v1"


def _as_ssot_config(cfg: Any) -> SSOTConfig:
    if isinstance(cfg, SSOTConfig):
        return cfg
    return SSOTConfig(
        name=str(getattr(cfg, "name", "ssot")),
        enabled=bool(getattr(cfg, "enabled", False)),
        mode=str(getattr(cfg, "mode", "auto")),
        strict_answer_extraction=bool(getattr(cfg, "strict_answer_extraction", True)),
        debug_log_examples=int(getattr(cfg, "debug_log_examples", 3)),
        force_enable_thinking=str(getattr(cfg, "force_enable_thinking", "auto")),
        prompt_variant=str(getattr(cfg, "prompt_variant", "faithful_v1")),
    )


def _clone_model_cfg_with_overrides(model_cfg: Any, **overrides: Any) -> Any:
    if model_cfg is None:
        return SimpleNamespace(**overrides)
    data: dict[str, Any] = {}
    if hasattr(model_cfg, "keys"):
        for key in model_cfg.keys():
            data[str(key)] = getattr(model_cfg, key)
    if hasattr(model_cfg, "__dict__"):
        data.update(vars(model_cfg))
    data.update(overrides)
    return SimpleNamespace(**data)


def _resolve_enable_thinking(cfg: SSOTConfig, backend: GenerationBackend) -> bool | None:
    override = str(cfg.force_enable_thinking).strip().lower()
    if override == "true":
        return True
    if override == "false":
        return False
    current = getattr(getattr(backend, "model_cfg", None), "enable_thinking", None)
    adapter = get_ssot_model_adapter(_get_backend_model_name(backend))
    if adapter.auto_enable_thinking is not None:
        return adapter.auto_enable_thinking
    return current


def _resolve_mode(cfg: SSOTConfig, explicit_mode: SSOTMode | str | None) -> SSOTMode:
    candidate = explicit_mode if explicit_mode is not None else cfg.mode
    if isinstance(candidate, SSOTMode):
        return candidate
    if candidate in {None, "auto"}:
        raise ValueError("SSOT mode must be set explicitly for this benchmark integration.")
    return SSOTMode(str(candidate))


def build_ssot_prompt(
    prompt: str,
    *,
    mode: SSOTMode,
    prompt_variant: str = "faithful_v1",
    model_name: str | None = None,
) -> PromptInput:
    if prompt_variant != "faithful_v1":
        raise ValueError(f"Unsupported SSoT prompt variant: {prompt_variant}")
    adapter = get_ssot_model_adapter(model_name)
    return [
        {
            "role": adapter.system_role,
            "content": build_system_prompt(mode, model_family=adapter.family),
        },
        {"role": adapter.user_role, "content": prompt},
    ]


def _fallback_extract_answer(text: str) -> str:
    without_random = _RANDOM_STRING_PATTERN.sub("", text)
    without_thinking = _THINKING_PATTERN.sub("", without_random)
    without_tags = _ANY_TAG_PATTERN.sub("", without_thinking)
    lines = [line.strip() for line in without_tags.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def extract_ssot_answer(
    text: str,
    *,
    strict: bool = True,
    model_name: str | None = None,
) -> tuple[str, bool]:
    adapter = get_ssot_model_adapter(model_name)
    normalized_text = adapter.normalize_output(text)
    match = _ANSWER_PATTERN.search(normalized_text)
    if match:
        return match.group(1).strip(), False
    unclosed_match = _UNCLOSED_ANSWER_PATTERN.search(normalized_text)
    if unclosed_match and not strict:
        return unclosed_match.group(1).strip(), True
    for extra_pattern in adapter.extra_answer_patterns:
        extra_match = extra_pattern.search(normalized_text)
        if extra_match:
            return extra_match.group(1).strip(), True
    if adapter.family == "gpt_oss":
        stripped = normalized_text.strip()
        if stripped and "\n" not in stripped and "<" not in stripped and "assistant" not in stripped.lower():
            return stripped, True
    if strict:
        return "", True
    fallback = _fallback_extract_answer(normalized_text)
    if fallback:
        return fallback, True
    return normalized_text.strip(), True


@dataclass
class SSOTWrappedBackend:
    backend: GenerationBackend
    cfg: SSOTConfig
    mode: SSOTMode
    _debug_examples_emitted: int = 0
    adapter: SSOTModelAdapter | None = None

    def __post_init__(self) -> None:
        self.adapter = get_ssot_model_adapter(_get_backend_model_name(self.backend))
        resolved_enable_thinking = _resolve_enable_thinking(self.cfg, self.backend)
        if hasattr(self.backend, "model_cfg"):
            overrides = dict(self.adapter.backend_overrides) if self.adapter is not None else {}
            overrides["enable_thinking"] = resolved_enable_thinking
            self.backend.model_cfg = _clone_model_cfg_with_overrides(
                getattr(self.backend, "model_cfg", None),
                **overrides,
            )

    @property
    def tokenizer(self) -> object:
        return self.backend.tokenizer

    def _debug_log(self, *, prompt: str, rewritten_prompt: PromptInput, raw_output: str, extracted_answer: str, degraded: bool) -> None:
        if self._debug_examples_emitted >= max(int(self.cfg.debug_log_examples), 0):
            return
        print(f"[SSOT:{self.mode.value}] prompt={prompt}", flush=True)
        print(f"[SSOT:{self.mode.value}] rewritten_prompt={rewritten_prompt}", flush=True)
        print(f"[SSOT:{self.mode.value}] raw_output={raw_output}", flush=True)
        print(
            f"[SSOT:{self.mode.value}] extracted_answer={extracted_answer} degraded_extraction={degraded}",
            flush=True,
        )
        self._debug_examples_emitted += 1

    def format_prompt(self, prompt: PromptInput) -> str:
        if not isinstance(prompt, str):
            return self.backend.format_prompt(prompt)
        return self.backend.format_prompt(
            build_ssot_prompt(
                prompt,
                mode=self.mode,
                prompt_variant=self.cfg.prompt_variant,
                model_name=_get_backend_model_name(self.backend),
            )
        )

    def strip_response(self, text: str) -> str:
        return self.backend.strip_response(text)

    def _rewrite_prompts(self, prompts: list[str]) -> list[PromptInput]:
        return [
            build_ssot_prompt(
                prompt,
                mode=self.mode,
                prompt_variant=self.cfg.prompt_variant,
                model_name=_get_backend_model_name(self.backend),
            )
            for prompt in prompts
        ]

    def generate_text(self, prompt: str, seed: int) -> str:
        return self.generate_text_batch([prompt], [seed])[0]

    def generate_text_batch(
        self,
        prompts: list[str],
        seeds: list[int],
        *,
        stop_strings: list[str] | None = None,
    ) -> list[str]:
        rewritten_prompts = self._rewrite_prompts(prompts)
        raw_outputs = self.backend.generate_text_batch(
            rewritten_prompts,
            seeds,
            stop_strings=stop_strings,
        )
        extracted_outputs: list[str] = []
        for prompt, rewritten_prompt, raw_output in zip(prompts, rewritten_prompts, raw_outputs, strict=True):
            stripped_output = self.backend.strip_response(raw_output)
            extracted_answer, degraded = extract_ssot_answer(
                stripped_output,
                strict=bool(self.cfg.strict_answer_extraction),
                model_name=_get_backend_model_name(self.backend),
            )
            self._debug_log(
                prompt=prompt,
                rewritten_prompt=rewritten_prompt,
                raw_output=raw_output,
                extracted_answer=extracted_answer,
                degraded=degraded,
            )
            extracted_outputs.append(extracted_answer)
        return extracted_outputs

    def score_prompt_continuation_pairs_batch(
        self,
        prompt_groups: list[list[str]],
        continuation_groups: list[list[str]],
    ) -> list[list[float]]:
        raise NotImplementedError("SSOT is not compatible with continuation scoring on rewritten prompts.")


def maybe_wrap_generation_backend(
    generation_backend: GenerationBackend | None,
    inference_method_cfg: Any | None,
    *,
    mode: SSOTMode | str | None = None,
) -> GenerationBackend | None:
    if generation_backend is None:
        return None
    cfg = _as_ssot_config(inference_method_cfg)
    if not cfg.enabled or str(cfg.name) != "ssot":
        return generation_backend
    resolved_mode = _resolve_mode(cfg, mode)
    return SSOTWrappedBackend(backend=generation_backend, cfg=cfg, mode=resolved_mode)


def _get_backend_model_name(backend: GenerationBackend) -> str | None:
    model_name = getattr(getattr(backend, "model_cfg", None), "checkpoint", None)
    if model_name:
        return str(model_name)
    model_name = getattr(backend, "model_name", None)
    if model_name:
        return str(model_name)
    model_name = getattr(getattr(backend, "eval_target_cfg", None), "base_checkpoint", None)
    if model_name:
        return str(model_name)
    return None
