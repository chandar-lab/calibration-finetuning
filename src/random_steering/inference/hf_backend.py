from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from random_steering.inference.base import PromptInput
from random_steering.inference.chat_format import format_prompt, strip_model_response


def _truncate_at_stop_strings(text: str, stop_strings: list[str] | None) -> str:
    if not stop_strings:
        return text
    stop_positions = [text.find(stop_string) for stop_string in stop_strings if stop_string]
    stop_positions = [position for position in stop_positions if position >= 0]
    if not stop_positions:
        return text
    return text[: min(stop_positions)]


@dataclass
class HFGenerationBackend:
    model: Any
    tokenizer: Any
    model_cfg: Any
    model_name: str | None = None

    def __post_init__(self) -> None:
        if self.model_name is None:
            self.model_name = str(getattr(self.model_cfg, "checkpoint", ""))

    def _token_ids(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "encode"):
            return list(self.tokenizer.encode(text, add_special_tokens=True))
        encoded = self.tokenizer(text)
        return list(encoded["input_ids"])

    def _tokenize_text_batch(self, texts: list[str], *, padding_side: str = "right") -> dict[str, torch.Tensor]:
        original_padding_side = getattr(self.tokenizer, "padding_side", None)
        restore_padding_side = original_padding_side is not None and original_padding_side != padding_side
        if restore_padding_side:
            self.tokenizer.padding_side = padding_side
        try:
            try:
                encoded = self.tokenizer(texts, return_tensors="pt", padding=True)
                if isinstance(encoded, dict) and "input_ids" in encoded:
                    return encoded
            except TypeError:
                pass
        finally:
            if restore_padding_side:
                self.tokenizer.padding_side = original_padding_side

        tokenized = [self._token_ids(text) for text in texts]
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        max_length = max(len(token_ids) for token_ids in tokenized)
        input_ids = []
        attention_mask = []
        for token_ids in tokenized:
            pad_length = max_length - len(token_ids)
            if padding_side == "left":
                input_ids.append(([int(pad_token_id)] * pad_length) + token_ids)
                attention_mask.append(([0] * pad_length) + ([1] * len(token_ids)))
            else:
                input_ids.append(token_ids + ([int(pad_token_id)] * pad_length))
                attention_mask.append(([1] * len(token_ids)) + ([0] * pad_length))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def _move_model_inputs(self, model_inputs: dict[str, Any]) -> dict[str, Any]:
        device = next(self.model.parameters()).device
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in model_inputs.items()}

    def _generation_config(self):
        generation_config = deepcopy(getattr(self.model, "generation_config", None))
        if generation_config is None:
            try:
                from transformers import GenerationConfig

                generation_config = GenerationConfig()
            except Exception:
                class _GenerationConfig:
                    pass

                generation_config = _GenerationConfig()
        generation_config.max_new_tokens = int(getattr(self.model_cfg, "max_new_tokens", 256))
        generation_config.do_sample = bool(getattr(self.model_cfg, "do_sample", True))
        generation_config.pad_token_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", 0)
        if generation_config.do_sample:
            generation_config.temperature = float(getattr(self.model_cfg, "temperature", 1.0))
            generation_config.top_p = float(getattr(self.model_cfg, "top_p", 1.0))
            top_k = int(getattr(self.model_cfg, "top_k", 0))
            generation_config.top_k = top_k if top_k > 0 else None
        else:
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None
        return generation_config

    def _build_generators(self, seeds: list[int]) -> list[torch.Generator]:
        device = next(self.model.parameters()).device
        generator_device = "cpu" if device.type == "cpu" else device.type
        generators: list[torch.Generator] = []
        for seed in seeds:
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(int(seed))
            generators.append(generator)
        return generators

    def format_prompt(self, prompt: PromptInput) -> str:
        return format_prompt(
            self.tokenizer,
            prompt,
            model_name=self.model_name,
            use_chat_template=bool(getattr(self.model_cfg, "use_chat_template", True)),
            enable_thinking=getattr(self.model_cfg, "enable_thinking", None),
            reasoning_effort=getattr(self.model_cfg, "reasoning_effort", None),
            generation_prefix=getattr(self.model_cfg, "generation_prefix", None),
        )

    def strip_response(self, text: str) -> str:
        return strip_model_response(text, model_name=self.model_name)

    def generate_text(self, prompt: PromptInput, seed: int) -> str:
        return self.generate_text_batch([prompt], [seed])[0]

    def generate_text_batch(
        self,
        prompts: list[PromptInput],
        seeds: list[int],
        *,
        stop_strings: list[str] | None = None,
    ) -> list[str]:
        if len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must have the same length")
        generation_config = self._generation_config()
        outputs: list[str] = []
        batch_size = max(int(getattr(self.model_cfg, "batch_size", 1)), 1)
        for start in range(0, len(prompts), batch_size):
            prompt_batch = prompts[start : start + batch_size]
            seed_batch = seeds[start : start + batch_size]
            formatted_prompts = [self.format_prompt(prompt) for prompt in prompt_batch]
            model_inputs = self._tokenize_text_batch(formatted_prompts, padding_side="left")
            prompt_width = int(model_inputs["input_ids"].shape[1])
            model_inputs = self._move_model_inputs(model_inputs)
            generators = self._build_generators(seed_batch)
            device = next(self.model.parameters()).device
            with torch.no_grad():
                try:
                    generated = self.model.generate(**model_inputs, generation_config=generation_config, generator=generators)
                except (TypeError, ValueError):
                    torch.manual_seed(int(seed_batch[0]))
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(int(seed_batch[0]))
                    generated = self.model.generate(**model_inputs, generation_config=generation_config)
            for row_index in range(len(prompt_batch)):
                decoded = self.tokenizer.decode(generated[row_index, prompt_width:], skip_special_tokens=True)
                outputs.append(_truncate_at_stop_strings(decoded, stop_strings).strip())
        return outputs

    def score_prompt_continuation_pairs_batch(
        self,
        prompt_groups: list[list[str]],
        continuation_groups: list[list[str]],
    ) -> list[list[float]]:
        if len(prompt_groups) != len(continuation_groups):
            raise ValueError("prompt_groups and continuation_groups must have the same length")
        if not prompt_groups:
            return []

        sequence_texts: list[str] = []
        sequence_specs: list[tuple[int, int, int]] = []
        for group_index, (prompt_group, continuation_group) in enumerate(
            zip(prompt_groups, continuation_groups, strict=True)
        ):
            if len(prompt_group) != len(continuation_group):
                raise ValueError("Each prompt group must match its continuation group length")
            for choice_index, (prompt_text, continuation_text) in enumerate(
                zip(prompt_group, continuation_group, strict=True)
            ):
                formatted_prompt = self.format_prompt(prompt_text)
                prompt_inputs = self._tokenize_text_batch([formatted_prompt])
                prompt_length = int(prompt_inputs["attention_mask"].sum(dim=1).item())
                sequence_texts.append(formatted_prompt + continuation_text)
                sequence_specs.append((group_index, choice_index, prompt_length))

        full_inputs = self._tokenize_text_batch(sequence_texts)
        full_lengths = full_inputs["attention_mask"].sum(dim=1).tolist()
        full_inputs = self._move_model_inputs(full_inputs)

        with torch.no_grad():
            logits = self.model(**full_inputs).logits

        scores_by_group: list[list[float]] = [[] for _ in prompt_groups]
        for sequence_index, (group_index, choice_index, prompt_length) in enumerate(sequence_specs):
            full_length = int(full_lengths[sequence_index])
            continuation_ids = full_inputs["input_ids"][sequence_index, prompt_length:full_length]
            if continuation_ids.numel() == 0:
                raise ValueError("Continuation tokenization must add at least one token")
            start = max(prompt_length - 1, 0)
            end = start + continuation_ids.shape[0]
            continuation_logits = logits[sequence_index, start:end]
            log_probs = torch.log_softmax(continuation_logits, dim=-1)
            token_log_probs = log_probs.gather(1, continuation_ids.unsqueeze(-1)).squeeze(-1)
            score = float(token_log_probs.sum().item())
            while len(scores_by_group[group_index]) <= choice_index:
                scores_by_group[group_index].append(float("-inf"))
            scores_by_group[group_index][choice_index] = score
        return scores_by_group
