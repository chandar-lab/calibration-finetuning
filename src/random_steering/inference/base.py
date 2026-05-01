from __future__ import annotations

from typing import Protocol

PromptInput = str | list[dict[str, str]]


class GenerationBackend(Protocol):
    tokenizer: object

    def format_prompt(self, prompt: PromptInput) -> str: ...

    def strip_response(self, text: str) -> str: ...

    def generate_text(self, prompt: PromptInput, seed: int) -> str: ...

    def generate_text_batch(
        self,
        prompts: list[PromptInput],
        seeds: list[int],
        *,
        stop_strings: list[str] | None = None,
    ) -> list[str]: ...

    def score_prompt_continuation_pairs_batch(
        self,
        prompt_groups: list[list[str]],
        continuation_groups: list[list[str]],
    ) -> list[list[float]]: ...
