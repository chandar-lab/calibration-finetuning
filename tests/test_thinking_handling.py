from __future__ import annotations

import unittest

from random_steering.distributions.registry import get_distribution
from random_steering.eval.protocol_runner import run_protocol
from random_steering.inference.chat_format import format_prompt, strip_thinking_trace


class _TokenizerSupportsThinking:
    chat_template = "dummy"

    def __init__(self) -> None:
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        _ = messages
        self.kwargs = kwargs
        return "formatted"


class _TokenizerNoThinkingKwarg:
    chat_template = "dummy"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _ = (messages, tokenize, add_generation_prompt)
        return "formatted-no-thinking-kwarg"


class _FakeSteering:
    def on_request_start(self, target_distribution, request_seed: int) -> None:
        _ = (target_distribution, request_seed)

    def install_hooks(self) -> None:
        return

    def remove_hooks(self) -> None:
        return


class _FakeEngine:
    def generate_text(self, prompt: str, seed: int) -> str:
        _ = (prompt, seed)
        return "<think>internal reasoning</think>\n3.14159"


class ThinkingHandlingTests(unittest.TestCase):
    def test_strip_thinking_trace_closed_block(self) -> None:
        text = "<think>reasoning</think>\n42"
        self.assertEqual(strip_thinking_trace(text), "42")

    def test_strip_thinking_trace_unclosed_block(self) -> None:
        text = "12.3 <think>reasoning without close"
        self.assertEqual(strip_thinking_trace(text), "12.3")

    def test_format_prompt_passes_enable_thinking_when_supported(self) -> None:
        tokenizer = _TokenizerSupportsThinking()
        formatted = format_prompt(tokenizer, "hello", enable_thinking=False)
        self.assertEqual(formatted, "formatted")
        self.assertIn("enable_thinking", tokenizer.kwargs)
        self.assertFalse(tokenizer.kwargs["enable_thinking"])

    def test_format_prompt_falls_back_when_enable_thinking_not_supported(self) -> None:
        tokenizer = _TokenizerNoThinkingKwarg()
        formatted = format_prompt(tokenizer, "hello", enable_thinking=True)
        self.assertEqual(formatted, "formatted-no-thinking-kwarg")

    def test_protocol_runner_parses_after_think_strip(self) -> None:
        spec = get_distribution("gaussian_0_1")
        records = run_protocol(
            engine=_FakeEngine(),
            steering_policy=_FakeSteering(),
            steering_name="none",
            spec=spec,
            protocol="independent",
            num_samples=1,
            base_seed=11,
        )
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].is_valid)
        self.assertAlmostEqual(records[0].parsed_value or 0.0, 3.14159, places=5)
        self.assertEqual(records[0].raw_text, "3.14159")


if __name__ == "__main__":
    unittest.main()
