from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import gc
import torch

from random_steering.inference.base import PromptInput
from random_steering.inference.chat_format import format_prompt, strip_model_response


@dataclass
class VLLMGenerationBackend:
    tokenizer: Any
    model_cfg: Any
    eval_target_cfg: Any
    inference_cfg: Any

    def __post_init__(self) -> None:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError("`vllm` is required for inference=vllm. Install it in the eval environment.") from exc

        model_path = str(self.eval_target_cfg.base_checkpoint)
        tokenizer_path = str(getattr(self.eval_target_cfg, "tokenizer_checkpoint", model_path))
        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "tokenizer": tokenizer_path,
            "trust_remote_code": bool(getattr(self.inference_cfg, "trust_remote_code", True)),
            "tensor_parallel_size": int(getattr(self.inference_cfg, "tensor_parallel_size", 1)),
            "gpu_memory_utilization": float(getattr(self.inference_cfg, "gpu_memory_utilization", 0.9)),
            "max_num_seqs": int(getattr(self.inference_cfg, "max_num_seqs", 256)),
            "enforce_eager": bool(getattr(self.inference_cfg, "enforce_eager", False)),
        }
        dtype = getattr(self.model_cfg, "dtype", None)
        if dtype is not None:
            llm_kwargs["dtype"] = str(dtype)
        self.llm = LLM(**llm_kwargs)

    def _format_prompt(self, prompt: PromptInput) -> str:
        return format_prompt(
            self.tokenizer,
            prompt,
            use_chat_template=bool(getattr(self.model_cfg, "use_chat_template", True)),
            enable_thinking=getattr(self.model_cfg, "enable_thinking", None),
            reasoning_effort=getattr(self.model_cfg, "reasoning_effort", None),
            generation_prefix=getattr(self.model_cfg, "generation_prefix", None),
        )

    def format_prompt(self, prompt: PromptInput) -> str:
        return self._format_prompt(prompt)

    def strip_response(self, text: str) -> str:
        return strip_model_response(text, model_name=str(getattr(self.eval_target_cfg, "base_checkpoint", "")))

    def _sampling_params(self, seed: int, *, stop_strings: list[str] | None = None):
        from vllm import SamplingParams

        do_sample = bool(getattr(self.model_cfg, "do_sample", True))
        params: dict[str, Any] = {
            "n": 1,
            "seed": int(seed),
            "max_tokens": int(getattr(self.model_cfg, "max_new_tokens", 256)),
            "skip_special_tokens": True,
        }
        if do_sample:
            params["temperature"] = float(getattr(self.model_cfg, "temperature", 1.0))
            params["top_p"] = float(getattr(self.model_cfg, "top_p", 1.0))
        else:
            params["temperature"] = 0.0
            params["top_p"] = 1.0
        if stop_strings:
            params["stop"] = list(stop_strings)
        return SamplingParams(**params)

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
        outputs: list[str] = []
        batch_size = max(int(getattr(self.model_cfg, "batch_size", 1)), 1)
        for start in range(0, len(prompts), batch_size):
            prompt_batch = prompts[start : start + batch_size]
            seed_batch = seeds[start : start + batch_size]
            formatted_prompts = [self._format_prompt(prompt) for prompt in prompt_batch]
            params_batch = [self._sampling_params(seed, stop_strings=stop_strings) for seed in seed_batch]
            try:
                request_outputs = self.llm.generate(formatted_prompts, params_batch, use_tqdm=False)
            except TypeError:
                request_outputs = self.llm.generate(prompts=formatted_prompts, sampling_params=params_batch, use_tqdm=False)
            for request_output in request_outputs:
                if not getattr(request_output, "outputs", None):
                    outputs.append("")
                    continue
                outputs.append(str(request_output.outputs[0].text).strip())
        return outputs

    def score_prompt_continuation_pairs_batch(
        self,
        prompt_groups: list[list[str]],
        continuation_groups: list[list[str]],
    ) -> list[list[float]]:
        raise NotImplementedError("vLLM backend does not support teacher-forced continuation scoring.")

    def close(self) -> None:
        llm = getattr(self, "llm", None)
        if llm is None:
            return
        try:
            if hasattr(llm, "sleep"):
                llm.sleep(level=2)
        except Exception:
            pass
        try:
            llm_engine = getattr(llm, "llm_engine", None)
            if llm_engine is not None and hasattr(llm_engine, "shutdown"):
                llm_engine.shutdown()
        except Exception:
            pass
        self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
