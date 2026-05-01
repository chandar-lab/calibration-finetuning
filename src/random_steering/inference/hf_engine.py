from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from random_steering.inference.chat_format import format_prompt
from random_steering.utils.hf import ensure_hf_home


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class HFEngine:
    model_cfg: object

    def __post_init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = getattr(self.model_cfg, "device", "cuda")

    def load(self) -> None:
        checkpoint = self.model_cfg.checkpoint
        trust_remote_code = bool(getattr(self.model_cfg, "trust_remote_code", True))
        dtype_str = str(getattr(self.model_cfg, "dtype", "bfloat16"))
        dtype = _DTYPE_MAP.get(dtype_str, torch.bfloat16)
        ensure_hf_home()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
        except TypeError:
            # Backward compatibility for transformers versions that still use torch_dtype.
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                trust_remote_code=trust_remote_code,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def generate_text(self, prompt: str, seed: int) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("HFEngine.load() must be called before generate_text().")

        enable_thinking = getattr(self.model_cfg, "enable_thinking", None)
        reasoning_effort = getattr(self.model_cfg, "reasoning_effort", None)
        generation_prefix = getattr(self.model_cfg, "generation_prefix", None)
        formatted_prompt = format_prompt(
            self.tokenizer,
            prompt,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            generation_prefix=generation_prefix,
        )
        model_inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        torch.manual_seed(int(seed))
        if self.model.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

        with torch.no_grad():
            output = self.model.generate(
                **model_inputs,
                max_new_tokens=int(getattr(self.model_cfg, "max_new_tokens", 256)),
                do_sample=bool(getattr(self.model_cfg, "do_sample", True)),
                temperature=float(getattr(self.model_cfg, "temperature", 1.0)),
                top_p=float(getattr(self.model_cfg, "top_p", 1.0)),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = output[0][model_inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
