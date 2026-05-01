from __future__ import annotations

from random_steering.distributions.parser import ParsedNumber, parse_batch_numbers, parse_single_number
from random_steering.distributions.validators import validate_value
from random_steering.inference.chat_format import strip_thinking_trace
from random_steering.prompts.templates import build_batch_prompt, build_independent_prompt
from random_steering.types import DistributionSpec, SampleRecord
from random_steering.utils.seed import request_seed


def _to_record(
    parsed: ParsedNumber,
    spec: DistributionSpec,
    protocol: str,
    request_index: int,
    seed: int,
    prompt: str,
    raw_text: str,
    steering_name: str,
    metadata: dict | None = None,
) -> SampleRecord:
    support_ok = False
    support_error = None
    if parsed.valid and parsed.value is not None:
        support_ok, support_error = validate_value(spec, parsed.value)

    error_code = parsed.error_code if parsed.error_code else support_error
    is_valid = parsed.valid and support_ok

    return SampleRecord(
        protocol=protocol,
        distribution_id=spec.distribution_id,
        request_index=request_index,
        seed=seed,
        prompt=prompt,
        raw_text=raw_text,
        parsed_value=parsed.value,
        is_valid=is_valid,
        support_ok=support_ok,
        error_code=error_code,
        steering_name=steering_name,
        metadata=metadata or {},
    )


def run_protocol(
    *,
    engine,
    steering_policy,
    steering_name: str,
    spec: DistributionSpec,
    protocol: str,
    num_samples: int,
    base_seed: int,
) -> list[SampleRecord]:
    if protocol == "batch":
        prompt = build_batch_prompt(spec, num_samples)
        steering_policy.on_request_start(spec, base_seed)
        steering_policy.install_hooks()
        try:
            raw_text = engine.generate_text(prompt, seed=base_seed)
        finally:
            steering_policy.remove_hooks()
        cleaned_text = strip_thinking_trace(raw_text)
        parsed_values = parse_batch_numbers(cleaned_text)
        records: list[SampleRecord] = []
        for idx in range(num_samples):
            if idx < len(parsed_values):
                parsed = parsed_values[idx]
            else:
                parsed = ParsedNumber(raw="", valid=False, value=None, error_code="missing_value")
            records.append(
                _to_record(
                    parsed,
                    spec,
                    protocol,
                    idx,
                    request_seed(base_seed, idx),
                    prompt,
                    cleaned_text,
                    steering_name,
                    metadata={"returned_count": len(parsed_values)},
                )
            )
        return records

    if protocol != "independent":
        raise ValueError(f"Unsupported protocol: {protocol}")

    records: list[SampleRecord] = []
    prompts = [build_independent_prompt(spec) for _ in range(num_samples)]
    seeds = [request_seed(base_seed, idx) for idx in range(num_samples)]
    if hasattr(engine, "generate_text_batch"):
        steering_policy.on_request_start(spec, base_seed)
        steering_policy.install_hooks()
        try:
            raw_texts = engine.generate_text_batch(prompts, seeds)
        finally:
            steering_policy.remove_hooks()
        for idx, (prompt, seed, raw_text) in enumerate(zip(prompts, seeds, raw_texts, strict=True)):
            cleaned_text = strip_thinking_trace(raw_text)
            parsed = parse_single_number(cleaned_text)
            records.append(
                _to_record(
                    parsed,
                    spec,
                    protocol,
                    idx,
                    seed,
                    prompt,
                    cleaned_text,
                    steering_name,
                )
            )
        return records

    for idx, (prompt, seed) in enumerate(zip(prompts, seeds, strict=True)):
        steering_policy.on_request_start(spec, seed)
        steering_policy.install_hooks()
        try:
            raw_text = engine.generate_text(prompt, seed=seed)
        finally:
            steering_policy.remove_hooks()
        cleaned_text = strip_thinking_trace(raw_text)
        parsed = parse_single_number(cleaned_text)
        records.append(
            _to_record(
                parsed,
                spec,
                protocol,
                idx,
                seed,
                prompt,
                cleaned_text,
                steering_name,
            )
        )

    return records
