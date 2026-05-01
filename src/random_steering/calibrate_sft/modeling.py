from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from random_steering.utils.distributed import (
    get_local_rank,
    is_distributed_enabled,
    is_main_process,
)
from random_steering.utils.hf import ensure_hf_home
from random_steering.inference.hf_backend import HFGenerationBackend
from random_steering.inference.vllm_backend import VLLMGenerationBackend


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class TrainingBundle:
    model: Any
    tokenizer: Any
    device: torch.device


@dataclass
class EvaluationAssets:
    tokenizer: Any
    hf_model: Any | None
    generation_backend: Any | None


def _resolve_dtype(model_cfg: Any) -> torch.dtype:
    dtype_name = str(getattr(model_cfg, "dtype", "bfloat16"))
    return _DTYPE_MAP.get(dtype_name, torch.bfloat16)


def _local_files_only(cfg: Any) -> bool:
    return bool(getattr(cfg, "local_files_only", False))


def _prepare_tokenizer_checkpoint(checkpoint: str) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        return checkpoint, None

    tokenizer_config_path = checkpoint_path / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return checkpoint, None

    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except Exception:
        return checkpoint, None

    if not isinstance(tokenizer_config.get("extra_special_tokens"), list):
        return checkpoint, None

    tokenizer_config.pop("extra_special_tokens", None)
    tmpdir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmpdir.name)
    for source_path in checkpoint_path.iterdir():
        if source_path.is_file():
            shutil.copy2(source_path, tmp_path / source_path.name)
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(tmp_path), tmpdir


def _load_tokenizer(checkpoint: str, *, trust_remote_code: bool, local_files_only: bool):
    ensure_hf_home()
    load_path, tmpdir = _prepare_tokenizer_checkpoint(checkpoint)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_training_device(device_name: str, *, distributed_cfg: Any | None) -> torch.device:
    if is_distributed_enabled(distributed_cfg):
        if device_name != "cuda":
            raise ValueError("Distributed training currently requires model.device=cuda.")
        return torch.device("cuda", get_local_rank())
    return torch.device(device_name)


def _load_base_model(
    checkpoint: str,
    *,
    trust_remote_code: bool,
    device_name: str,
    dtype: torch.dtype,
    local_files_only: bool,
    distributed_cfg: Any | None = None,
    use_device_map_auto: bool = True,
):
    ensure_hf_home()
    use_distributed = is_distributed_enabled(distributed_cfg)
    kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": None if use_distributed else ("auto" if (use_device_map_auto and device_name == "cuda") else None),
        "local_files_only": local_files_only,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=dtype,
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            **kwargs,
        )

    if use_distributed or device_name != "cuda" or not use_device_map_auto:
        model.to(_resolve_training_device(device_name, distributed_cfg=distributed_cfg))
    return model


def _unwrap_base_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    base_model = getattr(model, "base_model", None)
    if isinstance(base_model, nn.Module):
        return base_model
    return model


def _get_module_by_attr_chain(root: nn.Module | None, attr_chain: tuple[str, ...]) -> Any | None:
    current = root
    for attr in attr_chain:
        current = getattr(current, attr, None)
        if current is None:
            return None
    return current


def _infer_transformer_layer_classes(model: nn.Module) -> set[type[nn.Module]]:
    base_model = _unwrap_base_model(model)
    candidate_roots = [
        base_model,
        getattr(base_model, "model", None),
        getattr(base_model, "base_model", None),
    ]
    attr_chains = (
        ("layers",),
        ("model", "layers"),
        ("decoder", "layers"),
        ("language_model", "layers"),
        ("model", "language_model", "layers"),
        ("text_model", "layers"),
        ("model", "text_model", "layers"),
    )
    for root in candidate_roots:
        if root is None:
            continue
        for attr_chain in attr_chains:
            current = _get_module_by_attr_chain(root, attr_chain)
            if isinstance(current, nn.ModuleList) and len(current) > 0:
                return {current[0].__class__}
    raise ValueError("Could not infer transformer block classes for FSDP auto-wrapping.")


def _fsdp_mixed_precision(distributed_cfg: Any):
    from torch.distributed.fsdp import MixedPrecision

    precision_name = str(getattr(distributed_cfg, "mixed_precision", "bfloat16")).lower()
    if precision_name in {"none", "false"}:
        return None
    dtype = _DTYPE_MAP.get(precision_name, torch.bfloat16)
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def _fsdp_sharding_strategy(distributed_cfg: Any):
    from torch.distributed.fsdp import ShardingStrategy

    strategy_name = str(getattr(distributed_cfg, "sharding_strategy", "full_shard")).lower()
    mapping = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    if strategy_name not in mapping:
        raise ValueError(f"Unsupported FSDP sharding strategy: {strategy_name}")
    return mapping[strategy_name]


def _apply_activation_checkpointing_if_enabled(model: nn.Module, distributed_cfg: Any) -> None:
    if not bool(getattr(distributed_cfg, "activation_checkpointing", False)):
        return

    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
    except ImportError as exc:
        raise ImportError("Activation checkpointing requires torch checkpoint wrapper support.") from exc

    layer_classes = tuple(_infer_transformer_layer_classes(model))
    wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: isinstance(module, layer_classes),
    )


def _wrap_model_for_fsdp(model: nn.Module, distributed_cfg: Any, device: torch.device) -> nn.Module:
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    layer_classes = _infer_transformer_layer_classes(model)
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_classes)
    _apply_activation_checkpointing_if_enabled(model, distributed_cfg)
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=_fsdp_mixed_precision(distributed_cfg),
        sharding_strategy=_fsdp_sharding_strategy(distributed_cfg),
        use_orig_params=bool(getattr(distributed_cfg, "use_orig_params", True)),
        cpu_offload=CPUOffload(offload_params=bool(getattr(distributed_cfg, "cpu_offload", False))),
        sync_module_states=bool(getattr(distributed_cfg, "sync_module_states", True)),
        limit_all_gathers=bool(getattr(distributed_cfg, "limit_all_gathers", True)),
        device_id=device,
    )


def _cast_trainable_params_to_dtype(model: nn.Module, dtype: torch.dtype) -> None:
    for param in model.parameters():
        if param.requires_grad and param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype=dtype)


def load_training_bundle(model_cfg: Any, train_cfg: Any, distributed_cfg: Any | None = None) -> TrainingBundle:
    checkpoint = model_cfg.checkpoint
    trust_remote_code = bool(getattr(model_cfg, "trust_remote_code", True))
    device_name = str(getattr(model_cfg, "device", "cuda"))
    dtype = _resolve_dtype(model_cfg)
    local_files_only = _local_files_only(model_cfg)

    tokenizer = _load_tokenizer(
        checkpoint,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model = _load_base_model(
        checkpoint,
        trust_remote_code=trust_remote_code,
        device_name=device_name,
        dtype=dtype,
        local_files_only=local_files_only,
        distributed_cfg=distributed_cfg,
    )
    model.config.use_cache = False

    lora_cfg = getattr(train_cfg, "lora", None)
    if lora_cfg is not None and bool(getattr(lora_cfg, "enabled", True)):
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("PEFT is required for LoRA training. Install `peft`.") from exc

        config = LoraConfig(
            r=int(lora_cfg.rank),
            lora_alpha=int(lora_cfg.alpha),
            lora_dropout=float(lora_cfg.dropout),
            target_modules=list(lora_cfg.target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, config)
        _cast_trainable_params_to_dtype(model, dtype)

    device = _resolve_training_device(device_name, distributed_cfg=distributed_cfg)
    if is_distributed_enabled(distributed_cfg):
        model = _wrap_model_for_fsdp(model, distributed_cfg, device)

    return TrainingBundle(
        model=model,
        tokenizer=tokenizer,
        device=device if is_distributed_enabled(distributed_cfg) else next(model.parameters()).device,
    )


def load_evaluation_bundle(model_cfg: Any, eval_target_cfg: Any) -> TrainingBundle:
    base_checkpoint = str(eval_target_cfg.base_checkpoint)
    adapter_checkpoint = getattr(eval_target_cfg, "adapter_checkpoint", None)
    tokenizer_checkpoint = str(getattr(eval_target_cfg, "tokenizer_checkpoint", adapter_checkpoint or base_checkpoint))
    trust_remote_code = bool(getattr(model_cfg, "trust_remote_code", True))
    device_name = str(getattr(model_cfg, "device", "cuda"))
    dtype = _resolve_dtype(model_cfg)
    local_files_only = _local_files_only(eval_target_cfg) or _local_files_only(model_cfg)

    tokenizer = _load_tokenizer(
        tokenizer_checkpoint,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model = _load_base_model(
        base_checkpoint,
        trust_remote_code=trust_remote_code,
        device_name=device_name,
        dtype=dtype,
        local_files_only=local_files_only,
        use_device_map_auto=False,
    )

    if adapter_checkpoint:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("PEFT is required to load LoRA adapters for evaluation. Install `peft`.") from exc
        model = PeftModel.from_pretrained(
            model,
            adapter_checkpoint,
            is_trainable=False,
            local_files_only=local_files_only,
        )

    model.config.use_cache = False
    model.eval()
    return TrainingBundle(
        model=model,
        tokenizer=tokenizer,
        device=next(model.parameters()).device,
    )


def load_evaluation_assets(
    model_cfg: Any,
    eval_target_cfg: Any,
    inference_cfg: Any,
    *,
    require_generation_backend: bool,
    require_hf_model: bool,
) -> EvaluationAssets:
    backend_name = str(getattr(inference_cfg, "backend", "hf"))
    if backend_name not in {"hf", "vllm"}:
        raise ValueError(f"Unsupported inference backend: {backend_name}")

    needs_hf_model = bool(require_hf_model or (require_generation_backend and backend_name == "hf"))
    bundle: TrainingBundle | None = None
    if needs_hf_model:
        bundle = load_evaluation_bundle(model_cfg, eval_target_cfg)
        tokenizer = bundle.tokenizer
        hf_model = bundle.model
    else:
        tokenizer_checkpoint = str(
            getattr(
                eval_target_cfg,
                "tokenizer_checkpoint",
                getattr(eval_target_cfg, "adapter_checkpoint", None) or str(eval_target_cfg.base_checkpoint),
            )
        )
        trust_remote_code = bool(getattr(model_cfg, "trust_remote_code", True))
        local_files_only = _local_files_only(eval_target_cfg) or _local_files_only(model_cfg)
        tokenizer = _load_tokenizer(
            tokenizer_checkpoint,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        hf_model = None

    generation_backend = None
    if require_generation_backend:
        if backend_name == "hf":
            if hf_model is None:
                raise RuntimeError("HF generation backend requires an HF model.")
            generation_backend = HFGenerationBackend(
                model=hf_model,
                tokenizer=tokenizer,
                model_cfg=model_cfg,
            )
        else:
            generation_backend = VLLMGenerationBackend(
                tokenizer=tokenizer,
                model_cfg=model_cfg,
                eval_target_cfg=eval_target_cfg,
                inference_cfg=inference_cfg,
            )

    return EvaluationAssets(
        tokenizer=tokenizer,
        hf_model=hf_model,
        generation_backend=generation_backend,
    )


def save_training_bundle(bundle: TrainingBundle, output_dir: str | Path, distributed_cfg: Any | None = None) -> None:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if is_distributed_enabled(distributed_cfg):
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        if isinstance(bundle.model, FSDP):
            state_dict_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(bundle.model, StateDictType.FULL_STATE_DICT, state_dict_cfg):
                state_dict = bundle.model.state_dict()
            if is_main_process():
                bundle.model.module.save_pretrained(target_dir, state_dict=state_dict)
                bundle.tokenizer.save_pretrained(target_dir)
            return

    if is_main_process():
        bundle.model.save_pretrained(target_dir)
        bundle.tokenizer.save_pretrained(target_dir)
