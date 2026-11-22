"""Use Megatron's native DeepSeek-V4 hybrid attention implementation."""

import torch


def _hadamard_transform(x, scale=1.0):
    """Differentiable power-of-two Hadamard transform for the ROCm reference path."""
    shape, dtype = x.shape, x.dtype
    width = shape[-1]
    if width < 1 or width & (width - 1):
        raise ValueError(f"Hadamard width must be a power of two, got {width}")
    x = x.float()
    stride = 1
    while stride < width:
        left, right = x.reshape(*shape[:-1], -1, 2, stride).unbind(-2)
        x = torch.stack((left + right, left - right), dim=-2)
        stride *= 2
    return (x.reshape(shape) * scale).to(dtype)


def get_dsv4_spec(args, config, vp_stage=None):
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )

    if config.experimental_attention_variant != "dsv4_hybrid":
        raise ValueError("DeepSeek-V4 requires --experimental-attention-variant dsv4_hybrid")
    if args.mtp_num_layers:
        raise ValueError("DeepSeek-V4 MTP is not supported by this recipe")
    if torch.version.hip is not None:
        from megatron.core.transformer.experimental_attention_variant import dsa

        if dsa.hadamard_transform is None:
            dsa.hadamard_transform = _hadamard_transform
    return get_transformer_block_with_experimental_attention_variant_spec(config, vp_stage=vp_stage)


def validate_dsv41_args(args, hf):
    """Reject unsupported weight/lifecycle modes before allocating the model."""
    if getattr(hf, "quantization_config", None):
        raise ValueError("DeepSeek V4.1 requires BF16 train/rollout weights; keep frozen Engram tables in FP8")
    if getattr(args, "mtp_num_layers", 0):
        raise ValueError("DeepSeek V4.1 MTP/DSpark training is not supported")
    if getattr(args, "context_parallel_size", 1) != 1 or getattr(args, "pipeline_model_parallel_size", 1) != 1:
        raise ValueError("Vime's initial V4.1 integration requires CP=1 and PP=1")
    rollout_offload = getattr(args, "offload_rollout", None)
    implicit_offload = getattr(args, "colocate", False) and (
        rollout_offload is not False or getattr(args, "release_train", False)
    )
    if (
        rollout_offload
        or implicit_offload
        or getattr(args, "offload", False)
        or getattr(args, "vllm_enable_sleep_mode", False)
    ):
        raise ValueError("Keep the V4.1 rollout resident: Engram tables are not included in online weight updates")
    if getattr(args, "save_hf", None):
        raise ValueError("Use distributed checkpoints for V4.1; raw HF export does not copy frozen Engram tables")


def get_dsv41_spec(args, config, vp_stage=None):
    """Reuse Miles' V4.1 layer spec; Vime still owns training and weight sync."""
    from copy import copy

    from vime.utils.hf_config import load_hf_config

    hf = load_hf_config(args.hf_checkpoint, trust_remote_code=True)
    if hf.model_type not in {"deepseek_v41", "deepseek_v41_text"}:
        raise ValueError("DeepSeek V4.1 requires a deepseek_v41 HF checkpoint")
    validate_dsv41_args(args, hf)
    if config.experimental_attention_variant != "dsv4":
        raise ValueError("DeepSeek V4.1 requires the Miles Megatron fork and --experimental-attention-variant dsv4")
    try:
        from miles_plugins.models.deepseek_v4_1.deepseek_v4_1 import get_dsv41_spec as miles_spec
    except ModuleNotFoundError as exc:
        if exc.name not in {"miles", "miles_plugins"}:
            raise
        raise ImportError(
            "Install the optional Miles V4.1 runtime described in examples/README-deepseek-v4.md"
        ) from exc
    args = copy(args)
    args.dsv4_impl = "miles"
    args.miles_dsa_topk_backend = "torch"
    config.activation_func_clamp_shared_expert = True
    return miles_spec(args, config, vp_stage)
