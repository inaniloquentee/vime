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
