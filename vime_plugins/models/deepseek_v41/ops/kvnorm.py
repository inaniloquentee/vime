import torch

from vime_plugins.models.deepseek_v4_kernels.fake_quant import fake_quant_compressed_kv
from vime_plugins.models.deepseek_v4_kernels.rope import apply_rotary_emb
from vime_plugins.models.deepseek_v4_ops.qat import fp8_simulate_qat


def kv_norm_rope(kv, weight, eps, freqs_cis, rope_dim):
    x = kv.float()
    data = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()).contiguous()
    apply_rotary_emb(data[..., -rope_dim:], freqs_cis)
    return data.to(kv.dtype)


def kv_norm_rope_fp8(kv, weight, eps, freqs_cis, rope_dim):
    out = kv_norm_rope(kv, weight, eps, freqs_cis, rope_dim).clone()
    out[..., :-rope_dim] = fp8_simulate_qat(out[..., :-rope_dim], 64)
    return out


def compressed_kv_stored(latent, rope_dim):
    out = latent.clone()
    out[..., :-rope_dim] = fp8_simulate_qat(out[..., :-rope_dim], 64)
    return out
