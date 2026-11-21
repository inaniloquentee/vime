"""Check the converter against native Megatron DeepSeek-V4 layer names."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from vime.backends.megatron_utils.deepseek_v4_weights import _PENDING_ALPHAS, convert_deepseek_v4_to_hf


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron native layers need CUDA")
def test_four_layer_native_names_match_checkpoint():
    pytest.importorskip("megatron.core")
    pytest.importorskip("tests.unit_tests.test_utilities")
    from megatron.core.process_groups_config import ProcessGroupCollection
    from tests.unit_tests.test_utilities import Utils
    from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
        _build_dsv4_moe_layer,
        _make_config,
    )

    checkpoint = os.environ.get("DSV4_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("Set DSV4_CHECKPOINT to check the local Flash checkpoint")
    pytest.importorskip("transformer_engine")
    index = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())["weight_map"]
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    try:
        config = _make_config(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            v_head_dim=32,
            qk_pos_emb_head_dim=16,
            q_lora_rank=32,
            csa_compress_ratios=[0, 0, 4, 128],
            num_moe_experts=4,
            moe_ffn_hidden_size=256,
            moe_layer_freq=1,
            moe_router_topk=2,
            moe_router_score_function="sqrtsoftplus",
            moe_router_enable_expert_bias=True,
            moe_n_hash_layers=3,
            actual_vocab_size=128,
            activation_func=F.silu,
            gated_linear_unit=True,
            moe_grouped_gemm=True,
            qk_layernorm=True,
            enable_hyper_connections=True,
            mhc_num_residual_streams=4,
        )
        pg = ProcessGroupCollection.use_mpu_process_groups()
        _PENDING_ALPHAS.clear()
        for layer_id in range(4):
            layer = _build_dsv4_moe_layer(config, layer_id + 1, pg)
            exported = []
            for name, tensor in layer.named_parameters():
                mcore_name = f"module.module.decoder.layers.{layer_id}.{name}"
                exported.extend(key for key, _ in convert_deepseek_v4_to_hf(None, mcore_name, tensor))
            for name, tensor in layer.named_buffers():
                if name.endswith(("router.expert_bias", "router.tid2eid")):
                    mcore_name = f"module.module.decoder.layers.{layer_id}.{name}"
                    exported.extend(key for key, _ in convert_deepseek_v4_to_hf(None, mcore_name, tensor))
            assert not _PENDING_ALPHAS
            assert set(exported) <= set(index)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("width", [1, 32, 128, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"))],
)
def test_rocm_hadamard_matches_matrix_forward_and_backward(width, dtype, device):
    from vime_plugins.models.deepseek_v4 import _hadamard_transform

    matrix = torch.ones(1, 1, device=device)
    while matrix.size(0) < width:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    matrix *= width**-0.5
    x = torch.randn(2, 3, width * 2, dtype=dtype, device=device)[..., ::2].requires_grad_()
    actual = _hadamard_transform(x, scale=width**-0.5)
    expected = (x.float() @ matrix).to(dtype)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    upstream = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, x, upstream)[0]
    expected_grad = torch.autograd.grad(expected, x, upstream)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)
