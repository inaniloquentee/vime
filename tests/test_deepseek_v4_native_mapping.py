"""Check the converter against native Megatron DeepSeek-V4 layer names."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("nested", [False, True])
def test_v41_config_loads_text_without_global_hf_registration(tmp_path, nested):
    from transformers import AutoConfig

    from vime.utils.hf_config import load_hf_config

    text = {
        "model_type": "deepseek_v41_text",
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20,
    }
    config = {"model_type": "deepseek_v41", "text_config": text, "vision_config": {}} if nested else dict(text)
    config["quantization_config"] = {"quant_method": "fp8", "expert_dtype": "fp4"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    hf = load_hf_config(tmp_path)
    assert hf.model_type in {"deepseek_v41", "deepseek_v41_text"}
    assert hf.hidden_size == 5120 and hf.num_hidden_layers == 40
    assert hf.first_k_dense_replace == 0
    assert hf.quantization_config["expert_dtype"] == "fp4"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 32}))
    assert type(load_hf_config(tmp_path)) is type(AutoConfig.from_pretrained(tmp_path))


@pytest.mark.parametrize(
    "option,value,message",
    [
        ("mtp_num_layers", 1, "MTP/DSpark"),
        ("context_parallel_size", 2, "CP=1"),
        ("pipeline_model_parallel_size", 2, "PP=1"),
        ("offload_rollout", True, "rollout resident"),
        ("offload", True, "rollout resident"),
        ("colocate", True, "rollout resident"),
        ("vllm_enable_sleep_mode", True, "rollout resident"),
        ("save_hf", "output", "frozen Engram"),
    ],
)
def test_v41_rejects_unsupported_modes(option, value, message):
    from vime_plugins.models.deepseek_v4 import validate_dsv41_args

    with pytest.raises(ValueError, match=message):
        validate_dsv41_args(SimpleNamespace(**{option: value}), SimpleNamespace())


def test_v41_adapter_delegates_without_mutating_shared_args(tmp_path, monkeypatch):
    import sys
    import types

    from vime_plugins.models.deepseek_v4 import get_dsv41_spec

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41", "hidden_size": 5120}))
    args = SimpleNamespace(hf_checkpoint=str(tmp_path), mtp_num_layers=0)
    config = SimpleNamespace(experimental_attention_variant="dsv4")
    captured = []
    module = types.ModuleType("miles_plugins.models.deepseek_v4_1.deepseek_v4_1")
    module.get_dsv41_spec = lambda *values: captured.append(values) or "spec"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert get_dsv41_spec(args, config) == "spec"
    forwarded = captured[0][0]
    assert forwarded.dsv4_impl == "miles" and forwarded.miles_dsa_topk_backend == "torch"
    assert not hasattr(args, "dsv4_impl")
    assert config.activation_func_clamp_shared_expert


def test_v41_rejects_native_quantized_checkpoint():
    from vime_plugins.models.deepseek_v4 import validate_dsv41_args

    with pytest.raises(ValueError, match="BF16 train/rollout"):
        validate_dsv41_args(SimpleNamespace(), SimpleNamespace(quantization_config={"expert_dtype": "fp4"}))


@pytest.mark.skipif(not os.environ.get("DSV41_CHECKPOINT"), reason="Optional Miles runtime/checkpoint required")
def test_v41_real_recipe_builds_upstream_layer_spec():
    import argparse
    import subprocess

    from megatron.core import parallel_state
    from megatron.core.config import set_experimental_flag
    from megatron.training.arguments import add_megatron_arguments, core_transformer_config_from_args

    from vime_plugins.models.deepseek_v4 import get_dsv41_spec

    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    parser.add_argument("--model-name")
    parser.add_argument("--freeze-indexer", action="store_true")
    parser.add_argument("--freeze-params-name-list", nargs="+")
    result = subprocess.run(
        ["bash", "-c", 'source scripts/models/deepseek-v4.1-flash.sh && printf "%s\\n" "${MODEL_ARGS[@]}"'],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    args = parser.parse_args(result.stdout.splitlines())
    args.hf_checkpoint = os.environ["DSV41_CHECKPOINT"]
    args.params_dtype = torch.bfloat16
    set_experimental_flag(True)
    config = core_transformer_config_from_args(args)
    # Spec assembly needs rank metadata, but allocates no model/GPU tensors.
    parallel_state.set_pipeline_model_parallel_rank(0)
    parallel_state.set_pipeline_model_parallel_world_size(1)
    try:
        spec = get_dsv41_spec(args, config)
        assert len(spec.layer_specs) == args.num_layers
        assert all(layer.module.__name__ == "V41TransformerLayer" for layer in spec.layer_specs)
        assert args.freeze_indexer and args.moe_router_bias_update_rate == 0
    finally:
        parallel_state.set_pipeline_model_parallel_rank(None)
        parallel_state.set_pipeline_model_parallel_world_size(None)
