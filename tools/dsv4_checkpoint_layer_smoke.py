"""Run checkpoint-backed forward/backward on one V4 layer or a reduced V4 model."""

import argparse
import json
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig

from vime.backends.megatron_utils.deepseek_v4_weights import deepseek_v4_hf_tensor
from vime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
from vime_plugins.models.deepseek_v4 import get_dsv4_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--layer-index", type=int, default=2)
    scope.add_argument("--num-layers", type=int, help="Build the first N layers, embedding, and LM head")
    parser.add_argument("--seq-length", type=int, default=256)
    args = parser.parse_args()
    hf_config = json.loads((args.checkpoint / "config.json").read_text())
    num_layers = args.num_layers if args.num_layers is not None else hf_config["num_hidden_layers"]
    if not 1 <= num_layers <= hf_config["num_hidden_layers"]:
        parser.error("--num-layers must fit the checkpoint")
    if not 0 <= args.layer_index < hf_config["num_hidden_layers"] and args.num_layers is None:
        parser.error("--layer-index must fit the checkpoint")
    if args.seq_length <= 0:
        parser.error("--seq-length must be positive")
    torch.manual_seed(42)
    torch.distributed.init_process_group("nccl", store=torch.distributed.HashStore(), rank=0, world_size=1)
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
        config = MLATransformerConfig(
            experimental_attention_variant="dsv4_hybrid",
            dsa_kernel_backend="none",
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            multi_latent_attention=True,
            rope_type="rope",
            dsa_indexer_loss_coeff=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            num_layers=num_layers,
            perform_initialization=False,
            hidden_size=hf_config["hidden_size"],
            num_attention_heads=hf_config["num_attention_heads"],
            num_query_groups=hf_config["num_key_value_heads"],
            v_head_dim=hf_config["head_dim"],
            qk_pos_emb_head_dim=hf_config["qk_rope_head_dim"],
            q_lora_rank=hf_config["q_lora_rank"],
            output_projection_groups=hf_config["o_groups"],
            output_projection_lora_rank=hf_config["o_lora_rank"],
            csa_compress_ratios=hf_config["compress_ratios"][:num_layers],
            csa_window_size=hf_config["sliding_window"],
            dsa_indexer_n_heads=hf_config["index_n_heads"],
            dsa_indexer_head_dim=hf_config["index_head_dim"],
            dsa_indexer_topk=hf_config["index_topk"],
            num_moe_experts=hf_config["n_routed_experts"],
            moe_ffn_hidden_size=hf_config["moe_intermediate_size"],
            moe_shared_expert_intermediate_size=hf_config["moe_intermediate_size"] * hf_config["n_shared_experts"],
            moe_layer_freq=1,
            moe_router_topk=hf_config["num_experts_per_tok"],
            moe_router_score_function=hf_config["scoring_func"],
            moe_router_dtype="fp32",
            moe_router_topk_scaling_factor=hf_config["routed_scaling_factor"],
            moe_router_enable_expert_bias=True,
            moe_n_hash_layers=hf_config["num_hash_layers"],
            actual_vocab_size=hf_config["vocab_size"],
            activation_func=F.silu,
            gated_linear_unit=True,
            activation_func_clamp_value=hf_config["swiglu_limit"],
            moe_grouped_gemm=True,
            qk_layernorm=True,
            normalization="RMSNorm",
            enable_hyper_connections=True,
            mhc_num_residual_streams=hf_config["hc_mult"],
            mhc_sinkhorn_iterations=hf_config["hc_sinkhorn_iters"],
            layernorm_epsilon=hf_config["rms_norm_eps"],
            attention_latent_norm_epsilon=hf_config["rms_norm_eps"],
            csa_compress_rotary_base=hf_config["compress_rope_theta"],
            rotary_scaling_factor=hf_config["rope_scaling"]["factor"],
            original_max_position_embeddings=hf_config["rope_scaling"]["original_max_position_embeddings"],
        )
        layer_id = args.layer_index
        pg = ProcessGroupCollection.use_mpu_process_groups()
        spec = get_dsv4_spec(SimpleNamespace(mtp_num_layers=0), config)
        if args.num_layers is None:
            model = build_module(
                spec.layer_specs[layer_id], config=config, layer_number=layer_id + 1, pg_collection=pg
            ).cuda()
            prefix = f"decoder.layers.{layer_id}."
        else:
            model = GPTModel(
                config=config,
                transformer_layer_spec=spec,
                vocab_size=hf_config["vocab_size"],
                max_sequence_length=args.seq_length,
                position_embedding_type="rope",
                share_embeddings_and_output_weights=False,
                pg_collection=pg,
            ).cuda()
            prefix = ""
        reader = SafetensorReader(args.checkpoint)
        loaded = 0
        tensors = chain(
            model.named_parameters(),
            (
                (name, buffer)
                for name, buffer in model.named_buffers()
                if name.endswith(("router.expert_bias", "router.tid2eid"))
            ),
        )
        with torch.no_grad():
            for name, parameter in tensors:
                key = prefix + name
                tensor = deepseek_v4_hf_tensor(key, reader, hf_config)
                if tensor.shape != parameter.shape:
                    raise ValueError(f"{key}: checkpoint {tuple(tensor.shape)}, Megatron {tuple(parameter.shape)}")
                parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
                loaded += 1
        reader._files.clear()
        torch.cuda.empty_cache()
        print(f"loaded={loaded}", flush=True)
        input_ids = torch.randint(0, hf_config["vocab_size"], (1, args.seq_length), device="cuda")
        if args.num_layers is None:
            hidden = torch.randn(
                args.seq_length,
                1,
                config.hidden_size * hf_config["hc_mult"],
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            output, _ = model(hidden_states=hidden, attention_mask=None, input_ids=input_ids)
            loss = output.float().square().mean()
        else:
            positions = torch.arange(args.seq_length, device="cuda").unsqueeze(0)
            cu_seqlens = torch.tensor([0, args.seq_length], device="cuda", dtype=torch.int32)
            packed = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=args.seq_length,
                max_seqlen_kv=args.seq_length,
            )
            loss = (
                model(
                    input_ids,
                    positions,
                    attention_mask=None,
                    labels=input_ids.roll(-1, dims=1),
                    packed_seq_params=packed,
                )
                .float()
                .mean()
            )
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert torch.isfinite(loss) and gradients
        assert all(torch.isfinite(grad).all().item() for grad in gradients), "Non-finite gradient"
        grad_count = len(gradients)
        print(f"loaded={loaded} loss={loss.item():.6g} finite_gradient_tensors={grad_count}")
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
