#!/usr/bin/env bash

# Optional Miles/Megatron runtime; BF16 text training, MTP/DSpark disabled.
# Supply --hf-checkpoint "$DSV41_CHECKPOINT" to train.py as well.
: "${DSV41_CHECKPOINT:?Set DSV41_CHECKPOINT to a local BF16 V4.1 checkpoint}"
source "$(dirname "${BASH_SOURCE[0]}")/deepseek-v4-flash.sh"
readarray -t MODEL_ARGS < <(python - "$DSV41_CHECKPOINT" "${MODEL_ARGS[@]}" <<'PY'
import sys
from vime.utils.hf_config import load_hf_config

hf = load_hf_config(sys.argv[1], trust_remote_code=True)
if hf.model_type not in {"deepseek_v41", "deepseek_v41_text"}:
    raise ValueError("DSV41_CHECKPOINT must be a DeepSeek V4.1 checkpoint")
if getattr(hf, "quantization_config", None):
    raise ValueError("Use a BF16 V4.1 checkpoint; native FP4 refit is not supported")
args = sys.argv[2:]
replacements = {
    "--model-name": "deepseek_v41",
    "--experimental-attention-variant": "dsv4",
    "--num-layers": hf.num_hidden_layers,
    "--hidden-size": hf.hidden_size,
    "--q-lora-rank": hf.q_lora_rank,
    "--norm-epsilon": hf.rms_norm_eps,
    "--csa-compress-ratios": str(hf.compress_ratios[:hf.num_hidden_layers]).replace(" ", ""),
    "--dsa-indexer-n-heads": hf.index_n_heads,
    "--num-experts": hf.n_routed_experts,
    "--moe-ffn-hidden-size": hf.moe_intermediate_size,
    "--moe-shared-expert-intermediate-size": hf.moe_intermediate_size * hf.n_shared_experts,
}
# The Miles Megatron fork uses the earlier names of these shared V4 flags.
renames = {
    "--output-projection-groups": "--o-groups",
    "--output-projection-lora-rank": "--o-lora-rank",
    "--moe-num-hash-layers": "--moe-n-hash-layers",
    "--mhc-num-residual-streams": "--num-residual-streams",
}
for i, value in enumerate(args):
    if value in replacements:
        args[i + 1] = str(replacements[value])
    elif value == "--moe-num-hash-layers":
        args[i + 1] = "0"
    args[i] = renames.get(value, value)
args[args.index("get_dsv4_spec")] = "get_dsv41_spec"
args += ["--ffn-hidden-size", str(hf.moe_intermediate_size), "--kv-lora-rank", str(hf.head_dim),
         "--qk-head-dim", str(hf.head_dim), "--no-rope-fusion", "--no-bias-swiglu-fusion",
         "--attention-softmax-in-fp32", "--moe-router-pre-softmax", "--freeze-indexer",
         "--moe-router-bias-update-rate", "0", "--freeze-params-name-list", r"mlp\.router\.weight$"]
print("\n".join(args))
PY
)
# Propagate checkpoint/config errors from the process substitution.
[[ ${#MODEL_ARGS[@]} -gt 0 ]] || return 1
