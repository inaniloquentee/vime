# vime + RL-Kernel baseline OOM 后续执行方案

## 0. 背景结论

本轮没有直接使用 `vime-RLK.md` 中默认的 2xH100 预验证规模，不是因为只想做 smoke 或 fast A/B，而是因为 baseline 在原定配置下 OOM。

因此下一步的目标应从“按默认规模证明收益”调整为：

1. 先确认 baseline OOM 的具体阶段和可复现配置。
2. 找到 baseline 和 candidate 都能稳定跑完的最大非 OOM 公平配置。
3. 在这个配置上重新做 2xH100 A/B 预验证。
4. 只有当 2 卡结果稳定且收益明确时，才继续上 8xH100 benchmark。

当前 PDF 报告只能说明：

- RL-Kernel `linear_logp` 能在 vime candidate 中加载。
- `VIME_RL_KERNEL_STRICT=1` 下没有 fallback。
- fast debug-rollout workload 下 candidate 有训练侧收益迹象。

它不能替代正式 2 卡性能预验证。

本方案是 `vime-RLK.md` 在 baseline 默认配置 OOM 后的补充执行方案。除本文件明确改写的 OOM 降级路径外，代码来源、安装、模型下载、checkpoint 转换、baseline/candidate 开关方式和正式 8 卡 benchmark 边界仍以 `vime-RLK.md` 为准。

## 1. 处理原则

后续所有实验必须遵守以下原则：

- baseline 和 candidate 除 RL-Kernel 开关外必须使用完全相同的配置。
- 不允许只因为 candidate 能跑通，就使用 baseline 会 OOM 的配置做对比。
- 不允许把 train-only debug-rollout fast A/B 作为最终性能结论。
- 如果默认规模 OOM，应使用“最大非 OOM 公平配置”替代默认规模，并在报告中明确写出 OOM 原因和降级路径。
- 正式指标仍优先看 `mean_log_probs_time_s`、`peak_vram_gb`、`mean_step_time_s`、`train_rollout_logprob_abs_diff`、`raw_reward` 和 RL-Kernel runtime counters。
- 如果是全新环境，必须先完成 `vime-RLK.md` 中的仓库 checkout、RL-Kernel/vime 安装、模型与数据下载、Megatron `torch_dist` checkpoint 转换。

## 2. 第一阶段：复现并定位 baseline OOM

先用 `vime-RLK.md` 默认配置单独复现 baseline OOM，保留完整日志。

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2
export MEGATRON_TP=2
export MEGATRON_EP=2
export MEGATRON_CP=1
export ROLLOUT_NUM_GPUS_PER_ENGINE=2

export NUM_ROLLOUT=24
export ROLLOUT_BATCH_SIZE=2
export N_SAMPLES_PER_PROMPT=2
export GLOBAL_BATCH_SIZE=4
export MAX_TOKENS_PER_GPU=4096
export ROLLOUT_MAX_RESPONSE_LEN=1024
export VLLM_GPU_MEMORY_UTILIZATION=0.50

export VIME_CKPT_DIR=/root/Qwen3-30B-A3B_vime_tp2_dev
export VIME_DISABLE_SAVE=1
export VIME_SKIP_EVAL_BEFORE_TRAIN=1
export VIME_VLLM_ENFORCE_EAGER=1
export VIME_NO_GRAD_ACCUM_FUSION=1

unset VIME_RL_KERNEL VIME_RL_KERNEL_OPS VIME_RL_KERNEL_STRICT

bash scripts/run-qwen3-30B-A3B.sh 2>&1 | tee /workspace/vime-rlk-tp2-baseline-default-oom.log
```

需要记录：

- OOM 发生在 vLLM 初始化、rollout、wake up/offload、forward/backward、optimizer step，还是 logprob 计算阶段。
- OOM 前最后一次 GPU memory line。
- 是否两张卡同时 OOM，还是单卡不均衡。
- 是否存在 CUDA fragmentation 或 allocator reserve 明显高于 allocated。
- baseline 是否在第 0 step 前 OOM，还是训练若干 step 后 OOM。

如果日志不能清楚定位阶段，需要补充一次更短 run，只为定位 OOM，不用于性能统计。

## 3. 第二阶段：寻找最大非 OOM 公平配置

从 `vime-RLK.md` 已定义的第一档降级开始。该档优先保留 `MAX_TOKENS_PER_GPU=4096` 和 `ROLLOUT_MAX_RESPONSE_LEN=1024`，只降低 batch 规模。

### 3.1 第一档降级，优先尝试

```text
NUM_ROLLOUT=24
ROLLOUT_BATCH_SIZE=1
N_SAMPLES_PER_PROMPT=2
GLOBAL_BATCH_SIZE=2
MAX_TOKENS_PER_GPU=4096
ROLLOUT_MAX_RESPONSE_LEN=1024
VLLM_GPU_MEMORY_UTILIZATION=0.50
```

执行顺序：

1. baseline 跑通至少 24 train step。
2. candidate 使用完全相同配置跑通至少 24 train step。
3. 如果两者都成功，这一档就是新的 2xH100 OOM-compatible 预验证配置。

### 3.2 如果第一档仍 baseline OOM

继续按以下顺序降级，每次只改一个主要变量，直到 baseline 能稳定跑完 24 step。

```text
L1:
  ROLLOUT_BATCH_SIZE=1
  N_SAMPLES_PER_PROMPT=2
  GLOBAL_BATCH_SIZE=2
  MAX_TOKENS_PER_GPU=4096
  ROLLOUT_MAX_RESPONSE_LEN=1024
  VLLM_GPU_MEMORY_UTILIZATION=0.50

L2:
  ROLLOUT_BATCH_SIZE=1
  N_SAMPLES_PER_PROMPT=2
  GLOBAL_BATCH_SIZE=2
  MAX_TOKENS_PER_GPU=4096
  ROLLOUT_MAX_RESPONSE_LEN=1024
  VLLM_GPU_MEMORY_UTILIZATION=0.45

L3:
  ROLLOUT_BATCH_SIZE=1
  N_SAMPLES_PER_PROMPT=2
  GLOBAL_BATCH_SIZE=2
  MAX_TOKENS_PER_GPU=3072
  ROLLOUT_MAX_RESPONSE_LEN=1024
  VLLM_GPU_MEMORY_UTILIZATION=0.45

L4:
  ROLLOUT_BATCH_SIZE=1
  N_SAMPLES_PER_PROMPT=2
  GLOBAL_BATCH_SIZE=2
  MAX_TOKENS_PER_GPU=3072
  ROLLOUT_MAX_RESPONSE_LEN=768
  VLLM_GPU_MEMORY_UTILIZATION=0.45
```

停止条件：

- 找到 baseline 能连续跑完 24 train step 的最高档配置。
- candidate 也必须在同一档配置下跑完。
- 如果降到 L4 baseline 仍 OOM，则 2xH100 不适合作为 Qwen3-30B-A3B 的有效性能预验证环境，应直接转向 8xH100 或更高显存/更合理并行配置。

## 4. 第三阶段：正式 OOM-compatible 2 卡 A/B

找到最大非 OOM 配置后，重新跑正式 A/B。

baseline：

```bash
unset VIME_RL_KERNEL VIME_RL_KERNEL_OPS VIME_RL_KERNEL_STRICT
bash scripts/run-qwen3-30B-A3B.sh 2>&1 | tee /workspace/vime-rlk-tp2-baseline-oom-compatible-r1.log
```

candidate：

```bash
export VIME_RL_KERNEL=1
export VIME_RL_KERNEL_OPS=linear_logp
export VIME_RL_KERNEL_STRICT=1
bash scripts/run-qwen3-30B-A3B.sh 2>&1 | tee /workspace/vime-rlk-tp2-candidate-oom-compatible-r1.log
```

建议至少执行：

```text
baseline:  3 runs
candidate: 3 runs
每个 run 至少 24 train step
丢弃前 5 step warmup 后统计
```

如果时间有限，最低可接受口径：

```text
baseline:  1 successful run
candidate: 1 successful run
每个 run 至少 24 train step
丢弃前 5 step warmup 后统计
报告中明确标注为 single-run precheck，不作为宣传 benchmark
```

## 5. 必须统计的指标

正式 OOM-compatible 报告必须记录：

```text
oom_stage_if_any
oom_config_if_any
selected_non_oom_level
gpu_name
num_gpus
vime_commit
rl_kernel_commit
vime_pr
rl_kernel_pr
model
tp
ep
cp
num_rollout
rollout_batch_size
n_samples_per_prompt
global_batch_size
max_tokens_per_gpu
rollout_max_response_len
vllm_gpu_memory_utilization
selected_rl_kernel_backend
rl_kernel_fallback_count
rl_kernel_linear_logp_call_count_total
rl_kernel_linear_logp_call_count_delta
rl_kernel_linear_logp_token_count_total
rl_kernel_linear_logp_token_count_delta
rl_kernel_linear_logp_dispatch_elapsed_s_total
rl_kernel_linear_logp_tokens_per_call_delta
rl_kernel_linear_logp_tokens_per_call_total
rl_kernel_linear_logp_dispatch_elapsed_s_delta
first_successful_train_step
mean_step_time_s
p50_step_time_s
p90_step_time_s
mean_log_probs_time_s
p50_log_probs_time_s
p90_log_probs_time_s
peak_vram_gb
raw_reward_mean
train_rollout_logprob_abs_diff_mean
error_stack_if_failed
```

注意：

- `dispatch_elapsed_s` 只能作为 runtime counter，不作为 GPU kernel time 宣传。
- 如果没有 `mean_log_probs_time_s`，需要补充等价的 logprob 路径计时，不能只看 `actor_train_time`。
- 显存必须记录峰值，不能只记录若干 memory line 的最大观察值。

## 6. 验收线

candidate 必须满足：

```text
VIME_RL_KERNEL_STRICT=1 不报错
rl_kernel_fallback_count = 0
rl_kernel_linear_logp_call_count_delta > 0
rl_kernel_linear_logp_token_count_delta > 0
rl_kernel_linear_logp_dispatch_elapsed_s_delta > 0
tokens_per_call 不是极小空 workload
loss / logprob / reward 指标 finite
train_rollout_logprob_abs_diff 不持续高于 baseline
raw_reward 不低于 baseline 同量级
```

不允许出现：

```text
fallback 到 vime materialized logits 路径
target vocab shard 报错
TP collective hang
loss / logprob / reward NaN 或 Inf
candidate 质量指标明显劣于 baseline
rl_kernel_linear_logp_call_count_delta 长时间为 0
rl_kernel_linear_logp_token_count_delta 只覆盖极少 token
```

性能判断：

```text
优先通过：
  mean_log_probs_time_s 下降 >= 20%

或：
  peak_vram_gb 下降 >= 10%

或：
  mean_log_probs_time_s 和 peak_vram_gb 都有稳定小幅下降，
  且 mean_step_time_s 不明显变差。
```

如果 2xH100 的最大非 OOM 配置已经被压得很小，导致 RL-Kernel token delta 很低，则应把结论写成：

```text
2xH100 仅证明集成可用和无 fallback。
性能宣传价值需要转到 8xH100 正式 benchmark 判断。
```

## 7. 上 8 卡的条件

满足以下任一条件后再上 8xH100：

- 2xH100 OOM-compatible 配置下 candidate 相比 baseline 的 `mean_log_probs_time_s` 或 `peak_vram_gb` 有稳定收益。
- 2xH100 因 baseline OOM 只能降到过小 workload，但 candidate 已证明 strict 无 fallback、logprob diff 正常、token counter 正常。

上 8 卡时需要重新使用更接近正式 workload 的配置，不应复用 fast debug-rollout 结果。

## 8. 推荐下一步执行顺序

1. 用默认配置复现 baseline OOM，保存 `/workspace/vime-rlk-tp2-baseline-default-oom.log`。
2. 尝试 L1 降级配置，baseline 先跑 24 step。
3. 如果 L1 baseline 成功，用同一配置跑 candidate。
4. 如果 L1 baseline OOM，按 L2、L3、L4 顺序继续找最大非 OOM 配置。
5. 在最大非 OOM 配置上 baseline/candidate 各跑至少 1 次，最好各 3 次。
6. 丢弃前 5 step warmup，重新生成 OOM-compatible 2 卡报告。
7. 根据收益和 token workload 决定是否进入 8xH100 benchmark。

## 9. 与 vime-RLK.md 的覆盖关系

已覆盖：

- baseline/candidate 两组都必须跑，且除 RL-Kernel 开关外配置一致。
- baseline 默认配置 OOM 后，先使用 `vime-RLK.md` 的第一档降级。
- 降级后仍要求每组至少 24 train step。
- 建议 baseline/candidate 各 3 次 run，最低口径必须标注 single-run precheck。
- 丢弃前 5 step warmup 后统计。
- `VIME_RL_KERNEL_STRICT=1`、`fallback_count=0`、call/token/dispatch delta 均大于 0。
- `loss / logprob / reward` finite。
- `raw_reward`、`train_rollout_logprob_abs_diff` 不劣于 baseline 同量级。
- 继续以 `mean_log_probs_time_s` 或 `peak_vram_gb` 作为核心收益指标。
- 记录 `vime-RLK.md` 要求的版本、配置、runtime counter、step/logprob/VRAM/quality 指标。
- 如果 2 卡 workload 被 OOM 压得过小，不把 2 卡结果作为宣传 benchmark。

需要回看 `vime-RLK.md` 执行的前置部分：

- 仓库 clone、PR checkout、安装。
- 模型、数据、Megatron checkpoint 转换。
- 8xH100 正式 benchmark 的完整执行细节。
