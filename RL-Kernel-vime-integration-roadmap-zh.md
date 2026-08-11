# RL-Kernel 与 vime 集成 Roadmap

这份 Roadmap 面向 vime maintainer，目标是讨论 RL-Kernel 与 vime 的更深度集成。我们的基本定位是：vime 继续作为 RL orchestration framework，RL-Kernel 作为可选、可观测的算子后端，为用户提供两类能力：

1. **Fast Path**：用 RL-Kernel 降低 vime RL 训练中的算子级和系统级开销。
2. **Consistency Path**：让 rollout-training 训推一致性可测量、可诊断、可强制，并最终沉淀成训练与推理算子共享的数值契约。

已经完成的 `linear_logp` 集成证明了这个边界是可行的：vime 保持 Megatron training、vLLM rollout、weight sync、data flow 和算法执行；当后端可用时，RL-Kernel 替换 selected-logprob hot path。下一步是把这个 proof point 升级成一套可维护的集成接口。

## 摘要

vime 把 Megatron training 与 vLLM rollout 连接起来，这是 scalable RL post-training 非常合适的架构。但这个架构同时带来两个系统问题：

- **性能问题**：完整 RL step 不只受单个 kernel 影响，还会受 tensor-parallel 通信、rank skew、host-side 调度、GPU idle/wait、初始化噪声以及大量小框架 kernel 影响。
- **训推一致性问题**：在参数更新之前，同一个 checkpoint、同一批 token、同一个 mask 下，rollout 侧记录的 log probability 与 training 侧 teacher-forcing 重算的 log probability 应该一致；否则本来应该 on-policy 的训练会隐性变成 off-policy。

RL-Kernel 可以同时帮助 vime 解决这两类问题，但集成方式应该显式、可观测、可回退，而不是隐式魔法。用户应该能清楚选择：我要性能、我要一致性审计/强制，或者两者都要。

## 当前 Proof Point

第一阶段已经完成的集成是 `linear_logp`。

- vime 从 Megatron model 中取出 LM head weight、TP group、本地 vocab 范围和 global vocab size。
- 在 training-side logprob 计算时，vime 可以让 Megatron 返回 hidden states，而不是物化完整 logits。
- vime 将 hidden states、target token IDs 和 tensor-parallel metadata 传给 RL-Kernel。
- 当验证过的 CUDA path 可用时，RL-Kernel dispatch `FusedLinearLogpSM90Op`。
- fused path 直接计算 selected logprobs，不把完整 `[tokens, vocab]` logits 暴露给 Python framework layer。
- native fallback 仍然保留，并通过 `rl_kernel_fallback_count` 可观测。

已有 8xH100 Qwen3-30B-A3B 实验中，T1/T2/T3 配置均完成并且 `fallback=0`。在最大的 completed no-trace 配置中，`linear_logp` forward + backward CUDA 时间从 **33.96 ms** 降到 **18.50 ms**；单算子 peak reserved delta 从 **32342 MB** 降到 **26710 MB**。

这说明 operator-level 收益是真实的，但 claim 边界需要谨慎。当前端到端 wall time 主要被更大的 pipeline / distributed-system 因素主导，包括 TP/NCCL AllReduce、rank/device 不均衡、GPU idle/wait、host-side orchestration、profiling 窗口里的 CUDA/cuBLAS/NCCL lazy initialization，以及大量 PyTorch elementwise/reduce/copy 小 kernel。因此 Fast Path Roadmap 不能只停在 `linear_logp`，还要扩展到通信、调度和端到端 profiling 口径。

## 当前支持与已知限制

当前 vime 集成应被看作第一个已验证后端，而不是已经完成的通用算子后端框架。

| 领域 | 当前状态 | Roadmap 方向 |
|---|---|---|
| 已启用算子 | `linear_logp` | 只通过显式 capability 和 benchmark gate 扩展更多算子。 |
| 后端选择 | 通过环境变量走 registry 或 forced `triton` / `cuda_sm90` 风格 backend | 将稳定配置迁移到 CLI/config；环境变量保留为兼容别名。 |
| TP 支持 | `linear_logp` 会把 TP group、vocab start、global vocab size 传给 RL-Kernel | 用 sharding / reduction contract 泛化。 |
| CP 支持 | 需要 CP redistribution 时 `linear_logp` fallback | 先补 CP-aware contract 和 attention/logprob layout，再启用 strict path。 |
| Entropy | 请求 entropy 时 fallback 到 vime native path | 只有在数值 contract 兼容时再支持。 |
| Temperature | `rollout_temperature` 会在 selected-logprob 计算前应用 | 将 sampling metadata 纳入 Consistency Path contract。 |
| Fallback | warning + `rl_kernel_fallback_count`；strict mode 可以 raise | 用结构化 path-decision record 替换零散 fallback reason。 |
| Timing | dispatch counter 和可选 CUDA event timing | 标准化为 benchmark/telemetry surface。 |
| Memory | 可选 operator-level memory probe | 保留用于 operator claim；与 full-step memory claim 分开。 |

## 集成原则

- **vime 负责 orchestration**：training、rollout、data buffer、weight sync、sampling/reward flow 和算法级行为继续留在 vime。
- **RL-Kernel 负责 operator backend**：fused kernel、reduction-aware kernel、hardware-specific dispatch 和 operator-level contract 留在 RL-Kernel。
- **默认 opt-in**：在覆盖范围和生产信心足够前，不改变 vime 默认行为。
- **不允许 silent fallback**：每次路径选择都应该通过日志、counter 和 strict error 可见。
- **先保证一致性再谈更大 claim**：性能 claim 应限定在已测量的 surface；strict consistency 必须先可测量，再宣传为 guarantee。
- **先单卡，后多卡**：分布式一致性工作必须在对应算子的单卡对拍通过后推进。
- **两条 path 共用同一套 hook**：capability discovery、path selection、fallback、telemetry 和 benchmark wiring 不应分叉成两套机制。

## 用户侧开关

命名：

```bash
--rl-kernel-fast-path {off,auto,strict}
--rl-kernel-consistency-path {off,audit,strict}
```

环境变量别名：

```bash
VIME_RL_KERNEL_FAST_PATH=off|auto|strict
VIME_RL_KERNEL_CONSISTENCY_PATH=off|audit|strict
```

迁移期内，它们应与当前 `--enable-rl-kernel`、`--rl-kernel-ops`、`--rl-kernel-strict` 共存。兼容关系可以是：

- `--enable-rl-kernel` 等价于 `--rl-kernel-fast-path auto`。
- `--rl-kernel-strict` 对 enabled ops 映射为 `--rl-kernel-fast-path strict`。
- `--rl-kernel-ops linear_logp` 继续作为算子 allowlist。
- `--rl-kernel-consistency-path audit` 可以在不开 Fast Path 时单独启用，用于审计 native path 的 rollout-training mismatch。

两类开关是正交的：

| Fast Path | Consistency Path | 行为 |
|---|---|---|
| `off` | `off` | vime native 行为。 |
| `auto` | `off` | 满足条件时使用已验证 RL-Kernel fast path；否则 warning 并 fallback。 |
| `strict` | `off` | 强制使用 enabled fast path；不支持或不可用时 error。 |
| `off` | `audit` | 仍走 native execution，但计算并记录 consistency diagnostics。 |
| `off` | `strict` | 要求 metadata 和 consistency checks 完整；contract mismatch 时 fail。 |
| `auto` | `audit` | 满足条件时使用 fast path，同时报告 consistency metrics。 |
| `auto` | `strict` | 只使用满足 consistency contract 的 fast path；否则选择 contract-preserving reference path 或按 fallback policy fail。 |
| `strict` | `strict` | 对 enabled ops 同时要求 acceleration 和 consistency contract support。 |

这样用户心智比较清楚：Fast Path 回答“vime 能不能用 RL-Kernel 优化算子？”，Consistency Path 回答“vime 要不要测量或强制 rollout-training parity？”。

## Consistency Path 范围

Consistency Path 应落地 `home.pdf` 中的 rollout-training consistency contract。

核心度量是：

```text
dlogp = training-side recomputed logp - rollout-side old logp
```

这个指标只有在以下前提一致时才有意义：

- checkpoint 或 weight version；
- token IDs；
- active response/action mask；
- tokenizer 和 padding semantics；
- sampling metadata，例如 temperature、top-p、top-k；
- position IDs 和 cache metadata；
- quantization configuration；
- operator numeric contract，包括 accumulation dtype 和 reduction order。

第一版 diagnostics surface 应包含：

- `abs_dlogp` mean、max、p50、p90、p99；
- `ratio0 = exp(dlogp)`；
- `clipfrac0`；
- `approx_kl0`；
- distributed run 的 per-rank 版本；
- path 支持时的 per-op / per-stage attribution；
- mask coverage 和 active token count；
- checkpoint、tokenizer、sampling settings、quantization、reduction contract 的 metadata fingerprint。

设计文档中对 drift 来源的优先级排序是：

1. attention，尤其是 flash attention vs paged/chunked-prefill，以及 online-softmax merge order；
2. `lm_head + logp`，尤其是大词表 softmax；
3. matmul / projection，尤其是 split-K 和 quantized GEMM；
4. RMSNorm，单层影响较小但会逐层累积；
5. RoPE、SwiGLU、embedding，通常影响较小，但仍应进入对拍框架；
6. 逻辑层 mismatch，例如 mask、tokenizer、temperature、top-p/top-k、padding、position/cache metadata。

对 distributed drift，contract 需要显式描述 reduction：

- accumulation dtype，consistency-sensitive reduction 默认应优先考虑 fp32；
- canonical reduction order，例如 fixed binary tree 或 canonical rank order；
- downcast 发生的唯一位置；
- reduction 发生在 operator 内、NCCL，还是 custom engine；
- attention LSE merge 这类特殊 merge semantics，而不只是 sum reduction。

## Fast Path 范围

Fast Path 应从 `linear_logp` 开始，向 recent profiling 里暴露出的端到端瓶颈扩展。

重要瓶颈类别：

- **TP/NCCL AllReduce**：profiled actor training window 中最大的 GPU kernel 类别，并且有明显 rank/device imbalance。
- **GPU kernel 外的 idle/wait**：actor window 可以是秒级，但真实 GPU kernel 执行只占很小一部分。
- **Host-side orchestration 和 initialization**：如果 warmup 没控制好，profiling 窗口会混入 CUDA library loading、cuBLAS handle creation、NCCL communicator initialization。
- **大量 PyTorch 小 kernel**：elementwise/reduce/copy 目前不一定是一阶瓶颈，但通信问题缓解后会继续贡献 launch overhead 和 stream queueing。
- **Optimizer 和 grad norm**：可见，但在当前 profiled window 中仍小于 TP communication。
- **`linear_logp`**：已经优化到不再是 profiled small-scale actor window 的端到端主瓶颈，但它是集成与可观测性的好模板。

Fast Path 优先级：

1. 先把 profiling 做干净：capture 前 warm up CUDA、cuBLAS、NCCL、vLLM generation 和一次 dummy train step。
2. 为 rollout wait、forward、logprob、backward、optimizer、weight sync、communication 加 NVTX range。
3. 降低 TP AllReduce 开销：减少 collective 次数、改善 bucketing/fusion、overlap、topology-aware placement。
4. 评估小模型什么时候应避免 latency-bound TP，例如显存允许时使用 TP=1 + DP 或单 actor。
5. 融合 hot training path 上大量 repeated small elementwise/reduce/copy kernels。
6. 只有当 profiling evidence 或 consistency contract 支持时，才把 RL-Kernel 扩展到 `linear_logp` 之外。

## Proposed Architecture

### 1. vime 侧 Path Selection Layer

vime 应拥有一个薄的 path-selection layer，用来回答：

- 哪些 RL-Kernel ops 被启用；
- Fast Path 是 off、auto 还是 strict；
- Consistency Path 是 off、audit 还是 strict；
- 当前 model、dtype、parallelism、hardware、algorithm 条件是否满足；
- fallback 是否允许；
- fallback 或 strict failure 的原因是什么。

当前已有的 `is_rl_kernel_op_enabled`、fallback counters 和 `LinearLogpContext` 是很好的起点。下一版可以暴露结构化 decision object：

```text
op = linear_logp
selected_backend = cuda_sm90
path = fast
consistency_contract = unchecked|audited|strict
eligible = true|false
fallback_allowed = true|false
fallback_reason = ...
tokens = ...
shape = ...
tp = ...
cp = ...
dtype = ...
```

### 2. RL-Kernel 侧 Operator Capability Registry

RL-Kernel 应向 vime 暴露 operator capabilities：

- op name 和 version；
- supported dtypes；
- supported hardware；
- autograd support；
- TP/SP/CP support；
- entropy support；
- quantization support；
- reduction contract support；
- strict consistency support；
- benchmark tags。

这样可以避免 vime 硬编码 backend-specific 假设。

### 3. Rollout Sample 上的 Consistency Metadata

为了支持 Consistency Path，rollout samples 需要携带训练侧校验所需的 metadata：

```text
weight_version
checkpoint_id
tokenizer_fingerprint
sampling_config_fingerprint
mask_semantics
padding_semantics
position_metadata
cache_metadata
old_logprob_contract
quantization_contract
reduction_contract_fingerprint
```

training side 在计算 `dlogp` 之前应先校验这些字段。audit mode 下记录并计数 mismatch；strict mode 下应 early fail。

### 4. Reduction 与 Sharding Contracts

长期来看，RL-Kernel 应暴露显式 distributed operator contracts：

```text
Placement = Replicate | Shard(dim) | Partial(kind)
ShardingSpec = inputs -> outputs
ReductionSpec = acc_dtype, order, downcast_at, engine
```

示例：

- `linear_logp` TP：各 rank 在本地 vocab shard 上计算 partial statistics，再 canonical merge 得到 selected logprob。
- matmul TP：row/column-parallel sharding，并在需要 reduction 时显式输出 `Partial`。
- RMSNorm SP：token-wise norm 本地完成，但 SP 边界 collective 仍受同一份 reduction contract 约束。
- attention CP：`Partial(kind=logsumexp)`，使用固定 global block order 和 fp32 LSE buffers。

关键规则是：当 strict consistency 被请求时，train 与 rollout implementations 应共享同一份 numeric contract。

## Roadmap 阶段

### Phase 0: Maintainer Alignment

交付物：

- 确认 vime 继续作为 orchestration layer，RL-Kernel 作为 optional backend。
- 确认用户侧开关：`--rl-kernel-fast-path` 和 `--rl-kernel-consistency-path`。
- 决定这些 flags 是放在现有 RL-Kernel 参数下，还是放到更广义的 vime performance/consistency config 下。
- 定义 fallback policy 和 logging expectations。
- 选定第一组 upstreamable PR sequence。

需要和 maintainer 讨论的问题：

- strict consistency 应该 fail fast，还是当 fast backend 缺少 contract 时允许 fallback 到 reference path？
- consistency diagnostics 应全局启用，还是 per op 启用？
- 多少 rollout metadata 应进入稳定 `Sample` schema？
- 哪些 benchmark scripts 应成为 vime maintained examples，哪些应留在外部 RL-Kernel validation scripts？

### Phase 1: Productionize Existing `linear_logp` Integration

交付物：

- 保留当前 native fallback path。
- 将合适的私有环境变量提升为 documented CLI/config。
- 保留已有环境变量作为兼容别名。
- 增加 structured path decision logs。
- 保留 `fallback_count`、token count、dispatch time、CUDA event time 和 memory probe。
- 增加 strict fallback behavior、backend selection、temperature handling、TP metadata、empty-token cases 的测试。
- 文档化精确 support matrix：dtype、hardware、TP、CP、entropy、qkv format、full-gradient support。

验收标准：

- validated configs 在 strict mode 下 `fallback=0`。
- RL-Kernel 缺失或不支持时，native fallback 仍然正确。
- native vime tests 无显著回归。
- published benchmark scripts 不依赖未文档化的私有状态。

### Phase 2: Add Consistency Diagnostics

交付物：

- 实现共享的 `dlogp` diagnostic module。
- 计算 `ratio0`、`clipfrac0`、`approx_kl0` 和 `abs_dlogp` percentiles。
- 报告 active token counts 和 mask coverage。
- 为 distributed runs 增加 per-rank metrics。
- 增加 checkpoint/weight version、tokenizer、masks、padding、sampling settings、position/cache metadata、quantization 的 metadata validation。
- 支持不开 fast operators 时单独使用 `--rl-kernel-consistency-path audit`。
- 增加 debug dumps，允许 replay rollout batches 到 training-side recompute。

验收标准：

- zero-update single-GPU run 的 `dlogp` 在约定 dtype tolerance 内接近 0。
- 已知 metadata mismatch 能被捕获并归因。
- audit mode 低风险，不改变训练行为。

### Phase 3: Operator-Level Consistency Contracts

交付物：

- 在 RL-Kernel operator test surface 中加入 `ReductionSpec`、`ShardingSpec` 和 `train` / `infer` role tag。
- 将 cross-implementation tests 从 `logp` / `linear_logp` 扩展到更多算子。
- 为 attention、`lm_head/logp`、matmul、RMSNorm、RoPE、SwiGLU、embedding 增加 single-card train-vs-infer comparisons。
- 单卡通过后再加 distributed tests：matmul TP、RMSNorm SP、attention CP、`linear_logp` TP。
- 对 consistency-sensitive distributed ops 要求显式 sharding rules，不允许 silent full-replication fallback。

验收标准：

- 每个 enabled strict consistency op 都暴露 numeric contract。
- CI 可以配对 training-style 与 inference-style implementations。
- 在把 distributed drift 归因到通信之前，先测量 single-card op drift。

### Phase 4: Expand Fast Path Beyond `linear_logp`

交付物：

- 增加带 warmup 和 stable capture window 的 clean profiling scripts。
- 为 rollout、train compute、logprob、backward、optimizer、weight sync、communication 加 NVTX range。
- 增加 communication counters：AllReduce count/time、AllGather count/time、rank skew、overlap indicators。
- 在 profiling 显示 TP communication 主导时优化通信。
- 探索 fusion 高数量 PyTorch elementwise/reduce/copy hot paths。
- 只有当通信不再是一阶瓶颈后，再评估 optimizer / grad-norm fusion。
- 只有当算子既 hot 又有清晰 consistency story 时，才添加更多 RL-Kernel ops。

候选扩展方向：

- `linear_logp` 改进：CP support、更广 backend coverage、更好 full-gradient coverage、更多 shape coverage。
- `logp` 和 `lm_head/logp`：共享 softmax contract 和大词表 logprob path。
- matmul projections：TP-aware reduction contract 和 split-K policy。
- RMSNorm：CUDA implementation 和 SP boundary contract。
- attention：CP/LSE merge contract 和 deterministic merge order。
- RL losses：当 profiling 显示 overhead 或 launch pressure 时，考虑 ratio/KL、GRPO/PPO/DPO loss fragments。

验收标准：

- Fast Path 确实降低它所瞄准的 measured bottleneck。
- 只有 full-step measurements 稳定显示收益后，才做端到端 claim。
- GPU active time、communication time 和 rank skew 有改善，或有清楚解释。

### Phase 5: Unified Path Selection and Compatibility

交付物：

- RL-Kernel 到 vime 的稳定 capability query。
- 稳定的 vime path-decision report。
- 当前 `VIME_RL_KERNEL`、`VIME_RL_KERNEL_OPS`、`VIME_RL_KERNEL_STRICT` 的 compatibility aliases。
- Fast Path 与 Consistency Path 冲突时的清晰行为。
- 常见 recipes 文档：
  - native baseline；
  - fast `linear_logp`；
  - consistency audit only；
  - fast path plus consistency audit；
  - strict consistency CI run；
  - strict fast path benchmark run。

验收标准：

- 用户能从日志中准确理解实际跑了哪条 path。
- maintainer 能通过结构化 metrics 复现 fallback decision。
- migration 期间旧脚本继续可用。

### Phase 6: Benchmarks, CI, and Release Process

交付物：

- config parsing 和 fallback decisions 的 unit tests。
- diagnostic math 的 CPU reference tests。
- 每个 supported backend 的 GPU operator tests。
- TP/SP/CP contracts 的 distributed tests。
- 针对代表性 vime workloads 的 nightly 或 scheduled benchmark matrix。
- benchmark report template，明确分开 operator-level、actor-window 和 full-step claims。

推荐 benchmark metrics：

- operator CUDA time；
- forward + backward CUDA time；
- tokens per call；
- dispatch elapsed time；
- fallback count 和 reasons；
- peak allocated/reserved memory；
- full step time；
- rollout time；
- weight sync time；
- GPU active time；
- kernel count；
- AllReduce/AllGather time 和 count；
- rank skew；
- `dlogp` metrics；
- reward 和 loss sanity metrics。

Release rule：

- Operator-level speedup 只能作为 operator-level speedup 发布。
- Full-step speedup 只有在测量包含 rollout、communication 和 framework scheduling 后才能发布。
- Consistency claim 要说明是 audit、tolerance-based strict，还是 bitwise strict。

## 建议 PR Sequence

1. **Roadmap and naming RFC**
   - 加入 two-path design，并决定 public names。

2. **Path selection cleanup**
   - 添加 `fast_path` 和 `consistency_path` config parsing。
   - 保留当前 RL-Kernel flags 作为 compatibility aliases。

3. **Structured fallback and capability reporting**
   - 用 structured decision records 替代零散 fallback messages。
   - 保留已有 counters。

4. **`linear_logp` productionization**
   - 完成 docs、tests、support matrix 和 strict behavior。

5. **Consistency audit module**
   - 添加 `dlogp`、ratio、KL、percentile、per-rank metrics。
   - 添加 rollout sample metadata validation。

6. **Profiling hygiene**
   - 添加 warmup、NVTX ranges 和 stable profile capture scripts。

7. **Distributed contract prototype**
   - 添加 `ReductionSpec`、`ShardingSpec`，以及 `linear_logp` / matmul prototype。

8. **Fast Path expansion PRs**
   - 每次提交一个 op family，并带 profiling evidence 和 consistency tests。

## 分工建议

vime side：

- user-facing flags 和 config；
- path-selection policy；
- rollout sample metadata；
- native fallback behavior；
- training/rollout diagnostics；
- benchmark entry points；
- docs 和 examples。

RL-Kernel side：

- operator implementation；
- backend capability reporting；
- reduction 和 sharding contracts；
- cross-implementation operator tests；
- hardware-specific dispatch；
- operator-level benchmark tooling。

Joint：

- benchmark matrix；
- support matrix；
- CI coverage；
- release notes；
- path semantics 和 fallback policy 的 maintainer review。

## Non-Goals

- 不替换 Megatron 或 vLLM。
- 不让 RL-Kernel 成为 vime 必选依赖。
- 不静默改变 vime native paths 的数值行为。
- 不用 isolated operator measurements 直接 claim full-step speedup。
- 不把 unsupported cases 藏在不可观测 fallback 后面。
- 不要求所有用户先接受 strict bitwise consistency 才能使用 audit mode。

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| Fast kernels 改变 numerics | strict mode 必须依赖显式 consistency contracts 和 diagnostics。 |
| flags 太多导致用户困惑 | 保留两个高层开关，op-specific 细节放进 advanced docs。 |
| fallback 掩盖 coverage 缺口 | structured fallback reasons、strict mode、benchmark acceptance gates。 |
| profiling 混入初始化噪声 | 强制 warmup 和 stable-window profile scripts。 |
| operator fusion 后 TP communication 成为主瓶颈 | 把 communication 当成 Fast Path target，而不是外部问题。 |
| consistency metadata 让 samples 变重 | 默认只存 compact fingerprints；需要时再开启 verbose debug dumps。 |
| hardware-specific kernels 造成支持碎片化 | capability registry + documented support matrix。 |

## Maintainer 讨论议程

1. 确认 two-path model：Fast Path 与 Consistency Path。
2. 确认 flag names，以及如何从当前 RL-Kernel flags 迁移。
3. 确认 vime 负责 path selection，RL-Kernel 负责 backend capability reporting。
4. 确认第一个 strict support target：建议是 validated Hopper TP configs 上的 `linear_logp`。
5. 确认第一个 audit target：native 和 RL-Kernel `linear_logp` paths 的 `dlogp` metrics。
6. 决定多少 consistency metadata 应进入稳定 sample schema。
7. 选择足够小、适合 upstream review 的第一个 PR boundary。

## 目标状态

用户可以运行：

```bash
python train.py \
  --enable-rl-kernel \
  --rl-kernel-ops linear_logp \
  --rl-kernel-fast-path auto \
  --rl-kernel-consistency-path audit
```

并清楚看到：

- 哪些 RL-Kernel operators 实际运行；
- 哪些 operators fallback，以及原因是什么；
- fast path 是否命中 validated backend coverage；
- operator time 和 memory 改变了多少；
- rollout-training `dlogp` 是否保持在预期 tolerance 内；
- distributed ranks 是否均衡；
- end-to-end time 当前受 kernels、communication、rollout、weight sync 还是 host scheduling 限制。

生产或 CI 场景可以收紧同一套接口：

```bash
python train.py \
  --enable-rl-kernel \
  --rl-kernel-ops linear_logp \
  --rl-kernel-fast-path strict \
  --rl-kernel-consistency-path strict
```

这就是我们希望达到的集成形态：vime 继续作为 orchestration layer，RL-Kernel 成为一等但可选的 operator backend，用户得到两个容易理解的杠杆：让它更快、让它更一致，或者同时要求两者。
