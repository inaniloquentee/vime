# VIME 与 RL-Kernel：从训推编排到算子一致性

VIME 解决的是 **训推系统怎么跑稳**：Megatron 训练、vLLM rollout、router、Ray actor、
data buffer、权重同步、offload、sleep/wake、故障恢复，都在它的框架层里被组织起来。

RL-Kernel 解决的是 **关键算子怎么算得可解释、可复现、可审计**：
GEMM、Attention、LogP 这类会直接影响 policy loss / KL / advantage 的数学边界，
需要明确的 reduction order、dtype、mask、sharding、KV cache、LSE merge 和 tolerance contract。

---

## VIME 最新变化

VIME 的更新主线很清晰：它在继续强化生产级 rollout/training runtime。

| Commit | 更新 | 对一致性的意义 |
| --- | --- | --- |
| `ee16693c` | Docker 默认 CU13 镜像升级到 vLLM `0.25.1`，并更新 patch | VIME 跟随 vLLM 发布节奏，说明 rollout backend 会持续演进，算子契约不能散落在临时代码里 |
| `db6c87de` | 修复 Geo3K VLM multi-turn rollout，并补 e2e test | rollout 的输入形态更复杂，多轮、多模态路径更需要结构化元数据 |
| `617da1d3` | 优化 vLLM 权重同步，使用 packed update payload，覆盖 tensor/distributed 更新测试 | 权重同步更像生产数据通道，后续 consistency path 应借用这套 version/provenance，而不是另建一条黑盒路径 |
| `93182da0` | 修复 IPv6 vLLM engine 和 health-check URL | router/engine 编排更稳，说明 VIME 关心的是跨进程、跨节点服务可用性 |
| `f2755327` | 同步 Slime runtime safeguard，包括 reloadable process group memory check | 训练侧生命周期更可靠，算子一致性可以依赖更稳定的 process group 和内存状态 |
| `5cabf1f3` | 增加 on-policy distillation 示例，Qwen3-8B actor + Qwen3-32B vLLM teacher | VIME 正在覆盖 actor/ref/teacher 多模型关系，一致性不再只是 actor rollout vs actor training |
| `c0ed6d83` | 增加 ROCm GPU CI | 后端平台更多，算子 contract 需要表达 hardware/backend 差异 |
| `1fc199d9` | 文档化 cache-aware vLLM router support，并调整 rollout/types 小逻辑 | prefix cache 和路由策略进入框架视野，Attention consistency 需要看 KV/cache/provenance |

这些更新共同说明一件事：VIME 的重心是把训练和推理之间的 **系统边界** 做稳。
它越来越像一个生产编排层，而不是一个单算子实验场。

这正好给 RL-Kernel 留出了很清楚的位置：不要替 VIME 管 Ray、router、engine 和数据流，
而是在 VIME 已经收集到的真实上下文里，把关键算子变成可选择、可回退、可审计的数学边界。

---

## VIME 和 RL-Kernel 的本质区别

### VIME：框架层的一致性

VIME 关心的是一条 RL post-training 轨迹能否闭环：

```text
prompt
  -> vLLM rollout / router / engines
  -> Sample: tokens, logprobs, reward, masks, metadata
  -> data buffer
  -> Megatron actor/ref/teacher recompute logprob or train
  -> weight update back to vLLM
  -> next rollout
```

在这个闭环里，VIME 维护的是 **训练和推理之间的状态一致性**：

| VIME 维护的对象 | 典型位置 | 它回答的问题 |
| --- | --- | --- |
| rollout tokens / response length / loss mask | `vime/utils/types.py`, `vime/rollout/vllm_rollout.py` | 训练到底消费了哪几个 response token，哪些 token 参与 loss |
| rollout logprobs / teacher logprobs | `Sample.rollout_log_probs`, `Sample.teacher_log_probs`, Megatron `_get_rollout_data` | 训练侧是否能对同一条轨迹重算 logprob 并做 PPO/GRPO/OPD |
| weight version | `VLLMEngine.get_weight_version`, `Sample.weight_versions`, update weight path | rollout 时用的是哪一版 actor 权重 |
| top-p replay metadata | `rollout_top_p_token_ids`, `rollout_top_p_token_offsets` | 训练侧是否能复现 rollout 采样时的 nucleus 候选集合 |
| MoE routed experts | `rollout_routed_experts`, `fill_routing_replay` | MoE 训练侧能否复用 rollout 侧的专家路由 |
| prefix cache stats | `Sample.PrefixCacheInfo`, vLLM usage metadata | router/cache 是否影响了 rollout 的 runtime provenance |
| topology / process group | Megatron actor、vLLM engine、weight updater | TP/PP/DP/CP、offload、sleep/wake、NCCL/IPC 是否处在可控状态 |

所以 VIME 的一致性更像是：

> 这批数据是不是从正确的模型版本、正确的 rollout 配置、正确的 router/engine 路径来，
> 并且被训练侧以正确的 token/mask 重新消费？

### RL-Kernel：算子层的一致性

RL-Kernel 关心的是同一条数学语义在不同路径下是否仍然可比较、可解释：

```text
same semantic operator
  -> native / reference path
  -> optimized path
  -> deterministic path
  -> comparison report
  -> contract id + tolerance + provenance
```

它维护的是 **算子语义和数值契约**：

| RL-Kernel 维护的对象 | 典型位置 | 它回答的问题 |
| --- | --- | --- |
| operator identity | `kernel_registry.get_op("det_gemm")`, `"attention"`, `"linear_logp"` | 现在调用的到底是哪一个语义算子 |
| backend capability | CUDA / Triton / PyTorch backend descriptor | 这个后端支持哪些 dtype、硬件、shape、autograd 和并行模式 |
| numeric contract | accumulation dtype、reduction order、downcast point、tolerance | 结果应该按什么规则比较，漂移多大算正常 |
| deterministic rule | `det_gemm`, deterministic attention | batch size、padding、chunked prefill 是否会改变 reduction order |
| comparison report | attention comparison harness / drift stats | candidate path 和 reference path 的 out/LSE/dlogp 差异在哪里 |
| registry dispatch | `rl_engine/kernels/registry.py` | 在当前平台上选择哪个实现，失败后怎么降级 |

所以 RL-Kernel 的一致性更像是：

> 给定同一个 GEMM 或 Attention 语义，不同 backend、不同 batch 组织、不同 prefill/decode 形态，
> 是否仍满足同一个可命名的 contract？

---

## 为什么它们是互补关系

VIME 和 RL-Kernel 不应该争夺同一个职责。
更好的分工是：VIME 拥有上下文，RL-Kernel 拥有算子契约。

| 维度 | VIME | RL-Kernel | 合在一起之后 |
| --- | --- | --- | --- |
| 核心问题 | 训推闭环怎么跑 | 算子语义怎么保证 | 训推闭环里的关键算子可解释 |
| 管理对象 | samples、engines、router、weights、Ray actors | kernels、contracts、tolerances、comparison reports | 带 provenance 的 operator execution |
| 主要风险 | 权重没同步、token/mask 错位、engine 状态不一致 | reduction order 漂移、dtype 下沉不一致、mask/KV/LSE 语义不一致 | 知道 drift 是系统错位还是算子语义差异 |
| fallback 方式 | 回到原生 VIME 训练/rollout | 回到 reference/native op | fast path 失败不破坏训练，strict path 可以阻断 |
| 观测粒度 | rollout/job/batch/model version | operator/backend/tensor contract | 从 job 级问题定位到算子级原因 |

这就是故事的核心转折：

> VIME 已经能说“这条样本来自哪次 rollout、哪版权重、哪些 token”。
> RL-Kernel 让我们进一步说“这些 token 对应的 logprob、attention output、GEMM output
> 是按哪个 contract 算出来的，和另一路差多少”。

换句话说，VIME 负责 **把同一件事送到训练侧**，
RL-Kernel 负责 **证明训练侧和推理侧对这件事的计算含义一致**。

---

## 为什么后训练框架需要 RL-Kernel

当目标从 **能训练** 升级到 **训推一致、可复现、可审计、可安全加速** 时，
框架层就必须有一层像 RL-Kernel 这样的算子契约系统。

原因在于，RL 后训练里的误差不是停留在算子局部的。
一个很小的 logprob 漂移，会继续进入 ratio、KL、advantage weighting 和 policy loss；
而 logprob 漂移往往来自更底层的 GEMM、Attention、softmax reduction、KV cache merge、
dtype downcast 或并行 shard 合并。框架只知道 token、mask、weight version 还不够，
它还需要知道这些 token 在关键数学边界上到底按什么规则被算出来。

可以把原生 VIME 和接入 RL-Kernel 后的 VIME 分成两个层次：

| 层次 | 原生 VIME | 接入 RL-Kernel 后 |
| --- | --- | --- |
| 系统闭环 | 能把 vLLM rollout 数据送回 Megatron 训练 | 保持原有闭环，同时给关键算子附带 contract 和 provenance |
| 一致性粒度 | 权重版本、token、mask、logprob、routing/cache metadata | 进一步覆盖 operator identity、backend、dtype、reduction order、tolerance |
| 问题定位 | 发现 rollout logprob 和 training recompute 不一致 | 判断不一致来自权重/token 错位，还是 GEMM/Attention/LogP 算子 drift |
| 加速方式 | 依赖原生 Megatron/vLLM backend | fast path 可以先 audit，再 fallback，最后 strict；加速不再是黑盒替换 |
| 长期维护 | 随 vLLM/Megatron/CUDA/ROCm 更新继续适配 | 每次 backend 更新都有 capability、contract id、fingerprint 和 drift report |

所以，RL-Kernel 对 VIME 的价值不是“多一个 kernel 库”，而是把后训练框架从：

> 我能把 rollout 结果拿回来重新训练。

提升到：

> 我知道 rollout path 和 training path 在关键算子上是否满足同一个数值契约；
> 如果不满足，我知道差在哪里、差多少、能不能接受，以及应该 fallback 还是 strict failure。

这就是接入后的能力跃迁：

```text
原生 VIME:
  高性能训推编排框架

VIME + RL-Kernel:
  具备算子级一致性治理能力的后训练框架
```

尤其是 GEMM 和 Attention，它们不是普通优化点，而是后训练里最容易放大误差的两个数学源头。
GEMM 控制 hidden-to-logits、MLP、QKV projection 等大部分线性变换；
Attention 控制 prefill、chunked prefill、decode、paged KV cache 等推理形态的语义对齐。
把这两个算子纳入 RL-Kernel consistency path，VIME 才能从“系统级训推一致”
进一步走到“算子可解释的训推一致”。

---

## VIME 目前如何做框架层面的训推一致

VIME 现在已经有一套框架级 train-inference consistency 逻辑，只是它的粒度主要停在 runtime 和 batch 级。

### 权重版本一致性

训练侧 Megatron actor 在 `update_weights()` 中把当前 actor 权重同步给 vLLM engines。
同步通道可以是 disk、delta、IPC tensor、NCCL distributed 等模式。

vLLM engine 侧在成功更新后记录 `weight_version`，
rollout 生成的 `Sample` 也会保存 `weight_versions`。

这解决的是：

> rollout 用的 actor 权重，和训练认为的 old policy / rollout actor 是否是同一个版本？

对 RL-Kernel 来说，这个版本号应该进入 operator provenance。
否则即使 Attention/GEMM 数值完全一致，也无法判断它们是不是在同一版权重上比较。

### token、mask 和 logprob 对齐

rollout 侧通过 vLLM router 生成 response token，并把 token IDs、logprobs、finish reason、
usage metadata 写进 `Sample`。

训练侧 Megatron actor 在 `_get_rollout_data()` 中把 tokens、loss masks、rollout logprobs
搬到 GPU，并按 CP 规则调用 `slice_log_prob_with_cp`。

这解决的是：

> Megatron 训练时重算 logprob 的 token 范围，是否和 vLLM rollout 返回的 response 范围一致？

对 RL-Kernel 来说，`target_ids`、`loss_mask`、`total_lengths`、`response_lengths`
就是 LogP/GEMM consistency path 的最低必要元数据。

### 采样 replay 与 MoE routing replay

当 `top_p != 1.0` 时，vLLM rollout 会请求返回 top-p 候选 token 集合。
VIME 用 `rollout_top_p_token_ids` 和 `rollout_top_p_token_offsets` 保存 ragged 结构，
让训练侧有机会复现采样约束。

对于 MoE，vLLM 可以返回 routed experts，VIME 在 `fill_routing_replay()` 中把它喂给 Megatron
训练侧的 routing replay。

这解决的是：

> rollout 侧做过的随机采样裁剪和专家选择，训练侧是否知道？

对 RL-Kernel 来说，这些信息不一定直接进入 GEMM，但会进入 Attention/LogP comparison 的语义背景：
如果候选 token 集合或专家路由不同，dlogp drift 就不能简单归因到算子实现。

### router、prefix cache 和 runtime provenance

VIME 通过 vLLM router 管理 engine 拓扑，支持 consistent hash routing、cache-aware router、
engine pause/resume、sleep/wake、prefix cache reset 等 runtime 行为。

这些不是算子本身，却会影响 Attention consistency 的解释。
比如 chunked prefill、paged KV cache、prefix cache 命中、PD prefill/decode 分离，
都会改变 Attention 在推理侧的物理执行形态。

这解决的是：

> 同一条请求在 rollout 侧到底走了哪个 engine、哪种 cache/路由/worker 形态？

RL-Kernel 不应该自己猜这些，它应该由 VIME 的 vLLM_utils 和 rollout metadata 提供。

---

## VIME 还缺的那一层：算子级 consistency path

VIME 能让训练侧知道“我应该重算哪批 token 的 logprob”，
但它默认无法回答以下问题：

| 问题 | 为什么 VIME 框架层不够 |
| --- | --- |
| GEMM 在 rollout 和 training 中是否使用相同 reduction order？ | cuBLAS / fused kernel 可能随 shape、batch、split-K、TP shard 改变累加顺序 |
| Attention 的 full prefill、chunked prefill、paged KV、decode 是否语义一致？ | 物理 materialization 不同，mask offset、LSE merge、KV cache 拼接都可能引入差异 |
| BF16/FP16/FP32 的 downcast point 是否一致？ | 框架能看到 dtype，但不知道算子内部何时累加、何时转换 |
| TP/CP/SP 下的 partial result 怎么合并？ | 框架知道并行拓扑，但算子需要明确 sharding rule 和 merge semantics |
| drift 是正常数值误差还是语义错误？ | 没有 contract id 和 tolerance，就只能看一个裸 diff |

这就是 RL-Kernel consistency path 应该接入的位置：

```text
VIME batch/runtime metadata
  + Megatron tensors
  + vLLM rollout metadata
  -> rl_kernel_utils adapter
  -> RL-Kernel operator / reference / comparison
  -> OperatorResult + ExecutionDecision + drift report
```

注意这里有两个路径，不能混在一起：

| 路径 | 目的 | 是否替换 VIME 原生结果 |
| --- | --- | --- |
| fast path | 用 RL-Kernel optimized/deterministic backend 加速或稳定关键算子 | 可以替换，失败时可 fallback |
| consistency path | 对 native/reference/candidate 做审计、对比和 contract 检查 | 通常不替换，audit 模式只记录，strict 模式可阻断 |

`adapter.py` 不是“保存所有 input 的仓库”，它更像是 VIME 和 RL-Kernel 之间的一扇门：
VIME 把 tensor 和元数据借给它，它调用可选 RL-Kernel op，然后返回 value、decision 和 provenance。

`execution.py` 则更像是这扇门旁边的裁判：
它不做算子计算，而是根据 VIME 模式、RL-Kernel capability、dtype、backend、contract
决定这次该 native、audit-only、optimized、strict-fast、fallback-native 还是 strict-failure。

---

## 接入 GEMM consistency path 时，VIME 需要给 RL-Kernel 什么

GEMM 的核心不是“矩阵乘法能不能跑”，而是：

> 同一个 row 的输出，是否会因为 batch size、padding、chunked prefill、TP shard 或 backend
> 改变 reduction order？

RL-Kernel 的 `det_gemm` 目标是固定 K 维累加顺序，BF16 输入、FP32 accumulation、禁用 TF32、
禁用 split-K，让结果具备 batch-invariant 属性。

VIME 侧需要维护和传递的核心量可以分成四组：

| 核心量 | 来自哪里 | 对应 GEMM 的位置 |
| --- | --- | --- |
| `A` / activation / hidden states | Megatron forward 或训练侧 hook | GEMM 左矩阵，形如 `[M, K]`，其中 `M` 往往来自 active tokens |
| `B` / weight shard | Megatron 参数或 vLLM weight metadata | GEMM 右矩阵，形如 `[K, N]` 或 lm_head 的 `[V_local, H]` 转置语义 |
| `target_ids` / active mask | rollout batch: tokens、loss mask、response ranges | 如果是 fused linear-logp，决定只取哪些 vocabulary column / token row |
| shape + dtype + stride | tensor 本身和 metadata builder | 决定 backend capability、accumulation dtype、downcast point |
| TP/CP/SP context | Megatron mpu、rollout engine topology | 决定 `N` 或 vocab shard 范围、是否需要 all-reduce/all-gather |
| vocab shard metadata | `vocab_start_index`, `global_vocab_size` | lm_head GEMM 后 selected-token logprob 的列空间映射 |
| reduction contract | RL-Kernel capability | K 维累加顺序、是否 split-K、TF32 策略 |
| weight version | VIME weight updater / vLLM engine | 证明比较的是同一版权重 |

在 VIME 里，GEMM consistency 最自然的第一落点不是所有 linear 层，
而是 **linear-logp / lm_head**：

```text
hidden [active_tokens, hidden]
  @ lm_head_weight.T [hidden, vocab_shard]
  -> logits or selected logprobs
  -> PPO/GRPO logprob delta / KL
```

原因很简单：这个位置离 loss 最近。
如果这里的 GEMM 有 batch-dependent drift，它会直接变成 logprob drift，再进入 ratio、KL 和 advantage 加权。

---

## 接入 Attention consistency path 时，VIME 需要给 RL-Kernel 什么

Attention 的一致性比 GEMM 更像一个“同一语义、多种物理形态”的问题。

训练侧通常看到 full sequence 或 CP 切分后的训练视角；
rollout 侧可能看到 prefill、chunked prefill、decode、paged KV cache、prefix cache、PD 分离。
这些路径都应该落在同一个 attention 语义上：

```text
Q, K, V
  -> scores = QK^T * scale + mask
  -> softmax + LSE
  -> output = P @ V
```

VIME 侧需要维护和传递的核心量：

| 核心量 | 来自哪里 | 对应 Attention 的位置 |
| --- | --- | --- |
| `q`, `k`, `v` | Megatron attention hook 或 vLLM attention metadata | Attention 输入张量，QK/PV 两次 reduction 的源头 |
| `causal` / mask | model config、sequence layout、padding info | scores 上的 mask 语义 |
| `q_start`, `k_start`, `Sq`, `Skv` | chunked prefill / decode / CP metadata | causal offset 和全局 token 位置 |
| position / RoPE state | Megatron/vLLM position ids、RoPE config | Q/K 进入 attention 前是否处在同一位置语义 |
| KV cache layout | vLLM paged KV / prefix cache metadata | rollout 侧 K/V 的物理来源和拼接语义 |
| LSE | RL-Kernel attention path 或 vLLM backend 可导出时 | softmax merge 和跨 page/chunk 对齐的关键中间量 |
| head layout | `Hq`, `Hkv`, GQA group | Q head 到 KV head 的映射 |
| dtype + scale | tensor dtype、attention scale | accumulation 和 tolerance 判断 |
| weight version / engine provenance | VIME rollout metadata | 证明 attention 输入来自同一版模型和同一路 rollout |

Attention consistency path 的第一阶段应该优先比较三类形态：

| Reference | Candidate | 目的 |
| --- | --- | --- |
| full prefill | chunked-query prefill | 检查 chunk boundary 是否改变 attention-domain 语义 |
| full prefill | paged-KV prefill | 检查 KV page materialization 和 LSE merge |
| prefill tail | decode step | 检查 decode 使用 KV cache 后是否仍等价于 full context 对应位置 |

这里 `LSE` 特别重要。
只比较 final output 有时能发现 drift，但不容易定位问题发生在 softmax max/sum、mask offset、
page merge 还是 PV reduction。
导出 attention-domain LSE 后，comparison report 才能把错误定位到更细的阶段。

---

## 在 VIME 中应该怎样接线

当前分支已经有 RL-Kernel 的基础边界：

| 文件 | 角色 |
| --- | --- |
| `vime/utils/arguments.py` | 解析 `--rlk-fast`、`--rlk-consistency`、`--rl-kernel-ops`，把 fast path 和 consistency path 作为正交开关 |
| `vime/backends/rl_kernel_utils/adapter.py` | VIME 调 RL-Kernel op 的唯一生产入口，负责输入 payload、adapter protocol、fallback/noop/mock |
| `vime/backends/rl_kernel_utils/execution.py` | 执行决策和 capability/contract 记录，负责 native、audit、optimized、strict、fallback 的选择和日志 |
| `vime/backends/rl_kernel_utils/__init__.py` | 对外导出 adapter 与 execution 的统一边界 |

下一步对接 Attention/GEMM 时，保持这个结构：

```text
Megatron side metadata builder
        \
         -> rl_kernel_utils.{gemm, attention} payload
        /
vLLM side metadata builder
        |
        v
rl_kernel_utils.adapter
        |
        v
RL-Kernel registry / comparison / deterministic op
        |
        v
ExecutionDecision + OperatorResult + drift metrics
```

不要让 Megatron actor 或 vLLM rollout 直接 import `rl_engine`。
VIME 侧只应该知道：

- 我要不要启用 RL-Kernel；
- 哪些 op 被 allowlist；
- 这次是 fast、audit 还是 strict；
- 本 batch 的 tensor、shape、dtype、token、mask、weight version、topology 是什么；
- 如果 RL-Kernel 不可用，是否 fallback 到 VIME native path。

RL-Kernel 侧才应该知道：

- 哪个 backend 可用；
- 这个 backend 的 contract id 是什么；
- tolerance 怎么定义；
- reference 和 candidate 怎么比较；
- deterministic kernel 是否满足 batch-invariant。

---

## 落地顺序

### 第一阶段：先做“可观察”

目标不是立刻替换 VIME 训练路径，而是先把一致性信息记录下来。

需要做到：

1. `--rlk-consistency audit` 可以打开 Attention/GEMM 的审计路径；
2. 每次审计都记录 operator、stage、dtype、shape、parallel context、weight version、contract id；
3. audit 不改变训练结果，只输出 drift metrics 和 execution decision；
4. 所有失败都 fallback native，不影响 VIME 原有训练。

### 第二阶段：再做“可约束”

当 audit 数据稳定后，引入 strict：

1. `--rlk-consistency strict` 要求 rollout/training 两侧 contract id 匹配；
2. Attention 要求 mask、position、KV cache、LSE merge 语义完整；
3. GEMM 要求 dtype、accumulation、sharding、vocab range 完整；
4. contract 缺失或 mismatch 时返回 `strict-failure`。

### 第三阶段：最后做“可替换”

fast path 应该在 consistency path 之后更自然：

1. `--rlk-fast auto` 使用 eligible backend，失败 fallback；
2. `--rlk-fast strict` 要求 fast backend 存在且 contract 满足；
3. 对 loss 最近的 linear-logp/GEMM 优先替换；
4. Attention fast path 要谨慎，只在 mask/KV/LSE 语义能完整表达后再替换。
