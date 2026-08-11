# vime 昇腾生态 CANN 算子对接 Roadmap

Status: Draft / review v5

这份 Roadmap 面向 vime 在 Ascend NPU / CANN 生态里的算子级接入。它参考
RL-Align/vime#6 的 vime/RL-Kernel 集成路线，也参考现有训推一致性、
分布式算子契约和 dlogp 诊断设计。核心定位是：

- vime 继续负责 RL 后训练编排：Megatron training、vLLM Ascend rollout、Data Buffer、reward flow、weight sync、batching、调度、audit replay 和 fallback policy。
- CANN 算子后端只作为可选、可观测、可回退的 operator backend 接入 vime，不成为第二套训练框架。
- Fast Path 和 Consistency Path 要分开建模：一个回答“能不能更快”，一个回答“训推 logprob 能不能被测量、诊断、强制一致”。
- 所有 CANN strict claim 都必须经过数值契约、metadata provenance、batch invariance 和分布式 sharding/reduction 契约验证。

本轮补充参考：

- AscendNPU-IR: https://gitcode.com/Ascend/AscendNPU-IR
- Triton Ascend Kernels: https://gitcode.com/Ascend/triton-ascend-kernels
- CANNBot Skills: https://gitcode.com/cann/cannbot-skills

## 0. 现状判断

当前 vime 已经有 Ascend/NPU 路线，但它和 CUDA 主线不是同一套完整实现：

- `origin/ascend` 分支包含 NPU 文档、`Dockerfile.npu`、CANN 9.0.0 环境、vLLM Ascend、Megatron/MindSpeed/torch-npu patch、Qwen3 NPU 脚本，以及 GRPO/PPO 示例。
- 当前 PR4 的 RL-Kernel benchmark 分支是 CUDA-oriented proof point：`linear_logp` 入口只显式支持 Triton/CUDA/registry，并且 telemetry 里有 `torch.cuda.Event`、`.is_cuda`、CUDA memory probe 等假设。
- `triton-ascend-kernels` 已经提供基于 Triton-Ascend 的高性能算子包，并带 tests/benchmark。里面有和 vime/RL 后训练很近的候选：GRPO loss、fused/linear cross entropy、RMSNorm、deterministic batch-invariant linear/log-softmax 等。它应该作为短周期 POC、reference 或 test oracle 来源，但不能自动等价于 vime 已有 production CANN backend。
- Triton-Ascend 调测文档显示其编译链路大致是 Triton Python DSL -> TTIR -> Triton-Ascend backend IR -> BiSheng -> AscendNPU IR -> NPU binary。因此 AscendNPU-IR 更适合进入 compiler/IR 级诊断、归因和长期融合路线，而不是替换 vime operator adapter。
- CANNBot Skills 把昇腾算子开发拆成多条路径：Ascend C direct invoke、Ascend C registry/ACLNN、Catlass、PyPTO、TileLang、Triton-Ascend、torch.compile graph mode、Runtime migration 等。Roadmap 里不能只写一个笼统的 `backend=cann`，必须记录每个 backend instance 走哪条 implementation route。
- 因此，“vime 能在 Ascend 上跑 RL 后训练”和“vime 已经能用 CANN 自定义算子替换 RL 热点路径”是两件事。前者在 `origin/ascend` 有基础，后者还需要做专门的 operator backend 适配。

本 Roadmap 的目标不是把 CUDA 代码硬移植到 NPU，而是在 Ascend 分支上建立一条干净的 CANN 算子接入线。

## 0.1 与原 Roadmap 的关系

这份文档不应该成为一条平行 Roadmap，也不应该重复实现原 RL-Kernel/vime Roadmap 已经规划的公共能力。更准确的定位是：

```text
原 Roadmap = vime/RL-Kernel 的公共控制面和一致性基础设施
CANN Roadmap = 原 Roadmap 下的 Ascend/CANN 后端落地附录
```

必须复用原 Roadmap 的部分：

- `--rlk-fast` / `--rlk-consistency` 用户语义；
- operator adapter 的 vime 侧边界；
- capability descriptor schema；
- execution decision record；
- structured fallback；
- rollout sample metadata fingerprint；
- requested-vs-actual provenance；
- dlogp / ratio0 / clipfrac0 / approx_kl0 诊断模块；
- ShardingSpec / ReductionSpec / Placement；
- A0-A5 audit profile；
- batch-invariance gate；
- operator-level、actor-window、full-step 三层报告口径。

CANN 只应该新增这些后端专属能力：

- `backend=cann` 的 capability descriptor 实例；
- 可选的 route-specific descriptor 实例，例如 `backend=triton_ascend`，或 `backend=cann` + `implementation_route=aclnn_registry|ascendc_direct|triton_ascend|ascend_ir`；
- CANN op package 的 Docker/wheel/source-build 交付方式；
- CANN/Ascend/torch-npu/vLLM Ascend/Megatron/MindSpeed build-runtime fingerprint；
- `DeviceRuntime` 的 NPU 实现；
- CANN `linear_logp`、后续 CANN ops 及其 reference compare；
- HCCL / CANN reduction engine 的 contract 实例；
- Ascend CI、profiling、benchmark 矩阵。

如果某个任务已经在原 Roadmap 里作为公共任务存在，这份 CANN Roadmap 不另开同功能 PR，只在该公共接口上补 `cann` backend、Ascend metadata 字段和 Ascend 测试覆盖。

## 1. 用户入口

继承 issue #6 的两开关模型：

```bash
--rlk-fast {off,auto,strict}
--rlk-consistency {off,audit,strict}
```

建议新增或兼容以下高级配置：

```bash
--rl-kernel-ops linear_logp,logp,rms_norm,attention
--rl-kernel-backend auto|native|cuda|triton|triton_ascend|cann
--rl-kernel-cann-route auto|triton_ascend|ascendc_direct|aclnn_registry|ascend_ir
VIME_RL_KERNEL_BACKEND=auto|native|cuda|triton|triton_ascend|cann
VIME_RLK_CANN_ROUTE=auto|triton_ascend|ascendc_direct|aclnn_registry|ascend_ir
VIME_RLK_CANN_STRICT_BUILD=0|1
```

行为语义：

- `--rlk-fast off --rlk-consistency off`：Ascend native vime 路径，不改变默认行为。
- `--rlk-fast auto --rl-kernel-backend cann`：CANN 算子满足 capability 和 contract 时启用，否则 structured fallback。
- `--rlk-fast strict --rl-kernel-backend cann`：指定 CANN backend 必须可用；不可用或不支持当前 shape/dtype/parallel mode 时 fail。
- `--rl-kernel-backend triton_ascend`：把 Triton-Ascend 算子包或生成算子作为 Ascend fast path 使用，但仍必须挂在同一个 adapter 和 consistency contract 后面。
- `--rl-kernel-cann-route ...`：在 Ascend/CANN 伞形后端下选择具体实现路线，不改变 vime 对用户暴露的 orchestration 语义。
- `--rlk-consistency audit`：即使不启用 CANN fast path，也记录 dlogp 和 metadata/provenance 诊断。
- `--rlk-consistency strict`：要求 rollout/training metadata、numeric contract、batch layout、parallel placement 全部可验证；不满足则 early fail。

迁移期保留现有 `--enable-rl-kernel`、`--rl-kernel-ops`、`--rl-kernel-strict`，但内部统一落到 execution decision 对象，而不是散落在各个 hot path 里。

## 2. CANN 接入边界

vime 侧应提供一层薄的 operator adapter，避免 production path 直接耦合 RL-Kernel 的实验控制器、child-process runner 或 benchmark artifact layout。

建议边界：

```text
vime
  ├─ rollout / Data Buffer / audit replay / training loop
  ├─ RlkOperatorAdapter
  │    ├─ capability query
  │    ├─ execution decision
  │    ├─ metadata and numeric contract validation
  │    ├─ structured fallback
  │    └─ telemetry
  └─ backend calls
       ├─ native torch/torch_npu reference
       ├─ RL-Kernel CUDA/Triton backend
       ├─ RL-Kernel Triton-Ascend backend
       └─ RL-Kernel Ascend CANN backend
```

vime 不应该猜测 CANN 是否支持某个 op，而应该从 backend descriptor 读取：

- op name、backend id、implementation fingerprint；
- implementation route：`triton_ascend`、`ascendc_direct`、`aclnn_registry`、`catlass`、`pypto`、`tilelang` 或 `ascend_ir`；
- CANN/driver/torch-npu/vLLM Ascend/Megatron/MindSpeed 版本；
- 使用对应路线时的 Triton-Ascend、BiSheng、AscendNPU-IR、Catlass/PyPTO/TileLang 版本；
- dtype：fp32、bf16、fp16 等；
- shape range、alignment、workspace 需求；
- tiling policy、UB/workspace budget、grid limit 和 compile cache policy；
- forward-only / autograd / saved-state 策略；
- TP/SP/CP 支持矩阵；
- HCCL / in-op reduction / boundary reduction 语义；
- batch-invariant / deterministic claim；
- strict consistency eligibility；
- fallback reason 和 fallback 是否允许。

## 3. CANN 算子交付形态

CANN backend 不应只是一段 Python import 逻辑，而要有清楚的 build/runtime 生命周期。建议在第一版就明确三件事：

1. **交付方式**
   - Docker image 内置 CANN op package；
   - wheel 安装 CANN op package；
   - 源码编译作为开发路径。

2. **Python 绑定**
   - vime 只依赖稳定的 Python operator object；
   - 具体是 torch extension、torch-npu 自定义算子绑定、CANN/ACLNN wrapper，还是 RL-Kernel 自己的 registry，由 backend package 内部处理；
   - 如果路线是 Triton-Ascend，vime 应通过 `triton_ascend_kernels` 这类包边界或 RL-Kernel 自己的 wrapper 消费，而不是 import 零散 benchmark 脚本；
   - 如果路线是 Ascend C direct invoke，它只作为原型验证和 microbenchmark 路径；进入 production 前必须有稳定 PyTorch/ACLNN-facing wrapper 和 descriptor；
   - vime 不直接依赖 CANN 编译目录、算子源码路径或临时 build artifact。

3. **版本指纹**
   - CANN toolkit / kernels / NNAL-ATB；
   - Ascend driver；
   - torch、torch-npu；
   - triton-ascend、triton-ascend-kernels；
   - BiSheng compiler 和 AscendNPU-IR lowering path；
   - vLLM Ascend；
   - Megatron、MindSpeed、Megatron-Bridge patch；
   - CANN op package commit、compiler flags、op implementation id。

strict 或 benchmark 报告必须携带这些指纹。否则同一个 `backend=cann` 在不同机器上可能代表完全不同的数值和性能行为。

同时，vime 需要把 CUDA-specific 设施抽出来：

```text
DeviceRuntime:
  current_device()
  synchronize()
  event_timer_or_none()
  memory_allocated_or_none()
  reset_peak_memory_or_none()
  profiler_range()
```

CUDA backend 可以继续用 `torch.cuda.Event`；CANN backend 用 NPU runtime 能力。某个能力不可用时，应报 structured unavailable，而不是假装数值为 0。

## 3.1 Ascend 实现路线

`backend=cann` 不能隐藏具体实现路线。不同路线成熟度不同，准入门禁也不同。

| 路线 | 主要用途 | vime 准入规则 |
|---|---|---|
| `triton_ascend` | 快速 POC、复用现有 Triton-Ascend kernels、单卡对拍 | 只能通过 adapter 作为 Ascend fast path；必须通过 multi-shape correctness、no-PyTorch-fallback 和 benchmark-after-verify 门禁。 |
| `ascendc_direct` | `<<<>>>` kernel 直调原型和 microbenchmark | 不能单独作为 production route；只有在补齐稳定 Python/PyTorch 或 ACLNN-facing wrapper、descriptor、clean-build package 后才能升级。 |
| `aclnn_registry` | 标准 Ascend C/CANN 注册算子接入 | 长期 production CANN ops 的优先路线，因为它提供 framework-facing operator boundary。 |
| `catlass` / `blaze` | matmul-heavy kernels 和 epilogue fusion | profiling 证明 GEMM/epilogue 是瓶颈后，用于 `linear_logp`、projection、fused CE 等路线。 |
| `pypto` / `tilelang` | 其它 DSL 路线的快速探索 | 可用于探索，但 strict mode 必须满足同一套 descriptor、numeric contract 和 CI gate。 |
| `ascend_ir` | compiler/IR 诊断、lowering inspection、graph-level fusion 和长期优化 | 不是 Phase 0-2 依赖；当 Triton-Ascend/CANN 编译问题需要 IR 归因或稳定 op chain 值得融合时再进入。 |
| `torch_compile_npugraph` | graph capture/replay 和整图实验 | 除非仍能产出 operator-scoped decision、provenance 和 fallback，否则不进入 operator adapter。 |

每条 execution decision 都必须记录 route：

```text
implementation_route=triton_ascend|ascendc_direct|aclnn_registry|catlass|pypto|tilelang|ascend_ir|native
route_source_repo=...
route_source_commit=...
route_artifact=wheel|docker|source-build|direct-build|jit-cache
compile_stage=python|ttir|ttadapter_ir|bisheng|ascendnpu_ir|npubin|aclnn
```

## 3.2 Route-specific 准入门禁

Triton-Ascend 准入：

- 先评估现有 `triton-ascend-kernels` 是否能作为 `linear_logp`、`logp`、`linear_cross_entropy`、`grpo_loss`、RMSNorm、deterministic batch-invariant linear/log-softmax 的 reference、fast POC 或 test oracle。
- correctness verification 覆盖所有声明 shape 前，不允许报告 benchmark。
- fast path 必须检测 PyTorch fallback；如果 wrapper 静默用 `torch.*` 计算，不能被接受为 operator backend。
- 调试时记录 `TRITON_DEBUG`、`TRITON_INTERPRET`、`TRITON_ALWAYS_COMPILE`、JIT cache policy 和 intermediate IR dump。
- `ub overflow`、`grid > 65535`、bf16 interpreter 不支持、compile-stage failure 等，在 tiling 或 route 修复前都应成为 structured unsupported/fallback reason。

Ascend C direct/registry 准入：

- 必要时 kernel source 使用 `.asc`；direct invoke `<<<>>>` 是 prototype path，production 应暴露稳定 PyTorch 或 ACLNN/registry boundary。
- tiling 在 host 侧计算并 fingerprint；core count、vector/cube core 选择、UB size、block count 不允许硬编码。
- 非对齐 GM/UB 搬运和 vector API 必须对照官方 API 约束验证，API choice 写入 operator design record。
- Host code 不应包含本该在 device 上执行的 tensor-compute loops。
- PyTorch 暴露路径应包含干净的 `torch.ops` load 或等价 framework boundary，并在需要时提供 meta/shape 行为。

所有路线：

- release claim 必须来自 fresh checkout 或 pinned image。
- vendoring 或改造第三方 kernel 前必须保留 license/provenance 记录。
- 生成 artifact 和 debug dump 可作为可选证据保存，但 vime production import 不能依赖这些 artifact path。

## 4. Consistency Contract

核心指标沿用 vime/RL-Kernel 一致性设计：

```text
dlogp = training-side recomputed logp - rollout-side old logp
```

只在 active response/action tokens 上计算。prompt、padding、masked-out response positions 不进入聚合。

比较成立的前提：

- 同 checkpoint 或 weight version；
- 同 token IDs；
- 同 active loss/action mask；
- 同 tokenizer、chat template、padding semantics；
- 同 sampling metadata：temperature、top-p、top-k 等；
- 同 position IDs、packed sequence metadata、KV/cache metadata；
- 同 quantization/dequantization policy；
- 同 numeric contract：accumulation dtype、reduction order、downcast point、特殊 merge semantics；
- 同 sharding contract：TP/SP/CP 输入输出如何切、哪些张量是 Partial；
- 同 batch-invariance contract：同一个 sample 放进不同 batch、不同 padding/packing、不同 microbatch 时，strict 覆盖范围内的 logp/gradient 不应漂移出容差。

诊断指标：

- `abs_dlogp` mean、p50、p90、p99、max；
- `ratio0 = exp(dlogp)`；
- `clipfrac0`；
- `approx_kl0 = mean(exp(dlogp) - 1 - dlogp)`；
- active token count、mask coverage、zero-active-token cases；
- worst-token metadata：sample id、token position、rank、op、backend、contract id；
- per-rank drift；
- requested-vs-actual provenance；
- fallback count/reason；
- CANN build/runtime fingerprint。

strict 阈值不应硬编码在 vime 文档里。vime 应消费 RL-Kernel/CANN backend 暴露的 per-dtype tolerance contract，并在报告里写 contract id。

## 5. Sharding / Reduction 契约

一致性设计的关键判断是：通信不是算子之外的独立问题，而是分布式算子输出的一部分。CANN 接入必须从第一天就把这个观念写进接口。

建议抽象：

```text
Placement = Shard(dim) | Replicate | Partial(reduce_kind)

ShardingSpec:
  inputs:  每个输入张量的 Placement
  outputs: 每个输出张量的 Placement

ReductionSpec:
  kind: sum | max | logsumexp | custom
  accumulator_dtype: fp32 | bf16 | fp16
  order: canonical_rank | fixed_tree | backend_default
  engine: in_op | hccl | boundary | reference
  downcast_point: after_reduce | final_write | none
```

对 CANN 后端尤其要明确：

- HCCL 只负责搬运/归约，还是 CANN op 内部自己控制归约顺序；
- strict mode 下能否固定 rank order 或 fixed tree；
- SP RMSNorm 这种“算子内不归约、边界归约”的场景，边界通信是否也绑定同一个 ReductionSpec；
- CP attention 的 LSE merge 不是普通 sum reduction，要单独声明；
- 如果 fast path 使用 HCCL backend default order，它可以是 opportunistic fast，但不能自动声明 strict。

## 6. 算子优先级

优先级继承原一致性路线的漂移排序，并结合 Ascend/CANN 的工程价值。

| 优先级 | 算子/路径 | 为什么先做 | 第一阶段 strict 范围 |
|---|---|---|---|
| P0 | `linear_logp` | PR4 已证明 vime 边界可行；CANN 版可直接服务 RL logprob 热点 | BF16/FP32，CP=1，entropy=0，teacher-forcing，TP 元数据显式 |
| P0 | native audit dlogp | 不启用 CANN 也能先量 Ascend native 漂移 | 单卡、active token、zero-update |
| P0 | Triton-Ascend `linear_cross_entropy` / deterministic linear-logsoftmax probe | 现有 Triton-Ascend kernels 接近 vime logprob 路径，可缩短 POC/admission | 单卡、multi-shape verify、无 PyTorch fallback、只做 audit-only strict claim |
| P1 | `logp` / selected softmax | 训推一致性的核心指标；也可作为 `linear_logp` fallback/reference | 单卡，无 autograd 或 reference autograd |
| P1 | RMSNorm | 单层影响小但层层累积；SP 边界契约代表性强 | 单卡 + SP boundary |
| P1 | GRPO loss fragments | Triton-Ascend 已有 GRPO-loss-oriented kernels，且直接影响 RL 后训练稳定性 | CPU/torch_npu reference first；audit 未稳定前不做 production strict |
| P2 | matmul/projection epilogue | split-K、GEMM epilogue、dtype/downcast 容易造成漂移 | 单卡 role-pair，TP projection 后置 |
| P2 | attention / CP LSE merge | 影响最大，但工程面最宽，必须等 audit 和 metadata 成熟 | 先 audit，后 CP strict |
| P3 | RoPE / SwiGLU / embedding | 漂移通常较小，但应进入完整 forward-chain 对拍 | 单卡 role-pair |
| P3 | PyPTO/TileLang/Catlass exploratory variants | core adapter 和 audit gates 已存在后，用于 route 对比 | descriptor 和 CI gates 未达 CANN 规则前只做探索 |

顺序原则：先单卡算子对拍，再分布式算子，再通信/拓扑 sweep。不能在单卡 CANN op 还没验证时，把多卡 drift 直接归因给 HCCL。

## 7. Phase 0: Ascend 基线和分支策略

Deliverables:

- 以 `origin/ascend` 为 CANN 工作基线，明确不在 CUDA benchmark PR 上直接混入 NPU 改动。
- 固定最小验证环境：Ascend Atlas A2/A3、aarch64、Python 3.12、CANN 9.0.0、torch-npu、vLLM Ascend、Megatron/MindSpeed patch 版本。
- 记录 AscendNPU-IR、triton-ascend、triton-ascend-kernels 以及任何 CANNBot-derived operator template 的 source pin。
- 整理一页 `docs/en/get_started/NPU.md` / 中文对应文档里的 CANN 算子开发附录。
- 给 CANN operator package 定义安装方式：源码编译、wheel、Docker image 三种至少一种可复现。
- 增加 `DeviceRuntime` 薄抽象，先替换 RL-Kernel telemetry path 里的 CUDA-only event/memory/profiler 调用。
- 在文档里增加 implementation route matrix，让用户知道一次运行实际用了 `triton_ascend`、`ascendc_direct`、`aclnn_registry` 还是 native fallback。

Acceptance:

- native Ascend vime GRPO 示例可跑通。
- 训练、rollout、weight sync、HCCL 环境变量和 Ray NPU visible devices 行为可复现。
- 文档明确当前支持分支和不支持当前 CUDA PR4 分支直接启用 CANN。

## 8. Phase 1: 复用公共 Adapter，补 CANN Capability 和 Decision 实例

Deliverables:

- 依赖原 Roadmap 的公共 `RlkOperatorAdapter` 或等价薄层；不在 CANN 路线重复实现。
- 在公共 capability schema 下新增 CANN backend descriptor。
- 新增 route-specific descriptor 字段：implementation route、route source repo/commit、compile stage、route artifact、tiling policy、UB/workspace budget 和 debug dump availability。
- 在公共 structured execution decision record 中填充 CANN 字段，例如：

```text
op=linear_logp
requested_backend=cann
actual_backend=cann|triton_ascend|native|fallback
implementation_route=triton_ascend|ascendc_direct|aclnn_registry|ascend_ir
fast_mode=auto|strict
consistency_mode=off|audit|strict
eligible=true|false
contract_id=...
fallback_allowed=true|false
fallback_reason=...
hardware=ascend
cann_version=...
torch_npu_version=...
route_source_commit=...
compile_stage=...
```

- 复用公共 `enable_rl_kernel` / `rl_kernel_strict` / fallback counter 接入逻辑。
- CANN backend 不存在时，`auto` warning + fallback，`strict` fail。
- CANN backend descriptor 带完整 build/runtime fingerprint。

Acceptance:

- 默认 off 时 native 行为完全不变。
- 每个算子调用都有可解释 decision。
- 不允许 silent fallback。
- 当前无 CANN op 实现时，公共 audit mode 仍可独立运行；CANN 只补 Ascend provenance 和 runtime fingerprint。

## 9. Phase 2: CANN `linear_logp` POC

目标：复刻 CUDA `linear_logp` 的 vime 接入形态，但后端实现和 telemetry 全部 Ascend-native。

第一版实现可以优先尝试 `triton_ascend`，前提是它能比原生 Ascend C 更快满足 vime operator contract。Roadmap 应先评估已有 Triton-Ascend linear/cross-entropy/log-softmax 候选，再决定是否从零写 CANN 算子。如果 Triton-Ascend 无法覆盖 support matrix 或 strict contract，再用 `ascendc_direct` 做 prototype validation，并以 `aclnn_registry` 作为 production packaging 路线。

Inputs:

```text
hidden_states: [T, H]
lm_head_weight: [V_local, H]
target_ids: [T]
bias: optional
tp_group / hccl_group metadata
vocab_start_index
global_vocab_size
rollout_temperature
```

Deliverables:

- CANN `linear_logp` forward。
- Triton-Ascend 路线探针：现有或改造的 `linear_cross_entropy`、deterministic linear 和 log-softmax kernels。
- autograd backward 或 reference backward fallback。
- torch/torch_npu reference path。
- CANN backend descriptor。
- dtype/shape/alignment support matrix。
- fallback reasons：unsupported dtype、unsupported CP、entropy requested、missing TP metadata、no autograd、build missing。
- compile-stage fallback reasons：unsupported Triton DSL、Triton-Ascend lowering failure、BiSheng failure、AscendNPU IR failure、`ub overflow`、`grid > 65535`、JIT cache mismatch。
- claimed fast path 必须有 no-PyTorch-fallback check。
- NPU event/timer 通过 `DeviceRuntime` 接入；没有稳定 API 时先只报 host timer，并明确标注。
- memory probe 通过 `DeviceRuntime` 接入 torch_npu/NPU memory API；不可用时 structured unavailable。

Acceptance:

- 单卡 CANN vs torch_npu reference 在 active token 上通过 per-dtype tolerance。
- zero-update GRPO/SFT teacher-forcing audit 通过。
- 报告任何 benchmark 前，multi-shape verification 必须先通过。
- `--rlk-fast strict --rl-kernel-backend cann --rl-kernel-ops linear_logp` 在支持配置下 fallback=0。
- 不支持配置能给出稳定 fallback/fail reason。

## 10. Phase 3: 接入公共 Audit Harness 的 Ascend 扩展

Deliverables:

- 复用原 Roadmap 的 native audit-only dlogp 和 teacher-forcing replay path；不重复实现指标模块。
- 在公共 rollout sample metadata fingerprint 上补 Ascend 字段：
  checkpoint、tokenizer、sampling config、mask/padding、position/cache、quantization、parallel placement、backend contract。
- 在公共 requested-vs-actual provenance 上补 Ascend 字段：
  vLLM Ascend engine、router、server group、weight version、CANN build id、torch-npu runtime。
- 复用公共 batch-layout fingerprint：
  sequence lengths、packed order、padding side、active mask density、microbatch id、global batch shape、dynamic sampling group。
- 复用公共 result cube，并增加 Ascend/CANN axes：
  batch size、layout、dtype、TP/SP/CP、backend、cache policy、router policy、determinism policy。

Acceptance:

- 已知 metadata mismatch 会先被抓住，而不是误判为算子漂移。
- audit mode 不改变训练行为。
- 报告 worst token、per-rank drift、fallback reason 和 active token coverage。
- 同 sample 单独跑和混 batch 跑，在 strict 覆盖矩阵内通过 tolerance。

## 11. Phase 4: 单卡 CANN 算子对拍

Deliverables:

- 将 operator specs 从 `linear_logp/logp` 扩到：
  attention、lm_head/logp、projection matmul、RMSNorm、RoPE、SwiGLU、embedding、RL loss fragments。
- 每个 op 至少有两个 role：train reference 和 infer/serving style reference；CANN fast backend 是第三个实现。
- 对适用场景，把 Triton-Ascend deterministic operators 加进 admission table：`linear_batch_invariant`、`log_softmax_batch_invariant`、`mean_batch_invariant`、`rms_norm_batch_invariant`。
- 输出从 pass/fail 扩展成 drift attribution：每个 op 对 `dlogp` 或中间误差贡献多少。
- forward-chain 对拍：把前向链按真实顺序串起来，看小误差层层累积。

Acceptance:

- 每个 P0/P1 op 都有单卡 train-vs-infer 对拍。
- CANN backend 宣称 strict 前，必须通过单卡同 sample batch-invariance 测试。
- forward-chain 只用于 CI/admission/audit，不在 production 每次调用前运行。

## 12. Phase 5: 分布式 CANN/HCCL 契约

Deliverables:

- `linear_logp` TP CANN 测试：vocab shard、target id routing、HCCL group、global vocab size。
- TP matmul/projection 测试。
- SP RMSNorm boundary reduction 测试。
- CP attention LSE merge audit 测试。
- A0-A5 profile：
  A0 fully aligned reference；
  A1 arithmetic-only；
  A2 reduction/topology-only；
  A3 representation-only；
  A4 pairwise mismatch；
  A5 production mismatch。
- HCCL algorithm/topology/rank placement sweep。

Acceptance:

- 单卡没过的 op 不进入分布式归因。
- distributed drift 能归因到 sharding、reduction order、dtype/downcast、HCCL engine、metadata mismatch 或 batch layout。
- strict distributed run 有 reference path。
- overlap/bucket/fusion 改变 reduction grouping 时必须作为显式 axis，不允许作为 silent fast path。

## 13. Phase 6: Profiling 和性能扩展

Deliverables:

- Ascend profiling 脚本：warmup、stable window、actor train、rollout generate、weight sync、communication。
- CANN/NPU profiling 标记：
  forward、selected logprob、backward、optimizer、HCCL collectives、weight sync、host wait。
- Triton-Ascend compile/runtime profiling 要区分 Python/JIT compile time、BiSheng compile time、first-run cache fill、steady-state kernel time 和 host wait。
- operator-level、actor-window、full-step 三层报告分开。
- 只有 profiling 证明是瓶颈，且 contract 能描述，才扩展新 CANN fast op。

Acceptance:

- 不能用单算子 speedup 宣称 full-step speedup。
- 报告能区分 kernel compute、HCCL communication、rollout wait、weight sync、host orchestration。
- `linear_logp` 作为 proof point 保留，但不被过度宣称为端到端瓶颈。

## 14. Phase 7: CI、Benchmark 和发布规则

CI layers:

- CPU/PyTorch reference：指标数学和 metadata validation。
- torch_npu native：Ascend native audit。
- CANN single-op：P0/P1 operator tolerance。
- Triton-Ascend route CI：correctness before benchmark、multi-shape coverage、no PyTorch fallback、structured compile-stage failure capture。
- Ascend C route CI：clean source build、`.asc`/CMake checks、host-side tiling validation、dynamic core/UB assumptions、PyTorch 或 ACLNN wrapper smoke。
- distributed NPU：TP/SP/CP/HCCL contract。
- batch-invariance：同 sample 跨 batch size、padding/packing、active density、microbatch。
- production smoke：Qwen3-4B GRPO short run。
- scheduled benchmark：Qwen3-30B-A3B 或代表性 MoE。

发布规则：

- operator speedup 只能称 operator speedup。
- full-step speedup 必须有 full-step benchmark。
- strict consistency 必须声明覆盖的 dtype、shape、parallel mode、batch-layout matrix。
- CANN backend 版本、CANN/torch-npu/vLLM Ascend/Megatron/MindSpeed fingerprint 必须写进报告。
- fallback=0 只对当前覆盖矩阵成立，不自动外推。

## 15. 建议 PR 顺序

1. **Roadmap and boundary**
   - 落本文档，声明 CANN backend 不进入 CUDA PR4 主线。

2. **公共 adapter 依赖确认**
   - 确认原 Roadmap 的 backend selector、capability descriptor、execution decision、structured fallback 已落地或有 keeper issue。

3. **DeviceRuntime and CANN packaging**
   - 抽出 CUDA/NPU event、memory、profiler runtime；定义 CANN op wheel/Docker/source-build 指纹。

4. **Implementation route matrix**
   - 增加 `triton_ascend`、`ascendc_direct`、`aclnn_registry`、`catlass`、`pypto`、`tilelang` 和 `ascend_ir` 的 descriptor 字段与文档。

5. **Ascend audit extension**
   - 在公共 dlogp、metadata fingerprint、requested-vs-actual provenance 上补 Ascend 字段和测试。

6. **Triton-Ascend `linear_logp` route probe**
   - 在 vime contract 下评估现有或改造的 Triton-Ascend linear/cross-entropy/log-softmax kernels。

7. **CANN `linear_logp` POC**
   - forward、reference compare、support matrix、fallback reasons、compile-stage diagnostics。

8. **CANN `linear_logp` autograd and strict**
   - backward/saved-state、batch-invariance、TP metadata、strict mode。

9. **Single-card operator table**
   - `logp`、RMSNorm、matmul/projection、RoPE/SwiGLU/embedding、GRPO loss、deterministic Triton-Ascend candidates。

10. **Distributed contracts**
   - TP `linear_logp`、SP RMSNorm、CP attention audit、A0-A5 profile。

11. **Profiling and benchmarks**
   - Ascend stable-window profiling、operator/actor/full-step reports。

12. **Docs and release matrix**
   - NPU docs、Docker/wheel install、CI/benchmark coverage、known unsupported cases。

## 16. 非目标

- 不替换 Megatron、vLLM Ascend、MindSpeed 或 torch-npu。
- 不把 RL-Kernel/CANN 变成 vime 的必需依赖。
- 不在 CUDA benchmark 分支上混入大规模 NPU 条件分支。
- 不重复实现原 Roadmap 的 flags、adapter、decision、metadata、audit、result cube 或 report schema。
- 不把 CANN backend 的 import 成功当作 capability 成功。
- 不允许 silent fallback。
- 不在 audit mode 里强制 bitwise parity。
- 不用 isolated operator benchmark 代表 full-step 收益。
- 不在 metadata 不完整时宣称 strict consistency。
- 不把 Triton-Ascend prototype、direct-invoke executable 或 graph-mode experiment 当成 production CANN backend，除非已有 adapter descriptor、wrapper boundary、clean packaging 和 CI gates。
- 不在缺少 license/provenance review 的情况下 vendoring 或改造第三方 kernels。

## 17. 风险与缓解

| 风险 | 缓解 |
|---|---|
| CANN / torch-npu / vLLM Ascend 版本漂移 | backend descriptor 写完整 fingerprint；CI 固定镜像；release matrix 明确版本 |
| CANN op build 难复现 | Docker image、wheel、源码编译至少保留一条稳定路径 |
| implementation routes 混在一起 | 每个 descriptor 记录 `implementation_route`、route source commit、compile stage、artifact type 和 admission gate |
| Triton-Ascend POC 静默 fallback 到 PyTorch | fast path 宣称前增加 no-PyTorch-fallback check |
| Triton-Ascend kernels 命中 UB/grid/compiler limits | 把 `ub overflow`、`grid > 65535`、Triton lowering、BiSheng、AscendNPU IR failures 记录成 structured fallback reasons |
| direct-invoke prototype 被误认为 production integration | production 前强制 PyTorch 或 ACLNN/registry wrapper、clean package 和 descriptor |
| IR-level optimization 绕过 operator telemetry | AscendNPU-IR 和 graph-mode work 必须挂在 operator-scoped decisions 后面，或标记为 non-admitted research |
| fast op 数值不一致 | strict 只认 numeric contract；audit 先量，不直接 fail production |
| HCCL reduction 顺序不可控 | strict mode 使用 reference/in-op deterministic path，或把 HCCL order 声明为 audit axis |
| batch layout 影响 tiling 和 reduction | same-sample batch-invariance 作为 strict gate |
| metadata 过大 | 默认只存 fingerprint；debug 时再 dump verbose replay artifact |
| fallback 掩盖缺失能力 | decision record + fallback counter + strict fail |
| 单卡没过就测多卡导致误归因 | 强制单卡 admission 先于 distributed admission |
| profiler 混入编译/初始化噪声 | warmup + stable window + 分层报告 |
| custom rollout 缺 metadata | audit warning；strict 对该 sample fail 或禁用 strict claim |

## 18. Review Passes

### Review 1: 对齐 issue #6

- 保留 vime orchestration / RL-Kernel operator backend 的边界。
- 保留 Fast Path 与 Consistency Path 正交设计。
- 保留 capability reporting、numeric contract、metadata fingerprint、fallback telemetry。
- 避免把 RL-Kernel 实验控制器引入 vime production path。

### Review 2: 对齐训推一致性设计

- 使用 `dlogp = training logp - rollout old logp` 作为主指标。
- 明确 active token 口径。
- 保持“先单卡算子，再分布式算子，再通信归因”的顺序。
- 把通信视为分布式算子契约的一部分，而不是独立附属问题。
- 引入 ShardingSpec、ReductionSpec、Placement、batch-invariance 和 A0-A5。

### Review 3: Ascend/CANN 可执行性

- 承认当前可跑 RL 后训练的是 `origin/ascend`，不是当前 CUDA benchmark 分支。
- 第一颗 CANN op 选 `linear_logp`，因为 vime 接入边界已经被 CUDA proof point 验证。
- 不假设 CUDA event/memory probe 可复用，NPU telemetry 单独建。
- CANN 算子有交付形态、Python 绑定边界和 build/runtime fingerprint。
- HCCL/CANN strict claim 要经过 explicit contract，不走 backend default 静默假设。
- 文档区分 operator-level、actor-window、full-step claim。

### Review 4: 与原 Roadmap 去重

- CANN Roadmap 是原 Roadmap 的后端附录，不是平行实现。
- 公共能力只实现一次：flags、adapter、capability schema、execution decision、fallback、metadata、provenance、audit metrics、result cube。
- CANN PR 只补 backend descriptor、DeviceRuntime NPU 实现、CANN 算子、HCCL contract、Ascend CI/benchmark。
- 任何“新增 audit core / 新增 adapter skeleton”的表述都应改成“复用公共层并增加 Ascend 扩展”。

### Review 5: 对齐昇腾实现生态

- `backend=cann` 不再是模糊桶；每次运行都记录 `implementation_route`。
- Triton-Ascend 被明确为短周期 POC/reference 路线，特别适合先评估 RL-adjacent linear CE、GRPO loss、RMSNorm 和 deterministic batch-invariant kernels。
- Ascend C direct invoke 保留为 prototype/microbenchmark，ACLNN/registry 是 production wrapper 的优先路线。
- AscendNPU-IR 定位为 compiler/IR diagnostics 和长期 fusion 基础设施，不替代 vime adapter。

### Review 6: 验证和发布门禁

- Triton-Ascend 与原生 CANN 路线都必须先过 correctness gate，再报告 performance gate。
- multi-shape verification、no-PyTorch-fallback、clean-build reproducibility、compile-stage diagnostics、license/provenance review 都已显式化。
- UB overflow、grid limit、JIT cache、BiSheng、AscendNPU IR failures 都是 structured unsupported/fallback reasons。
- direct-invoke 和 graph-mode experiments 没有 wrapper、descriptor 和 CI admission 前，不能宣称 production backend support。

## 19. 期望终态

用户可以在 Ascend 环境里运行：

```bash
python train.py \
  --enable-rl-kernel \
  --rl-kernel-backend triton_ascend \
  --rl-kernel-ops linear_logp \
  --rlk-fast auto \
  --rlk-consistency audit
```

并清楚看到：

- CANN backend 是否被选中；
- 哪个 op 实际走 CANN，哪个 op fallback；
- fallback 原因；
- implementation route、source commit、compile stage、artifact type；
- CANN / torch-npu / CANN op build fingerprint；
- active-token dlogp 指标；
- per-rank drift；
- worst token；
- batch-invariance 覆盖矩阵；
- operator-level、actor-window、full-step 三层性能报告；
- 当前 strict consistency 是否只是 audit、tolerance strict，还是 bitwise deterministic op claim。

当用户切到：

```bash
python train.py \
  --enable-rl-kernel \
  --rl-kernel-backend cann \
  --rl-kernel-cann-route aclnn_registry \
  --rl-kernel-ops linear_logp,logp,rms_norm \
  --rlk-fast strict \
  --rlk-consistency strict
```

vime 只有在所有 enabled CANN ops 同时满足 capability、numeric contract、metadata completeness、batch-invariance 和分布式契约时才成功；否则选择 contract-preserving reference path，或按 strict policy fail。
