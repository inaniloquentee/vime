# VIME vs VIME + RL-Kernel Nsight Systems Trace 分析报告

## 分析范围

输入 trace：

- `cuda_qwen3_4b_full_actor_train_r1.nsys-rep`
- `cuda_qwen3_4b_full_actor_train_r2.nsys-rep`
- `vime/trace` 下的测试配置说明

测试配置关键信息：

- 模型：Qwen3-4B
- GPU：2 x NVIDIA H100 80GB HBM3
- 并行：TP=2，PP=1，DP=1
- batch：global_batch_size=4，micro_batch_size=1
- rollout：rollout_batch_size=2，n_samples_per_prompt=2，max_response_len=512
- dtype：参数 bf16，attention softmax fp32
- recompute：full / uniform / 1 layer

trace 身份判断：从 kernel 名称看，r1 包含 `linear_logp_probs_bf16_forward_kernel` 和 `linear_logp_local_probs_bf16_to_dlogits_row_kernel`；r2 包含 `fused_linear_logp_sm90_kernel` 和 `fused_linear_logp_sm90_combine_kernel`。因此我把 r1 理解为 save-probs / 非 SM90 fused 的 linear-logp 路径，把 r2 理解为 fused-tile SM90 的 RL-Kernel 路径。如果 r1 原本想表示纯 VIME，需要注意：r1 trace 里仍然出现了自定义的 `linear_logp_*` CUDA kernel。

## 核心结论

这两份 trace 里，主要瓶颈不是 RL-Kernel 的 linear-logp kernel。GPU 侧最大开销是 TP=2 带来的 NCCL AllReduce 通信，并且存在明显的 rank/device 不均衡。更大的端到端问题是：在被捕获的 actor 训练窗口内，GPU 大部分时间处于空闲/等待状态，同时 trace 里还混入了 CUDA、cuBLAS、NCCL 的 lazy initialization。

关键现象：

- r1 的 actor range 大约是每个 rank 6.49 s；r2 大约是 6.35-6.40 s。端到端 actor range 只提升了约 1-2%。
- 双卡 GPU kernel 总时间从 r1 的 451.1 ms 降到 r2 的 283.0 ms，但没有等比例转化为 wall-clock 提升，因为 GPU active time 只占 2-GPU actor 窗口容量的 3.45%（r1）和 2.21%（r2）。
- NCCL AllReduce 是最大的 GPU kernel 类别：r1 为 331.8 ms，r2 为 174.8 ms。
- RL-Kernel linear-logp 在完整 trace 里占比很小：r1 为 2.26 ms，r2 为 5.39 ms。即使继续大幅优化这块，也很难撬动 6.4 s 级别的 actor wall time。
- GPU memcpy/memset 可以忽略：两份 trace 都不到 1 ms。
- 当前 capture 包含初始化开销：`cuLibraryLoadData` 在 r1 中为 6.75 s，在 r2 中为 6.00 s；`cuBLAS:cublasCreate_v2` NVTX range 在 r1 中为 4.19 s，在 r2 中为 3.69 s。因此这份 trace 不能直接当作干净的 steady-state per-step profile。

## 顶层时间对比

| 指标 | r1 | r2 | 变化 |
|---|---:|---:|---:|
| Actor NVTX range，rank 0/1 | 6.496 s / 6.488 s | 6.349 s / 6.405 s | 约快 1-2% |
| CUDA kernel capture wall | 6.362 s | 6.344 s | 基本相同 |
| 双卡 GPU kernel 总时间 | 451.1 ms | 283.0 ms | -37.3% |
| 2-GPU active-time 容量占用 | 3.45% | 2.21% | r2 GPU 工作量更低 |
| GPU kernel 数量 | 11,946 | 8,508 | -28.8% |
| GPU memcpy/memset 时间 | 0.90 ms | 0.86 ms | 可忽略 |

最重要的矛盾是：actor 窗口约 6.4 s，但真实 GPU kernel 执行只有几百毫秒。这说明瓶颈主要在 kernel 之外，例如等待、调度、初始化、rollout 管线或通信同步。

## GPU Kernel 分类拆解

### r1

| 类别 | 时间 | 占比 | 次数 |
|---|---:|---:|---:|
| NCCL AllReduce | 331.8 ms | 73.6% | 463 |
| PyTorch elementwise/reduce/copy | 59.8 ms | 13.3% | 8,278 |
| GEMM / attention nvJitLink kernels | 38.2 ms | 8.5% | 1,536 |
| NCCL AllGather | 8.2 ms | 1.8% | 9 |
| Norm kernels | 5.7 ms | 1.3% | 906 |
| Triton fused pointwise | 3.8 ms | 0.8% | 635 |
| RL-Kernel linear-logp | 2.3 ms | 0.5% | 6 |

### r2

| 类别 | 时间 | 占比 | 次数 |
|---|---:|---:|---:|
| NCCL AllReduce | 174.8 ms | 61.8% | 292 |
| PyTorch elementwise/reduce/copy | 49.0 ms | 17.3% | 6,148 |
| TransformerEngine Adam | 21.9 ms | 7.8% | 101 |
| GEMM / attention nvJitLink kernels | 19.6 ms | 6.9% | 854 |
| RL-Kernel linear-logp | 5.4 ms | 1.9% | 6 |
| NCCL AllGather | 3.6 ms | 1.3% | 7 |
| Norm / grad norm | 6.3 ms | 2.2% | 662 |

RL-Kernel 改变了 kernel mix，也减少了总 kernel 数量和总 GPU kernel 时间。但优化之后，剩下最大的桶仍然是 TP 通信。

## Device / Rank 不均衡

每张卡的 active time：

| Trace | GPU 0 active | GPU 1 active | 现象 |
|---|---:|---:|---|
| r1 | 174.6 ms | 273.4 ms | GPU 1 在 NCCL 上花更多时间 |
| r2 | 37.6 ms | 245.4 ms | 不均衡更明显，GPU 1 承担了几乎全部 NCCL/optimizer 时间 |

这个不对称是很强的信号：NCCL kernel duration 里包含了等待另一个 rank，或者 collective 内部在等同步。在 r1 中，GPU 1 的 AllReduce 为 215.5 ms，而 GPU 0 是 116.4 ms；在 r2 中，GPU 1 的 AllReduce 是 169.2 ms，而 GPU 0 只有 5.6 ms。

在 TP=2、micro_batch_size=1 的设置下，频繁的小 AllReduce 很容易变成 latency-bound。减少 AllReduce 次数、增大每次通信承载的有效计算量，或者改进通信 overlap/bucketing，收益大概率高于继续调 linear-logp。

## Host 侧与初始化开销

CUDA API summary 里 lazy loading 非常显眼：

| API / NVTX | r1 | r2 | 说明 |
|---|---:|---:|---|
| `cuLibraryLoadData` | 6.75 s | 6.00 s | CUDA library/JIT loading，不是稳定训练开销 |
| `cuBLAS:cublasCreate_v2` | 4.19 s | 3.69 s | cuBLAS handle/init 开销混进了 capture |
| `NCCL:ncclCommInitRankConfig` | r1 不在 top | r2 为 0.577 s | r2 capture 包含 NCCL init |
| `cudaLaunchKernel` | 78.8 ms | 40.4 ms | r2 更低，但不是 wall-clock 主因 |
| `cudaStreamSynchronize` | 10.2 ms | 14.1 ms | 不是主要问题 |

由于 capture 包含初始化，完整 actor range 不是干净的稳定训练测量窗口。这里看到的端到端提升应该被当作保守且有噪声的结果。

## 瓶颈排序

1. TP 通信 / NCCL AllReduce
   - 两份 trace 中最大的 GPU kernel 类别。
   - r1 占 GPU kernel 时间 73.6%。
   - r2 占 GPU kernel 时间 61.8%。
   - 明显的 GPU 间不均衡说明 collective 内部有等待或 rank skew。

2. CUDA kernel 之外的 GPU 饥饿
   - Actor 窗口约 6.4 s，但 GPU active time 只占 2-GPU 容量的 2-3.5%。
   - 可能来源包括 rollout/Ray 调度等待、host 侧编排、CUDA/cuBLAS/NCCL lazy init，以及 collective 周围的同步等待。

3. 大量小 PyTorch elementwise/reduce/copy kernel
   - r1 有 8,278 个，r2 仍有 6,148 个。
   - 目前不是一阶瓶颈，但在通信问题解决后，会继续贡献 launch overhead 和 stream queueing。

4. r2 中的 optimizer / grad norm
   - Adam + grad norm 约 25 ms。
   - 可见，但仍远小于 NCCL AllReduce。

5. RL-Kernel linear-logp
   - 在这两份 trace 里不是瓶颈。
   - r2 的 fused SM90 logp 总共只有 5.4 ms。

## 优化建议

### 性能优化优先级

- 优先减少 TP AllReduce 开销，再继续调 linear-logp。
- 如果 Qwen3-4B full training 在当前 recompute 设置下能放进单张 H100 80GB，可以试 `tensor_model_parallel_size=1`，对比 TP=1/DP=2 或 TP=1 单 actor。对 4B 模型和 micro_batch_size=1 来说，TP=2 很可能被通信 latency 主导。
- 如果显存允许，增大 micro-batch 或 gradient accumulation，让每次 collective 对应更多有效计算。
- 检查梯度 reduction 是否做了 bucket/fusion，以及 backward compute 和 communication overlap 是否真的打开。
- 用 `nvidia-smi topo -m` 检查双 H100 的拓扑，确认走的是 NVLink/NVSwitch 路径，并确认 NCCL P2P 生效。
- 给 forward、logprob、backward、optimizer、通信阶段分别加 NVTX range。下一轮 trace 要能区分“等 rollout”和“训练计算/通信”。

### Profiling 方法建议

- 在 `cudaProfilerStart` 之前先 warm up：
  - 初始化 CUDA context
  - 创建 cuBLAS handle
  - 初始化 NCCL communicator
  - 跑一次 dummy forward/backward/update
  - 如果 rollout 在被测路径里，也跑一次 dummy vLLM generation
- 只抓稳定 train step，例如配置里提到的 step 3-10。
- 分开抓两类 trace：一份看完整 pipeline，另一份只看 actor training 短窗口。当前 trace 混合了训练、初始化和 idle/wait time。
- 系统级瓶颈还没解决前，优先用 Nsight Systems 看 pipeline/通信；Nsight Compute 适合等系统瓶颈收敛后再做单 kernel 调优。

## 最终判断

RL-Kernel 确实减少了 GPU 工作量，但在当前 2 x H100、TP=2 的设置下，wall-clock 瓶颈主要是 NCCL AllReduce 和 kernel 之外的等待/初始化。linear-logp 已经太小，不是限制端到端速度的主因。下一步最值得优化的是 TP 通信量/通信次数和 rank 不均衡，同时要清理 profiling 窗口，把 lazy initialization 从测量区间里排除掉。
