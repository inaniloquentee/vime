# Roadmap: Ascend/CANN Operator Backend Integration for vime

Status: Draft / reviewed v5

This roadmap covers operator-level integration between vime and the Ascend NPU / CANN ecosystem. It builds on the vime/RL-Kernel integration roadmap in RL-Align/vime#6 and on the existing rollout-training consistency design around distributed operator contracts and dlogp diagnostics.

The core position is:

- vime continues to own RL post-training orchestration: Megatron training, vLLM Ascend rollout, Data Buffer, reward flow, weight sync, batching, scheduling, audit replay, and fallback policy.
- The CANN operator backend is an optional, observable, and fallback-aware operator backend. It must not become a second training framework.
- Fast Path and Consistency Path are separate concerns. Fast Path answers whether an operator can run faster. Consistency Path answers whether rollout/training logprobs can be measured, diagnosed, and enforced.
- Every strict CANN claim must be backed by numeric contracts, metadata provenance, batch-invariance checks, and distributed sharding/reduction contracts.

Reference inputs reviewed for this pass:

- AscendNPU-IR: https://gitcode.com/Ascend/AscendNPU-IR
- Triton Ascend Kernels: https://gitcode.com/Ascend/triton-ascend-kernels
- CANNBot Skills: https://gitcode.com/cann/cannbot-skills

## 0. Current State

vime already has an Ascend/NPU path, but it is not the same implementation surface as the CUDA-oriented benchmark branch:

- `origin/ascend` contains NPU docs, `Dockerfile.npu`, a CANN 9.0.0 environment, vLLM Ascend integration, Megatron/MindSpeed/torch-npu patches, Qwen3 NPU scripts, and GRPO/PPO examples.
- The current PR4 RL-Kernel benchmark branch is a CUDA-oriented proof point. Its `linear_logp` entry point explicitly targets Triton/CUDA/registry paths, and telemetry still contains assumptions such as `torch.cuda.Event`, `.is_cuda`, and CUDA memory probes.
- `triton-ascend-kernels` already provides a Python package of high-performance Triton-Ascend kernels with tests and benchmarks, including RL-adjacent pieces such as GRPO loss, fused/linear cross entropy, RMSNorm, and deterministic batch-invariant linear/log-softmax kernels. This should be treated as a short-cycle candidate route and reference source, not as proof that vime has a production CANN backend.
- The Triton-Ascend debug documentation describes a lowering path from Triton Python DSL to TTIR, Triton-Ascend backend IR, BiSheng compilation, AscendNPU IR, and finally NPU binaries. This makes AscendNPU-IR relevant as a compiler/IR-level route for long-term diagnostics and fusion, not a replacement for the vime operator adapter.
- CANNBot Skills separates multiple Ascend operator development paths: Ascend C direct invoke, Ascend C registry/ACLNN integration, Catlass, PyPTO, TileLang, Triton-Ascend, torch.compile graph mode, model inference optimization, and runtime migration. The vime roadmap must therefore record which implementation route each backend instance uses.
- Therefore, "vime can run RL post-training on Ascend" and "vime can already replace RL hot paths with custom CANN operators" are different claims. The former has a base on `origin/ascend`; the latter requires a dedicated operator backend integration layer.

This roadmap is not about directly porting CUDA code to NPU. It is about establishing a clean CANN operator integration line on top of the Ascend branch.

## 0.1 Relationship With The Original Roadmap

This document should not become a parallel roadmap, and it should not reimplement shared capabilities already planned in the RL-Kernel/vime roadmap. The intended relationship is:

```text
Original roadmap = shared vime/RL-Kernel control plane and consistency infrastructure
CANN roadmap     = Ascend/CANN backend appendix under the original roadmap
```

The CANN work must reuse these shared pieces from the original roadmap:

- `--rlk-fast` / `--rlk-consistency` user semantics
- the vime-owned operator adapter boundary
- capability descriptor schema
- execution decision record
- structured fallback
- rollout sample metadata fingerprints
- requested-vs-actual provenance
- dlogp / ratio0 / clipfrac0 / approx_kl0 diagnostics
- ShardingSpec / ReductionSpec / Placement
- A0-A5 audit profiles
- batch-invariance gates
- operator-level, actor-window, and full-step report schemas

The CANN work should only add backend-specific pieces:

- a `backend=cann` capability descriptor instance
- optional route-specific descriptor instances such as `backend=triton_ascend` or `backend=cann` with `implementation_route=aclnn_registry|ascendc_direct|triton_ascend|ascend_ir`
- Docker/wheel/source-build packaging for CANN op packages
- CANN/Ascend/torch-npu/vLLM Ascend/Megatron/MindSpeed build-runtime fingerprints
- an NPU implementation of `DeviceRuntime`
- CANN `linear_logp`, later CANN ops, and their reference comparisons
- HCCL / CANN reduction engine contract instances
- Ascend CI, profiling, and benchmark matrices

If a task already exists as shared work in the original roadmap, the CANN roadmap should not open another PR for the same feature. It should extend the shared interface with a `cann` backend, Ascend metadata fields, and Ascend test coverage.

## 1. User-Facing Entry Points

Reuse the two-switch model from issue #6:

```bash
--rlk-fast {off,auto,strict}
--rlk-consistency {off,audit,strict}
```

Add or keep compatibility with the following advanced controls:

```bash
--rl-kernel-ops linear_logp,logp,rms_norm,attention
--rl-kernel-backend auto|native|cuda|triton|triton_ascend|cann
--rl-kernel-cann-route auto|triton_ascend|ascendc_direct|aclnn_registry|ascend_ir
VIME_RL_KERNEL_BACKEND=auto|native|cuda|triton|triton_ascend|cann
VIME_RLK_CANN_ROUTE=auto|triton_ascend|ascendc_direct|aclnn_registry|ascend_ir
VIME_RLK_CANN_STRICT_BUILD=0|1
```

Semantics:

- `--rlk-fast off --rlk-consistency off`: use the native Ascend vime path and preserve default behavior.
- `--rlk-fast auto --rl-kernel-backend cann`: enable CANN operators when capability and contract checks pass; otherwise use structured fallback.
- `--rlk-fast strict --rl-kernel-backend cann`: require the CANN backend; fail if it is unavailable or does not support the current shape, dtype, or parallel mode.
- `--rl-kernel-backend triton_ascend`: use a Triton-Ascend kernel package or generated Triton-Ascend operator as an Ascend fast path, still behind the same adapter and consistency contract.
- `--rl-kernel-cann-route ...`: select the concrete implementation route under the Ascend/CANN umbrella without changing vime's public orchestration semantics.
- `--rlk-consistency audit`: record dlogp and metadata/provenance diagnostics even when the CANN fast path is disabled.
- `--rlk-consistency strict`: require rollout/training metadata, numeric contracts, batch layout, and parallel placement to be verifiable; otherwise fail early.

During migration, keep existing flags such as `--enable-rl-kernel`, `--rl-kernel-ops`, and `--rl-kernel-strict`, but normalize them into one execution decision object instead of scattering checks across hot paths.

## 2. CANN Integration Boundary

vime should expose a thin operator adapter layer so production code does not directly couple to RL-Kernel experiment controllers, child-process runners, or benchmark artifact layouts.

Suggested boundary:

```text
vime
  +- rollout / Data Buffer / audit replay / training loop
  +- RlkOperatorAdapter
  |    +- capability query
  |    +- execution decision
  |    +- metadata and numeric contract validation
  |    +- structured fallback
  |    +- telemetry
  +- backend calls
       +- native torch/torch_npu reference
       +- RL-Kernel CUDA/Triton backend
       +- RL-Kernel Triton-Ascend backend
       +- RL-Kernel Ascend CANN backend
```

vime should not guess whether CANN supports an operator. It should read this information from a backend descriptor:

- op name, backend id, and implementation fingerprint
- implementation route: `triton_ascend`, `ascendc_direct`, `aclnn_registry`, `catlass`, `pypto`, `tilelang`, or `ascend_ir`
- CANN, driver, torch-npu, vLLM Ascend, Megatron, and MindSpeed versions
- Triton-Ascend, BiSheng, AscendNPU-IR, Catlass/PyPTO/TileLang versions when those routes are used
- supported dtypes such as fp32, bf16, and fp16
- shape ranges, alignment constraints, and workspace requirements
- tiling policy, UB/workspace budget, grid limits, and compile cache policy
- forward-only / autograd / saved-state strategy
- TP/SP/CP support matrix
- HCCL / in-op reduction / boundary reduction semantics
- batch-invariant and deterministic claims
- strict-consistency eligibility
- fallback reason and whether fallback is allowed

## 3. CANN Operator Delivery Model

The CANN backend should not be just a Python import path. It needs a clear build/runtime lifecycle from the first version.

1. Delivery mode
   - CANN op package preinstalled in a Docker image.
   - CANN op package installed from a wheel.
   - Source build retained as a development path.

2. Python binding
   - vime should depend only on a stable Python operator object.
   - Whether the backend uses a torch extension, torch-npu custom operator binding, CANN/ACLNN wrapper, or RL-Kernel's own registry is an internal backend-package detail.
   - If the route is Triton-Ascend, vime should consume it through a package boundary such as `triton_ascend_kernels` or an RL-Kernel-owned wrapper, not by importing loose benchmark scripts.
   - If the route is Ascend C direct invoke, it is allowed for prototype and microbenchmark admission; production use requires a stable PyTorch/ACLNN-facing wrapper and descriptor.
   - vime should not directly depend on CANN build directories, operator source paths, or temporary build artifacts.

3. Version fingerprint
   - CANN toolkit / kernels / NNAL-ATB
   - Ascend driver
   - torch and torch-npu
   - triton-ascend and triton-ascend-kernels
   - BiSheng compiler and AscendNPU-IR lowering path
   - vLLM Ascend
   - Megatron, MindSpeed, and Megatron-Bridge patches
   - CANN op package commit, compiler flags, and op implementation id

Strict runs and benchmark reports must carry these fingerprints. Otherwise, the same `backend=cann` label may describe completely different numeric and performance behavior across machines.

vime also needs to factor CUDA-specific runtime hooks into a backend-neutral surface:

```text
DeviceRuntime:
  current_device()
  synchronize()
  event_timer_or_none()
  memory_allocated_or_none()
  reset_peak_memory_or_none()
  profiler_range()
```

The CUDA backend can keep using `torch.cuda.Event`. The CANN backend should use NPU runtime capabilities. If a capability is unavailable, report it as structured unavailable instead of pretending the value is zero.

## 3.1 Ascend Implementation Routes

`backend=cann` must not hide the implementation route. Each route has a different maturity profile and a different admission gate.

| Route | Primary use | vime admission rule |
|---|---|---|
| `triton_ascend` | Fast POC, reuse of existing Triton-Ascend kernels, and single-device comparison | Allowed as an Ascend fast path only through the adapter. Must pass multi-shape correctness, no-PyTorch-fallback checks, and benchmark-after-verify gates. |
| `ascendc_direct` | Direct `<<<>>>` kernel prototyping and microbenchmarks | Not a production route by itself. It can graduate only after a stable Python/PyTorch or ACLNN-facing wrapper, descriptor, and clean-build package exist. |
| `aclnn_registry` | Standard registered Ascend C/CANN operator integration | Preferred route for long-lived production CANN ops because it gives a framework-facing operator boundary. |
| `catlass` / `blaze` | Matmul-heavy kernels and epilogue fusion | Consider for `linear_logp`, projection, and fused CE once profiling proves the GEMM/epilogue path is the bottleneck. |
| `pypto` / `tilelang` | Alternative DSL routes for rapid operator exploration | Useful for exploration, but strict mode requires the same descriptor, numeric contract, and CI gates as any CANN route. |
| `ascend_ir` | Compiler/IR diagnostics, lowering inspection, graph-level fusion, and long-term optimization | Not a Phase 0-2 dependency. It becomes relevant when Triton-Ascend or CANN compilation issues need IR-level attribution or when stable operator chains are candidates for fusion. |
| `torch_compile_npugraph` | Graph capture/replay and whole-graph experiments | Keep out of the operator adapter unless a graph-mode call can still produce operator-scoped decisions, provenance, and fallback. |

Route selection must be captured in every execution decision:

```text
implementation_route=triton_ascend|ascendc_direct|aclnn_registry|catlass|pypto|tilelang|ascend_ir|native
route_source_repo=...
route_source_commit=...
route_artifact=wheel|docker|source-build|direct-build|jit-cache
compile_stage=python|ttir|ttadapter_ir|bisheng|ascendnpu_ir|npubin|aclnn
```

## 3.2 Route-Specific Admission Gates

Triton-Ascend admission:

- First evaluate whether existing `triton-ascend-kernels` operators can serve as reference, fast POC, or test oracle candidates for `linear_logp`, `logp`, `linear_cross_entropy`, `grpo_loss`, RMSNorm, and deterministic batch-invariant linear/log-softmax.
- Do not benchmark until correctness verification has passed for all declared shapes.
- Detect PyTorch fallback in the fast path; a wrapper that silently computes with `torch.*` is not an accepted operator backend.
- Record `TRITON_DEBUG`, `TRITON_INTERPRET`, `TRITON_ALWAYS_COMPILE`, JIT cache policy, and any dumped intermediate IR when used for debugging.
- Treat `ub overflow`, `grid > 65535`, unsupported bf16 interpreter behavior, and compile-stage failures as structured unsupported/fallback reasons until the tiling or route is fixed.

Ascend C direct/registry admission:

- Kernel sources use `.asc` where required; direct invoke `<<<>>>` is a prototype path, while production integration should expose a stable PyTorch or ACLNN/registry boundary.
- Tiling is computed on the host side and fingerprinted; core count, vector/cube core choice, UB size, and block count must not be hardcoded.
- Non-aligned GM/UB movement and vector APIs must be validated against official API constraints. API choice belongs in the operator design record.
- Host code must not contain tensor-compute loops that should execute on device.
- PyTorch exposure must include a clean `torch.ops` loading path or equivalent framework boundary, plus meta/shape behavior where needed by vime tests.

All routes:

- Build from a fresh checkout or pinned image for release claims.
- Preserve third-party license/provenance records before vendoring or adapting kernels.
- Store generated artifacts and debug dumps as optional evidence, but do not make vime production imports depend on those artifact paths.

## 4. Consistency Contract

The primary diagnostic is:

```text
dlogp = training-side recomputed logp - rollout-side old logp
```

Compute it only over active response/action tokens. Prompt tokens, padding, and masked-out response positions must not enter aggregation.

The comparison is valid only when these inputs match:

- checkpoint or weight version
- token IDs
- active loss/action mask
- tokenizer, chat template, and padding semantics
- sampling metadata such as temperature, top-p, and top-k
- position IDs, packed sequence metadata, and KV/cache metadata
- quantization/dequantization policy
- numeric contract: accumulation dtype, reduction order, downcast point, and special merge semantics
- sharding contract: how TP/SP/CP split inputs and outputs, and which tensors are Partial
- batch-invariance contract: for the same sample under different batch sizes, padding/packing, and microbatches, strict-covered logp/gradient values should stay within tolerance

Diagnostics:

- `abs_dlogp` mean, p50, p90, p99, and max
- `ratio0 = exp(dlogp)`
- `clipfrac0`
- `approx_kl0 = mean(exp(dlogp) - 1 - dlogp)`
- active token count, mask coverage, and zero-active-token cases
- worst-token metadata: sample id, token position, rank, op, backend, and contract id
- per-rank drift
- requested-vs-actual provenance
- fallback count and reason
- CANN build/runtime fingerprint

Strict thresholds should not be hardcoded in vime docs. vime should consume per-dtype tolerance contracts exposed by the RL-Kernel/CANN backend and include the contract id in reports.

## 5. Sharding And Reduction Contracts

Communication is not a separate problem outside the operator. It is part of the distributed operator output contract. CANN integration must encode this from day one.

Suggested abstractions:

```text
Placement = Shard(dim) | Replicate | Partial(reduce_kind)

ShardingSpec:
  inputs:  Placement for each input tensor
  outputs: Placement for each output tensor

ReductionSpec:
  kind: sum | max | logsumexp | custom
  accumulator_dtype: fp32 | bf16 | fp16
  order: canonical_rank | fixed_tree | backend_default
  engine: in_op | hccl | boundary | reference
  downcast_point: after_reduce | final_write | none
```

For the CANN backend, explicitly define:

- whether HCCL only transports/reduces data or the CANN op controls reduction order internally
- whether strict mode can fix rank order or use a fixed tree
- whether SP RMSNorm boundary communication shares the same ReductionSpec as the operator boundary
- CP attention LSE merge semantics, which are not ordinary sum reductions
- whether HCCL backend default order is only opportunistic fast rather than strict

## 6. Operator Priority

Priority follows the original consistency-oriented drift ordering and the engineering value of Ascend/CANN.

| Priority | Operator / Path | Why First | First Strict Scope |
|---|---|---|---|
| P0 | `linear_logp` | PR4 already proved the vime boundary; a CANN version directly serves the RL logprob hot path | BF16/FP32, CP=1, entropy=0, teacher-forcing, explicit TP metadata |
| P0 | native audit dlogp | Measures Ascend native drift before enabling any CANN fast path | single device, active tokens, zero update |
| P0 | Triton-Ascend `linear_cross_entropy` / deterministic linear-logsoftmax probe | Existing Triton-Ascend kernels are close to vime's logprob path and can shorten POC/admission | single device, multi-shape verify, no PyTorch fallback, audit-only strict claim |
| P1 | `logp` / selected softmax | Core metric for rollout-training consistency and a fallback/reference for `linear_logp` | single device, no autograd or reference autograd |
| P1 | RMSNorm | Per-layer effect is small but accumulates; SP boundary contracts are representative | single device + SP boundary |
| P1 | GRPO loss fragments | Triton-Ascend already has GRPO-loss-oriented kernels; these directly touch RL post-training stability | CPU/torch_npu reference first, no production strict until audit is stable |
| P2 | matmul/projection epilogue | split-K, GEMM epilogue, dtype, and downcast points can cause drift | single-device role pair, post-TP projection |
| P2 | attention / CP LSE merge | High impact but broad engineering surface; requires mature audit and metadata first | audit first, CP strict later |
| P3 | RoPE / SwiGLU / embedding | Usually smaller drift but needed for full forward-chain comparison | single-device role pair |
| P3 | PyPTO/TileLang/Catlass exploratory variants | Useful for route comparison after the core adapter and audit gates exist | exploratory only until descriptor and CI gates match CANN rules |

Ordering principle: validate single-device operators first, then distributed operators, then communication/topology sweeps. Do not attribute multi-device drift to HCCL before the single-device CANN operator has passed validation.

## 7. Phase 0: Ascend Baseline And Branch Strategy

Deliverables:

- Use `origin/ascend` as the CANN work baseline, and explicitly avoid mixing large NPU changes into the CUDA benchmark PR.
- Pin the minimum validation environment: Ascend Atlas A2/A3, aarch64, Python 3.12, CANN 9.0.0, torch-npu, vLLM Ascend, and Megatron/MindSpeed patch versions.
- Record reference-source pins for AscendNPU-IR, triton-ascend, triton-ascend-kernels, and any CANNBot-derived operator templates used for design.
- Add a CANN operator development appendix to `docs/en/get_started/NPU.md` and the corresponding Chinese docs.
- Define at least one reproducible installation path for the CANN operator package: source build, wheel, or Docker image.
- Add a thin `DeviceRuntime` abstraction and use it to replace CUDA-only event, memory, and profiler calls in RL-Kernel telemetry paths.
- Add an "implementation route matrix" to docs so users know whether a run used `triton_ascend`, `ascendc_direct`, `aclnn_registry`, or native fallback.

Acceptance:

- A native Ascend vime GRPO example can run.
- Training, rollout, weight sync, HCCL environment variables, and Ray NPU visible-device behavior are reproducible.
- Docs clearly state the supported branch and that the current CUDA PR4 branch does not directly enable CANN.

## 8. Phase 1: Reuse The Shared Adapter, Add CANN Capability And Decision Instances

Deliverables:

- Depend on the shared `RlkOperatorAdapter` or equivalent thin layer from the original roadmap. Do not reimplement it in the CANN line.
- Add a CANN backend descriptor under the shared capability schema.
- Add route-specific descriptor fields: implementation route, route source repo/commit, compile stage, route artifact, tiling policy, UB/workspace budget, and debug dump availability.
- Fill CANN fields in the shared structured execution decision record, for example:

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

- Reuse shared `enable_rl_kernel` / `rl_kernel_strict` / fallback counter logic.
- When the CANN backend is absent, use warning + fallback in `auto` mode and fail in `strict` mode.
- Ensure the CANN backend descriptor carries complete build/runtime fingerprints.

Acceptance:

- Native behavior is unchanged when the feature is off.
- Every operator call has an explainable decision.
- Silent fallback is not allowed.
- Before any CANN op exists, shared audit mode can still run independently; CANN only adds Ascend provenance and runtime fingerprints.

## 9. Phase 2: CANN `linear_logp` POC

Goal: reproduce the vime integration shape of CUDA `linear_logp`, while using Ascend-native backend implementation and telemetry.

The first implementation may use `triton_ascend` if it can satisfy the vime operator contract faster than a native Ascend C implementation. The roadmap should first evaluate existing Triton-Ascend linear/cross-entropy/log-softmax candidates before writing a new CANN operator from scratch. If Triton-Ascend cannot satisfy the support matrix or strict contract, fall back to `ascendc_direct` for prototype validation and `aclnn_registry` for production packaging.

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

- CANN `linear_logp` forward.
- Triton-Ascend route probe for existing or adapted `linear_cross_entropy`, deterministic linear, and log-softmax kernels.
- Autograd backward or reference backward fallback.
- torch/torch_npu reference path.
- CANN backend descriptor.
- dtype/shape/alignment support matrix.
- fallback reasons: unsupported dtype, unsupported CP, entropy requested, missing TP metadata, no autograd, build missing.
- compile-stage fallback reasons: unsupported Triton DSL, Triton-Ascend lowering failure, BiSheng failure, AscendNPU IR failure, `ub overflow`, `grid > 65535`, or JIT cache mismatch.
- no-PyTorch-fallback check for the claimed fast path.
- NPU event/timer through `DeviceRuntime`; if there is no stable API, report host timing and mark it clearly.
- memory probe through `DeviceRuntime` and torch_npu/NPU memory APIs; if unavailable, report structured unavailable.

Acceptance:

- Single-device CANN vs torch_npu reference passes per-dtype tolerance over active tokens.
- zero-update GRPO/SFT teacher-forcing audit passes.
- Multi-shape verification passes before any performance benchmark is reported.
- `--rlk-fast strict --rl-kernel-backend cann --rl-kernel-ops linear_logp` has fallback=0 for supported configurations.
- Unsupported configurations produce stable fallback/fail reasons.

## 10. Phase 3: Extend The Shared Audit Harness For Ascend

Deliverables:

- Reuse the original roadmap's native audit-only dlogp and teacher-forcing replay path. Do not reimplement the metrics core.
- Add Ascend fields to the shared rollout sample metadata fingerprint: checkpoint, tokenizer, sampling config, mask/padding, position/cache, quantization, parallel placement, and backend contract.
- Add Ascend fields to the shared requested-vs-actual provenance: vLLM Ascend engine, router, server group, weight version, CANN build id, and torch-npu runtime.
- Reuse the shared batch-layout fingerprint: sequence lengths, packed order, padding side, active mask density, microbatch id, global batch shape, and dynamic sampling group.
- Reuse the shared result cube and add Ascend/CANN axes: batch size, layout, dtype, TP/SP/CP, backend, cache policy, router policy, and determinism policy.

Acceptance:

- Known metadata mismatches are caught before being misdiagnosed as operator drift.
- Audit mode does not change training behavior.
- Reports include worst token, per-rank drift, fallback reason, and active-token coverage.
- The same sample passes tolerance both alone and inside mixed batches within the strict coverage matrix.

## 11. Phase 4: Single-Device CANN Operator Comparison

Deliverables:

- Expand operator specs from `linear_logp/logp` to attention, lm_head/logp, projection matmul, RMSNorm, RoPE, SwiGLU, embedding, and RL loss fragments.
- Each op has at least two roles: train reference and infer/serving-style reference. The CANN fast backend is the third implementation.
- Include Triton-Ascend deterministic operators in the admission table where applicable: `linear_batch_invariant`, `log_softmax_batch_invariant`, `mean_batch_invariant`, and `rms_norm_batch_invariant`.
- Move from pass/fail to drift attribution: how much each op contributes to `dlogp` or intermediate error.
- Add forward-chain comparison by running the forward path in realistic order and observing accumulated error.

Acceptance:

- Each P0/P1 op has single-device train-vs-infer comparison.
- Before claiming strict CANN behavior, the backend passes same-sample batch-invariance tests on one device.
- Forward-chain comparison is used for CI/admission/audit, not before every production call.

## 12. Phase 5: Distributed CANN/HCCL Contracts

Deliverables:

- TP CANN tests for `linear_logp`: vocab shard, target id routing, HCCL group, and global vocab size.
- TP matmul/projection tests.
- SP RMSNorm boundary reduction tests.
- CP attention LSE merge audit tests.
- A0-A5 profiles:
  - A0 fully aligned reference
  - A1 arithmetic-only
  - A2 reduction/topology-only
  - A3 representation-only
  - A4 pairwise mismatch
  - A5 production mismatch
- HCCL algorithm/topology/rank-placement sweeps.

Acceptance:

- Operators that fail single-device validation do not enter distributed attribution.
- Distributed drift can be attributed to sharding, reduction order, dtype/downcast, HCCL engine, metadata mismatch, or batch layout.
- Strict distributed runs have a reference path.
- If overlap, bucket, or fusion changes reduction grouping, it must be an explicit axis and not a silent fast path.

## 13. Phase 6: Profiling And Performance Expansion

Deliverables:

- Ascend profiling scripts covering warmup, stable window, actor train, rollout generate, weight sync, and communication.
- CANN/NPU profiling markers for forward, selected logprob, backward, optimizer, HCCL collectives, weight sync, and host wait.
- Triton-Ascend compile/runtime profiling should separate Python/JIT compile time, BiSheng compile time, first-run cache fill, steady-state kernel time, and host wait.
- Separate operator-level, actor-window, and full-step reports.
- Add a new CANN fast op only when profiling proves it is a bottleneck and the contract can describe its behavior.

Acceptance:

- Do not use single-operator speedup to claim full-step speedup.
- Reports distinguish kernel compute, HCCL communication, rollout wait, weight sync, and host orchestration.
- Keep `linear_logp` as a proof point, but do not overstate it as an end-to-end bottleneck without full-step data.

## 14. Phase 7: CI, Benchmark, And Release Rules

CI layers:

- CPU/PyTorch reference for metric math and metadata validation.
- torch_npu native for Ascend native audit.
- CANN single-op for P0/P1 operator tolerance.
- Triton-Ascend route CI: correctness before benchmark, multi-shape coverage, no PyTorch fallback, structured compile-stage failure capture.
- Ascend C route CI: clean source build, `.asc`/CMake checks, host-side tiling validation, dynamic core/UB assumptions, PyTorch or ACLNN wrapper smoke.
- distributed NPU for TP/SP/CP/HCCL contracts.
- batch-invariance across batch size, padding/packing, active density, and microbatch.
- production smoke, for example a short Qwen3-4B GRPO run.
- scheduled benchmark, for example Qwen3-30B-A3B or a representative MoE model.

Release rules:

- Operator speedup can only be called operator speedup.
- Full-step speedup requires a full-step benchmark.
- Strict consistency must declare the covered dtype, shape, parallel mode, and batch-layout matrix.
- CANN backend version and CANN/torch-npu/vLLM Ascend/Megatron/MindSpeed fingerprints must be included in reports.
- fallback=0 is valid only for the covered matrix and must not be generalized automatically.

## 15. Suggested PR Order

1. Roadmap and boundary
   - Land this roadmap and state that the CANN backend does not enter the CUDA PR4 line.

2. Shared adapter dependency confirmation
   - Confirm that the original roadmap's backend selector, capability descriptor, execution decision, and structured fallback have landed or have keeper issues.

3. DeviceRuntime and CANN packaging
   - Extract CUDA/NPU event, memory, and profiler runtime hooks; define CANN op wheel/Docker/source-build fingerprints.

4. Implementation route matrix
   - Add descriptor fields and docs for `triton_ascend`, `ascendc_direct`, `aclnn_registry`, `catlass`, `pypto`, `tilelang`, and `ascend_ir`.

5. Ascend audit extension
   - Add Ascend fields and tests on top of shared dlogp, metadata fingerprint, and requested-vs-actual provenance.

6. Triton-Ascend `linear_logp` route probe
   - Evaluate existing or adapted Triton-Ascend linear/cross-entropy/log-softmax kernels under vime's contract.

7. CANN `linear_logp` POC
   - Implement forward, reference comparison, support matrix, fallback reasons, and compile-stage diagnostics.

8. CANN `linear_logp` autograd and strict
   - Add backward/saved-state, batch-invariance, TP metadata, and strict mode.

9. Single-device operator table
   - Add specs for `logp`, RMSNorm, matmul/projection, RoPE/SwiGLU, embedding, GRPO loss, and deterministic Triton-Ascend candidates.

10. Distributed contracts
   - Add TP `linear_logp`, SP RMSNorm, CP attention audit, and A0-A5 profiles.

11. Profiling and benchmarks
   - Add Ascend stable-window profiling and operator/actor/full-step reports.

12. Docs and release matrix
    - Add NPU docs, Docker/wheel install instructions, CI/benchmark coverage, and known unsupported cases.

## 16. Non-Goals

- Do not replace Megatron, vLLM Ascend, MindSpeed, or torch-npu.
- Do not make RL-Kernel/CANN a required vime dependency.
- Do not mix large NPU conditionals into the CUDA benchmark branch.
- Do not reimplement flags, adapter, decision, metadata, audit, result cube, or report schema from the original roadmap.
- Do not treat successful import of the CANN backend as successful capability validation.
- Do not allow silent fallback.
- Do not enforce bitwise parity in audit mode.
- Do not use isolated operator benchmarks as evidence for full-step gains.
- Do not claim strict consistency when metadata is incomplete.
- Do not treat a Triton-Ascend prototype, direct-invoke executable, or graph-mode experiment as a production CANN backend until it has the adapter descriptor, wrapper boundary, clean packaging, and CI gates.
- Do not vendor or adapt third-party kernels without explicit license/provenance review.

## 17. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| CANN / torch-npu / vLLM Ascend version drift | Put complete fingerprints in backend descriptors; pin CI images; publish a release matrix |
| CANN op build is hard to reproduce | Keep at least one stable path among Docker image, wheel, and source build |
| Implementation routes blur together | Record `implementation_route`, route source commit, compile stage, artifact type, and admission gate in every descriptor |
| Triton-Ascend POC silently falls back to PyTorch | Add no-PyTorch-fallback checks before declaring a fast path |
| Triton-Ascend kernels hit UB/grid/compiler limits | Capture `ub overflow`, `grid > 65535`, Triton lowering, BiSheng, and AscendNPU IR failures as structured fallback reasons |
| Direct-invoke prototype is mistaken for production integration | Require PyTorch or ACLNN/registry wrapper, clean package, and descriptor before production use |
| IR-level optimization bypasses operator telemetry | Keep AscendNPU-IR and graph-mode work behind operator-scoped decisions or mark it as non-admitted research |
| Fast op is numerically inconsistent | Strict mode accepts only numeric contracts; audit measures first and does not directly fail production |
| HCCL reduction order is not controllable | Use reference/in-op deterministic paths for strict mode, or declare HCCL order as an audit axis |
| Batch layout affects tiling and reduction | Use same-sample batch-invariance as a strict gate |
| Metadata becomes too large | Store fingerprints by default; dump verbose replay artifacts only in debug mode |
| Fallback hides missing capability | Use decision records, fallback counters, and strict failures |
| Multi-device tests start before single-device validation | Require single-device admission before distributed admission |
| Profiling includes compile/init noise | Use warmup, stable windows, and layered reports |
| Custom rollout lacks metadata | Warn in audit mode; fail that sample or disable strict claim in strict mode |

## 18. Review Passes

### Review 1: Alignment With Issue #6

- Preserves the boundary between vime orchestration and RL-Kernel operator backends.
- Preserves the orthogonal Fast Path and Consistency Path model.
- Preserves capability reporting, numeric contracts, metadata fingerprints, and fallback telemetry.
- Avoids pulling RL-Kernel experiment controllers into the vime production path.

### Review 2: Alignment With Rollout-Training Consistency Design

- Uses `dlogp = training logp - rollout old logp` as the primary metric.
- Defines active-token scope.
- Keeps the order of single-device operator validation, distributed operator validation, then communication attribution.
- Treats communication as part of the distributed operator contract, not an unrelated side channel.
- Introduces ShardingSpec, ReductionSpec, Placement, batch-invariance, and A0-A5 profiles.

### Review 3: Ascend/CANN Executability

- Acknowledges that the runnable RL post-training base is currently `origin/ascend`, not the CUDA benchmark branch.
- Chooses `linear_logp` as the first CANN op because the vime integration boundary was already proven by the CUDA proof point.
- Does not assume CUDA event or memory probes can be reused; NPU telemetry gets its own runtime layer.
- Requires delivery form, Python binding boundary, and build/runtime fingerprints for CANN ops.
- Requires explicit contracts for HCCL/CANN strict claims instead of backend-default assumptions.
- Separates operator-level, actor-window, and full-step claims.

### Review 4: Deduplication Against The Original Roadmap

- The CANN roadmap is a backend appendix under the original roadmap, not a parallel implementation.
- Shared capabilities are implemented once: flags, adapter, capability schema, execution decision, fallback, metadata, provenance, audit metrics, and result cube.
- CANN PRs only add backend descriptor, DeviceRuntime NPU implementation, CANN ops, HCCL contracts, and Ascend CI/benchmark coverage.
- Any wording like "add a new audit core" or "add a new adapter skeleton" should be changed to "reuse the shared layer and add Ascend extensions."

### Review 5: Alignment With Ascend Implementation Ecosystem

- `backend=cann` is no longer a vague bucket; every run records `implementation_route`.
- Triton-Ascend is explicitly admitted as a short-cycle POC/reference route, especially for RL-adjacent linear CE, GRPO loss, RMSNorm, and deterministic batch-invariant kernels.
- Ascend C direct invoke is kept as prototype/microbenchmark infrastructure, while ACLNN/registry remains the preferred production wrapper.
- AscendNPU-IR is positioned as compiler/IR diagnostics and long-term fusion infrastructure, not a replacement for the vime adapter.

### Review 6: Verification And Release Gates

- Correctness gates precede performance gates for Triton-Ascend and native CANN routes.
- Multi-shape verification, no-PyTorch-fallback checks, clean-build reproducibility, compile-stage diagnostics, and license/provenance checks are now explicit.
- UB overflow, grid-limit, JIT cache, BiSheng, and AscendNPU IR failures are structured unsupported/fallback reasons rather than ad hoc errors.
- Direct-invoke and graph-mode experiments cannot be marketed as production backend support without wrapper, descriptor, and CI admission.

## 19. Expected End State

Users can run this in an Ascend environment:

```bash
python train.py \
  --enable-rl-kernel \
  --rl-kernel-backend triton_ascend \
  --rl-kernel-ops linear_logp \
  --rlk-fast auto \
  --rlk-consistency audit
```

and clearly see:

- whether the CANN backend was selected
- which op used CANN and which op fell back
- fallback reasons
- implementation route, source commit, compile stage, and artifact type
- CANN / torch-npu / CANN op build fingerprints
- active-token dlogp metrics
- per-rank drift
- worst token
- batch-invariance coverage matrix
- operator-level, actor-window, and full-step performance reports
- whether the current strict-consistency claim is audit-only, tolerance strict, or bitwise deterministic

When users switch to:

```bash
python train.py \
  --enable-rl-kernel \
  --rl-kernel-backend cann \
  --rl-kernel-cann-route aclnn_registry \
  --rl-kernel-ops linear_logp,logp,rms_norm \
  --rlk-fast strict \
  --rlk-consistency strict
```

vime succeeds only when all enabled CANN ops satisfy capability checks, numeric contracts, metadata completeness, batch-invariance gates, and distributed contracts. Otherwise it chooses a contract-preserving reference path or fails according to strict policy.
