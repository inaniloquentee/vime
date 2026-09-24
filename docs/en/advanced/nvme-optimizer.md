# NVMe optimizer-state streaming

NVMe streaming trades optimizer-step time for GPU memory. It keeps FP32 main
parameters and Adam moments in files, loading and updating one bucket at a time.
It is separate from moving the entire training actor out of GPU memory between
training and rollout. This feature does not add whole-actor disk offload or Muon.

## Enable streaming

Use Vime's patched Megatron runtime, including
`OptimizerConfig.defer_main_param_initialization`, with a dedicated writable NVMe
directory for each independent job:

```bash
--optimizer adam \
--use-distributed-optimizer \
--stream-optimizer-state-to-disk \
--offload-train-disk-dir /root/nvme/job-NEW \
--offload-train-disk-chunk-mb 64 \
--stream-optimizer-state-moment-dtype fp32
```

Do not share the scratch directory between concurrent jobs: each rank cleans its
scratch subtree during initialization. Deferred initialization preserves each main
tensor's shape, dtype, device and identity while releasing its CUDA storage. The
largest individual FP32 shard must still fit briefly; initialization then
materializes one main-only bucket at a time. Native-FP32 model parameters remain
GPU-resident.

The supported path is Adam with Megatron DistributedOptimizer, without
precision-aware optimizer mode, CPU optimizer offload, Megatron FSDP, FP8 model
parameters, Muon or stateless Adam. FP32 moment storage has full-model coverage;
BF16 moment storage has component coverage. Other storage dtypes are not covered
by the full-model validation.

## Checkpoint and resume

Use synchronous saving and resume with the same model, optimizer-state dtypes and
parallel topology. The checkpoint includes per-rank bucket files and manifests;
the streamed state is saved before Megatron publishes its checkpoint tracker.
The format is not resharded when TP/PP/DP/CP layout changes.

A checkpoint without streamed optimizer state cannot resume that state silently.
`--no-load-optim` explicitly accepts starting a fresh optimizer; it is not a
checkpoint roundtrip. When extending a run's rollout budget, use Megatron's
`--use-checkpoint-opt-param-scheduler` if the saved scheduler should be retained.

## Variable global batches

Add `--variable-global-batch-size` to keep a trailing partial optimizer step.
For example, 24 samples with `--global-batch-size 16` produce steps of 16 and 8.
An explicit `--global-batch-size-schedule 16,8` enables the variable mode as well.
Without these flags the existing fixed-batch behavior is unchanged.

## Validation

In a dedicated eight-GPU Vime container with prepared model/data caches:

```bash
export NCCL_NVLS_ENABLE=0
export TMPDIR=/root/nvme
python -m pytest -q tests/fast-gpu/test_nvme_optimizer_main_init.py
python -m pytest -q tests/fast-gpu/test_nvme_stream.py
torchrun --standalone --nproc_per_node=8 tests/fast-gpu/nvme_8gpu_smoke.py
python tests/test_qwen3_4B_ckpt.py \
  --save-optimizer nvme --load-optimizer nvme \
  --skip-prepare --checkpoint-dir /root/nvme/checkpoint-NEW
```

`tests/test_qwen3_4B_nvme_benchmark.py` provides manual capture, replay, comparison
and short-soak phases; run it with `--help`. Its synthetic within-group rewards
ensure nonzero gradients, not task-quality evaluation. Run CPU stub tests and
actual Megatron/GPU tests in separate Python processes.

The launchers manage Ray/vLLM processes, so use a dedicated container. GPU
preflight checks are not resource reservations. Timing and memory depend on the
model, topology, filesystem and page cache; these tests do not establish raw NVMe
bandwidth, long-duration endurance or power-loss atomicity.
