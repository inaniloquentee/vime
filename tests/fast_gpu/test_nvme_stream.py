"""GPU tests for bucketed NVMe optimizer state streaming.

These tests deliberately exercise the file-backed transfer layer without requiring a
full Megatron model: the production DistributedOptimizer binding is tested through
the same _Bucket and checkpoint primitives it uses.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from vime_plugins.optimizers.nvme_stream import (
    NVMeOptimizerStateStore,
    _Bucket,
    _Entry,
    _Stager,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_bucket(path, values, *, lr=1e-3):
    param = torch.nn.Parameter(values.clone().cuda())
    optimizer = torch.optim.Adam([param], lr=lr, betas=(0.9, 0.95), eps=1e-8)
    param.grad = torch.linspace(0.1, 1.0, param.numel(), device="cuda").view_as(param)
    optimizer.step()  # materializes Adam moments and the step counter
    entry = _Entry(param, param, 0)
    bucket = _Bucket(
        str(path),
        [entry],
        optimizer,
        _Stager(1 << 20),
        {segment: torch.float32 for segment in ("main", "exp_avg", "exp_avg_sq")},
    )
    return param, optimizer, bucket


def _state_copy(optimizer, param):
    state = optimizer.state[param]
    return param.detach().clone(), state["exp_avg"].detach().clone(), state["exp_avg_sq"].detach().clone()


def test_bucket_fetch_step_matches_in_memory_adam(tmp_path):
    initial = torch.linspace(-1.0, 1.0, 4096)
    streamed, streamed_opt, bucket = _make_bucket(tmp_path / "stream.bin", initial)

    reference = torch.nn.Parameter(initial.cuda())
    reference_opt = torch.optim.Adam([reference], lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    for _ in range(2):
        reference.grad = torch.linspace(0.1, 1.0, reference.numel(), device="cuda").view_as(reference)
        reference_opt.step()

    # Persist the state created by the first step, evict it, then run the second
    # step after reading the state back from NVMe.
    bucket.flush()
    assert bucket.moments_ready
    bucket.fetch()
    streamed.grad = torch.linspace(0.1, 1.0, streamed.numel(), device="cuda").view_as(streamed)
    streamed_opt.step()

    expected = _state_copy(reference_opt, reference)
    actual = _state_copy(streamed_opt, streamed)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)


def test_nvme_checkpoint_round_trip_restores_main_and_moments(tmp_path):
    initial = torch.linspace(-2.0, 2.0, 2048)
    param, optimizer, bucket = _make_bucket(tmp_path / "live.bin", initial)
    bucket.flush()

    store = object.__new__(NVMeOptimizerStateStore)
    store._rank = 0
    store._instance = 0
    store.uid = 7
    store.dir = str(tmp_path / "live")
    store.dtypes = bucket.dtypes
    store.buckets = [bucket]
    store._fp32_adam = None
    store._allow_fresh_state = False
    store.save_to(str(tmp_path / "checkpoint"))

    restored_param = torch.nn.Parameter(initial.cuda())
    restored_opt = torch.optim.Adam([restored_param], lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    restored_entry = _Entry(restored_param, restored_param, 0)
    restored_bucket = _Bucket(
        str(tmp_path / "restored.bin"),
        [restored_entry],
        restored_opt,
        _Stager(1 << 20),
        bucket.dtypes,
    )
    restored_store = object.__new__(NVMeOptimizerStateStore)
    restored_store._rank = 0
    restored_store._instance = 0
    restored_store.uid = 7
    restored_store.dir = str(tmp_path / "restored")
    restored_store.dtypes = bucket.dtypes
    restored_store.buckets = [restored_bucket]
    restored_store._fp32_adam = None
    restored_store._allow_fresh_state = False
    assert restored_store.load_from(str(tmp_path / "checkpoint"))

    restored_bucket.fetch()
    bucket.fetch()
    for lhs, rhs in zip(
        _state_copy(restored_opt, restored_param),
        _state_copy(optimizer, param),
    ):
        torch.testing.assert_close(lhs, rhs, atol=0.0, rtol=0.0)
    assert restored_opt.param_groups[0].get("step", 0) == optimizer.param_groups[0].get("step", 0)
    os.close(bucket.fd)
    os.close(restored_bucket.fd)


def test_bf16_moment_storage_round_trip(tmp_path):
    initial = torch.linspace(-1.0, 1.0, 4096)
    param = torch.nn.Parameter(initial.cuda())
    optimizer = torch.optim.Adam([param], lr=1e-3)
    param.grad = torch.linspace(0.1, 1.0, param.numel(), device="cuda").view_as(param)
    optimizer.step()
    entry = _Entry(param, param, 0)
    bucket = _Bucket(
        str(tmp_path / "bf16.bin"),
        [entry],
        optimizer,
        _Stager(1 << 20),
        {"main": torch.float32, "exp_avg": torch.bfloat16, "exp_avg_sq": torch.bfloat16},
    )
    expected_main = param.detach().clone()
    expected_avg = optimizer.state[param]["exp_avg"].detach().clone()
    expected_sq = optimizer.state[param]["exp_avg_sq"].detach().clone()
    bucket.flush()
    bucket.fetch()
    torch.testing.assert_close(param, expected_main, atol=0, rtol=0)
    torch.testing.assert_close(
        optimizer.state[param]["exp_avg"], expected_avg, atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        optimizer.state[param]["exp_avg_sq"], expected_sq, atol=2e-3, rtol=2e-3
    )
    os.close(bucket.fd)
