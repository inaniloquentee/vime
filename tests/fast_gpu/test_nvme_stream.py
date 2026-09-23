"""GPU tests for bucketed NVMe optimizer state streaming.

These tests deliberately exercise the file-backed transfer layer without requiring a
full Megatron model: the production DistributedOptimizer binding is tested through
the same _Bucket and checkpoint primitives it uses.
"""

import importlib.util
import os
import sys
import types

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
    restored_step = restored_opt.state[restored_param]["step"]
    original_step = optimizer.state[param]["step"]
    torch.testing.assert_close(restored_step.cpu(), original_step.cpu(), atol=0.0, rtol=0.0)

    # Verify that a resumed optimizer can perform the next update exactly.
    grad = torch.linspace(0.2, 0.9, restored_param.numel(), device="cuda")
    restored_param.grad = grad
    restored_opt.step()
    param.grad = grad
    optimizer.step()
    torch.testing.assert_close(restored_param, param, atol=0.0, rtol=0.0)
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


def _load_checkpoint_wrapper():
    megatron = types.ModuleType("megatron")
    training = types.ModuleType("megatron.training")
    checkpointing = types.ModuleType("megatron.training.checkpointing")
    global_vars = types.ModuleType("megatron.training.global_vars")
    checkpointing.load_checkpoint = lambda *args, **kwargs: None
    checkpointing.save_checkpoint = lambda *args, **kwargs: None
    global_vars.get_args = lambda: None
    training.__path__ = []
    megatron.__path__ = []
    modules = {
        "megatron": megatron,
        "megatron.training": training,
        "megatron.training.checkpointing": checkpointing,
        "megatron.training.global_vars": global_vars,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    spec = importlib.util.spec_from_file_location(
        "vime_checkpoint_test_module",
        os.path.join(os.path.dirname(__file__), "../../vime/backends/megatron_utils/checkpoint.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    for name, old in previous.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old
    return module


def test_checkpoint_wrapper_saves_streamed_state_before_tracker(tmp_path, monkeypatch):
    checkpointing = _load_checkpoint_wrapper()
    events = []

    class Store:
        def save_to(self, base):
            events.append(("nvme", base))

    args = types.SimpleNamespace(save=str(tmp_path), no_save_optim=False)
    optimizer = types.SimpleNamespace(_nvme_state_store=Store())
    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(
        checkpointing,
        "_save_checkpoint_megatron",
        lambda *a, **k: events.append(("megatron",)) or "saved",
    )

    result = checkpointing.save_checkpoint(3, None, optimizer, None)

    assert result == "saved"
    assert [event[0] for event in events] == ["nvme", "megatron"]
    assert events[0][1].endswith("iter_0000003")


def test_checkpoint_wrapper_honors_no_load_optim(tmp_path, monkeypatch):
    checkpointing = _load_checkpoint_wrapper()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("3")
    events = []

    class Store:
        def load_from(self, base):
            events.append(base)
            raise AssertionError("NVMe state must not load with --no-load-optim")

    args = types.SimpleNamespace(load=str(tmp_path), no_load_optim=True)
    optimizer = types.SimpleNamespace(_nvme_state_store=Store())
    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(
        checkpointing,
        "_load_checkpoint_megatron",
        lambda **kwargs: (3, 0),
    )

    result = checkpointing.load_checkpoint(None, optimizer, None, None)

    assert result == (3, 0)
    assert events == []
