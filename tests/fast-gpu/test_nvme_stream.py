"""GPU tests for bucketed NVMe optimizer state streaming.

These tests deliberately exercise the file-backed transfer layer without requiring a
full Megatron model: the production DistributedOptimizer binding is tested through
the same _Bucket and checkpoint primitives it uses.
"""

import importlib.util
import json
import os
import sys
import types

import pytest
import torch

from vime_plugins.optimizers.nvme_stream import NVMeOptimizerStateStore, _Bucket, _Entry, _Stager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_bucket(path, values, *, lr=1e-3, optimizer_type=torch.optim.Adam):
    param = torch.nn.Parameter(values.clone().cuda())
    optimizer = optimizer_type([param], lr=lr, betas=(0.9, 0.95), eps=1e-8)
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
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("optimizer_backend", ["torch", "megatron"])
def test_nvme_checkpoint_round_trip_restores_main_and_moments(tmp_path, optimizer_backend):
    if optimizer_backend == "megatron":
        from megatron.core.optimizer import Adam as optimizer_type
    else:
        optimizer_type = torch.optim.Adam
    initial = torch.linspace(-2.0, 2.0, 2048)
    param, optimizer, bucket = _make_bucket(tmp_path / "live.bin", initial, optimizer_type=optimizer_type)
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
    restored_opt = optimizer_type([restored_param], lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
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
        strict=True,
    ):
        torch.testing.assert_close(lhs, rhs, atol=0.0, rtol=0.0)
    # Torch Adam tracks per-parameter steps; Megatron's fused Adam tracks groups.
    restored_step = restored_opt.state[restored_param].get("step", restored_opt.param_groups[0].get("step"))
    original_step = optimizer.state[param].get("step", optimizer.param_groups[0].get("step"))
    assert float(restored_step) == float(original_step) == 1

    # Verify that a resumed optimizer can perform the next update exactly.
    grad = torch.linspace(0.2, 0.9, restored_param.numel(), device="cuda")
    restored_param.grad = grad
    restored_opt.step()
    param.grad = grad
    optimizer.step()
    for actual, expected in zip(_state_copy(restored_opt, restored_param), _state_copy(optimizer, param), strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    os.close(bucket.fd)
    os.close(restored_bucket.fd)


@pytest.mark.parametrize("group_indices", [(1,), (1, 4)])
def test_legacy_checkpoint_uses_bucket_local_group_steps(tmp_path, group_indices):
    """Miles-style manifests store steps in bucket-local, not global, group order."""
    from megatron.core.optimizer import Adam

    def make_store(name):
        params = [torch.nn.Parameter(torch.linspace(-1.0, 1.0, 1024, device="cuda")) for _ in group_indices]
        adam = Adam([{"params": [param]} for param in params], lr=1e-3)
        bucket = _Bucket(
            str(tmp_path / f"{name}.bin"),
            [_Entry(param, param, group_index) for param, group_index in zip(params, group_indices, strict=True)],
            adam,
            _Stager(1 << 20),
            {segment: torch.float32 for segment in ("main", "exp_avg", "exp_avg_sq")},
        )
        store = object.__new__(NVMeOptimizerStateStore)
        store._rank = store._instance = store.uid = 0
        store.dir = str(tmp_path / name)
        store.dtypes = bucket.dtypes
        store.buckets = [bucket]
        store._fp32_adam = None
        store._allow_fresh_state = False
        return store, bucket, adam, params

    store, bucket, adam, params = make_store("live")
    restored, restored_bucket, restored_adam, restored_params = make_store("restored")
    try:
        for param in params:
            param.grad = torch.full_like(param, 0.25)
        adam.step()
        # Distinct counters expose both out-of-range indexing and wrong-group aliases.
        for index, group in enumerate(adam.param_groups):
            group["step"] = 2 + index * 3
        bucket.flush()
        checkpoint = tmp_path / "checkpoint"
        store.save_to(str(checkpoint))
        manifest_path = checkpoint / store.relative_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for meta in manifest["buckets"]:
            del meta["state_steps"]
        manifest_path.write_text(json.dumps(manifest))

        assert restored.load_from(str(checkpoint))
        assert [group["step"] for group in restored_adam.param_groups] == [
            group["step"] for group in adam.param_groups
        ]
        bucket.fetch()
        restored_bucket.fetch()
        for original, loaded in zip(params, restored_params, strict=True):
            for actual, expected in zip(_state_copy(restored_adam, loaded), _state_copy(adam, original), strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            original.grad = torch.full_like(original, 0.5)
            loaded.grad = original.grad.clone()
        adam.step()
        restored_adam.step()
        for original, loaded in zip(params, restored_params, strict=True):
            for actual, expected in zip(_state_copy(restored_adam, loaded), _state_copy(adam, original), strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        os.close(bucket.fd)
        os.close(restored_bucket.fd)


@pytest.mark.parametrize("missing_payload", [False, True])
def test_native_fp32_checkpoint_requires_resident_optimizer_state(tmp_path, missing_payload):
    def make_store():
        store = object.__new__(NVMeOptimizerStateStore)
        store._rank = store._instance = store.uid = 0
        store.dtypes = {segment: torch.float32 for segment in ("main", "exp_avg", "exp_avg_sq")}
        store.buckets = []
        store._allow_fresh_state = False
        param = torch.nn.Parameter(torch.ones(32, device="cuda"))
        store._fp32_adam = torch.optim.Adam([param], lr=1e-3)
        return store, param

    original, param = make_store()
    param.grad = torch.full_like(param, 0.25)
    original._fp32_adam.step()
    original.save_to(str(tmp_path))
    if missing_payload:
        (tmp_path / original.relative_dir / "fp32_resident_optimizer.pt").unlink()

    restored, restored_param = make_store()
    if missing_payload:
        with pytest.raises(FileNotFoundError, match="fp32_resident_optimizer.pt"):
            restored.load_from(str(tmp_path))
    else:
        assert restored.load_from(str(tmp_path))
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                restored._fp32_adam.state[restored_param][key],
                original._fp32_adam.state[param][key],
                atol=0,
                rtol=0,
            )


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
    torch.testing.assert_close(optimizer.state[param]["exp_avg"], expected_avg, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(optimizer.state[param]["exp_avg_sq"], expected_sq, atol=2e-3, rtol=2e-3)
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
