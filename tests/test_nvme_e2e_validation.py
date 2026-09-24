"""CPU regression coverage for the NVMe E2E checkpoint assertions and commands."""

import asyncio
import json
from types import SimpleNamespace

import pytest

import test_qwen3_4B_ckpt as e2e


@pytest.fixture
def checkpoint(tmp_path):
    for rank in range(e2e.NUM_GPUS):
        root = tmp_path / "iter_0000001" / f"rank{rank:05d}" / "opt0_0"
        root.mkdir(parents=True)
        # Deliberately unaligned parameters to exercise per-entry padding.
        (root / "bucket00000.bin").write_bytes(bytes(3 * 4096 * 2))
        manifest = {
            "dtypes": {name: "torch.float32" for name in ("main", "exp_avg", "exp_avg_sq")},
            "buckets": [
                dict(numel=5, entry_numels=[2, 3], state_steps=[4, 4], steps=[0], file="bucket00000.bin")
            ],
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_complete_checkpoint(checkpoint):
    summary = e2e.assert_nvme_checkpoint(str(checkpoint), 1, 4)
    assert summary["ranks"] == 8
    assert summary["bucket_files"] == 8
    assert summary["bytes"] == 8 * 3 * 4096 * 2


def test_fused_adam_group_counters(checkpoint):
    for path in checkpoint.rglob("manifest.json"):
        manifest = json.loads(path.read_text())
        manifest["buckets"][0]["state_steps"] = [0, 0]
        manifest["buckets"][0]["steps"] = [4]
        path.write_text(json.dumps(manifest))
    e2e.assert_nvme_checkpoint(str(checkpoint), 1, 4)


@pytest.mark.parametrize("corruption", ["missing_rank", "missing_bucket", "truncated", "steps", "dtype", "layout"])
def test_reject_incomplete_checkpoint(checkpoint, corruption):
    root = checkpoint / "iter_0000001" / "rank00007" / "opt0_0"
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if corruption == "missing_rank":
        path.unlink()
    elif corruption == "missing_bucket":
        (root / "bucket00000.bin").unlink()
    elif corruption == "truncated":
        (root / "bucket00000.bin").write_bytes(b"short")
    else:
        if corruption == "steps":
            manifest["buckets"][0]["state_steps"] = [0, 0]
        elif corruption == "dtype":
            manifest["dtypes"]["exp_avg"] = "torch.bfloat16"
        else:
            manifest["buckets"][0]["numel"] += 1
        path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        e2e.assert_nvme_checkpoint(str(checkpoint), 1, 4)


def test_reject_wrong_iteration(checkpoint):
    with pytest.raises(AssertionError):
        e2e.assert_nvme_checkpoint(str(checkpoint), 2, 6)


@pytest.mark.parametrize("placement", ["cpu", "gpu", "nvme"])
def test_roundtrip_commands(monkeypatch, placement):
    calls = []
    monkeypatch.setattr(e2e.U, "execute_train", lambda **kwargs: calls.append(kwargs["train_args"]))
    monkeypatch.setattr(e2e.U, "get_default_wandb_args", lambda *args: "")
    e2e.execute("save", optimizer=placement, checkpoint_dir="/tmp/test checkpoint")
    e2e.execute("load", optimizer=placement, checkpoint_dir="/tmp/test checkpoint")
    save, load = calls
    assert "--load '/tmp/test checkpoint' --ckpt-step 1" in load
    if placement == "nvme":
        assert "--rollout-batch-size 12 --n-samples-per-prompt 2" in save
        assert "--variable-global-batch-size" in save
        assert "--save '/tmp/test checkpoint_resumed' --save-interval 1" in load
        assert "--stream-optimizer-state-to-disk" in save
        assert "--custom-rm-path test_qwen3_4B_ckpt.nvme_test_reward" in save
        assert "--ci-save-grad-norm" in save
    else:
        assert "--variable-global-batch-size" not in save
        assert "--save " not in load
        assert "--num-rollout 2" in save


def test_synthetic_rewards_vary_within_every_prompt():
    args = SimpleNamespace(n_samples_per_prompt=2)
    samples = [SimpleNamespace(index=i) for i in range(24)]
    assert asyncio.run(e2e.nvme_test_reward(args, samples)) == [0.0, 1.0] * 12
    assert asyncio.run(e2e.nvme_test_reward(args, samples[1])) == 1.0


@pytest.mark.parametrize("norms", [[0.1, 0.2], [0.0, 0.2], [float("nan"), 0.2], [float("inf"), 0.2], [0.1]])
def test_gradient_evidence(tmp_path, norms):
    import torch

    checkpoint_dir = str(tmp_path / "checkpoint")
    grad_dir = tmp_path / "checkpoint_grad_norms"
    grad_dir.mkdir()
    for i, norm in enumerate(norms):
        torch.save(norm, grad_dir / f"save_0_{i}.pt")
    if norms == [0.1, 0.2]:
        e2e.assert_nvme_grad_norms(checkpoint_dir, "save", 2)
    else:
        with pytest.raises(AssertionError):
            e2e.assert_nvme_grad_norms(checkpoint_dir, "save", 2)
