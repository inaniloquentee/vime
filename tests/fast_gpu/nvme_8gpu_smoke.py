"""Eight-GPU NVMe bucket smoke test.

Run with:
  torchrun --standalone --nproc_per_node=8 tests/fast_gpu/nvme_8gpu_smoke.py
"""
import os
import tempfile

import torch
import torch.distributed as dist

from vime_plugins.optimizers.nvme_stream import _Bucket, _Entry, _Stager


def main():
    # Some RL_Kernel NVSwitch nodes expose a broken NVLS fabric; the test
    # validates optimizer/NVMe behavior, so use ordinary NCCL collectives.
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    with tempfile.TemporaryDirectory(prefix=f"vime-nvme-r{rank}-") as root:
        param = torch.nn.Parameter(torch.linspace(-1, 1, 16384, device=device))
        opt = torch.optim.Adam([param], lr=1e-3, betas=(0.9, 0.95))
        entry = _Entry(param, param, 0)
        bucket = _Bucket(
            os.path.join(root, "bucket.bin"),
            [entry],
            opt,
            _Stager(1 << 20),
            {name: torch.float32 for name in ("main", "exp_avg", "exp_avg_sq")},
        )
        grad = torch.linspace(0.1, 1.0, param.numel(), device=device)
        param.grad = grad
        opt.step()
        bucket.flush()
        bucket.fetch()
        param.grad = grad
        bucket.adam.step()
        checksum = param.detach().float().sum()
        dist.all_reduce(checksum, op=dist.ReduceOp.MAX)
        local = param.detach().float().sum()
        if not torch.allclose(local, checksum, atol=0, rtol=0):
            raise RuntimeError(f"rank {rank}: streamed update diverged")
    dist.barrier()
    if rank == 0:
        print("8-GPU NVMe bucket smoke: PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
