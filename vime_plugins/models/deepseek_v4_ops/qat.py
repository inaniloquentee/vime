import torch


def fp8_simulate(x: torch.Tensor, block_size: int):
    flat = x.float().reshape(-1, x.shape[-1])
    if flat.shape[-1] % block_size:
        return x
    view = flat.view(-1, flat.shape[-1] // block_size, block_size)
    scale = view.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 448.0
    q = (view / scale).clamp(-448.0, 448.0).round() * scale
    return q.view_as(flat).to(x.dtype).view_as(x)


class _QAT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, block_size=128):
        return fp8_simulate(x, block_size)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


fp8_simulate_qat = _QAT.apply
