import torch


def _quantize(x: torch.Tensor, block: int, max_value: float) -> torch.Tensor:
    flat = x.float().reshape(-1, x.shape[-1])
    if flat.shape[-1] % block:
        return x
    view = flat.view(-1, flat.shape[-1] // block, block)
    scale = view.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / max_value
    q = (view / scale).clamp(-max_value, max_value).round() * scale
    return q.view_as(flat).to(x.dtype).view_as(x)


def fake_quant_fp4_forward(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    y = _quantize(x, 32, 6.0)
    if out is not None:
        out.copy_(y)
        return out
    return y


def fake_quant_fp4(x: torch.Tensor) -> torch.Tensor:
    return x + (fake_quant_fp4_forward(x) - x).detach()


def fake_quant_compressed_kv(x: torch.Tensor) -> torch.Tensor:
    y = _quantize(x, 16, 6.0)
    return x + (y - x).detach()


def fake_quant_fp8(x: torch.Tensor) -> torch.Tensor:
    y = _quantize(x, 32, 448.0)
    return x + (y - x).detach()
