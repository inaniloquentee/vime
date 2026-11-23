import torch


def mhc_aggregate(x, pre, n):
    s, b, nc = x.shape
    return (pre.float().unsqueeze(-1) * x.view(s, b, n, nc // n).float()).sum(dim=2).to(x.dtype)


def mhc_mix(x, original_residual, h_res, h_post, n):
    s, b, d = x.shape
    residual = original_residual.view(s, b, n, d // n).float()
    mixed = torch.einsum('sbij,sbid->sbjd', h_res.float(), residual)
    out = h_post.float().unsqueeze(-1) * x.float().unsqueeze(2) + mixed
    return out.view(s, b, d).to(original_residual.dtype)
