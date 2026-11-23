import torch


def torch_dsa_topk(logits: torch.Tensor, topk: int, row_starts=None, row_ends=None) -> torch.Tensor:
    k = min(topk, logits.shape[-1])
    score, indices = torch.topk(logits, k, dim=-1)
    indices = indices.to(torch.int32).masked_fill(score == -torch.inf, -1)
    if k < topk:
        indices = torch.cat([indices, indices.new_full((*indices.shape[:-1], topk - k), -1)], dim=-1)
    return indices


def get_dsa_topk_fn(topk_backend: str):
    if topk_backend != 'torch':
        raise ValueError(f'Vime V4.1 supports only torch DSA top-k, got {topk_backend}')
    return torch_dsa_topk
