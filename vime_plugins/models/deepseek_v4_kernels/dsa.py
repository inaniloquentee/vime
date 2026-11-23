import torch


def indexer_logits_sbhd(q, k, weights, row_starts, row_ends, clean_logits=True, config=None):
    # q [S,B,H,D], k [K,B,D], weights [S,B,H] -> scores [B,S,K]
    qb = q.permute(1, 0, 2, 3).float()
    kb = k.permute(1, 0, 2).float()
    wb = weights.permute(1, 0, 2).float()
    scores = torch.einsum('bshd,bkd,bsh->bsk', qb, kb, wb)
    if clean_logits:
        positions = torch.arange(k.shape[0], device=q.device).view(1, 1, -1)
        lens = row_ends.to(torch.long).view(1, -1, 1)
        scores = scores.masked_fill(positions >= lens, -torch.inf)
    return scores


def sparse_attention(q, kv, indices, sm_scale, attn_sink=None):
    # Small-sequence V4.1 reference path. q [B,S,H,D], kv [B,K,1,D].
    if kv.shape[2] != 1:
        raise ValueError(f'Vime V4.1 sparse attention expects one KV group, got {kv.shape[2]}')
    idx = indices.squeeze(2) if indices.ndim == 4 else indices
    bsz, seqlen, topk = idx.shape
    _, kv_len, _, dim = kv.shape
    safe = idx.clamp_min(0).clamp_max(max(kv_len - 1, 0))
    base = kv[:, :, 0, :]
    gather = torch.gather(base.unsqueeze(1).expand(bsz, seqlen, kv_len, dim), 2, safe.unsqueeze(-1).expand(bsz, seqlen, topk, dim))
    scores = (q.unsqueeze(3).float() * gather.unsqueeze(2).float()).sum(-1) * sm_scale
    scores = scores.masked_fill(~idx.ge(0).unsqueeze(2), -torch.inf)
    if attn_sink is not None:
        sink = attn_sink.float().view(1, 1, -1, 1).expand(bsz, seqlen, -1, 1)
        probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :topk]
    else:
        probs = torch.softmax(scores, dim=-1)
    return (probs.unsqueeze(-1) * gather.unsqueeze(2)).sum(dim=3).to(q.dtype)
