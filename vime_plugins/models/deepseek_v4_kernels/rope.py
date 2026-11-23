import torch


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    y = x
    z = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    freqs = freqs_cis.conj() if inverse else freqs_cis
    if z.ndim == 3:
        freqs = freqs.view(1, z.size(1), z.size(-1))
    else:
        freqs = freqs.view(1, z.size(1), 1, z.size(-1))
    y.copy_(torch.view_as_real(z * freqs).flatten(-2))
    return y
