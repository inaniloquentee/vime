import logging
import torch.distributed as dist

logger = logging.getLogger(__name__)
FORCED_PARAM_DTYPE_ATTR = "_vime_forced_param_dtype"

def enforce_marked_param_dtypes(model_chunks):
    updated = []
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            target = getattr(param, FORCED_PARAM_DTYPE_ATTR, None)
            if target is None:
                continue
            if param.dtype != target:
                param.data = param.data.to(dtype=target)
            updated.append(name)
    if updated and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
        logger.info("Enforced marked parameter dtypes for %d tensors.", len(updated))
    return updated
