from importlib import import_module

import torch


def patch_rocm_checkpoint_writer():
    """Avoid pinned-memory preloads before async checkpoint workers fork on ROCm."""
    if not torch.version.hip:
        return

    # Megatron moved its optional async writer to nvidia-resiliency-ext.
    for module_name in (
        "megatron.core.dist_checkpointing.strategies.filesystem_async",
        "nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async",
    ):
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                continue
            raise
        writer = module.FileSystemWriterAsync
        if getattr(writer, "_vime_rocm_patched", False):
            return

        def preload_tensors(*args, _preload_tensors=writer.preload_tensors, **kwargs):
            if len(args) > 1 and isinstance(args[-1], bool):
                args = args[:-1] + (False,)
            else:
                kwargs["non_blocking"] = False
            return _preload_tensors(*args, **kwargs)

        module.FileSystemWriterAsync = type(
            "ROCmFileSystemWriterAsync",
            (writer,),
            {"_vime_rocm_patched": True, "preload_tensors": staticmethod(preload_tensors)},
        )
        return
