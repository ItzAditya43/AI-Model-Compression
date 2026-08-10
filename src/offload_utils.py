"""Manual layer-wise GPU offload for structural surgery.

accelerate's automatic device_map="auto" dispatch conflicts with us replacing
submodules mid-pipeline (pruning/decomposing weights in place) -- its hooks
and our module surgery fight over device placement and fragment VRAM.

Instead we hand-roll the same idea at a coarser grain: each top-level piece
(embedding, one decoder layer, final norm) normally rests on CPU. A forward
pre-hook moves it (and its inputs) to GPU just before its forward() runs; a
forward hook moves the module back to CPU right after, freeing VRAM. Only
~1 layer's worth of weights (~150-300MB) is ever resident on GPU at once,
so a 6GB fp16 model can be driven entirely from a 3.68GB card -- slower than
if it fit outright, but without touching system RAM/swap at all.
"""
import torch


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, tuple):
        return tuple(_to_device(v, device) for v in x)
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


def _pre_hook(module, args, kwargs, device):
    module.to(device)
    return _to_device(args, device), _to_device(kwargs, device)


def _post_hook(module, args, kwargs, output, device):
    module.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    return output


def enable_layerwise_gpu_offload(model, device):
    """Register offload hooks on embed_tokens, every decoder layer, and the final
    norm. Model should already be resting on CPU. Returns hook handles (rarely
    need to be removed -- the offloaded modules keep working with them attached
    even after later surgery, since hooks are per-layer, not per-submodule)."""
    if device != "cuda":
        return []

    handles = []
    targets = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    for module in targets:
        h1 = module.register_forward_pre_hook(
            lambda m, a, kw, d=device: _pre_hook(m, a, kw, d), with_kwargs=True
        )
        h2 = module.register_forward_hook(
            lambda m, a, kw, out, d=device: _post_hook(m, a, kw, out, d), with_kwargs=True
        )
        handles.extend([h1, h2])
    model.to("cpu")
    return handles
