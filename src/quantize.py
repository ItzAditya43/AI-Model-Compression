"""Activation-aware quantization with mixed precision.

We use bitsandbytes NF4 (weight-only 4-bit, activation-aware via its blockwise
quantization + double quant) for the bulk of the linear layers, and explicitly
keep sensitive modules -- lm_head -- in fp16, since it's cheap (small param
count relative to the whole model) but disproportionately hurts quality when
quantized (matches common AWQ/GPTQ practice of skipping salient weights).

Quantization is applied in-place to the live model object rather than via a
save-to-disk-then-reload-through-BitsAndBytesConfig round trip: this pipeline's
model contains custom LowRankLinear modules from the decomposition stage, which
transformers' AutoModelForCausalLM.from_pretrained cannot reconstruct from a
checkpoint (it rebuilds the *original* architecture from config and silently
drops/reinitializes anything that doesn't match -- exactly the bug that broke
the disk round-trip approach here). Walking the live module tree and swapping
each nn.Linear for a bnb.nn.Linear4bit sidesteps that entirely.
"""
import torch
import torch.nn as nn
import bitsandbytes as bnb


def save_fp16(model, tokenizer, path):
    model.half()
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[quantize] saved fp16 checkpoint to {path}")


def quantize_in_place(model, device, skip_substrings=("lm_head",)):
    """Replace every nn.Linear in the model (including ones nested inside
    LowRankLinear) with a 4-bit NF4 bitsandbytes Linear, except modules whose
    name matches skip_substrings (kept fp16 -- the mixed-precision part)."""
    model.to("cpu")

    def _replace(module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if any(s in full_name for s in skip_substrings):
                continue
            if isinstance(child, nn.Linear):
                new_layer = bnb.nn.Linear4bit(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=torch.float16, quant_type="nf4",
                )
                new_layer.weight = bnb.nn.Params4bit(
                    child.weight.data.clone(), requires_grad=False, quant_type="nf4",
                )
                if child.bias is not None:
                    new_layer.bias = nn.Parameter(child.bias.data.clone())
                setattr(module, name, new_layer)
            else:
                _replace(child, full_name)

    _replace(model)
    if device == "cuda":
        model.to(device)  # triggers the actual NF4 quantization on each Params4bit
    model.eval()
    return model


def quantized_disk_size_gb(model):
    """Estimate on-disk footprint: 4-bit for quantized linear weights, fp16 elsewhere."""
    total_bytes = 0
    for p in model.parameters():
        if p.dtype == torch.uint8:
            total_bytes += p.numel()  # nf4 packs 2 params/byte; numel() already counts packed bytes
        else:
            total_bytes += p.numel() * p.element_size()
    return total_bytes / 1e9
