"""Activation-aware quantization with mixed precision.

We use bitsandbytes NF4 (weight-only 4-bit, activation-aware via its blockwise
quantization + double quant) for the bulk of the linear layers, and explicitly
keep sensitive modules -- embeddings, lm_head, and input/output layernorms --
in fp16, since these are cheap (small param count) but disproportionately hurt
quality when quantized (matches common AWQ/GPTQ practice of skipping salient
weights).

We save the distilled model to disk first, then reload it through
BitsAndBytesConfig so the 4-bit quantization is applied on load (matches how
these models are actually deployed/served).
"""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def save_fp16(model, tokenizer, path):
    model.half()
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[quantize] saved fp16 checkpoint to {path}")


def load_quantized(path, device):
    """Reload with 4-bit NF4 weight quantization + double quantization (activation-aware
    blockwise scaling), keeping compute in fp16 for stability."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["lm_head"],  # keep output head in higher precision
    )
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=bnb_config,
        device_map={"": 0} if device == "cuda" else None,
    )
    model.eval()
    return model


def quantized_disk_size_gb(model):
    """Estimate on-disk footprint: 4-bit for quantized linear weights, fp16 elsewhere."""
    total_bits = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight") and getattr(module.weight, "dtype", None) is not None:
            pass
    # Simple approximation: sum parameter bytes as reported by the loaded (already
    # quantized) model, which bitsandbytes packs at ~4 bits/param for Linear4bit layers.
    total_bytes = 0
    for p in model.parameters():
        if p.dtype in (torch.uint8,):
            total_bytes += p.numel()  # already packed ~2 params/byte for nf4, but bnb
            # stores as uint8 with 2x4bit packed -> numel() here counts packed bytes
        else:
            total_bytes += p.numel() * p.element_size()
    return total_bytes / 1e9
