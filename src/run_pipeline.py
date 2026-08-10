"""End-to-end pipeline: prune -> decompose -> distill -> quantize, with a full
eval snapshot saved after every stage so we can see where quality is won/lost.

Structural surgery (pruning, SVD decomposition) needs real fp16 weight access,
but the fp16 3B model (6.2GB) doesn't fit in this GPU's 3.68GB VRAM, and the
machine's system RAM is too full to hold it either (loading it on CPU thrashed
swap badly). Instead we use manual layer-wise GPU offload (see offload_utils.py):
the model rests on CPU, and each layer is moved to GPU just for its own forward
pass, then back -- all compute happens on GPU, never more than ~1 layer's
weights resident in VRAM at once, and system RAM is barely touched.
"""
import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from configs.config import (
    BASE_MODEL, DEVICE, PRUNE_LAYER_KEEP_RATIO, PRUNE_FFN_KEEP_RATIO,
    SVD_RANK_RATIO, DISTILL_STEPS, DISTILL_LR, DISTILL_TEMPERATURE,
)
from eval.harness import run_full_eval, save_result
from src.prune import run_pruning, get_calibration_batches
from src.decompose import run_decomposition
from src.distill import run_distillation, load_frozen_teacher
from src.quantize import save_fp16, load_quantized
from src.offload_utils import enable_layerwise_gpu_offload


def free_gpu():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def eval_checkpoint_int4(path, tokenizer, tag, result_path):
    """Reload a saved fp16 checkpoint in 4-bit on GPU just to run the (fast) eval harness."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=bnb_config,
        device_map={"": 0} if DEVICE == "cuda" else None,
    )
    result = run_full_eval(model, tokenizer, DEVICE, tag=tag)
    save_result(result, result_path)
    del model
    free_gpu()


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Stage 1: structured pruning (depth + width), layer-wise GPU offload ----
    print("=== Loading model (rests on CPU, layers offloaded to GPU one at a time) ===")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map="cpu")
    enable_layerwise_gpu_offload(model, DEVICE)

    print("\n=== Stage: Structured Pruning ===")
    model = run_pruning(model, tokenizer, "cpu", PRUNE_LAYER_KEEP_RATIO, PRUNE_FFN_KEEP_RATIO)
    model.save_pretrained("checkpoints/pruned")
    tokenizer.save_pretrained("checkpoints/pruned")
    del model
    gc.collect()
    free_gpu()

    eval_checkpoint_int4("checkpoints/pruned", tokenizer, "pruned", "results/pruned/metrics.json")

    # ---- Stage 2: low-rank decomposition, layer-wise GPU offload ----
    print("\n=== Loading pruned checkpoint for SVD decomposition ===")
    model = AutoModelForCausalLM.from_pretrained("checkpoints/pruned", dtype=torch.float16, device_map="cpu")
    enable_layerwise_gpu_offload(model, DEVICE)
    calib_batches = get_calibration_batches(tokenizer, "cpu", n_batches=16)

    print("\n=== Stage: Low-Rank Decomposition ===")
    model = run_decomposition(model, calib_batches, SVD_RANK_RATIO)
    model.save_pretrained("checkpoints/decomposed")
    tokenizer.save_pretrained("checkpoints/decomposed")
    del model, calib_batches
    gc.collect()
    free_gpu()

    eval_checkpoint_int4("checkpoints/decomposed", tokenizer, "pruned+decomposed", "results/decomposed/metrics.json")

    # ---- Stage 3: knowledge distillation recovery, on GPU (QLoRA: 4-bit + LoRA adapters) ----
    print("\n=== Stage: Knowledge Distillation (GPU, QLoRA) ===")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    student = AutoModelForCausalLM.from_pretrained(
        "checkpoints/decomposed", quantization_config=bnb_config,
        device_map={"": 0} if DEVICE == "cuda" else None,
    )
    teacher = load_frozen_teacher(BASE_MODEL, DEVICE)
    model = run_distillation(
        student, teacher, tokenizer, DEVICE,
        n_steps=DISTILL_STEPS, lr=DISTILL_LR, temperature=DISTILL_TEMPERATURE,
    )
    del teacher
    free_gpu()
    result = run_full_eval(model, tokenizer, DEVICE, tag="distilled")
    save_result(result, "results/distilled/metrics.json")

    save_fp16(model, tokenizer, "checkpoints/distilled_fp16")
    del model, student
    free_gpu()

    # ---- Stage 4: activation-aware + mixed precision quantization ----
    print("\n=== Stage: Quantization ===")
    model = load_quantized("checkpoints/distilled_fp16", DEVICE)
    result = run_full_eval(model, tokenizer, DEVICE, tag="final_compressed_int4")
    save_result(result, "results/quantized/metrics.json")
    model.save_pretrained("checkpoints/final_quantized")

    print("\n=== Pipeline complete. See results/*/metrics.json for the full trail. ===")


if __name__ == "__main__":
    main()
