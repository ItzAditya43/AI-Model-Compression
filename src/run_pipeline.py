"""End-to-end pipeline: prune -> decompose -> distill -> quantize, with a full
eval snapshot saved after every stage so we can see where quality is won/lost.

Structural surgery (pruning, SVD decomposition) needs real fp16 weight access,
but the fp16 3B model (6.2GB) doesn't fit in this GPU's 3.68GB VRAM, and the
machine's system RAM is too full to hold it either (loading it on CPU thrashed
swap badly). Instead we use manual layer-wise GPU offload (see offload_utils.py):
the model rests on CPU, and each layer is moved to GPU just for its own forward
pass, then back -- all compute happens on GPU, never more than ~1 layer's
weights resident in VRAM at once, and system RAM is barely touched.

From the decomposition stage onward, the model contains custom LowRankLinear
modules that transformers' AutoModelForCausalLM.from_pretrained cannot
reconstruct from a checkpoint (it silently rebuilds the *original* architecture
from config instead). So rather than saving to disk and reloading through the
generic HF loader between stages, the live Python model object is carried
forward in-memory through pruning -> decomposition -> distillation ->
quantization; checkpoints are still written to disk after each stage, but only
as artifacts, not as something this pipeline reloads.
"""
import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from configs.config import (
    BASE_MODEL, DEVICE, PRUNE_LAYER_KEEP_RATIO, PRUNE_FFN_KEEP_RATIO,
    SVD_RANK_RATIO, DISTILL_STEPS, DISTILL_LR, DISTILL_TEMPERATURE,
    CALIB_N_BATCHES, EVAL_PPL_SAMPLES, EVAL_HELLASWAG_SAMPLES,
)
from eval.harness import run_full_eval, save_result
from src.prune import run_pruning, get_calibration_batches
from src.decompose import run_decomposition
from src.distill import (
    get_distill_dataset, precompute_teacher_targets, train_student_on_targets, load_frozen_teacher,
)
from src.quantize import quantize_in_place
from src.offload_utils import enable_layerwise_gpu_offload


def free_gpu():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def eval_pruned_int4(path, tokenizer, tag, result_path):
    """Pruning only changes nn.Linear shapes (no custom modules), so the
    checkpoint IS safely reloadable via the generic HF path -- reload in 4-bit
    for a fast, realistic-footprint eval."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=bnb_config,
        device_map={"": 0} if DEVICE == "cuda" else None,
    )
    result = run_full_eval(model, tokenizer, DEVICE, tag=tag,
                            n_ppl_samples=EVAL_PPL_SAMPLES, n_hellaswag=EVAL_HELLASWAG_SAMPLES)
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
    model = run_pruning(model, tokenizer, "cpu", PRUNE_LAYER_KEEP_RATIO, PRUNE_FFN_KEEP_RATIO, n_batches=CALIB_N_BATCHES)
    model.save_pretrained("checkpoints/pruned")
    tokenizer.save_pretrained("checkpoints/pruned")
    del model
    gc.collect()
    free_gpu()

    eval_pruned_int4("checkpoints/pruned", tokenizer, "pruned", "results/pruned/metrics.json")

    # ---- Stage 2: low-rank decomposition, layer-wise GPU offload ----
    print("\n=== Loading pruned checkpoint for SVD decomposition ===")
    model = AutoModelForCausalLM.from_pretrained("checkpoints/pruned", dtype=torch.float16, device_map="cpu")
    handles = enable_layerwise_gpu_offload(model, DEVICE)
    calib_batches = get_calibration_batches(tokenizer, "cpu", n_batches=CALIB_N_BATCHES)

    print("\n=== Stage: Low-Rank Decomposition ===")
    model = run_decomposition(model, calib_batches, SVD_RANK_RATIO)
    del calib_batches
    gc.collect()

    # Eval directly on the live (still layer-offloaded) model -- no disk round trip.
    result = run_full_eval(model, tokenizer, DEVICE, tag="pruned+decomposed",
                            n_ppl_samples=EVAL_PPL_SAMPLES, n_hellaswag=EVAL_HELLASWAG_SAMPLES)
    save_result(result, "results/decomposed/metrics.json")

    # Drop the per-layer offload hooks (also needed before torch.save: the hooks
    # are lambda closures, which the pickler underneath torch.save can't handle).
    for h in handles:
        h.remove()
    model.to("cpu")
    free_gpu()

    torch.save(model, "checkpoints/decomposed_model.pt")  # artifact only; not reloaded by this pipeline
    print("[checkpoint] saved checkpoints/decomposed_model.pt")

    # ---- Stage 3: knowledge distillation recovery (offline: teacher and student never coexist in VRAM) ----
    # A 4-bit teacher (1.7GB) and a 4-bit student (0.75GB) together leave no
    # headroom for even one training step's activations on a 3.68GB card. So:
    # teacher runs alone first to precompute target distributions (top-k
    # log-probs, cached on CPU), gets freed completely, THEN the student is
    # quantized and trained alone against those cached targets.
    print("\n=== Stage: Knowledge Distillation ===")
    distill_dataset = get_distill_dataset(tokenizer, n_examples=DISTILL_STEPS + 50, seq_len=128)

    teacher = load_frozen_teacher(BASE_MODEL, DEVICE)
    print("[distill] precomputing teacher targets (teacher-only pass)...")
    teacher_targets = precompute_teacher_targets(teacher, distill_dataset, DEVICE)
    del teacher
    free_gpu()

    model = quantize_in_place(model, DEVICE)
    free_gpu()
    model = train_student_on_targets(
        model, distill_dataset, teacher_targets, DEVICE,
        n_steps=DISTILL_STEPS, lr=DISTILL_LR, temperature=DISTILL_TEMPERATURE,
    )
    del teacher_targets
    free_gpu()
    result = run_full_eval(model, tokenizer, DEVICE, tag="distilled",
                            n_ppl_samples=EVAL_PPL_SAMPLES, n_hellaswag=EVAL_HELLASWAG_SAMPLES)
    save_result(result, "results/distilled/metrics.json")

    torch.save(model, "checkpoints/distilled_int4_lora.pt")
    print("[checkpoint] saved checkpoints/distilled_int4_lora.pt")

    # ---- Stage 4: final quantized artifact ----
    # The student was already quantized to 4-bit (NF4, lm_head kept fp16) before
    # distillation and trained that way end-to-end (QLoRA) -- the model coming
    # out of distillation IS the final activation-aware, mixed-precision
    # quantized artifact. No further requantization step needed.
    print("\n=== Stage: Quantization ===")
    result = run_full_eval(model, tokenizer, DEVICE, tag="final_compressed_int4",
                            n_ppl_samples=EVAL_PPL_SAMPLES, n_hellaswag=EVAL_HELLASWAG_SAMPLES)
    save_result(result, "results/quantized/metrics.json")
    torch.save(model, "checkpoints/final_quantized.pt")

    print("\n=== Pipeline complete. See results/*/metrics.json for the full trail. ===")


if __name__ == "__main__":
    main()
