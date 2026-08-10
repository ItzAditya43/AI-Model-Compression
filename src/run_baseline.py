"""Stage 0: load the original model and record baseline metrics."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from configs.config import BASE_MODEL, DEVICE, EVAL_PPL_SAMPLES, EVAL_HELLASWAG_SAMPLES
from eval.harness import run_full_eval, save_result


def record_fp16_reference():
    """Param count / disk size the original model would need at fp16, without
    actually materializing weights in VRAM (meta device = shapes only, no memory)."""
    config = AutoConfig.from_pretrained(BASE_MODEL)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
    n_params = sum(p.numel() for p in model.parameters())
    size_gb = n_params * 2 / 1e9  # fp16 = 2 bytes/param
    ref = {"n_params": n_params, "n_params_billions": n_params / 1e9, "weights_size_gb": size_gb}
    print("fp16 reference:", ref)
    save_result(ref, "results/baseline/fp16_reference.json")
    del model


def main():
    record_fp16_reference()

    print(f"Loading baseline model {BASE_MODEL} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 4GB VRAM can't hold fp16 3B (~6GB) + activations, so baseline is measured
    # in 4-bit too -- this is the fair "what you'd actually deploy" comparison point.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map={"": 0} if DEVICE == "cuda" else None,
    )

    result = run_full_eval(model, tokenizer, DEVICE, tag="baseline_qwen2.5-3b-int4",
                            n_ppl_samples=EVAL_PPL_SAMPLES, n_hellaswag=EVAL_HELLASWAG_SAMPLES)
    save_result(result, "results/baseline/metrics.json")


if __name__ == "__main__":
    main()
