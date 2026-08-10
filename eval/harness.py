"""Shared evaluation harness: perplexity, task accuracy, latency, memory, size.

Used identically at every pipeline stage so numbers are comparable stage-to-stage.
"""
import json
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset


@torch.no_grad()
def compute_perplexity(model, tokenizer, device, n_samples=40, seq_len=512):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()][:n_samples * 3])
    enc = tokenizer(text, return_tensors="pt").input_ids[0]

    nlls = []
    n_chunks = min(n_samples, len(enc) // seq_len)
    for i in range(n_chunks):
        chunk = enc[i * seq_len:(i + 1) * seq_len].unsqueeze(0).to(device)
        out = model(chunk, labels=chunk)
        nlls.append(out.loss.float() * seq_len)
    ppl = torch.exp(torch.stack(nlls).sum() / (n_chunks * seq_len))
    return ppl.item()


@torch.no_grad()
def eval_multiple_choice(model, tokenizer, device, dataset_name="hellaswag", n_samples=100):
    """Cloze-style eval: score each candidate ending by summed log-likelihood, pick best."""
    if dataset_name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True).select(range(n_samples))
        correct = 0
        for ex in ds:
            ctx = ex["ctx"]
            endings = ex["endings"]
            label = int(ex["label"])
            scores = []
            for ending in endings:
                full = ctx + " " + ending
                enc = tokenizer(full, return_tensors="pt").to(device)
                ctx_len = len(tokenizer(ctx, return_tensors="pt").input_ids[0])
                out = model(**enc, labels=enc.input_ids)
                logits = model(**enc).logits[0, ctx_len - 1:-1]
                target = enc.input_ids[0, ctx_len:]
                if len(target) == 0:
                    scores.append(-1e9)
                    continue
                logprobs = F.log_softmax(logits.float(), dim=-1)
                token_lp = logprobs.gather(1, target.unsqueeze(1)).squeeze(1)
                scores.append(token_lp.mean().item())
            pred = int(torch.tensor(scores).argmax())
            correct += int(pred == label)
        return correct / len(ds)
    raise ValueError(dataset_name)


def measure_latency(model, tokenizer, device, prompt="The future of artificial intelligence is", n_tokens=64, n_runs=5):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    # warmup
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=8, do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=n_tokens, do_sample=False)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        n_gen = out.shape[1] - enc.input_ids.shape[1]
        times.append(n_gen / elapsed)
    return sum(times) / len(times)  # tokens/sec


def measure_memory(device):
    if device != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / 1e9  # GB


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_disk_size_gb(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / 1e9


def run_full_eval(model, tokenizer, device, tag, n_ppl_samples=40, n_hellaswag=100):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.eval()

    print(f"[{tag}] computing perplexity...")
    ppl = compute_perplexity(model, tokenizer, device, n_samples=n_ppl_samples)

    print(f"[{tag}] computing hellaswag accuracy...")
    acc = eval_multiple_choice(model, tokenizer, device, "hellaswag", n_samples=n_hellaswag)

    print(f"[{tag}] measuring latency...")
    tok_per_s = measure_latency(model, tokenizer, device)

    mem_gb = measure_memory(device)
    n_params = count_params(model)
    size_gb = model_disk_size_gb(model)

    result = {
        "tag": tag,
        "perplexity_wikitext2": ppl,
        "hellaswag_acc": acc,
        "tokens_per_sec": tok_per_s,
        "peak_vram_gb": mem_gb,
        "n_params": n_params,
        "n_params_billions": n_params / 1e9,
        "weights_size_gb": size_gb,
    }
    print(json.dumps(result, indent=2))
    return result


def save_result(result, path):
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
