"""Structured pruning: depth (layer removal) + width (FFN intermediate dim) pruning.

Depth pruning: score each transformer block by the cosine-similarity between its
input and output hidden states on calibration data (ShortGPT-style). Low similarity
= block changes the representation a lot = important. High similarity = block is
nearly a no-op = safe to remove. We drop the least-important blocks.

Width pruning: for the surviving blocks, score each FFN intermediate neuron by
mean activation magnitude on calibration data, and keep only the top-k.
"""
import torch
import torch.nn as nn
from datasets import load_dataset


@torch.no_grad()
def get_calibration_batches(tokenizer, device, n_batches=16, seq_len=256):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 200][:n_batches]
    batches = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=seq_len).to(device)
        if enc.input_ids.shape[1] < 16:
            continue
        batches.append(enc)
    return batches


@torch.no_grad()
def score_layer_importance(model, tokenizer, device, calib_batches):
    """Return list of importance scores per decoder layer (lower = more removable)."""
    layers = model.model.layers
    n_layers = len(layers)
    sims = torch.zeros(n_layers)
    counts = 0

    hooks = []
    io_cache = {}

    def make_hook(idx):
        def hook(module, inp, out):
            hidden_in = inp[0].detach()
            hidden_out = out[0].detach() if isinstance(out, tuple) else out.detach()
            io_cache[idx] = (hidden_in, hidden_out)
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    for batch in calib_batches:
        io_cache.clear()
        model.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
        for i in range(n_layers):
            hin, hout = io_cache[i]
            cos = torch.nn.functional.cosine_similarity(
                hin.flatten(0, 1), hout.flatten(0, 1), dim=-1
            ).mean().cpu()
            sims[i] += cos
        counts += 1
        io_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    sims /= counts
    # importance = 1 - similarity (a block that barely changes its input is unimportant)
    importance = 1.0 - sims
    return importance.tolist()


def prune_depth(model, tokenizer, device, keep_ratio, calib_batches):
    layers = model.model.layers
    n_layers = len(layers)
    n_keep = max(1, int(n_layers * keep_ratio))

    importance = score_layer_importance(model, tokenizer, device, calib_batches)
    ranked = sorted(range(n_layers), key=lambda i: importance[i], reverse=True)
    keep_idx = sorted(ranked[:n_keep])
    dropped = sorted(set(range(n_layers)) - set(keep_idx))

    print(f"[prune_depth] {n_layers} -> {n_keep} layers. Dropped indices: {dropped}")

    new_layers = nn.ModuleList([layers[i] for i in keep_idx])
    for new_idx, layer in enumerate(new_layers):
        layer.layer_idx = new_idx
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = new_idx
    model.model.layers = new_layers
    model.config.num_hidden_layers = n_keep
    if getattr(model.config, "layer_types", None) is not None:
        model.config.layer_types = [model.config.layer_types[i] for i in keep_idx]
    return model, keep_idx, dropped


@torch.no_grad()
def score_ffn_neurons(model, calib_batches, layer):
    """Score FFN intermediate neurons by mean abs activation of gate_proj output."""
    acts = []

    def hook(module, inp, out):
        acts.append(out.detach().abs().mean(dim=(0, 1)).cpu())

    h = layer.mlp.gate_proj.register_forward_hook(hook)
    for batch in calib_batches:
        model.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    h.remove()

    return torch.stack(acts).mean(dim=0)


def prune_width(model, calib_batches, keep_ratio):
    """Shrink FFN intermediate dimension on every remaining layer by activation magnitude."""
    layers = model.model.layers
    final_n_keep = None
    for li, layer in enumerate(layers):
        mlp = layer.mlp
        inter_dim = mlp.gate_proj.out_features
        n_keep = max(1, int(inter_dim * keep_ratio))

        scores = score_ffn_neurons(model, calib_batches, layer)
        keep_idx = torch.topk(scores, n_keep).indices.sort().values

        gate_w = mlp.gate_proj.weight.data[keep_idx, :]
        up_w = mlp.up_proj.weight.data[keep_idx, :]
        down_w = mlp.down_proj.weight.data[:, keep_idx]

        new_gate = nn.Linear(mlp.gate_proj.in_features, n_keep, bias=False,
                              dtype=gate_w.dtype, device=gate_w.device)
        new_up = nn.Linear(mlp.up_proj.in_features, n_keep, bias=False,
                            dtype=up_w.dtype, device=up_w.device)
        new_down = nn.Linear(n_keep, mlp.down_proj.out_features, bias=False,
                              dtype=down_w.dtype, device=down_w.device)
        new_gate.weight.data = gate_w.clone()
        new_up.weight.data = up_w.clone()
        new_down.weight.data = down_w.clone()

        mlp.gate_proj = new_gate
        mlp.up_proj = new_up
        mlp.down_proj = new_down

        print(f"[prune_width] layer {li}: FFN {inter_dim} -> {n_keep}")
        final_n_keep = n_keep

    if final_n_keep is not None:
        model.config.intermediate_size = final_n_keep
    return model


def run_pruning(model, tokenizer, device, depth_keep_ratio, width_keep_ratio, n_batches=16):
    calib_batches = get_calibration_batches(tokenizer, device, n_batches=n_batches)
    model, keep_idx, dropped = prune_depth(model, tokenizer, device, depth_keep_ratio, calib_batches)
    # recompute calibration activations through the now-shorter model before width pruning
    model = prune_width(model, calib_batches, width_keep_ratio)
    return model
