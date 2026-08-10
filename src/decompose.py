"""Low-rank decomposition of linear layers via activation-weighted SVD (ASVD-style).

Plain SVD on raw weights wastes rank on directions the model rarely activates.
ASVD instead scales columns by input-activation magnitude before decomposing, so
the truncated rank is spent where it actually matters, then rescales back.
Each target nn.Linear(in, out) becomes nn.Linear(in, r) -> nn.Linear(r, out).
"""
import torch
import torch.nn as nn

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@torch.no_grad()
def collect_input_scales(model, calib_batches, target_modules=TARGET_MODULES):
    """Per-module average abs input activation, used to weight SVD columns."""
    scales = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach().flatten(0, 1).abs().mean(dim=0).cpu()
            if name in scales:
                scales[name] = torch.maximum(scales[name], x)
            else:
                scales[name] = x
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            hooks.append(module.register_forward_hook(make_hook(name)))

    for batch in calib_batches:
        model.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for h in hooks:
        h.remove()
    return scales


class LowRankLinear(nn.Module):
    def __init__(self, U, S, Vt, bias=None):
        super().__init__()
        r = S.shape[0]
        self.down = nn.Linear(Vt.shape[1], r, bias=False, dtype=U.dtype, device=U.device)
        self.up = nn.Linear(r, U.shape[0], bias=bias is not None, dtype=U.dtype, device=U.device)
        self.down.weight.data = (torch.diag(S) @ Vt)
        self.up.weight.data = U
        if bias is not None:
            self.up.bias.data = bias

    def forward(self, x):
        return self.up(self.down(x))


@torch.no_grad()
def decompose_linear(linear: nn.Linear, rank_ratio: float, input_scale: torch.Tensor = None):
    W = linear.weight.data.float()  # (out, in)
    out_dim, in_dim = W.shape
    max_rank = min(out_dim, in_dim)
    r = max(8, int(max_rank * rank_ratio))

    if input_scale is not None:
        s = input_scale.float().clamp(min=1e-5).to(W.device)
        W_scaled = W * s.unsqueeze(0)  # scale columns (input dim) by activation magnitude
    else:
        s = None
        W_scaled = W

    U, S, Vt = torch.linalg.svd(W_scaled, full_matrices=False)
    U, S, Vt = U[:, :r], S[:r], Vt[:r, :]

    if s is not None:
        Vt = Vt / s.unsqueeze(0)  # undo the column scaling

    dtype = linear.weight.dtype
    new_module = LowRankLinear(
        U.to(dtype), S.to(dtype), Vt.to(dtype),
        bias=linear.bias.data.clone() if linear.bias is not None else None,
    )
    return new_module, r


def run_decomposition(model, calib_batches, rank_ratio, target_modules=TARGET_MODULES):
    print("[decompose] scoring input activations for ASVD weighting...")
    scales = collect_input_scales(model, calib_batches, target_modules)

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue

        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        input_scale = scales.get(name)

        new_module, r = decompose_linear(module, rank_ratio, input_scale)
        setattr(parent, child_name, new_module)
        replaced += 1
        print(f"[decompose] {name}: {module.in_features}x{module.out_features} -> rank {r}")

    print(f"[decompose] replaced {replaced} linear layers with low-rank factors")
    return model
