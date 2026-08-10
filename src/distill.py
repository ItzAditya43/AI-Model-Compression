"""Knowledge distillation: recover capability lost to pruning/decomposition.

Teacher = original Qwen2.5-3B-Instruct (frozen, 4-bit). Student = the
pruned+decomposed model, further quantized to 4-bit and trained with LoRA
adapters (full fine-tuning of even the shrunk model would not fit in 4GB).

Distillation runs OFFLINE rather than the usual side-by-side teacher+student
forward pass: a 4-bit teacher (~1.7GB) and a 4-bit student (~0.75GB) together
already consume nearly all of a 3.68GB card before any activation memory
exists, so there's no room left for even a single training step. Instead the
teacher's output distribution (top-k log-probs, not the full ~152k-vocab
logits) is precomputed once over the training set with the teacher alone in
VRAM, the teacher is freed completely, and only then is the student loaded
and trained against those cached targets. Same distillation objective, just
decoupled in time instead of requiring both models to coexist.

Loss = alpha * KL(student || teacher, over teacher's top-k) + (1 - alpha) * next-token CE.
"""
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model

TEACHER_TOPK = 50


def load_frozen_teacher(base_model_name, device):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb_config, device_map={"": 0} if device == "cuda" else None,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def wrap_student_with_lora(student, target_modules=("down", "up")):
    # After decomposition, every attention/FFN projection (q_proj, v_proj, ...)
    # is a LowRankLinear wrapper whose *name* still ends in e.g. "q_proj" -- if
    # that string were in target_modules, peft would match the wrapper itself
    # (a custom class it doesn't know how to adapt) instead of its inner
    # "<name>.down"/"<name>.up" leaf Linears, which is what we actually want.
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=list(target_modules), task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora_config)
    student.print_trainable_parameters()
    return student


def get_distill_dataset(tokenizer, n_examples=2000, seq_len=128):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 100][:n_examples]

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=seq_len, padding="max_length")

    hf_ds = Dataset.from_dict({"text": texts})
    hf_ds = hf_ds.map(tokenize, batched=True, remove_columns=["text"])
    hf_ds.set_format(type="torch")
    return hf_ds


@torch.no_grad()
def precompute_teacher_targets(teacher, dataset, device, topk=TEACHER_TOPK, batch_size=1):
    """One pass over the dataset with only the teacher in VRAM. Stores top-k
    log-probs + indices per position on CPU (cheap: ~batch*seq*topk floats,
    vs. batch*seq*vocab for full logits)."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    targets = []
    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        topk_vals, topk_idx = log_probs.topk(topk, dim=-1)
        targets.append((topk_vals.cpu(), topk_idx.cpu()))
        if device == "cuda":
            torch.cuda.empty_cache()
        if i % 20 == 0:
            print(f"[distill] precomputed teacher targets {i}/{len(loader)}")
    return targets


def distillation_loss_topk(student_logits, teacher_topk_vals, teacher_topk_idx, labels,
                            temperature=2.0, alpha=0.5):
    student_logits = student_logits[:, :-1, :]
    teacher_topk_vals = teacher_topk_vals[:, :-1, :]
    teacher_topk_idx = teacher_topk_idx[:, :-1, :]
    target = labels[:, 1:]

    ce_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)).float(), target.reshape(-1),
        ignore_index=-100,
    )

    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    student_topk_log_probs = student_log_probs.gather(-1, teacher_topk_idx)
    teacher_topk_probs = F.softmax(teacher_topk_vals / temperature, dim=-1)

    kd_loss = F.kl_div(student_topk_log_probs, teacher_topk_probs, reduction="batchmean") * (temperature ** 2)

    return alpha * kd_loss + (1 - alpha) * ce_loss, ce_loss.item(), kd_loss.item()


def train_student_on_targets(student, dataset, teacher_targets, device, n_steps=300, lr=1e-4,
                              batch_size=1, temperature=2.0, log_every=20):
    """Phase 2: teacher is already freed by this point -- student (already
    quantized by the caller) is trained alone against the cached targets."""
    student = wrap_student_with_lora(student)
    student.to(device)
    student.train()

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimizer = AdamW([p for p in student.parameters() if p.requires_grad], lr=lr)

    step = 0
    for batch, (topk_vals, topk_idx) in zip(loader, teacher_targets):
        if step >= n_steps:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        topk_vals = topk_vals.to(device)
        topk_idx = topk_idx.to(device)

        student_out = student(input_ids=input_ids, attention_mask=attention_mask)

        loss, ce, kd = distillation_loss_topk(
            student_out.logits, topk_vals, topk_idx, labels, temperature=temperature,
        )

        optimizer.zero_grad()
        loss.backward()
        loss_value = loss.item()
        optimizer.step()

        del student_out, loss, topk_vals, topk_idx
        if device == "cuda":
            torch.cuda.empty_cache()

        if step % log_every == 0:
            print(f"[distill] step {step}/{n_steps} loss={loss_value:.4f} ce={ce:.4f} kd={kd:.4f}")
        step += 1

    # Not merging: peft's merge path assumes it can add the LoRA delta directly
    # into base_layer.weight.data, but that data is the packed 4-bit buffer here
    # (not a dequantized matrix), which is a shape mismatch. Leaving the adapters
    # attached to the 4-bit base is valid for eval/inference regardless -- and is
    # how QLoRA models are commonly deployed anyway (adapter + quantized base,
    # loaded together, never merged back to full precision).
    student.eval()
    return student
