"""Knowledge distillation: recover capability lost to pruning/decomposition.

Teacher = original Qwen2.5-3B-Instruct (frozen, loaded in 4-bit to fit VRAM
alongside the student). Student = the pruned+decomposed model, trained with
LoRA adapters (full fine-tuning of even the shrunk model plus a teacher forward
pass would not fit in 4GB).

Loss = alpha * KL(student || teacher) + (1 - alpha) * next-token CE on the text.
"""
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


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


def wrap_student_with_lora(student, target_modules=("q_proj", "v_proj", "up", "down")):
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=list(target_modules), task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora_config)
    student.print_trainable_parameters()
    return student


def get_distill_dataset(tokenizer, n_examples=2000, seq_len=256):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 100][:n_examples]

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=seq_len, padding="max_length")

    from datasets import Dataset
    hf_ds = Dataset.from_dict({"text": texts})
    hf_ds = hf_ds.map(tokenize, batched=True, remove_columns=["text"])
    hf_ds.set_format(type="torch")
    return hf_ds


def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    student_logits = student_logits[:, :-1, :]
    teacher_logits = teacher_logits[:, :-1, :]
    target = labels[:, 1:]

    ce_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)).float(), target.reshape(-1),
        ignore_index=-100,
    )

    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    return alpha * kd_loss + (1 - alpha) * ce_loss, ce_loss.item(), kd_loss.item()


def run_distillation(student, teacher, tokenizer, device, n_steps=300, lr=1e-4,
                      batch_size=1, temperature=2.0, log_every=20):
    student = wrap_student_with_lora(student)
    student.to(device)
    student.train()

    dataset = get_distill_dataset(tokenizer, n_examples=n_steps * batch_size + 50)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW([p for p in student.parameters() if p.requires_grad], lr=lr)

    step = 0
    for batch in loader:
        if step >= n_steps:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()

        with torch.no_grad():
            teacher_out = teacher(input_ids=input_ids, attention_mask=attention_mask)

        student_out = student(input_ids=input_ids, attention_mask=attention_mask)

        loss, ce, kd = distillation_loss(
            student_out.logits, teacher_out.logits, labels, temperature=temperature,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            print(f"[distill] step {step}/{n_steps} loss={loss.item():.4f} ce={ce:.4f} kd={kd:.4f}")
        step += 1

    student = student.merge_and_unload()
    student.eval()
    return student
