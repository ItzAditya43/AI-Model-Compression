# AI Model Compressor

A from-scratch research pipeline that takes a real instruction-tuned LLM and tries to
shrink it dramatically while keeping most of its capability — on a laptop GPU with
**4GB of VRAM**. Every stage (pruning, low-rank decomposition, knowledge distillation,
quantization) is implemented directly against PyTorch/HuggingFace internals rather than
wrapping an existing "one-command" compression library, because the point was to
understand and control what each technique actually does to the model.

## Why

Local/on-device LLM inference is constrained almost entirely by two numbers: how many
parameters a model has, and how many bits each one costs. Off-the-shelf models are sized
for datacenter GPUs, not an RTX 3050 laptop GPU. This project asks: how much of a
model's capability survives if you aggressively attack both numbers at once, using
techniques that are each individually well-established in the literature, chained
together into one pipeline?

## Hardware reality check (read this first)

The original plan was a 7-8B model. The actual GPU available for this project is a
laptop **RTX 3050 with 4GB VRAM** (3.68GB usable, and often less — see below). An
8B model doesn't fit in 4-bit quantized form, let alone fp16, so the base model was
changed to **Qwen2.5-3B-Instruct** (3.09B params, 6.17GB at fp16) — big enough to be a
real compression target, small enough that every pipeline stage is actually completable
on this hardware.

Even at 3B, this GPU is tight enough that most of the engineering effort in this repo
went into *fitting the pipeline into the hardware* as much as into the compression
techniques themselves:

- **VRAM (3.68GB usable)**: fp16 weights for a 3B model are 6.2GB — already over
  budget before any activations or gradients. Structural surgery (pruning,
  decomposition) needs real, unquantized weight access, so it can't just run 4-bit.
- **System RAM (15GB, frequently ~14GB already in use)**: naively loading the fp16
  model into normal CPU memory to work around the VRAM limit caused severe swap
  thrashing (observed: 80-90% I/O wait, 4M+ major page faults) rather than finishing in
  reasonable time.
- **The fix**: a hand-rolled *layer-wise GPU offload* (`src/offload_utils.py`). The
  model rests on CPU; a forward pre-hook moves exactly one transformer layer (weights +
  inputs) onto the GPU right before that layer's `forward()` runs, and a post-hook moves
  it back immediately after. At any instant, only ~1 layer's worth of weights (a few
  hundred MB) sits in VRAM. This is conceptually what `accelerate`'s automatic
  `device_map="auto"` offloading does, but implemented manually because the automatic
  version's hooks conflicted with this project's in-place module surgery (swapping
  `nn.Linear` layers for pruned/decomposed replacements mid-pipeline) and fragmented
  VRAM into repeated OOMs.
- **Calibration passes skip the LM head.** Early runs OOM'd computing the full
  ~152k-vocab output projection on every calibration batch — memory that pruning/
  decomposition scoring never actually needed, since they only read hidden states.
  Calling `model.model(...)` instead of `model(...)` during calibration fixed this.
- **Distillation uses QLoRA** (4-bit frozen base + LoRA adapters) rather than full
  fine-tuning, so the compressed student and the full-precision teacher can both be
  resident in VRAM at once for the forward/backward pass.

None of this changes what the compression techniques do — it changes how they had to be
*implemented* to run at all on this machine, and is arguably as representative of "real
local AI compression work" as the compression math itself.

## Pipeline

Base model: **Qwen2.5-3B-Instruct**. Every stage is scored against the same eval
harness (`eval/harness.py`) so the numbers are directly comparable stage to stage.

1. **Baseline** (`src/run_baseline.py`) — load the unmodified model (4-bit, since
   that's what's actually deployable on this GPU) and record perplexity, HellaSwag
   accuracy, generation speed, and VRAM footprint. This is the reference point for
   "% capability retained."

2. **Structured pruning** (`src/prune.py`)
   - *Depth pruning*: score every transformer block by the cosine similarity between
     its input and output hidden states on calibration data (ShortGPT-style). A block
     whose output barely differs from its input is nearly a no-op and safe to remove;
     the least-important blocks are dropped entirely.
   - *Width pruning*: for the surviving blocks, score each FFN intermediate neuron by
     mean activation magnitude on calibration data and keep only the top-k, physically
     shrinking `gate_proj`/`up_proj`/`down_proj`.

3. **Low-rank decomposition** (`src/decompose.py`) — activation-weighted SVD (ASVD-style)
   on every attention and FFN linear layer. Columns are scaled by their calibration
   input-activation magnitude before decomposing (so truncated rank is spent where the
   model actually uses it, not wasted on rarely-activated directions), then the scaling
   is undone after truncation. Each `Linear(in, out)` becomes
   `Linear(in, r) -> Linear(r, out)` at a fraction of the original rank.

4. **Knowledge distillation** (`src/distill.py`) — after two rounds of lossy structural
   surgery, capability drops significantly. The original (frozen, 4-bit) model acts as
   teacher; the pruned+decomposed model is wrapped in LoRA adapters and trained with a
   combined loss: `alpha * KL(student || teacher) + (1-alpha) * next-token CE`, so the
   student is pulled back toward the teacher's output distribution, not just toward the
   raw text.

5. **Activation-aware + mixed-precision quantization** (`src/quantize.py`) — the
   distilled model is quantized to 4-bit (NF4, double quantization) for the bulk of
   linear layers, while `lm_head` is explicitly kept out of the 4-bit path since it's
   cheap (relatively few params) but disproportionately quality-sensitive.

## Evaluation

Every stage is measured identically (`eval/harness.py`):

- **Perplexity** on WikiText-2
- **HellaSwag accuracy** (cloze-style, log-likelihood scoring over candidate endings)
- **Generation throughput** (tokens/sec, greedy decoding)
- **Peak VRAM**
- **Parameter count** and **on-disk weight size**

## Results

*(Filled in after the full pipeline run completes — see `results/*/metrics.json` for
the raw numbers at every stage: `baseline/`, `pruned/`, `decomposed/`, `distilled/`,
`quantized/`.)*

| Stage | Params | Size | Perplexity ↓ | HellaSwag ↑ | Tok/s | Peak VRAM |
|---|---|---|---|---|---|---|
| Baseline (int4) | 3.09B | 2.01GB | 13.13 | 58.0% | 20.9 | 3.18GB |
| Pruned | — | — | — | — | — | — |
| + Decomposed | — | — | — | — | — | — |
| + Distilled | — | — | — | — | — | — |
| Final (quantized) | — | — | — | — | — | — |

## Repo layout

```
configs/config.py     # all pipeline hyperparameters (keep ratios, rank ratio, distill steps, ...)
eval/harness.py        # shared perplexity / HellaSwag / latency / memory eval, used at every stage
src/prune.py            # depth + width structured pruning
src/decompose.py        # activation-weighted SVD low-rank decomposition
src/distill.py           # QLoRA knowledge distillation from the frozen original model
src/quantize.py          # 4-bit NF4 + mixed precision quantization
src/offload_utils.py    # manual layer-wise GPU offload for running structural surgery on a 4GB GPU
src/run_baseline.py     # stage 0: baseline metrics
src/run_pipeline.py      # orchestrates prune -> decompose -> distill -> quantize, with eval after each stage
results/*/metrics.json  # eval snapshot after each stage
```

## Running it

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.46" accelerate datasets peft sentencepiece protobuf autoawq bitsandbytes scipy

python src/run_baseline.py
python src/run_pipeline.py
```

Tune target compression aggressiveness in `configs/config.py`
(`PRUNE_LAYER_KEEP_RATIO`, `PRUNE_FFN_KEEP_RATIO`, `SVD_RANK_RATIO`, `DISTILL_STEPS`).
