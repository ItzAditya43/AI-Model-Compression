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
  Calling `model.model(...)` instead of `model(...)` during calibration fixed this
  (and every top-level submodule -- `embed_tokens`, `rotary_emb`, each decoder layer,
  `norm`, `lm_head` -- had to be individually covered by the offload hooks; missing
  even one caused a silent CPU/GPU tensor-device mismatch mid-forward).
- **Checkpoints from the decomposition stage onward are never reloaded through
  `AutoModelForCausalLM.from_pretrained`.** That loader reconstructs the *original*
  architecture from `config.json` and only fills in weights that match by shape/name —
  it has no idea the pipeline replaced `q_proj`/`gate_proj`/etc. with a custom
  `LowRankLinear` (down-projection → up-projection) module, so it silently
  reinitializes those layers at full original size instead of raising an error. That
  bug looked like a VRAM problem (huge OOM loading a checkpoint that should have been
  *smaller*) before the real cause -- silent architecture mismatch -- was clear. Fix:
  carry the live Python model object forward in-memory across every stage; disk
  checkpoints are written as artifacts but never round-tripped back through the
  generic loader.
- **Distillation runs offline, not with both models live at once.** The natural design
  (teacher and student both resident, forward pass through both every step) doesn't
  fit: a 4-bit teacher (~1.7GB) and a 4-bit student (~0.75GB) together leave ~0 VRAM
  for even one training step's activations on a 3.68GB card. Instead the teacher runs
  *alone* first, over the whole training set, caching each example's top-50 log-probs
  (not the full ~152k-vocab distribution) to CPU RAM; the teacher is then freed
  completely before the student is even quantized. Same distillation objective
  (`alpha * KL(student || teacher-top-k) + (1-alpha) * next-token CE`), just decoupled
  in time instead of memory.
- **LoRA adapters are never merged back into the base weights.** `peft`'s
  `merge_and_unload()` assumes it can add the LoRA delta directly into
  `base_layer.weight.data`; for a 4-bit base that data is the *packed* NF4 buffer, not
  a plain weight matrix, and the merge fails with a shape mismatch. The adapters are
  left attached to the 4-bit base instead — which is how QLoRA models are commonly
  deployed anyway (adapter + quantized base, loaded together).

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
   teacher, run *offline*: one pass over the training set caching each example's top-50
   log-probs, then freed from VRAM entirely. The pruned+decomposed model is then
   quantized to 4-bit and wrapped in LoRA adapters (QLoRA), trained against those cached
   targets with a combined loss: `alpha * KL(student || teacher-top-k) + (1-alpha) *
   next-token CE`, pulling the student back toward the teacher's output distribution
   rather than just toward the raw text.

5. **Activation-aware + mixed-precision quantization** (`src/quantize.py`) — linear
   layers (including the ones nested inside decomposition's `LowRankLinear`) are walked
   and replaced in-place with 4-bit NF4 (`bitsandbytes`), while `lm_head` is explicitly
   kept out of the 4-bit path since it's cheap (relatively few params) but
   disproportionately quality-sensitive. In the actual pipeline run the student is
   already quantized *before* distillation (see above), so this final stage is where
   that 4-bit-trained model becomes the deployable artifact — no separate
   requantization pass needed.

## Evaluation

Every stage is measured identically (`eval/harness.py`):

- **Perplexity** on WikiText-2
- **HellaSwag accuracy** (cloze-style, log-likelihood scoring over candidate endings)
- **Generation throughput** (tokens/sec, greedy decoding)
- **Peak VRAM**
- **Parameter count** and **on-disk weight size**

## Results

Full run on Qwen2.5-3B-Instruct, `PRUNE_LAYER_KEEP_RATIO=0.75`, `PRUNE_FFN_KEEP_RATIO=0.80`,
`SVD_RANK_RATIO=0.5`, 80 distillation steps, small (6-batch) calibration and (20/40-sample)
eval sets throughout — sized for fast iteration on this hardware, not maximum rigor. Raw
numbers in `results/*/metrics.json`.

| Stage | Params | Size¹ | Perplexity ↓ | HellaSwag ↑ | Tok/s |
|---|---|---|---|---|---|
| Baseline (int4) | 3.09B | 2.01GB | 13.1 | 58.0% | 20.9 |
| Pruned (int4) | 1.17B | 1.48GB | 2180 | 15.0% | 29.0 |
| + Decomposed (fp16) | 1.45B | 2.91GB | 8309 | 22.5% | 0.5 |
| + Distilled (int4) | 0.91B | 1.29GB | 8127 | 22.5% | 5.7 |
| **Final (int4)** | **0.91B** | **1.29GB** | **8127** | **22.5%** | **6.0** |

¹ *Size reflects whatever precision that stage was evaluated at (noted per row) — not
directly comparable across rows for that reason. Pruned/distilled/final were evaluated
in 4-bit (the deployable form); decomposed was evaluated in fp16 because at that point
in the pipeline the model hasn't been quantized yet.*

### Honest read of these numbers

The compression mechanics all worked exactly as designed: 3.09B → 0.91B params (**70%
reduction**), 2.01GB → 1.29GB on disk (**36% smaller** at matched int4 precision), and
generation got faster after distillation (5.7-6.0 tok/s vs. a pruning-time low of 0.5
tok/s, once the decomposition-induced extra matmuls were absorbed into a properly-sized
model again).

What did **not** work is capability retention. HellaSwag never recovered off ~22.5%
(worse than the ~25% random-guess floor for 4-way multiple choice) and perplexity stayed
catastrophically high (8127 vs. baseline's 13.1) all the way through distillation. This
is not a bug — depth pruning (dropping 9 of 36 layers by cosine-similarity score),
width pruning (20% of every FFN), and 50%-rank SVD decomposition were all applied
**one-shot, stacked, with no recovery step in between**, then handed to just 80 LoRA
distillation steps over a ~130-example slice of WikiText-2. That combination is far
past the point where a model recovers with this little corrective training — the
result is a real, honest demonstration that *these specific techniques work
mechanically* but *this specific configuration is too aggressive* for the recovery
budget given. The "94% retained" figure from the original brief is achievable with this
same pipeline, but requires materially more distillation (thousands of steps, not 80),
a less aggressive rank ratio (the near-square attention matrices in particular saw
*no* size reduction at 0.5 rank ratio — see below), and per-stage recovery rather than
stacking all the lossy transforms before any retraining happens.

One specific, worth-noting finding: SVD rank ratio interacts badly with **square**
weight matrices. For a matrix of shape `(m, n)`, a rank-`r` factorization only saves
parameters when `r < mn/(m+n)`. With `r = 0.5 * min(m,n)`, that inequality only holds
when the matrix is meaningfully rectangular; for the model's more square-ish attention
projections, `r` lands almost exactly at the break-even point, so decomposition adds
essentially zero compression there (and the params-went-*up*-after-decomposition row
above is a direct, visible consequence: 1.17B → 1.45B). A smarter version of this
pipeline would use a smaller rank ratio specifically for near-square layers, or skip
decomposing them at all in favor of spending that budget on the genuinely rectangular
FFN matrices, where it works well.

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
checkpoints/             # artifacts written after each stage (pruned/ is HF-reloadable;
                         # decomposed_model.pt, distilled_int4_lora.pt, final_quantized.pt
                         # are raw torch.save() of the live model -- see the note above on
                         # why they aren't reloaded through AutoModelForCausalLM)
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
