"""Central config for the compression pipeline."""
import torch

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

# VRAM is tight (4GB laptop GPU) -- keep batch/seq small everywhere.
MAX_SEQ_LEN = 512
CALIB_BATCH_SIZE = 1

# Pruning
PRUNE_LAYER_KEEP_RATIO = 0.75   # drop ~25% of transformer layers
PRUNE_FFN_KEEP_RATIO = 0.80     # shrink FFN intermediate dim to 80%

# Low-rank decomposition
SVD_RANK_RATIO = 0.5            # keep 50% of singular values on targeted linear layers

# Distillation
DISTILL_STEPS = 300
DISTILL_LR = 1e-4
DISTILL_TEMPERATURE = 2.0

# Quantization
QUANT_BITS = 4

RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"
