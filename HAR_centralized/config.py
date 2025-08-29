from pathlib import Path
import os
import torch
from typing import List, Optional
import shutil
from common.utils import BOLD, RESET


# Experiment name (used for naming output models/results)
EXPERIMENT_NAME = "HAR_centralized"

# Project root (two levels up from this file: .../MTSSL_for_Label-Efficient_Learning)
ROOT = Path(__file__).resolve().parents[1]
CUR_DIR = Path(__file__).resolve().parent

# Data paths (can be overridden with environment variable DATA_DIR)
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data" / "UCI_HAR"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", CUR_DIR / "outputs"))
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# Ensure output directories exist
for _d in (OUTPUT_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Misc / reproducibility
SEED: int = int(os.environ.get("SEED", 0))

# Device
# Prefer CUDA, then Apple's MPS (macOS GPU), otherwise CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Dataloader
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", 64))
BATCH_SIZE_TRANSFER: int = int(os.environ.get("BATCH_SIZE_TRANSFER", 32))
NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", 4))
PIN_MEMORY: bool = True if DEVICE.type == "cuda" else False
VERBOSE: bool = bool(int(os.environ.get("VERBOSE", True)))

# Augmentation / data settings
NOISE_RATIO: float = float(os.environ.get("NOISE_RATIO", 0.5))

# Model heads: [aug_classification, reconstruction, contrastive_learning, contrastive_features]
# Provide a typed list and validate length == 4
MODEL_HEADS: List[bool] = [True, True, True, True]
if len(MODEL_HEADS) != 4 or sum(MODEL_HEADS) == 0:
    raise ValueError("MODEL_HEADS must be a list of 4 booleans with at least one True value: [aug_classification, reconstruction, contrastive_learning, contrastive_features]")

# Whether to use dynamic weighted loss for multi-task learning
USE_WEIGHTED_LOSS: bool = bool(int(os.environ.get("USE_WEIGHTED_LOSS", True)))
CONTRASTIVE_LOSS: str = os.environ.get("CONTRASTIVE_LOSS", "ntxent")  # 'ntxent' or 'barlow' or 'vicreg'

# disable weighted loss if only one task is active
if sum(MODEL_HEADS) == 1 and USE_WEIGHTED_LOSS:
    # print a warning
    print("⚠️  Only one task heads are active; disabling USE_WEIGHTED_LOSS.")
    USE_WEIGHTED_LOSS = False

# Encoder / model params
ENCODER_NAME: str = os.environ.get("ENCODER_NAME", "FCN")  # 'FCN', 'DeepConvLSTM' or 'Transformer'
if ENCODER_NAME not in ["FCN", "DeepConvLSTM", "Transformer"]:
    raise ValueError("ENCODER_NAME must be one of 'FCN', 'DeepConvLSTM', or 'Transformer'")
OUT_DIM: int = int(os.environ.get("OUT_DIM", 128))  # output dimension of the encoder (representation vector)

# Training hyperparameters
LR: float = float(os.environ.get("LR", 1e-3))
EPOCHS: int = int(os.environ.get("EPOCHS", 200))
DOWNSTREAM_EPOCHS: int = int(os.environ.get("DOWNSTREAM_EPOCHS", 100))  # epochs for downstream classifier training


# Logging / checkpoints
SAVE_BEST_ONLY: bool = bool(int(os.environ.get("SAVE_BEST_ONLY", True)))
CHECKPOINT_FREQ: int = int(os.environ.get("CHECKPOINT_FREQ", 20))  # save every N epochs

# Dataset-specific parameters
UCIHAR_NUM_CLASSES: int = 6  # number of classes for UCI HAR classification
UCIHAR_FEATURE_DIM: int = 561  # number of features for UCI HAR dataset (fixed)
UCIHAR_NUM_CHANNELS: int = 9  # number of channels for UCI HAR dataset (fixed)
UCIHAR_SEQ_LEN: int = 128  # sequence length for UCI HAR dataset (fixed)


# make the experiment name unique by appending seed and active heads
def make_experiment_name(base_name: str = EXPERIMENT_NAME, seed: int = SEED, heads: List[bool] = MODEL_HEADS) -> str:
    heads_str = ''.join(['1' if h else '0' for h in heads])
    return f"{base_name}_s{seed}_h{heads_str}"

EXPERIMENT_NAME = make_experiment_name()

def load_config_from_cli() -> dict:
    import argparse

    parser = argparse.ArgumentParser(description="HAR Centralized Training Configuration")
    parser.add_argument("--experiment_name", type=str, default=EXPERIMENT_NAME, help="Base name for the experiment")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--batch_size_transfer", type=int, default=BATCH_SIZE_TRANSFER, help="Batch size for transfer learning")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="Number of workers for data loading")
    parser.add_argument("--noise_ratio", type=float, default=NOISE_RATIO, help="Noise ratio for data augmentation")
    parser.add_argument("--model_heads", type=str, default=''.join(['1' if h else '0' for h in MODEL_HEADS]), help="String of 4 chars (1 or 0) indicating which model heads to use")
    parser.add_argument("--use_weighted_loss", type=int, default=int(USE_WEIGHTED_LOSS), help="Whether to use dynamic weighted loss (1=True, 0=False)")
    parser.add_argument("--contrastive_loss", type=str, default=CONTRASTIVE_LOSS, help="Type of contrastive loss to use (ntxent, barlow, vicreg)")
    parser.add_argument("--encoder_name", type=str, default=ENCODER_NAME, help="Name of the encoder model (FCN, DeepConvLSTM, Transformer)")
    parser.add_argument("--out_dim", type=int, default=OUT_DIM, help="Output dimension of the encoder")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--downstream_epochs", type=int, default=DOWNSTREAM_EPOCHS, help="Number of epochs for downstream classifier training")
    parser.add_argument("--save_best_only", type=int, default=int(SAVE_BEST_ONLY), help="Whether to save only the best model (1=True, 0=False)")
    parser.add_argument("--checkpoint_freq", type=int, default=CHECKPOINT_FREQ, help="Frequency (in epochs) to save checkpoints")
    parser.add_argument("--verbose", type=int, default=int(VERBOSE), help="Verbosity level (1=True, 0=False)")
    args = parser.parse_args()

    # make the experiment name
    if args.experiment_name != EXPERIMENT_NAME:
        args.experiment_name = make_experiment_name(base_name=args.experiment_name, seed=args.seed, heads=[c == '1' for c in args.model_heads])

    # Process model_heads argument
    if len(args.model_heads) != 4 or any(c not in '01' for c in args.model_heads):
        raise ValueError("model_heads must be a string of 4 characters (1 or 0), e.g. '1101'")
    model_heads = [c == '1' for c in args.model_heads]

    # Determine weighted loss
    use_weighted = bool(int(args.use_weighted_loss))
    if sum(model_heads) == 1 and use_weighted:
        print("⚠️  Only one task head is active; disabling use_weighted_loss.")
        use_weighted = False

    # Build config dict
    cfg = {
        'EXPERIMENT_NAME': args.experiment_name,
        'SEED': int(args.seed),
        'BATCH_SIZE': int(args.batch_size),
        'BATCH_SIZE_TRANSFER': int(args.batch_size_transfer),
        'NUM_WORKERS': int(args.num_workers),
        'NOISE_RATIO': float(args.noise_ratio),
        'MODEL_HEADS': model_heads,
        'USE_WEIGHTED_LOSS': use_weighted,
        'CONTRASTIVE_LOSS': args.contrastive_loss,
        'ENCODER_NAME': args.encoder_name,
        'OUT_DIM': int(args.out_dim),
        'LR': float(args.lr),
        'EPOCHS': int(args.epochs),
        'DOWNSTREAM_EPOCHS': int(args.downstream_epochs),
        'SAVE_BEST_ONLY': bool(int(args.save_best_only)),
        'CHECKPOINT_FREQ': int(args.checkpoint_freq),
        'VERBOSE': bool(int(args.verbose)),
    }

    # Update module-level variables so other modules reading config get the overridden values
    globals().update(cfg)

    return cfg

# Expose a compact, pretty-printed config summary
def print_summary() -> str:
    """Return a nicely formatted multi-line configuration summary."""
    # Ensure EXPERIMENT_NAME reflects the current SEED and MODEL_HEADS
    global EXPERIMENT_NAME
    EXPERIMENT_NAME = make_experiment_name(base_name=EXPERIMENT_NAME.split("_s")[0], seed=SEED, heads=MODEL_HEADS)

    items = [
        ("Experiment", EXPERIMENT_NAME),
        ("Seed", SEED),
        ("Device", str(DEVICE)),
        ("Batch size", BATCH_SIZE),
        ("Batch size (transfer)", BATCH_SIZE_TRANSFER),
        ("Num workers", NUM_WORKERS),
        ("Pin memory", PIN_MEMORY),
        ("Model heads (1111 -> aug,rec,cont,feat)", ''.join(['1' if h else '0' for h in MODEL_HEADS])),
        ("Use weighted loss", USE_WEIGHTED_LOSS),
        ("Contrastive loss", CONTRASTIVE_LOSS),
        ("Noise ratio (rec task)", NOISE_RATIO),
        ("Encoder", ENCODER_NAME),
        ("Embedding size", OUT_DIM),
        ("Learning rate", LR),
        ("Epochs", EPOCHS),
        ("Downstream Epochs", DOWNSTREAM_EPOCHS),
        ("Verbose", VERBOSE),
    ]

    # print

    # Column widths for neat alignment
    key_width = max(len(k) for k, _ in items)
    terminal_width = shutil.get_terminal_size().columns

    print("━" * terminal_width)
    print(BOLD + "⚙️ CONFIGURATION SUMMARY" + RESET)
    print("─" * terminal_width)
    for k, v in items:
        print(f"{k.ljust(key_width)} : {v}")
    print("─" * terminal_width)

# if __name__ == "__main__":
#     print_summary()
#     load_config_from_cli()
#     print("\nLoaded Configuration from CLI:")
#     print_summary()

