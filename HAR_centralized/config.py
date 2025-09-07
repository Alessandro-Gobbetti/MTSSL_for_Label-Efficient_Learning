# ...existing code...
from dataclasses import dataclass, field, asdict
from pathlib import Path
import os
import shutil
from typing import List
from common.utils import check_gpu, BOLD, RESET

ROOT = Path(__file__).resolve().parents[1]
CUR_DIR = Path(__file__).resolve().parent

@dataclass
class Config:
    # Paths / meta
    EXPERIMENT_NAME: str = "experiment"

    DATASET_DIR: Path = Path(os.environ.get("DATA_DIR", ROOT / "data" / "UCI_HAR"))
    OUTPUT_DIR: Path = Path(os.environ.get("OUTPUT_DIR", CUR_DIR / "outputs"))
    MODELS_DIR: Path = Path(os.environ.get("MODEL_DIR", CUR_DIR / "outputs" / "models"))
    RESULTS_DIR: Path = Path(os.environ.get("RESULTS_DIR", CUR_DIR / "outputs" / "results"))

    # Ensure output directories exist
    for _d in [OUTPUT_DIR, MODELS_DIR, RESULTS_DIR]:
        _d.mkdir(parents=True, exist_ok=True)

    # Repro / device
    SEED: int = int(os.environ.get("SEED", 0))
    gpu: int = int(os.environ.get("GPU", -2))  # -1 cpu, -2 multigpu, <0 specific gpu>
    DEVICE: object = check_gpu(gpu)

    # Data
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", 64))
    BATCH_SIZE_TRANSFER: int = int(os.environ.get("BATCH_SIZE_TRANSFER", 32))
    NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", 0))
    PIN_MEMORY: bool = True if DEVICE.type == "cuda" else False

    # Model / training
    LR: float = float(os.environ.get("LR", 1e-3))
    EPOCHS: int = int(os.environ.get("EPOCHS", 200))
    DOWNSTREAM_EPOCHS: int = int(os.environ.get("DOWNSTREAM_EPOCHS", 100))
    VERBOSE: bool = bool(int(os.environ.get("VERBOSE", 1)))

    # Multi-task
    NUM_HEADS: int = 4
    MODEL_HEADS: List[bool] = field(default_factory=lambda: [True, True, True, True])
    USE_WEIGHTED_LOSS: bool = bool(int(os.environ.get("USE_WEIGHTED_LOSS", 1)))
    CONTRASTIVE_LOSS: str = os.environ.get("CONTRASTIVE_LOSS", "ntxent")

    # Checkpoint
    SAVE_BEST_ONLY: bool = bool(int(os.environ.get("SAVE_BEST_ONLY", 1)))
    CHECKPOINT_FREQ: int = int(os.environ.get("CHECKPOINT_FREQ", 20))
    IS_FULLY_SUPERVISED: bool = bool(int(os.environ.get("IS_FULLY_SUPERVISED", 0)))

    # Derived / helper methods
    def device(self):
        if self.DEVICE is None:
            self.DEVICE = check_gpu(self.gpu)
        return self.DEVICE

    def make_experiment_name(self):
        self.EXPERIMENT_NAME = self.EXPERIMENT_NAME.split('_s-')[0]
        heads = ''.join('1' if h else '0' for h in self.MODEL_HEADS)
        self.EXPERIMENT_NAME = f"{self.EXPERIMENT_NAME}_s-{self.SEED}_h-{heads}"
        return self.EXPERIMENT_NAME

    def to_env(self):
        """Export selected config keys to os.environ for subprocesses."""
        for k,v in asdict(self).items():
            # set boolean as int
            if isinstance(v, bool):
                v = int(v)
            os.environ[f"{k.upper()}"] = str(v)
            print(f"Set env variable {k.upper()}={v}")

    # after init
    def __post_init__(self):
        self.EXPERIMENT_NAME = self.make_experiment_name()

    def get_summary(self):
        items = [
            ("Experiment", self.EXPERIMENT_NAME),
            ("Seed", self.SEED),
            ("Device", str(self.DEVICE)),
            ("Batch size", self.BATCH_SIZE),
            ("Batch size (transfer)", self.BATCH_SIZE_TRANSFER),
            ("Num workers", self.NUM_WORKERS),
            ("LR", self.LR),
            ("Epochs", self.EPOCHS),
            ("Downstream Epochs", self.DOWNSTREAM_EPOCHS),
            ("Verbose", self.VERBOSE),
        ]
        return items

    def print_summary(self):
        items = self.get_summary()
        
        key_w = max(len(k) for k,_ in items)
        tw = shutil.get_terminal_size().columns
        print("━" * tw)
        print(BOLD + "⚙️ CONFIGURATION SUMMARY" + RESET)
        print("─" * tw)
        for k,v in items:
            print(f"{k.ljust(key_w)} : {v}")
        print("─" * tw)

    def add_cli_args(self, parser):
        """Add base arguments to an argparse parser."""
        parser.add_argument("--experiment_name", type=str, default=self.EXPERIMENT_NAME, help="Base name for the experiment")
        parser.add_argument("--seed", type=int, default=self.SEED, help="Random seed for reproducibility")
        parser.add_argument("--batch_size", type=int, default=self.BATCH_SIZE, help="Batch size for training")
        parser.add_argument("--batch_size_transfer", type=int, default=self.BATCH_SIZE_TRANSFER, help="Batch size for transfer learning")
        parser.add_argument("--num_workers", type=int, default=self.NUM_WORKERS, help="Number of workers for data loading")
        parser.add_argument("--model_heads", type=str, default=''.join(['1' if h else '0' for h in self.MODEL_HEADS]), help="String of 4 chars (1 or 0) indicating which model heads to use")
        parser.add_argument("--use_weighted_loss", type=int, default=int(self.USE_WEIGHTED_LOSS), help="Whether to use dynamic weighted loss (1=True, 0=False)")
        parser.add_argument("--contrastive_loss", type=str, default=self.CONTRASTIVE_LOSS, help="Type of contrastive loss to use (ntxent, barlow, vicreg)")
        parser.add_argument("--lr", type=float, default=self.LR, help="Learning rate")
        parser.add_argument("--epochs", type=int, default=self.EPOCHS, help="Number of training epochs")
        parser.add_argument("--downstream_epochs", type=int, default=self.DOWNSTREAM_EPOCHS, help="Number of epochs for downstream classifier training")
        parser.add_argument("--save_best_only", type=int, default=int(self.SAVE_BEST_ONLY), help="Whether to save only the best model (1=True, 0=False)")
        parser.add_argument("--checkpoint_freq", type=int, default=self.CHECKPOINT_FREQ, help="Frequency (in epochs) to save checkpoints")
        parser.add_argument("--is_fully_supervised", type=int, default=int(self.IS_FULLY_SUPERVISED), help="Whether the model is fully supervised (1=True, 0=False)")
        parser.add_argument("--gpu", type=int, default=self.gpu, help="GPU ID to use (-1 for CPU, -2 for multi-GPU)")
        parser.add_argument("--verbose", type=int, default=int(self.VERBOSE), help="Verbosity level (1=True, 0=False)")
        return parser

    
    def load_config_from_cli(self):
        import argparse
        parser = argparse.ArgumentParser(description="HAR Centralized Training Configuration")
        parser = self.add_cli_args(parser)
        args = parser.parse_args()
        self.update_from_args(args)

    def update_from_args(self, args):
        """Update configuration from command line arguments."""

        if args.experiment_name != self.EXPERIMENT_NAME:
            self.EXPERIMENT_NAME = self.make_experiment_name(args.experiment_name)

        if len(args.model_heads) != self.NUM_HEADS or any(c not in '01' for c in args.model_heads):
            raise ValueError(f"model_heads must be a string of {self.NUM_HEADS} characters (1 or 0), e.g. '1101'")
        self.MODEL_HEADS = [c == '1' for c in args.model_heads]
        
        self.USE_WEIGHTED_LOSS = bool(args.use_weighted_loss)
        if sum(self.MODEL_HEADS) == 1 and self.USE_WEIGHTED_LOSS:
            print("⚠️  Only one task head is active; disabling use_weighted_loss.")
            self.USE_WEIGHTED_LOSS = False

        self.SEED = args.seed
        self.BATCH_SIZE = args.batch_size
        self.BATCH_SIZE_TRANSFER = args.batch_size_transfer
        self.NUM_WORKERS = args.num_workers
        self.MODEL_HEADS = [int(h) for h in args.model_heads]
        self.CONTRASTIVE_LOSS = args.contrastive_loss
        self.LR = args.lr
        self.EPOCHS = args.epochs
        self.DOWNSTREAM_EPOCHS = args.downstream_epochs
        self.SAVE_BEST_ONLY = bool(args.save_best_only)
        self.CHECKPOINT_FREQ = args.checkpoint_freq
        self.IS_FULLY_SUPERVISED = bool(args.is_fully_supervised)
        self.gpu = args.gpu
        self.VERBOSE = bool(args.verbose)

        self.DEVICE = check_gpu(self.gpu)

        self.to_env()

        # set all as os variables
        # for k,v in asdict(self).items():
        #     os.environ[f"{k.upper()}"] = str(v)
        #     print(f"Set environment variable {k.upper()}={v}")




# federated config extends this config class
# 

# class ir_config(Config):
#     def __init__(self):
#         super().__init__()
#         self.EXPERIMENT_NAME = "IR_centralized"
#         self.EXPERIMENT_NAME = super().make_experiment_name(self.EXPERIMENT_NAME)

#         ROOT = Path(__file__).resolve().parents[1]
#         CUR_DIR = Path(__file__).resolve().parent
#         self.DATASET_DIR: Path = field(default_factory=lambda: Path(os.environ.get("DATASET_DIR", ROOT / "data" / "STL10")))
#         self.OUTPUT_DIR: Path = field(default_factory=lambda: Path(os.environ.get("OUTPUT_DIR", CUR_DIR / "outputs")))
#         self.MODELS_DIR: Path = field(default_factory=lambda: Path(os.environ.get("MODELS_DIR", CUR_DIR / "outputs" / "models")))
#         self.RESULTS_DIR: Path = field(default_factory=lambda: Path(os.environ.get("RESULTS_DIR", CUR_DIR / "outputs" / "results")))

#         # STL-10 specific parameters
#         self.STL10_NUM_CLASSES = 10
#         self.ENCODER_EPOCH = 200

#         self.NUM_HEADS = 3
#         self.MODEL_HEADS = [True, True, True]  # contrastive, rotation, mask

#         self.BATCH_SIZE = 512
#         self.BATCH_SIZE_TRANSFER = 512


#     def add_cli_args(self, parser):
#         """Add command line arguments specific to this configuration."""
#         parser = super().add_cli_args(parser)
#         parser.add_argument("--stl10_num_classes", type=int, default=self.STL10_NUM_CLASSES)
#         parser.add_argument("--encoder_epoch", type=int, default=self.ENCODER_EPOCH)
#         return parser

#     # expand load from cli
#     def load_config_from_cli(self):
#         import argparse
#         parser = argparse.ArgumentParser(description="IR Centralized Training Configuration")
#         parser = self.add_cli_args(parser)
#         args = parser.parse_args()
#         self.update_from_args(args)

#     def update_from_args(self, args):
#         super().update_from_args(args)
#         self.STL10_NUM_CLASSES = args.stl10_num_classes
#         self.ENCODER_EPOCH = args.encoder_epoch

