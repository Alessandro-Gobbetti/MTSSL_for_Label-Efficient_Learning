from dataclasses import field
from HAR_centralized.config import Config
from pathlib import Path
from dataclasses import dataclass, field
import os

ROOT = Path(__file__).resolve().parents[1]
CUR_DIR = Path(__file__).resolve().parent

@dataclass
class ir_config(Config):
    def __init__(self):
        super().__init__()
        self.EXPERIMENT_NAME = "IR_centralized"
        self.EXPERIMENT_NAME = super().make_experiment_name()

        ROOT = Path(__file__).resolve().parents[1]
        CUR_DIR = Path(__file__).resolve().parent
        self.DATASET_DIR: Path = Path(os.environ.get("DATASET_DIR", ROOT / "data" / "STL10"))
        self.OUTPUT_DIR: Path = Path(os.environ.get("OUTPUT_DIR", CUR_DIR / "outputs"))
        self.MODELS_DIR: Path = Path(os.environ.get("MODELS_DIR", CUR_DIR / "outputs" / "models"))
        self.RESULTS_DIR: Path = Path(os.environ.get("RESULTS_DIR", CUR_DIR / "outputs" / "results"))

        # Ensure output directories exist
        for _d in [self.OUTPUT_DIR, self.MODELS_DIR, self.RESULTS_DIR]:
            _d.mkdir(parents=True, exist_ok=True)

        # STL-10 specific parameters
        self.STL10_NUM_CLASSES = 10
        self.ENCODER_EPOCH = 200

        self.NUM_HEADS = 3
        self.MODEL_HEADS = [True, True, True]  # contrastive, rotation, mask

        self.BATCH_SIZE = 512
        self.BATCH_SIZE_TRANSFER = 512


    def add_cli_args(self, parser):
        """Add command line arguments specific to this configuration."""
        parser = super().add_cli_args(parser)
        parser.add_argument("--stl10_num_classes", type=int, default=self.STL10_NUM_CLASSES)
        parser.add_argument("--encoder_epoch", type=int, default=self.ENCODER_EPOCH)
        return parser

    # expand load from cli
    def load_config_from_cli(self):
        import argparse
        parser = argparse.ArgumentParser(description="IR Centralized Training Configuration")
        parser = self.add_cli_args(parser)
        args = parser.parse_args()
        self.update_from_args(args)

    def update_from_args(self, args):
        super().update_from_args(args)
        self.STL10_NUM_CLASSES = args.stl10_num_classes
        self.ENCODER_EPOCH = args.encoder_epoch

        super().make_experiment_name()
        self.to_env()

    def get_summary(self):
        summary = super().get_summary()
        summary.extend([
            ("stl10_num_classes", self.STL10_NUM_CLASSES),
            ("encoder_epoch", self.ENCODER_EPOCH),
        ])
        return summary