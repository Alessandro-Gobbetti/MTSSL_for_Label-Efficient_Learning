from HAR_centralized.config import Config
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
import os


@dataclass
class har_config(Config):
    def __init__(self):
        super().__init__()
        self.EXPERIMENT_NAME = "HAR_centralized"
        self.EXPERIMENT_NAME = super().make_experiment_name()


        ROOT = Path(__file__).resolve().parents[1]
        CUR_DIR = Path(__file__).resolve().parent
        self.DATASET_DIR: Path = Path(os.environ.get("DATASET_DIR", ROOT / "data" / "UCI_HAR"))
        self.OUTPUT_DIR: Path = Path(os.environ.get("OUTPUT_DIR", CUR_DIR / "outputs"))
        self.MODELS_DIR: Path = Path(os.environ.get("MODELS_DIR", CUR_DIR / "outputs" / "models"))
        self.RESULTS_DIR: Path = Path(os.environ.get("RESULTS_DIR", CUR_DIR / "outputs" / "results"))

        for _d in [self.OUTPUT_DIR, self.MODELS_DIR, self.RESULTS_DIR]:
            _d.mkdir(parents=True, exist_ok=True)

        # UCI HAR specific parameters
        self.UCIHAR_NUM_CHANNELS = 9
        self.UCIHAR_SEQ_LEN = 128
        self.UCIHAR_NUM_CLASSES = 6
        self.UCIHAR_FEATURE_DIM = 561

        # Model architecture
        self.ENCODER_NAME = os.environ.get("ENCODER_NAME", "FCN")
        self.OUT_DIM = int(os.environ.get("OUT_DIM", 128))
        self.NUM_HEADS = 4

        self.NOISE_RATIO = 0.5

    def add_cli_args(self, parser):
        """Add command line arguments specific to this configuration."""
        parser = super().add_cli_args(parser)
        parser.add_argument("--out_dim", type=int, default=self.OUT_DIM)
        parser.add_argument("--noise_ratio", type=float, default=self.NOISE_RATIO, help="Noise ratio for data augmentation in the range [0, 1]")
        return parser

    # expand load from cli
    def load_config_from_cli(self):
        import argparse
        parser = argparse.ArgumentParser(description="HAR Centralized Training Configuration")
        parser = self.add_cli_args(parser)
        args = parser.parse_args()
        self.update_from_args(args)

    def update_from_args(self, args):
        super().update_from_args(args)
        self.OUT_DIM = args.out_dim
        self.NOISE_RATIO = args.noise_ratio
        self.to_env()

    def get_summary(self):
        items = super().get_summary()
        # append UCI HAR specific parameters
        items.extend([
            ("Encoder", self.ENCODER_NAME),
            ("Embedding size", self.OUT_DIM),
        ])
        return items