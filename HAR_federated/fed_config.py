from HAR_centralized.har_config import har_config
import os
from dataclasses import dataclass, field
import os


@dataclass
class federated_config(har_config):
    def __init__(self):
        super().__init__()
        self.EXPERIMENT_NAME = "har_federated"
        self.EXPERIMENT_NAME = super().make_experiment_name()

        self.LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 2))
        self.N_ROUNDS = int(os.environ.get("N_ROUNDS", 100))  # number of global rounds
        self.N_CLIENTS = int(os.environ.get("N_CLIENTS", 16))  # number of clients
        self.PORT = '8018'
        self.IP = '0.0.0.0'

    def add_cli_args(self, parser):
        parser = super().add_cli_args(parser)
        parser.add_argument("--local_epochs", type=int, default=self.LOCAL_EPOCHS)
        parser.add_argument("--n_rounds", type=int, default=self.N_ROUNDS)
        parser.add_argument("--n_clients", type=int, default=self.N_CLIENTS)
        parser.add_argument("--port", type=str, default=self.PORT)
        parser.add_argument("--ip", type=str, default=self.IP)
        return parser
    
    def load_config_from_cli(self):
        import argparse
        parser = argparse.ArgumentParser(description="HAR Federated Training Configuration")
        parser = self.add_cli_args(parser)
        args = parser.parse_args()
        self.update_from_args(args)

    def update_from_args(self, args):
        super().update_from_args(args)
        self.LOCAL_EPOCHS = args.local_epochs
        self.N_ROUNDS = args.n_rounds
        self.N_CLIENTS = args.n_clients
        self.PORT = args.port
        self.IP = args.ip

        self.to_env()

    def get_summary(self):
        summary = super().get_summary()
        summary.extend([
            ("Local Epochs", self.LOCAL_EPOCHS),
            ("Global Rounds", self.N_ROUNDS),
            ("Number of Clients", self.N_CLIENTS),
            ("Server Address", f"{self.IP}:{self.PORT}"),
        ])
        return summary