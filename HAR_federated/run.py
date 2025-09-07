import os
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "true")
import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # already set; ignore
    pass

import subprocess
import sys
import time
from pathlib import Path
from fed_config import federated_config
from data_processing.HAR_precompute_augs import generate_client_dataloaders

def main():

    cfg = federated_config()
    # Load configuration
    cfg.load_config_from_cli()

    # Print summary if verbose
    if cfg.VERBOSE:
        cfg.print_summary()

    # Set k_folds
    k_folds = 5
    n_clients = 16  # Assuming this is defined in the config

    script_dir = Path(__file__).resolve().parent
    server_script = script_dir / "server.py"
    client_script = script_dir / "client.py"

    for fold in range(k_folds):
        print(f"Running fold {fold}")

        # Generate the dataset
        generate_client_dataloaders(cfg, is_save_val=True)

        # Start the server
        print("Starting server...")
        server_process = subprocess.Popen(
            [sys.executable, str(server_script), "--fold", str(fold)],
            cwd=str(script_dir),
            env=os.environ,
            stdout=None,
            stderr=None
        )
        time.sleep(4)  # Wait for the server to start

        # Start the clients
        client_processes = []
        for i in range(n_clients):
            print(f"Starting client ID {i}")
            client_process = subprocess.Popen(
                [sys.executable, str(client_script), "--id", str(i), "--fold", str(fold)],
                cwd=str(script_dir),
                env=os.environ,
                stdout=None,
                stderr=None
            )
            client_processes.append(client_process)

        # Wait for all clients to finish
        for client_process in client_processes:
            client_process.wait()

        # Terminate the server
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()