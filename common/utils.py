import random
import numpy as np
import torch
import os


BOLD = '\033[1m'
RESET = '\033[0m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = "\033[96m"
MAGENTA = "\033[95m"


def set_seed(seed: int, verbose: bool = True):
    """
    Set the random seed for reproducibility across various libraries and frameworks.
    """
    random.seed(seed)                          # Python random module
    np.random.seed(seed)                       # NumPy
    torch.manual_seed(seed)                    # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)           # PyTorch GPU (single)
        torch.cuda.manual_seed_all(seed)       # For multi-GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)            # PyTorch MPS (Apple Silicon)

    torch.backends.cudnn.deterministic = True  # Makes CUDA deterministic (may impact speed)
    torch.backends.cudnn.benchmark = False     # Disables optimization for non-deterministic algorithms

    # Set PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)

    if verbose:
        print(f"[Seed set to {seed}]")

def check_gpu(gpu, client_id:int = 0):
    """
    Selects and returns the appropriate torch device (CPU, CUDA, or MPS) based on the provided GPU index and client ID.

    Args:
        gpu (int): GPU index to use. 
            - If -1, selects CPU.
            - If -2, selects a GPU based on client_id (for multi-GPU setups).
            - Otherwise, selects the specified GPU index if CUDA is available.
        client_id (int, optional): Client identifier used to select a GPU when gpu is -2. Defaults to 0.

    Returns:
        torch.device: The selected device (CPU, CUDA, or MPS).

    Notes:
        - If CUDA is not available but MPS is, selects MPS device.
        - Falls back to CPU if neither CUDA nor MPS is available.
    """
    if gpu == -1:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        if gpu == -2: # multiple gpu
            # assert client_id >=0, "client_id must be passed to select the respective GPU"
            n_total_gpus = torch.cuda.device_count()
            device = torch.device('cuda:' + str(int(client_id % n_total_gpus)))
        else:
            device = torch.device('cuda:' + str(gpu))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device