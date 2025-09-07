"""
CFL implementation of fedavg, server side.

Code to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
run the appopriate client code (client.py).
"""

from typing import List, Tuple, Union, Optional, Dict
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from logging import WARNING
from collections import OrderedDict
import json
import time
from functools import reduce

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(current_dir, '../../centralized_learning/transformations'))
from fed_config import federated_config
import common.utils as utils
from common.models import MultiTaskModel
from common.encoders import FCN

import flwr as fl
from flwr.common import Parameters, Scalar, Metrics
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
)
FOLD = None
print("s"*20)


# Config_client
def fit_config(
        server_round: int,
        cfg=federated_config()
    ) -> Dict[str, Scalar]:
    """
        Generate training configuration dict for each round.
    """
    config = {
        "current_round": server_round,
        "local_epochs": cfg.LOCAL_EPOCHS,
        "tot_rounds": cfg.N_ROUNDS,
        "min_latent_space": 0,
    }
    return config

# Custom weighted average function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    # validities = [num_examples * m["validity"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

# Custom strategy to save model after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model  # used for saving checkpoints
        self.path = path # saving model path

    # Override configure_fit method to select the fit clients
    # def configure_fit(self, server_round, parameters, client_manager, train_clients_ids=None):
    #     """Configure fit for the server round."""

    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        ################################################################################
        # Federated averaging aggregation
        ################################################################################
        # Federated averaging - from traditional code
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_parameters_global = ndarrays_to_parameters(aggregate(weights_results))   # Global aggregation - traditional - no clustering
        
        # Aggregate custom metrics if aggregation fn was provided   NO FIT METRICS AGGREGATION FN PROVIDED - SKIPPED FOR NOW
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
            
        ################################################################################
        # Save model
        ################################################################################
        if aggregated_parameters_global is not None:

            print(f"Saving round {server_round} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters_global)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            global FOLD
            torch.save(self.model.state_dict(), f"checkpoints/{self.path}/{cfg.EXPERIMENT_NAME}_fold_{FOLD}_n_clients_{cfg.n_clients}_round_{server_round}.pth")
        
        return aggregated_parameters_global, aggregated_metrics

def main() -> None:
    # Get arguments
    parser = argparse.ArgumentParser(description='FedAvg - Server')
    parser.add_argument('--fold', type=int, default=0, help='Fold number of the cross-validation')
    args = parser.parse_args()

    
    cfg = federated_config()
    utils.set_seed(cfg.SEED)

    exp_path = f"{cfg.SEED}/{cfg.EXPERIMENT_NAME}"
    os.makedirs(f"results/{exp_path}", exist_ok=True)
    os.makedirs(f"histories/{exp_path}", exist_ok=True)
    os.makedirs(f"checkpoints/{exp_path}", exist_ok=True)
    os.makedirs(f"images/{exp_path}", exist_ok=True)
    # in_channels = utils.get_in_channels()
    task_heads = cfg.MODEL_HEADS
    print(f"Task heads: {task_heads}")

    encoder = FCN(9, out_size=1)
    model = MultiTaskModel(9, 17, feature_dim=561, task_heads=cfg.MODEL_HEADS, encoder=encoder)
    model.to(cfg.DEVICE)

    # save the fold as global variable
    global FOLD
    FOLD = args.fold

    # Define strategy
    strategy = SaveModelStrategy(
        # self defined
        model=model,
        path=exp_path,
        # super
        min_fit_clients=cfg.N_CLIENTS, # always all training
        min_evaluate_clients=cfg.N_CLIENTS, # always all evaluating
        min_available_clients=cfg.N_CLIENTS, # always all available
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
    )

    # Start Flower server and (finish all training and evaluation)
    history = fl.server.start_server(
        server_address=f"{cfg.IP}:{cfg.PORT}",   # 0.0.0.0 listens to all available interfaces
        config=fl.server.ServerConfig(num_rounds=cfg.N_ROUNDS),
        strategy=strategy,
    )

    # Convert history to list
    loss = [k[1] for k in history.losses_distributed]
    accuracy = [k[1] for k in history.metrics_distributed['accuracy']]

    # Save loss and accuracy to a file
    print(f"Saving metrics to as .json in histories folder...")
    with open(f'histories/{exp_path}/distributed_metrics_{args.fold}.json', 'w') as f:
        json.dump({'loss': loss, 'accuracy': accuracy}, f)

    
    time.sleep(1)
    
if __name__ == "__main__":
    main()
