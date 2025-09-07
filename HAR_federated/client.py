"""
This code creates a Flower client that can be used to train a model locally and share the updated 
model with the server. When it is started, it connects to the Flower server and waits for instructions.
If the server sends a model, the client trains the model locally and sends back the updated model.
If abilitated, at the end of the training the client evaluates the last model, and plots the 
metrics during the training.

This is code is set to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
have this code running.
"""

import argparse
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import flwr as fl

import sys
import os

from common.losses import set_up_loss_functions
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(current_dir, '../../centralized_learning/transformations'))
from fed_config import federated_config
import common.utils as utils
from common.models import MultiTaskModel
from common.encoders import FCN, get_encoder_from_name



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
        model,
        loss_weight_params,
        client_id,
        device,
        cfg=federated_config()
        ):
        self.model = model
        self.loss_weight_params = loss_weight_params if cfg.USE_WEIGHTED_LOSS else None
        self.client_id = client_id # [0,cfg.n_clients]
        self.device = device
        self.drifting_log = []
        self.cfg = cfg
        self.task_heads = cfg.MODEL_HEADS

        # plot
        self.metrics = {
            "rounds": [],
            "loss": [],
            "accuracy": []
        }

        self.cur_train_loader = self.load_current_data(0, train=True)
        self.cur_val_loader = self.load_current_data(0, train=False)
        
        # if cfg.training_drifting:
        #     drifting_log = np.load(f'../../data/cur_datasets/drifting_log.npy', allow_pickle=True).item()
        #     self.drifting_log = drifting_log[self.client_id]

    def load_current_data(self,
                          cur_round,
                          train=True) -> DataLoader:

        # load dataloader from pt file
        loader = os.path.join(self.cfg.DATASET_DIR, f'client_dataloaders/train_loader_client_{self.client_id}.pt')
        # loader = f'../../data/cur_datasets/train_loader_client_{self.client_id}.pt'
        # if os.path.exists(loader):
        
        d = torch.load(loader, weights_only=False)
        print(f'Loading data from {loader} for client {self.client_id}...')
        return d
        

        # load raw data
        if not cfg.training_drifting:
            cur_data = np.load(f'../../data/cur_datasets/client_{self.client_id}.npy', allow_pickle=True).item()
        else:
            load_index = max([index for index in self.drifting_log if index <= cur_round], default=0)
            cur_data = np.load(f'../../data/cur_datasets/client_{self.client_id}_round_{load_index}.npy', allow_pickle=True).item()

        cur_features = cur_data['train_features'] if not cfg.training_drifting else cur_data['features']
        cur_features = cur_features.unsqueeze(1) if utils.get_in_channels() == 1 else cur_features

        cur_labels = cur_data['train_labels'] if not cfg.training_drifting else cur_data['labels']

        # Split the data into training and testing subsets
        train_features, val_features, train_labels, val_labels = train_test_split(
            cur_features, cur_labels, test_size=cfg.client_eval_ratio, random_state=cfg.random_seed
        )
        
        # reduce client data
        if cfg.n_samples_clients > 0:
            train_features = train_features[:cfg.n_samples_clients]
            train_labels = train_labels[:cfg.n_samples_clients]
        
        # print(f"client shape data: {train_features.shape}, {val_features.shape}")

        if train:
            train_dataset = models.CombinedDataset(train_features, train_labels, transform=None)
            return DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        else:
            val_dataset = models.CombinedDataset(val_features, val_labels, transform=None)
            return DataLoader(val_dataset, batch_size=cfg.test_batch_size, shuffle=False)

    # override
    def get_parameters(self, config):
        """Return model parameters and loss weight parameters as a list."""
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        if self.cfg.USE_WEIGHTED_LOSS:
            loss_weights = [val.detach().cpu().numpy() for _, val in self.loss_weight_params.items()]
            return params + loss_weights
        else:
            return params

    # override
    def set_parameters(self, parameters):
        # get the params and loss weights from the parameters
        num_params = len(self.model.state_dict())
        params = parameters[:num_params]
        loss_weights = parameters[num_params:]
        # set the model parameters
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        # set the loss weight parameters

        if self.cfg.USE_WEIGHTED_LOSS:
            loss_weights_dict = zip(self.loss_weight_params.keys(), loss_weights)
            loss_weights_state_dict = OrderedDict({
                k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=True)
                for k, v in loss_weights_dict
            })
            self.loss_weight_params = loss_weights_state_dict
        

    # override
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        cur_round = config["current_round"]
        # cur_train_loader = self.load_current_data(cur_round, train=True)
        cur_train_loader = self.cur_train_loader

        # if cur_round > -1:
        #     # print in blue the loss_weight_params
        #     loss_weights = [val.detach().cpu().numpy() for _, val in self.loss_weight_params.items()]
        #     print(f"\033[1m\033[94mClient {self.client_id} - Round {cur_round} - Loss Weights: {loss_weights}\033[0m")
        
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + (list(self.loss_weight_params.values()) if self.cfg.USE_WEIGHTED_LOSS else []), lr=self.cfg.LR
        )
        # torch.optim.Adam([{"params": self.model.parameters()},
        #                   {"params": list(self.loss_weight_params.values())}], lr=cfg.lr),
        criterion_classification, criterion_reconstruction, criterion_contrastive, criterion_features = set_up_loss_functions(self.cfg)

        # Train the model
        for epoch in range(config["local_epochs"]):
            self.model.train()
            self.model.to(self.device)

            self.model.run_epoch(
                dataloader=cur_train_loader,
                optimizer=optimizer,
                a_dict=self.loss_weight_params,
                cfg=self.cfg,
                criterion_classification=criterion_classification,
                criterion_reconstruction=criterion_reconstruction,
                criterion_contrastive=criterion_contrastive,
                criterion_features=criterion_features,
                is_training=True,
                disable_tqdm=True
            )
        
        # print the updated loss weights
        if self.cfg.USE_WEIGHTED_LOSS:
            loss_weights = [val.detach().cpu().numpy() for _, val in self.loss_weight_params.items()]
            print(f"\033[1m\033[94mCLIENT {self.client_id} - Round {cur_round} - Updated Loss Weights: {loss_weights}\033[0m")

        return self.get_parameters(config), len(cur_train_loader.dataset), {}
    
    # override
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        cur_round = config["current_round"]
        # cur_val_loader = self.load_current_data(cur_round, train=False)
        cur_val_loader = self.cur_val_loader

        # loss = evaluate_model(
        #     model=self.model,
        #     device=self.device,
        #     test_loader=cur_val_loader,
        #     loss_weights=self.loss_weight_params if cfg.USE_WEIGHTED_LOSS else None,
        #     task_heads=self.task_heads,
        #     client_id=self.client_id,
        # )

        loss=1

        # loss_trad, accuracy_trad, f1_score_trad = models.simple_test(self.model, self.device, cur_val_loader)

        # # quick check results and save for plot
        print(f"Client {self.client_id} - Round {cur_round} - Loss: {loss:.4f}, Accuracy: {loss:.4f}")
        self.metrics["rounds"].append(cur_round)
        self.metrics["loss"].append(loss)
        self.metrics["accuracy"].append(loss)
        np.save(f"results/{self.cfg.SEED}/{self.cfg.EXPERIMENT_NAME}/client_{self.client_id}_metrics.npy", self.metrics)

        return float(loss), len(cur_val_loader), {
            "accuracy": float(loss),  # Placeholder, as we are not calculating accuracy here
            "f1_score": float(loss)   # Placeholder, as we are not calculating F1 score here
        }

# main
def main() -> None:

    cfg = federated_config()

    # Get client id
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--id",
        type=int,
        choices=range(0, cfg.N_CLIENTS),
        required=True,
        help="Specifies the artificial data partition",
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=False,
        default=0,
        help="Specifies the fold number of the cross-validation",
    )
    args = parser.parse_args()

    # Load device, model and data
    utils.set_seed(cfg.SEED)
    device = utils.check_gpu(cfg.gpu, client_id=args.id)

    encoder = get_encoder_from_name(cfg)
    model = MultiTaskModel(9, 17, feature_dim=561, task_heads=cfg.MODEL_HEADS, encoder=encoder)
    # load model from file 
    # model_path = os.path.join(current_dir, f"checkpoints/0/FCN/UCI_HAR/{cfg.strategy}/feature_skew_strict_fold_{args.fold}_n_clients_{cfg.n_clients}_round_200.pth")
    # model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    # model = model.to(device)



    # in_channels = utils.get_in_channels()
    # model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
    #                                       input_size=cfg.input_size).to(device)

    a_dict = {}
    a1 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
    a_dict['a1'] = a1
    a2 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
    a_dict['a2'] = a2
    a3 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
    a_dict['a3'] = a3
    a4 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
    a_dict['a4'] = a4

    if not cfg.USE_WEIGHTED_LOSS:
        a_dict = {}

    # Start Flower client
    client = FlowerClient(model=model,
                          loss_weight_params=a_dict,
                          client_id=args.id,
                          device=device
                          ).to_client()

    fl.client.start_client(server_address=f"{cfg.IP}:{cfg.PORT}", client=client) # local host

if __name__ == "__main__":
    print("-" * 20)
    main()

