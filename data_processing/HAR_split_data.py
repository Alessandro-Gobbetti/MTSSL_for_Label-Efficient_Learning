###########################################################################################################################
# Author: Alessandro Gobbetti
# Reference: Master's Thesis
#   Title: "Multi-task Self-Supervised Methods for Label-Efficient Learning"
#   Subtitle: "Combining Contrastive and Pretext-Based Learning for Effective Encoders from Unlabeled and Federated Data in Human Activity Recognition and Beyond"
#   Università della Svizzera italiana (USI), 2025
# 
# Repository: https://github.com/Alessandro-Gobbetti/MTSSL_for_Label-Efficient_Learning
#
# Description:
#   This file contains functions to load and split the UCI HAR dataset for training and evaluating machine learning models.
#   The splitting strategy is client-based, ensuring that data from the same client does not appear in both training and validation/test sets.
#   This file also defines scenarios for low-labeled data regimes.
#   Specifically, the training/validation splitting strategy remains the same across the main training and transfer learning tasks.
#   Suggestions for different splitting strategies are welcome (e.g., transfer learning as user-specific fine-tuning).
# 
# Note:
#   This file is part of the codebase accompanying my master's thesis.
#   Please cite the thesis when reusing this work.

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import shutil
from typing import Literal
from common.utils import BOLD, RESET

# data dir
pwd = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(pwd, '../data/UCI_HAR/')


###########################################################################################################################
# dataset class

class dataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples.astype(np.float32)  # set as float
        self.labels = labels.astype(np.int64)  # set as int

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        # apply the transformation
        return sample, target

    def __len__(self):
        return len(self.samples)

class dataset_features(Dataset):
    def __init__(self, samples, features, labels, clients=None):
        self.samples = samples.astype(np.float32)  # set as float
        self.features = features.astype(np.float32)  # set as float
        self.labels = labels.astype(np.int64)  # set as int
        self.clients = clients
        if self.clients is not None:
            self.clients = self.clients.to_numpy().astype(np.int64)
    
    def __getitem__(self, index):
        sample, feature, target = self.samples[index], self.features[index], self.labels[index]
        if self.clients is not None:
            return sample, feature, target, self.clients[index]

        return sample, feature, target

    def __len__(self):
        return len(self.samples)

###########################################################################################################################
# functions to load the data

def load_raw_data(train_test):
    """
    Load the UCI HAR dataset 
    :param train_test: 'train' or 'test' to load the corresponding dataset
    :return: the data, labels, and clients
    """

    label_names = pd.read_csv(
        f'{DATA_DIR}/activity_labels.txt', header=None, sep=' ', index_col=0)
    label_names.columns = ['label_name']
    label_names = label_names['label_name'].to_dict().values()

    if train_test == 'train':
        data_dir = f'{DATA_DIR}/'
        data_dir = os.path.join(data_dir, 'train/Inertial Signals')
        labels_path = f'{DATA_DIR}/train/y_train.txt'
        clients_path = f'{DATA_DIR}/train/subject_train.txt'
    elif train_test == 'test':
        data_dir = f'{DATA_DIR}/'
        data_dir = os.path.join(data_dir, 'test/Inertial Signals')
        labels_path = f'{DATA_DIR}/test/y_test.txt'
        clients_path = f'{DATA_DIR}/test/subject_test.txt'
    else:
        raise ValueError("train_test must be either 'train' or 'test'")

    # load clients
    data_files = os.listdir(data_dir)
    data_files = [os.path.join(data_dir, f) for f in data_files]
    # sort the files
    data_files = sorted(data_files)
    # load the data
    data = [np.loadtxt(f) for f in data_files]
    data = np.array(data)
    data = np.swapaxes(data, 0, 1)
    data = np.swapaxes(data, 1, 2)

    # load labels
    labels = pd.read_csv(labels_path, header=None)
    labels.columns = ['label']
    labels = np.array(labels).reshape(-1) - 1 # labels start from 1
    # load clients
    clients = pd.read_csv(clients_path, header=None)
    clients.columns = ['client']
    clients = clients['client']

    return data, labels, clients, list(label_names)




def split_train_val_clients(clients, train_val_ratio=0.8, seed=None):
    """
    Splits the clients into train and validation sets.
    :param clients: the clients to split
    :param train_val_ratio: the ratio of training to validation clients
    :return: the train and validation clients
    """
    # pick 80% of the clients for training and 20% for validation
    n_clients = len(clients)
    n_train_clients = int(n_clients * train_val_ratio)
    
    rng = np.random.default_rng(seed)  
    shuffled_clients = rng.permutation(clients)
    train_clients = shuffled_clients[:n_train_clients]
    val_clients = shuffled_clients[n_train_clients:]
    
    return train_clients, val_clients


def split_data_few_shots_all_clients(data, data_test, labels, labels_test, clients_train, clients_test, n_samples_per_activity_per_client=2, train_val_ratio=0.8, seed=None, batch_size=64, batch_size_transfer=32, verbose=True):
    """
    Splits the dataset into train, validation, and test sets for learning robust embeddings and transfer learning.

    For transfer learning, the data is created by selecting a specified number of samples (`n_samples_per_activity_per_client`) 
    for each activity from each client.

    :param data_train: the training data
    :param data_test: the testing data
    :param labels_train: the training labels
    :param labels_test: the testing labels
    :param clients_train: the training clients
    :param clients_test: the testing clients
    :param n_samples_per_activity_per_client: the number of samples per activity per client or 'all' to use all samples
    :return: Train, validation, and test DataLoaders for embedding learning and transfer learning tasks.
    """

    clients_training = np.unique(clients_train)
    train_clients, val_clients = split_train_val_clients(clients_training, train_val_ratio=train_val_ratio, seed=seed)
    
    # split the data into train and validation
    data_train = data[clients_train.isin(train_clients)]
    labels_train = labels[clients_train.isin(train_clients)]
    data_val = data[clients_train.isin(val_clients)]
    labels_val = labels[clients_train.isin(val_clients)]

    # transfer learning data
    activities = np.unique(labels_train)
    # select n_samples_per_activity_per_client samples for each activity from each client
    data_transfer = []
    labels_transfer = []
    clients_transfer = []
    
    for client in clients_training:
        # print in red
        # get the data for the client
        client_data = data[clients_train == client]
        client_labels = labels[clients_train == client]

        # for each activity, select n_samples_per_activity_per_client samples
        for activity in activities:
            # get the data for the activity
            activity_data = client_data[client_labels == activity]
            # select random n_samples_per_activity_per_client samples
            if n_samples_per_activity_per_client != 'all':
                rng = np.random.default_rng(seed)  
                activity_indices = rng.choice(activity_data.shape[0], n_samples_per_activity_per_client, replace=False)
                activity_data = activity_data[activity_indices]

            n_samples = activity_data.shape[0]
            # add the data to the transfer learning data
            data_transfer.extend(activity_data)
            labels_transfer.extend([activity] * n_samples)
            clients_transfer.extend([client] * n_samples)
                
    # split transfer data into train and validation
    data_transfer = np.array(data_transfer)
    labels_transfer = np.array(labels_transfer)
    # clients_trainsfer = np.array(clients_trainsfer)
    clients_transfer = pd.Series(clients_transfer)
    
    data_transfer_train = data_transfer[clients_transfer.isin(train_clients)]
    labels_transfer_train = labels_transfer[clients_transfer.isin(train_clients)]
    data_transfer_val = data_transfer[clients_transfer.isin(val_clients)]
    labels_transfer_val = labels_transfer[clients_transfer.isin(val_clients)]

    # create the data loaders
    train_dataset = dataset(data_train, labels_train)
    val_dataset = dataset(data_val, labels_val)
    test_dataset = dataset(data_test, labels_test)
    transfer_train_dataset = dataset(data_transfer_train, labels_transfer_train)
    transfer_val_dataset = dataset(data_transfer_val, labels_transfer_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    transfer_train_loader = DataLoader(transfer_train_dataset, batch_size=batch_size_transfer, shuffle=True)
    transfer_val_loader = DataLoader(transfer_val_dataset, batch_size=batch_size_transfer, shuffle=False)


    if verbose:

        terminal_width = shutil.get_terminal_size().columns
        print("━" * terminal_width)
        print(f"{BOLD}📊 Dataset Summary{RESET}")
        print(f"{'Split':<25}{'Data Shape':<20}{'Labels Shape':<15}")
        print("─" * terminal_width)
        print(f"{'Train':<25}{str(data_train.shape):<20}{str(labels_train.shape):<15}")
        print(f"{'Validation':<25}{str(data_val.shape):<20}{str(labels_val.shape):<15}")
        print(f"{'Test':<25}{str(data_test.shape):<20}{str(labels_test.shape):<15}")
        print(f"{'Transfer':<25}{str(data_transfer.shape):<20}{str(labels_transfer.shape):<15}")
        print(f"{' └─ Transfer Train':<25}{str(data_transfer_train.shape):<20}{str(labels_transfer_train.shape):<15}")
        print(f"{' └─ Transfer Val':<25}{str(data_transfer_val.shape):<20}{str(labels_transfer_val.shape):<15}")
        print("━" * terminal_width)

        print(f"{BOLD}👥 Client Distribution{RESET}")
        print(f"{'Group':<25}{'Count':<6}{'Client IDs':<15}")
        unique_clients_test = np.sort(np.unique(clients_test))
        print("─" * terminal_width)
        print(f"{'Total clients':<25}{len(clients_training)+len(unique_clients_test):<6}")
        print(f"{' └─ Train/Val clients':<25}{len(clients_training):<6}{np.sort(clients_training)}")
        print(f"{' └─ Test clients':<25}{len(unique_clients_test):<6}{unique_clients_test}")
        print(f"{'Train clients':<25}{len(train_clients):<6}{np.sort(train_clients)}")
        print(f"{'Val clients':<25}{len(val_clients):<6}{np.sort(val_clients)}")
        unique_transfer_clients = np.sort(np.unique(clients_transfer[clients_transfer.isin(train_clients)]))
        unique_transfer_val_clients = np.sort(np.unique(clients_transfer[clients_transfer.isin(val_clients)]))
        print(f"{'Transfer Train clients':<25}{len(unique_transfer_clients):<6}{unique_transfer_clients}")
        print(f"{'Transfer Val clients':<25}{len(unique_transfer_val_clients):<6}{unique_transfer_val_clients}")
        print("─" * terminal_width)

    return train_loader, val_loader, test_loader, transfer_train_loader, transfer_val_loader

def split_data_few_shots_all_clients_features(features, data, data_test, labels, labels_test, clients_train, clients_test, n_samples_per_activity_per_client=2, train_val_ratio=0.8, seed=None, batch_size=64, batch_size_transfer=32, include_client_ids=False, verbose=True):
    train_loader, val_loader, test_loader, transfer_train_loader, transfer_val_loader = split_data_few_shots_all_clients(data, data_test, labels, labels_test, clients_train, clients_test, n_samples_per_activity_per_client=n_samples_per_activity_per_client, train_val_ratio=train_val_ratio, seed=seed, batch_size=batch_size, batch_size_transfer=batch_size_transfer, verbose=verbose)
    # add the features to the data loaders

    clients_training = np.unique(clients_train)
    train_clients, val_clients = split_train_val_clients(clients_training, train_val_ratio=train_val_ratio, seed=seed)
    # split the data into train and validation
    data_train = data[clients_train.isin(train_clients)]
    labels_train = labels[clients_train.isin(train_clients)]
    data_val = data[clients_train.isin(val_clients)]
    labels_val = labels[clients_train.isin(val_clients)]
    if include_client_ids:
        clients_t = clients_train[clients_train.isin(train_clients)]
        clients_v = clients_train[clients_train.isin(val_clients)]

        train_dataset = dataset_features(data_train, features[clients_train.isin(train_clients)], labels_train, clients_t)
        val_dataset = dataset_features(data_val, features[clients_train.isin(val_clients)], labels_val, clients_v)
    else:
        train_dataset = dataset_features(data_train, features[clients_train.isin(train_clients)], labels_train)
        val_dataset = dataset_features(data_val, features[clients_train.isin(val_clients)], labels_val)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, transfer_train_loader, transfer_val_loader



def load_UCIHAR_all(split_strategy : Literal['few_samples', 'few_samples_features'],
                    n_samples_per_activity_per_client=2, 
                    seed : int = 42, 
                    batch_size : int = 64,
                    batch_size_transfer : int = 32, 
                    include_client_ids : bool = False, 
                    verbose : bool = True):
    """
    Load the UCI HAR dataset and split it into train, validation, and test sets.
    :param split_strategy: the splitting strategy to use
    :param n_samples_per_activity_per_client: the number of samples per activity per client for transfer learning or 'all' to use all samples
    :param seed: the random seed
    :param batch_size: the batch size for the data loaders
    :param batch_size_transfer: the batch size for the transfer learning data loaders
    :param include_client_ids: whether to include client IDs in the dataset (only for 'few_samples_features' strategy)
    :param verbose: whether to print the dataset summary
    :return: Train, validation, and test DataLoaders for embedding learning and transfer learning
    """

    # load train data
    data_train, labels_train, clients_train, _ = load_raw_data('train')
    data_test, labels_test, clients_test, _ = load_raw_data('test')
    
    # split the data into train, validation, and test sets
    if split_strategy == 'few_samples':
        return split_data_few_shots_all_clients(data_train, data_test, labels_train, labels_test, clients_train, clients_test, n_samples_per_activity_per_client=n_samples_per_activity_per_client, seed=seed, batch_size=batch_size, batch_size_transfer=batch_size_transfer, verbose=verbose)
    elif split_strategy == 'few_samples_features':
        features = pd.read_csv(f'{DATA_DIR}/train/X_train.txt', header=None, sep='\s+')
        features = np.array(features)
        return split_data_few_shots_all_clients_features(features, data_train, data_test, labels_train, labels_test, clients_train, clients_test, n_samples_per_activity_per_client=n_samples_per_activity_per_client, seed=seed, batch_size=batch_size, batch_size_transfer=batch_size_transfer, include_client_ids=include_client_ids, verbose=verbose)
    else:
        raise ValueError("Invalid split strategy. Choose from 'few_samples', or 'few_samples_features'.")
    