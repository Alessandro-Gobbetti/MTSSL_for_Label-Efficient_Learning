from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
import shutil
from data_processing.HAR_augmentations import gen_aug, aug_list
from common.utils import set_seed


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, original, masked, labels, augs_1, augs_2, raw_data, features):
        self.original = original
        self.masked = masked
        self.labels = labels
        self.augs_1 = augs_1
        self.augs_2 = augs_2
        self.raw_data = raw_data
        self.features = features
        
    def __len__(self):
        return len(self.original)
    
    def __getitem__(self, idx):
        # get the batch
        original = self.original[idx]
        masked = self.masked[idx]
        labels = self.labels[idx]
        augs_1 = self.augs_1[idx]
        augs_2 = self.augs_2[idx]
        raw_data = self.raw_data[idx]
        features = self.features[idx]
        # return the batch
        return original, masked, labels, augs_1, augs_2, raw_data, features
    
############################################################################################################################
# contrastive augmentations and masked data precomputation

def precompute_augmentations_contrastive(loader, auglist, seed, verbose=True):
    set_seed(seed, verbose=False)
    # rng to choose the augmentations
    rng = np.random.default_rng(seed)
    augs_1 = []
    augs_2 = []
    labels_1 = []
    for i, (x_batch,_, _) in tqdm(enumerate(loader), desc="Precomputing contrastive augmentations", total=len(loader), disable=not verbose, leave=False):
        aug_indices_1 = rng.integers(0, len(auglist), size=x_batch.shape[0])
        x_batch1 = torch.stack([gen_aug(x_batch[i:i+1, :, :], auglist[aug_idx]) for i, aug_idx in enumerate(aug_indices_1)]).squeeze(1)

        x_batch2 = torch.stack([gen_aug(x_batch[i:i+1, :, :], rng.choice(auglist)) for i in range(x_batch.shape[0])]).squeeze(1)
        augs_1.append(x_batch1)
        augs_2.append(x_batch2)
        labels_1.append(torch.tensor(aug_indices_1, dtype=torch.long))

    augs_1 = torch.cat(augs_1, dim=0)
    augs_2 = torch.cat(augs_2, dim=0)
    labels_1 = torch.cat(labels_1, dim=0)
    return augs_1, augs_2, labels_1

def precompute_mask(loader, noise_ratio, seed, verbose=True):
    set_seed(seed, verbose=False)
    # rng to choose the augmentations
    rng = np.random.default_rng(seed)
    original = []
    masked = []
    for i, (x_batch,_, _) in tqdm(enumerate(loader), desc="Precomputing masked data", total=len(loader), disable=not verbose, leave=False):

        # get the batch size
        batch_size = x_batch.shape[0]
        # get the number of samples
        num_samples = x_batch.shape[1]
        # get the number of features
        num_features = x_batch.shape[2]
        # create a mask
        mask = rng.random((batch_size, num_samples, num_features)) < noise_ratio
        # create the masked version of the batch
        masked_batch = x_batch.clone()
        masked_batch[mask] = 0
        original.append(x_batch)
        masked.append(masked_batch)
    original = torch.cat(original, dim=0)
    masked = torch.cat(masked, dim=0)
    return original, masked


def get_mtl_dataloader(train_loader, val_loader, cfg):
    """
    Given the original train and val dataloaders, precompute the augmentations for multi-task learning.
    Return new train and val dataloaders with the precomputed augmentations.
    1. Augmentation classification (task 1)
    2. Reconstruction (task 2)
    3. Contrastive learning (task 3)
    4. Contrastive features (task 4)
    """

    set_seed(cfg.SEED, verbose=False)
    original_train, masked_train = precompute_mask(train_loader, cfg.NOISE_RATIO, cfg.SEED)
    original_val, masked_val = precompute_mask(val_loader, cfg.NOISE_RATIO, cfg.SEED)

    set_seed(cfg.SEED, verbose=False)
    augs_1_train, augs_2_train, labels_1_train = precompute_augmentations_contrastive(train_loader, aug_list, cfg.SEED)
    augs_1_val, augs_2_val, labels_1_val = precompute_augmentations_contrastive(val_loader, aug_list, cfg.SEED)


    raw_data_train = train_loader.dataset.samples
    features_train = train_loader.dataset.features
    raw_data_val = val_loader.dataset.samples
    features_val = val_loader.dataset.features

    # create the dataset
    train_dataset = MultiTaskDataset(original_train, masked_train, labels_1_train, augs_1_train, augs_2_train, raw_data_train, features_train)
    val_dataset = MultiTaskDataset(original_val, masked_val, labels_1_val, augs_1_val, augs_2_val, raw_data_val, features_val)
    # create the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=True)

    if cfg.VERBOSE:
        terminal_width = shutil.get_terminal_size().columns
        print(f"✅ Precomputed multi-task augmentations:")
        print(f"     ─ Train dataset size: {len(train_dataset)}")
        print(f"     ─ Val dataset size: {len(val_dataset)}")
        print("─" * terminal_width)


    return train_loader, val_loader


if __name__ == "__main__":
    from data_processing.HAR_split_data import load_UCIHAR_all

    SEED = 0
    BATCH_SIZE = 64
    NOISE_RATIO = 0.3
    VERBOSE = True

    set_seed(SEED)
    train_loader, val_loader, test_loader, transfer_train_loader, transfer_val_loader = load_UCIHAR_all("few_samples_features", 1, seed=SEED, batch_size=BATCH_SIZE, batch_size_transfer=32, verbose=VERBOSE)

    train_loader, val_loader = get_mtl_dataloader(train_loader, val_loader, NOISE_RATIO, SEED, BATCH_SIZE, verbose=VERBOSE)