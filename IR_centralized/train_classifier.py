import os
from copy import deepcopy
import shutil
from matplotlib import transforms
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from common.encoders import get_encoder_from_name
from common.utils import set_seed
from config import ir_config
from data_processing.HAR_split_data import load_UCIHAR_all
from data_processing.HAR_precompute_augs import get_mtl_dataloader, aug_list
from common.losses import print_simple_epoch_summary, set_up_loss_functions, compute_multi_task_loss, print_epoch_summary
from common.models import IR_Classifier, MultiTaskIR, print_downstream_regime_summary, save_model, print_training_summary, load_encoder_from_epoch
from torchvision.datasets import STL10
from torchvision import transforms
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    cfg = ir_config()
    cfg.load_config_from_cli()
    if cfg.VERBOSE:
        cfg.print_summary()
    
    results = pd.DataFrame(columns=["data_fraction", "test_loss", "test_accuracy", "test_f1"])

    aug_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.RandomResizedCrop(size=96, scale=(0.8, 1.0)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))
                                       ])
    norm_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])
    
    class dataset_transform(Dataset):
            def __init__(self, images, labels, transform=None):
                self.images = images
                self.labels = labels
                self.transform = transform

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                img, label = self.images[idx], self.labels[idx]
                if self.transform:
                    img = self.transform(img)
                return img, label
    train_img_data = STL10(root=cfg.DATASET_DIR, split='train', download=True)
    test_img_data = STL10(root=cfg.DATASET_DIR, split='test', download=True, transform=norm_transform)

    for KEEP_FRACTION in [0.2, 0.5, 1.0]:


        # --------- Reducing Data ---------
        if KEEP_FRACTION < 1.0:
            # Get labels from the underlying dataset
            labels = np.array(train_img_data.labels)
            indices = np.arange(len(train_img_data))
            keep_indices, _ = train_test_split(indices, train_size=KEEP_FRACTION, stratify=labels, random_state=42)
            labels = labels[keep_indices]
            indices = indices[keep_indices]
        else:
            labels = np.array(train_img_data.labels)
            indices = np.arange(len(train_img_data))
        

        set_seed(cfg.SEED, verbose=cfg.VERBOSE)
        train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

        

        val_img_data = dataset_transform(
            images=[train_img_data[i][0] for i in val_indices],
            labels=[train_img_data[i][1] for i in val_indices],
            transform=aug_transforms
        )
        train_img_data = dataset_transform(
            images=[train_img_data[i][0] for i in train_indices],
            labels=[train_img_data[i][1] for i in train_indices],
            transform=aug_transforms
        )

        if cfg.VERBOSE:
            print(f"➡️  Downstream training with {KEEP_FRACTION*100:<.0f}% of the data:")
            print(f"    - Training samples:   {len(train_img_data)}")
            print(f"    - Validation samples: {len(val_img_data)}")
            print(f"    - Test samples:       {len(test_img_data)}")

        # init loaders
        train_loader = DataLoader(train_img_data, batch_size=cfg.BATCH_SIZE_TRANSFER, shuffle=True, num_workers=cfg.NUM_WORKERS)
        val_loader = DataLoader(val_img_data, batch_size=cfg.BATCH_SIZE_TRANSFER, shuffle=False, num_workers=cfg.NUM_WORKERS)
        test_loader = DataLoader(test_img_data, batch_size=cfg.BATCH_SIZE_TRANSFER, shuffle=False, num_workers=cfg.NUM_WORKERS)

        
        
        # --------- Set up model ---------
        # load the saved encoder
        model = MultiTaskIR(
            is_contrastive=cfg.MODEL_HEADS[0],
            is_rotation=cfg.MODEL_HEADS[1],
            is_mask=cfg.MODEL_HEADS[2],
        )

        # load the encoder
        if cfg.IS_FULLY_SUPERVISED:
            # print a message
            print("🔄  Using fully supervised training, training the full model.")
            encoder = model.encoder
        else:
            # load the encoder from a checkpoint
            epoch=cfg.ENCODER_EPOCH
            save_dir = os.path.join(cfg.MODELS_DIR, cfg.EXPERIMENT_NAME)
            path = os.path.join(save_dir, f"{cfg.EXPERIMENT_NAME}_epoch_{epoch+1:03}.pth")
            model.load_state_dict(torch.load(path, map_location=cfg.DEVICE), strict=False)
            encoder = model.encoder
        encoder.to(cfg.DEVICE)

        model_classifier = IR_Classifier(
            encoder=encoder,
            num_classes=cfg.STL10_NUM_CLASSES
        )
        if cfg.IS_FULLY_SUPERVISED:
            # set the encoder to be trainable
            for param in model_classifier.parameters():
                param.requires_grad = True
        model_classifier.to(cfg.DEVICE)

        optimizer = torch.optim.Adam(model_classifier.parameters(), lr=cfg.LR)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        best_epoch = -1
        val_losses = []
        val_accuracies = []
        best_model_state = None

        for epoch in tqdm(range(cfg.DOWNSTREAM_EPOCHS), desc="Downstream Epochs", leave=False, disable=not cfg.VERBOSE):
            model_classifier.train()

            train_loss, train_accuracy, train_f1 = model_classifier.run_epoch(
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=cfg.DEVICE,
                is_training=True
            )

            val_loss, val_accuracy, val_f1 = model_classifier.run_epoch(
                loader=val_loader,
                criterion=criterion,
                device=cfg.DEVICE,
                is_training=False
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_epoch = epoch + 1
                best_model_state = deepcopy(model_classifier.state_dict())
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)


        # run test with the best model
        model_classifier.load_state_dict(best_model_state)
        test_loss, test_accuracy, test_f1 = model_classifier.run_epoch(
            loader=test_loader,
            criterion=criterion,
            device=cfg.DEVICE,
            is_training=False
        )

        new_row = pd.DataFrame([{
            "data_fraction": KEEP_FRACTION,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_f1": test_f1
        }])
        if results.empty:
            results = new_row
        else:
            results = pd.concat([results, new_row], ignore_index=True)

    save_path = cfg.RESULTS_DIR
    os.makedirs(save_path, exist_ok=True)
    results.to_csv(os.path.join(save_path, f"{cfg.EXPERIMENT_NAME}.csv"), index=False)

    if cfg.VERBOSE:
        print("➡️  Downstream results:")
        print(results)
    print("✅ Downstream training completed.")