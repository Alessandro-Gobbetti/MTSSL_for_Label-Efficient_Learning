import shutil
import torch
import torch.nn as nn
from tqdm import tqdm
from common.encoders import get_encoder_from_name
from common.utils import BOLD, RESET, GREEN, YELLOW, CYAN, MAGENTA
from common.losses import compute_multi_task_loss
from data_processing.HAR_precompute_augs import aug_list
from sklearn.metrics import f1_score



class Classifier(torch.nn.Module):
    """
    A simple classifier that uses a pre-trained encoder to generate feature embeddings,
    followed by a fully connected layer head for classification.
    """
    def __init__(self, encoder, num_classes):
        super(Classifier, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.out_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def run_epoch(self, loader, criterion, optimizer=None, device="cpu", is_training=True):
        """
        Run a single epoch for training, validation, or testing.

        Args:
            loader (DataLoader): DataLoader for the current epoch.
            model (torch.nn.Module): The model to train or evaluate.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer, optional): Optimizer for training (ignored if is_training=False).
            device (str): Device to run the computation on.
            is_training (bool): Whether this is a training epoch (True) or evaluation (False).

        Returns:
            tuple: Average loss and accuracy for the epoch.
        """
        if is_training:
            self.train()
        else:
            self.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        # Use torch.no_grad() for evaluation
        context = torch.enable_grad() if is_training else torch.no_grad()

        with context:
            for batch in loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                if is_training:
                    optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)

                if is_training:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct / total
        # Compute F1 score if needed
        f1 = f1_score(all_labels, all_predictions, average='macro')

        return avg_loss, accuracy, f1
    

###########################################################################################################################

class Projector(nn.Module):
    """
    This is a simple MLP projector used to reduce the dimensionality of the input data.
    """
    def __init__(self, in_dim, out_dim):
        super(Projector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        # flatten the input tensor
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        return x


class MultiTaskModel(nn.Module):
    def __init__(self,
                 n_channels: int,
                 n_aug_classes: int,
                 feature_dim: int,
                 out_dim: int = 128,
                 encoder: nn.Module = None,
                 task_heads: list[bool] = None
                ):
        
        super(MultiTaskModel, self).__init__()
        if encoder is not None:
            self.encoder = encoder
            print(f'Using provided encoder: {encoder.__class__.__name__}')
        else:
            # default to FCN
            from common.encoders import FCN
            self.encoder = FCN(n_channels, out_size=1)
        

        # If task_heads is provided, use it. Otherwise, use individual flags (default True except features)
        if task_heads is not None:
            assert len(task_heads) == 4, "task_heads must be a list of 4 booleans"
            self.has_aug_classifier, self.has_reconstruction, self.has_contrastive, self.has_features = task_heads
        else:
            self.has_aug_classifier, self.has_reconstruction, self.has_contrastive, self.has_features = True


        # task specific heads
        # task 1: augmentation classification
        if self.has_aug_classifier:
            self.task_aug_classifier = Projector(self.encoder.out_dim, n_aug_classes)

        # task 2: reconstruction
        if self.has_reconstruction:
            self.task_reconstruction = nn.Sequential(
                nn.Linear(self.encoder.out_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 128*n_channels),
            )

        # task 3: contrastive learning
        if self.has_contrastive:
            # from copy import deepcopy
            # self.encoder_contrastive = deepcopy(self.encoder)
            self.task_contrastive = Projector(self.encoder.out_dim, out_dim)

        # task 4: contrastive features
        if self.has_features:
            from common.encoders import FeatureMLPEncoder
            self.encoder_features = FeatureMLPEncoder(feature_dim, out_dim)
            self.task_features = Projector(self.encoder.out_dim, out_dim)

    def forward(self, x_masked, x_augmentations, x_aug_contr1, x_aug_contr2, x_raw, x_features):
        
        outputs = {}
        
        # Task 1: Augmentation Classification
        if self.has_aug_classifier and x_augmentations is not None:
            # encode the augmentations
            z_aug = self.encoder(x_augmentations)
            z_aug = self.task_aug_classifier(z_aug)
            outputs['aug'] = z_aug

        # Task 2: Reconstruction
        if self.has_reconstruction and x_masked is not None:
            # encode the masked input
            z_masked = self.encoder(x_masked)
            z_masked = z_masked.view(z_masked.size(0), -1)
            # flatten the output
            z_recon = self.task_reconstruction(z_masked)
            # reshape to (batch_size, 128, n_channels)
            z_recon = z_recon.view(z_recon.size(0), 128, -1)
            outputs['recon'] = z_recon

        # Task 3: Contrastive Learning
        if self.has_contrastive and x_aug_contr1 is not None and x_aug_contr2 is not None:
            # encode the contrastive inputs
            z1_contrastive = self.encoder(x_aug_contr1)
            z1_contrastive = self.task_contrastive(z1_contrastive)
            z2_contrastive = self.encoder(x_aug_contr2)
            # z2_contrastive = self.encoder_contrastive(x_aug_contr2)
            z2_contrastive = self.task_contrastive(z2_contrastive)
            outputs['contrastive'] = (z1_contrastive, z2_contrastive)

        if self.has_features and x_raw is not None and x_features is not None:
            z1_feature = self.encoder(x_raw)
            z1_feature = z1_feature.view(z1_feature.size(0), -1)
            z1_feature = self.task_features(z1_feature)
            z2_feature = self.encoder_features(x_features)
            outputs['features'] = (z1_feature, z2_feature)

        return outputs

    def run_epoch(self, dataloader, optimizer, a_dict, cfg, criterion_classification, criterion_reconstruction, criterion_contrastive, criterion_features, is_training=True):
        """
        Run a single epoch for training or validation.

        Args:
            model (torch.nn.Module): The multitask model.
            dataloader (DataLoader): DataLoader for the current epoch.
            optimizer (torch.optim.Optimizer): Optimizer for training (ignored if is_training=False).
            a_dict (dict): Dictionary of learnable loss weights.
            cfg (module): Configuration module.
            criterion_classification (torch.nn.Module): Classification loss function.
            criterion_reconstruction (torch.nn.Module): Reconstruction loss function.
            criterion_contrastive (torch.nn.Module): Contrastive loss function.
            criterion_features (torch.nn.Module): Features loss function.
            is_training (bool): Whether this is a training epoch (True) or validation epoch (False).

        Returns:
            tuple: Average losses for the epoch (total, class, recon, contrastive, features).
        """
        if is_training:
            self.train()
        else:
            self.eval()

        total_loss = 0.0
        loss_class = 0.0
        loss_recon = 0.0
        loss_contrastive = 0.0
        loss_features = 0.0
        num_batches = 0

        # Use torch.no_grad() for validation
        context = torch.enable_grad() if is_training else torch.no_grad()

        with context:
            for i, (orig, masked, aug_labels, augs_1, augs_2, raw, features) in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc=f"{'Training' if is_training else 'Validation'} Epoch",
                leave=False,
                disable=not cfg.VERBOSE,
            ):
                orig = orig.squeeze(0).to(torch.float32).to(cfg.DEVICE)
                masked = masked.squeeze(0).to(torch.float32).to(cfg.DEVICE)
                aug_labels = aug_labels.squeeze(0).to(torch.long).to(cfg.DEVICE)
                augs_1 = augs_1.squeeze(0).to(torch.float32).to(cfg.DEVICE)
                augs_2 = augs_2.squeeze(0).to(torch.float32).to(cfg.DEVICE)
                raw = raw.squeeze(0).to(torch.float32).to(cfg.DEVICE)
                features = features.squeeze(0).to(torch.float32).to(cfg.DEVICE)

                if is_training:
                    optimizer.zero_grad()

                outputs = self(masked, augs_1, augs_1, augs_2, raw, features)
                out_recon = outputs.get('recon')
                out_class = outputs.get('aug')
                out_contr1, out_contr2 = outputs.get('contrastive', (None, None))
                out_features1, out_features2 = outputs.get('features', (None, None))

                # Compute the loss
                loss, losses = compute_multi_task_loss(
                    out_class, out_recon, out_contr1, out_contr2, out_features1, out_features2,
                    aug_labels, orig, a_dict, cfg,
                    criterion_classification, criterion_reconstruction, criterion_contrastive, criterion_features
                )

                # Update per-task accumulators when present
                if losses.get('classification') is not None:
                    loss_class += losses['classification'].item()
                if losses.get('reconstruction') is not None:
                    loss_recon += losses['reconstruction'].item()
                if losses.get('contrastive') is not None:
                    loss_contrastive += losses['contrastive'].item()
                if losses.get('features') is not None:
                    loss_features += losses['features'].item()

                # Accumulate the total loss
                total_loss += loss.item()
                num_batches += 1

                # Backward pass and optimizer step (only for training)
                if is_training:
                    loss.backward()
                    optimizer.step()

        # Average the losses
        total_loss /= num_batches
        loss_class /= num_batches
        loss_recon /= num_batches
        loss_contrastive /= num_batches
        loss_features /= num_batches

        return total_loss, loss_class, loss_recon, loss_contrastive, loss_features


def save_model(model, epoch, cfg):
    import os
    import torch
    # check if epoch is a number
    save_dir = os.path.join(cfg.MODELS_DIR, cfg.EXPERIMENT_NAME)
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(epoch, int):
        save_path = os.path.join(save_dir, f"{cfg.EXPERIMENT_NAME}_epoch_{epoch+1:03}.pth")
    else:
        save_path = os.path.join(save_dir, f"{cfg.EXPERIMENT_NAME}_best.pth")
    torch.save(model.state_dict(), save_path)
    # print(f"Model saved to {save_path}")

def load_encoder_from_epoch(epoch, cfg):
    import os
    import torch
    import re
    save_dir = os.path.join(cfg.MODELS_DIR, cfg.EXPERIMENT_NAME)
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(epoch, int):
        save_path = os.path.join(save_dir, f"{cfg.EXPERIMENT_NAME}_epoch_{epoch+1:03}.pth")
    else:
        save_path = os.path.join(save_dir, f"{cfg.EXPERIMENT_NAME}_best.pth")

    encoder = get_encoder_from_name(cfg)
    encoder.to(cfg.DEVICE)

    # load the model from the models directory
    model_multitask = MultiTaskModel(
        n_channels=cfg.UCIHAR_NUM_CHANNELS,
        n_aug_classes=len(aug_list),
        feature_dim=cfg.UCIHAR_FEATURE_DIM,
        out_dim=cfg.OUT_DIM,
        encoder=encoder,
        task_heads=cfg.MODEL_HEADS
    )
    model_multitask.load_state_dict(torch.load(save_path, map_location=cfg.DEVICE))
    model_multitask.to(cfg.DEVICE)
    model_multitask.eval()
    return model_multitask.encoder

def print_training_summary(best_epoch, best_val_loss, cfg):
    """
    Print a formatted summary of the training process.

    Args:
        best_epoch (int): The epoch with the best validation loss.
        best_val_loss (float): The best validation loss achieved.
        cfg (module): Configuration module.
    """

    print("\n" + "━" * shutil.get_terminal_size().columns)
    print(f"{BOLD}{CYAN}🎉 Training Complete!{RESET}")
    print("━" * shutil.get_terminal_size().columns)
    print(f"{GREEN}Best Epoch: {best_epoch+1:03}/{cfg.EPOCHS:03}{RESET}")
    print(f"{YELLOW}Best Validation Loss: {best_val_loss:.4f}{RESET}")
    print(f"{MAGENTA}Model saved to: {cfg.MODELS_DIR}{RESET}")
    print("━" * shutil.get_terminal_size().columns)

def print_downstream_regime_summary(regime, best_epoch, best_val_loss, best_val_acc, test_loss, test_accuracy, test_f1, cfg):
    """
    Print a formatted summary of the downstream training process.

    Args:
        regime (str): The downstream training regime.
        best_epoch (int): The epoch with the best validation loss.
        best_val_loss (float): The best validation loss achieved.
        best_val_acc (float): The best validation accuracy achieved.
        cfg (module): Configuration module.
    """

    print("\n" + "─" * shutil.get_terminal_size().columns)
    print(f"{BOLD}{CYAN}🎉 Regime {regime} Results:{RESET}")
    print(f"{GREEN}Best Epoch: {best_epoch+1:03}/{cfg.DOWNSTREAM_EPOCHS:03}{RESET}")
    print(f"{YELLOW}Best Validation Loss: {best_val_loss:.4f}{RESET}")
    print(f"{YELLOW}Best Validation Accuracy: {best_val_acc:.4f}{RESET}")
    print(f"{MAGENTA}Test Loss: {test_loss:.4f}{RESET}")
    print(f"{MAGENTA}Test Accuracy: {test_accuracy:.4f}{RESET}")
    print(f"{MAGENTA}Test F1 Score: {test_f1:.4f}{RESET}")
    print("━" * shutil.get_terminal_size().columns)
    