import torch
import torch.nn.functional as F
import numpy as np
from common.utils import BOLD, RESET, GREEN, YELLOW, CYAN, MAGENTA

##########################################################################################################################

class NTXentLoss(torch.nn.Module):
    """
    Contrastive loss function for NT-Xent (Normalized Temperature-scaled Cross Entropy Loss).
    """

    def __init__(self, device, batch_size, temperature=0.1, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
    
##########################################################################################################################

class BarlowTwinsLoss(torch.nn.Module):
    """
    Barlow Twins loss function.
    """

    def __init__(self, lambd=1e-5, batch_size=256):
        super(BarlowTwinsLoss, self).__init__()
        self.lambd = lambd
        self.batch_size = batch_size

    def forward(self, zis, zjs):
        # Normalize the representations along the batch dimension
        zis = (zis - zis.mean(dim=0)) / zis.std(dim=0)
        zjs = (zjs - zjs.mean(dim=0)) / zjs.std(dim=0)

        # Compute the cross-correlation matrix
        c = torch.mm(zis.T, zjs) / zis.size(0)

        c_diff = (c - torch.eye(c.size(0)).to(c.device)).pow(2)
        # Apply lambda to off-diagonal elements
        c_diff[~torch.eye(c_diff.size(0), dtype=bool)] *= self.lambd

        # Compute the loss
        loss = c_diff.sum()

        return loss
    

###########################################################################################################################

class VICRegLoss(torch.nn.Module):
    """
    https://arxiv.org/abs/2105.04906
    VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning
    """
    def __init__(self, batch_size=256):
        super(VICRegLoss, self).__init__()
        self.batch_size = batch_size
        self.lambd = 25
        self.mu = 25
        self.nu = 1

    def forward(self, z_a, z_b):
        # invariance loss
        sim_loss = F.mse_loss(z_a, z_b)

        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))

        # covariance loss
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = (z_a.T @ z_a) / (self.batch_size - 1)
        cov_z_b = (z_b.T @ z_b) / (self.batch_size - 1)
        D = z_a.size(1)
        off_diagonal = lambda x: x[~torch.eye(x.size(0), dtype=bool)].view(x.size(0), -1)
        cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D + off_diagonal(cov_z_b).pow_(2).sum() / D

        # Compute the final loss
        loss = self.lambd * sim_loss + self.mu * std_loss + self.nu * cov_loss
        return loss
    



###########################################################################################################################
# Dynamic weighted multi-task loss functions
    
def set_up_loss_functions(cfg):
    loss_name = cfg.CONTRASTIVE_LOSS.lower()
    if loss_name == "ntxent":
        criterion_contrastive = NTXentLoss(cfg.DEVICE, cfg.BATCH_SIZE,
                                        temperature=0.1,
                                        use_cosine_similarity=True)
    elif loss_name == 'barlow':
        criterion_contrastive = BarlowTwinsLoss(batch_size=cfg.BATCH_SIZE)
    elif loss_name == 'vicreg':
        criterion_contrastive = VICRegLoss(batch_size=cfg.BATCH_SIZE)
    else:
        raise ValueError(f"Unknown CONTRASTIVE_LOSS: {cfg.CONTRASTIVE_LOSS}")

    criterion_classification = torch.nn.CrossEntropyLoss()
    criterion_reconstruction = torch.nn.MSELoss()
    criterion_features = VICRegLoss(batch_size=cfg.BATCH_SIZE)
    return criterion_classification, criterion_reconstruction, criterion_contrastive, criterion_features


def compute_multi_task_loss(out_class, out_recon, out_contr1, out_contr2, out_features1, out_features2,
                            aug_labels, orig, a_dict, cfg,
                            criterion_classification, criterion_reconstruction, criterion_contrastive, criterion_features):
    """
    Compute total multi-task loss and return per-task losses.

    Returns: (total_loss: torch.Tensor, losses: dict)
    losses keys: 'classification','reconstruction','contrastive','barlow' (values or None)
    """
    device = cfg.DEVICE
    total = torch.tensor(0.0, device=device)
    losses = {'classification': None, 'reconstruction': None, 'contrastive': None, 'features': None}

    # classification / augmentation label head
    if cfg.MODEL_HEADS[0] and out_class is not None:
        losses['classification'] = criterion_classification(out_class, aug_labels)
        if cfg.USE_WEIGHTED_LOSS:
            a = a_dict.get('a1')
            loss_classification = (1 / (2 * a**2)) * losses['classification'] + torch.log(a)
        else:
            loss_classification = losses['classification']
        total = total + loss_classification

    # reconstruction head
    if cfg.MODEL_HEADS[1] and out_recon is not None:
        losses['reconstruction'] = criterion_reconstruction(out_recon, orig)
        if cfg.USE_WEIGHTED_LOSS:
            a = a_dict.get('a2')
            loss_reconstruction = (1 / (2 * a**2)) * losses['reconstruction'] + torch.log(a)
        else:
            loss_reconstruction = losses['reconstruction']
        total = total + loss_reconstruction

    # contrastive head
    if cfg.MODEL_HEADS[2] and out_contr1 is not None and out_contr2 is not None:
        losses['contrastive'] = criterion_contrastive(out_contr1, out_contr2)
        if cfg.USE_WEIGHTED_LOSS:
            a = a_dict.get('a3')
            loss_contrastive = (1 / (2 * a**2)) * losses['contrastive'] + torch.log(a)
        else:
            loss_contrastive = losses['contrastive']
        total = total + loss_contrastive

    # features / barlow/vicreg head
    if cfg.MODEL_HEADS[3] and out_features1 is not None and out_features2 is not None:
        losses['features'] = criterion_features(out_features1, out_features2)
        if cfg.USE_WEIGHTED_LOSS:
            a = a_dict.get('a4')
            loss_features = (1 / (2 * a**2)) * losses['features'] + torch.log(a)
        else:
            loss_features = losses['features']
        total = total + loss_features

    # sanity check
    if not any(cfg.MODEL_HEADS):
        raise ValueError("At least one model head must be enabled.")

    return total, losses

def print_simple_epoch_summary(epoch, epoch_total, train_loss, train_accuracy, val_loss, val_accuracy):
    text = f"{BOLD}{CYAN}📈 Epoch {epoch:03}/{epoch_total:03}{RESET} │ {GREEN}Train: {train_loss:7.4f}{RESET}"
    if train_accuracy:
        text += f" (Acc: {train_accuracy*100:5.2f}%) │ "
    text += f"{YELLOW}Val: {val_loss:7.4f}{RESET}"
    if val_accuracy:
        text += f" (Acc: {val_accuracy*100:5.2f}%)"
    print(text)

def print_epoch_summary(epoch, train_metrics, val_metrics, a_dict, cfg):   
    
    train_loss, train_loss_class, train_loss_recon, train_loss_contrastive, train_loss_features = train_metrics
    val_loss, val_loss_class, val_loss_recon, val_loss_contrastive, val_loss_features = val_metrics

    output = [f"{BOLD}{CYAN}📈 Epoch {epoch+1:03}/{cfg.EPOCHS:03}{RESET} │ ",
              f"{GREEN}Train: {train_loss:7.4f}{RESET} ("]

    train_parts = []
    if cfg.MODEL_HEADS[0]:
        train_parts.append(f"C: {train_loss_class:4.2f}")
    if cfg.MODEL_HEADS[1]:
        train_parts.append(f"R: {train_loss_recon:4.2f}")
    if cfg.MODEL_HEADS[2]:
        train_parts.append(f"K: {train_loss_contrastive:4.2f}")
    if cfg.MODEL_HEADS[3]:
        train_parts.append(f"F: {train_loss_features:5.2f}")
    output.append(", ".join(train_parts))

    output.append(f") │ {YELLOW}Val: {val_loss:7.4f}{RESET} (")
    val_parts = []
    if cfg.MODEL_HEADS[0]:
        val_parts.append(f"C: {val_loss_class:4.2f}")
    if cfg.MODEL_HEADS[1]:
        val_parts.append(f"R: {val_loss_recon:4.2f}")
    if cfg.MODEL_HEADS[2]:
        val_parts.append(f"K: {val_loss_contrastive:4.2f}")
    if cfg.MODEL_HEADS[3]:
        val_parts.append(f"F: {val_loss_features:5.2f}")
    output.append(", ".join(val_parts))

    weights = []
    if cfg.MODEL_HEADS[0]:
        weights.append(f"a1={a_dict['a1'].item():.2f}")
    if cfg.MODEL_HEADS[1]:
        weights.append(f"a2={a_dict['a2'].item():.2f}")
    if cfg.MODEL_HEADS[2]:
        weights.append(f"a3={a_dict['a3'].item():.2f}")
    if cfg.MODEL_HEADS[3]:
        weights.append(f"a4={a_dict['a4'].item():.2f}")

    if weights:
        output.append(f") │ {MAGENTA}Weights:{RESET} {', '.join(weights)}")

    print("".join(output))