import os
from pathlib import Path

import torch

from common.utils import set_seed, check_gpu
from data_processing.IR_augmentations import UnifiedTransform
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from common.models import MultiTaskIR, IR_Classifier, print_training_summary, save_model
from common.losses import NTXentLoss, print_simple_epoch_summary


from config import ir_config

if __name__ == '__main__':

    cfg = ir_config()
    cfg.load_config_from_cli()
    if cfg.VERBOSE:
        cfg.print_summary()

    set_seed(cfg.SEED)
    transform = UnifiedTransform(is_contrastive=cfg.MODEL_HEADS[0], is_rotation=cfg.MODEL_HEADS[1], is_mask=cfg.MODEL_HEADS[2])


    unlabeled_data = STL10(root=cfg.DATASET_DIR, split='unlabeled', download=True, transform=transform)
    val_data = STL10(root=cfg.DATASET_DIR, split='train', download=True, transform=transform)

    print(f"Unlabeled data size: {len(unlabeled_data)}")
    print(f"Train data (contrastive) size: {len(val_data)}")

    unlabeled_loader = DataLoader(unlabeled_data, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=True)
    validation_loader = DataLoader(val_data, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=True)

    model = MultiTaskIR(
        is_contrastive=cfg.MODEL_HEADS[0],
        is_rotation=cfg.MODEL_HEADS[1],
        is_mask=cfg.MODEL_HEADS[2],
    )
    model.to(cfg.DEVICE)

    def create_loss_weight_params(model_heads, init_value: float = 1.0, device=None):
            """Create learnable loss-weight parameters for active model heads.

            Args:
                model_heads (Sequence[bool]): list/tuple of booleans of length 4 indicating active heads
                init_value (float): initial scalar value for each Parameter
                device: torch device (optional)

            Returns:
                params (list[torch.nn.Parameter]), params_dict (dict[str, torch.nn.Parameter])
            """
            import torch as _torch
            names = ['a1', 'a2', 'a3']
            if device is None:
                device = _torch.device('cpu')

            params = []
            params_dict = {}
            for i, active in enumerate(model_heads):
                if active:
                    p = _torch.nn.Parameter(_torch.tensor(init_value, dtype=_torch.float32, device=device), requires_grad=True)
                    params.append(p)
                    params_dict[names[i]] = p
            return params, params_dict


    # loss weights learnable parameters (created via helper for reuse)
    a_params, a_dict = create_loss_weight_params(cfg.MODEL_HEADS, init_value=1.0, device=cfg.DEVICE)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': a_params} 
    ], lr=0.001)

    criterion_contr = NTXentLoss(device=cfg.DEVICE, batch_size=cfg.BATCH_SIZE)
    criterion_rot = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_mask = torch.nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(cfg.EPOCHS):
        train_loss = model.run_epoch(
            unlabeled_loader, optimizer, a_dict, cfg,
            criterion_contr, criterion_rot, criterion_mask, is_training=True, disable_tqdm=False
        )
        val_loss = model.run_epoch(
            validation_loader, optimizer, a_dict, cfg,
            criterion_contr, criterion_rot, criterion_mask, is_training=False, disable_tqdm=False
        )

        if cfg.VERBOSE:
            print_simple_epoch_summary(epoch, cfg.EPOCHS, train_loss, None, val_loss, None)

        # save the model every cfg.CHECKPOINT_FREQ epochs
        if not cfg.SAVE_BEST_ONLY:
            if (epoch + 1) % cfg.CHECKPOINT_FREQ == 0:
                save_model(model, epoch, cfg)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_model(model, 'best', cfg)
        # save last model
        if epoch == cfg.EPOCHS - 1:
            save_model(val_loss, epoch, cfg)

    if cfg.VERBOSE:
        print_training_summary(best_epoch, best_val_loss, cfg)
