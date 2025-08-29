import os
import shutil
from tqdm import tqdm
import torch
import numpy as np
from common.encoders import get_encoder_from_name
from common.utils import set_seed
import config as cfg
from data_processing.HAR_split_data import load_UCIHAR_all
from data_processing.HAR_precompute_augs import get_mtl_dataloader, aug_list
from common.losses import set_up_loss_functions, compute_multi_task_loss, print_epoch_summary
from common.models import save_model, print_training_summary


if __name__ == '__main__':

    cfg.load_config_from_cli()
    if cfg.VERBOSE:
        cfg.print_summary()


    # set seed for reproducibility
    set_seed(cfg.SEED, verbose=cfg.VERBOSE)

    # import data loaders
    train_loader, val_loader, _, _, _ = load_UCIHAR_all("few_samples_features", 1, seed=cfg.SEED, batch_size=cfg.BATCH_SIZE, batch_size_transfer=cfg.BATCH_SIZE_TRANSFER, verbose=cfg.VERBOSE)

    # precompute augmentations for multi-task learning
    train_loader_mtl, val_loader_mtl = get_mtl_dataloader(train_loader, val_loader, cfg=cfg)

    # model
    from common.models import MultiTaskModel

    encoder = get_encoder_from_name(cfg)
    encoder.to(cfg.DEVICE)

    model_multitask = MultiTaskModel(
        n_channels=cfg.UCIHAR_NUM_CHANNELS,
        n_aug_classes=len(aug_list),
        feature_dim=cfg.UCIHAR_FEATURE_DIM,
        out_dim=cfg.OUT_DIM,
        encoder=encoder,
        task_heads=cfg.MODEL_HEADS
    )


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
        names = ['a1', 'a2', 'a3', 'a4']
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
        {"params": model_multitask.parameters()},
        {"params": a_params}
    ], lr=cfg.LR)



    # SETTING UP LOSSES
    criterion_classification, criterion_reconstruction, criterion_contrastive, criterion_features = set_up_loss_functions(cfg)
    model_multitask.to(cfg.DEVICE)
    criterion_contrastive.to(cfg.DEVICE)
    criterion_classification.to(cfg.DEVICE)
    criterion_reconstruction.to(cfg.DEVICE)
    criterion_features.to(cfg.DEVICE)

    
    

    best_val_loss = float('inf')
    best_epoch = -1
    for epoch in range(cfg.EPOCHS):
        # Training
        train_metrics = model_multitask.run_epoch(
            train_loader_mtl, optimizer, a_dict, cfg,
            criterion_classification, criterion_reconstruction, criterion_contrastive, criterion_features,
            is_training=True
        )

        # Validation
        val_metrics = model_multitask.run_epoch(
            val_loader_mtl, optimizer, a_dict, cfg,
            criterion_classification, criterion_reconstruction, criterion_contrastive, criterion_features,
            is_training=False
        )



        # Print summaries
        if cfg.VERBOSE:
            print_epoch_summary(epoch, train_metrics, val_metrics, a_dict, cfg)
        
        # save the model every cfg.CHECKPOINT_FREQ epochs
        if not cfg.SAVE_BEST_ONLY:
            if (epoch + 1) % cfg.CHECKPOINT_FREQ == 0:
                save_model(model_multitask, epoch, cfg)
        if val_metrics[0] < best_val_loss:
            best_val_loss = val_metrics[0]
            best_epoch = epoch
            save_model(model_multitask, 'best', cfg)
        # save last model
        if epoch == cfg.EPOCHS - 1:
            save_model(model_multitask, epoch, cfg)
    
    if cfg.VERBOSE:
        print_training_summary(best_epoch, best_val_loss , cfg)

        

   