import os
from copy import deepcopy
import shutil
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from common.encoders import get_encoder_from_name
from common.utils import set_seed
import config as cfg
from data_processing.HAR_split_data import load_UCIHAR_all
from data_processing.HAR_precompute_augs import get_mtl_dataloader, aug_list
from common.losses import print_simple_epoch_summary, set_up_loss_functions, compute_multi_task_loss, print_epoch_summary
from common.models import Classifier, print_downstream_regime_summary, save_model, print_training_summary, load_encoder_from_epoch

if __name__ == '__main__':

    cfg.load_config_from_cli()
    if cfg.VERBOSE:
        cfg.print_summary()

    # load the saved encoder
    # cfg.ENCODER_PATH
    if cfg.IS_FULLY_SUPERVISED:
        # print a message
        print("🔄  Using fully supervised training, training the full model.")
        encoder = get_encoder_from_name(cfg)
    else:
        encoder = load_encoder_from_epoch('best', cfg)

    results = pd.DataFrame(columns=["n_act", "test_loss", "test_accuracy", "test_f1"])

    for n_act in [1, 2, 3, 4, 5, 6, "all"]:
        set_seed(cfg.SEED, verbose=cfg.VERBOSE)
        _, _, test_loader, transfer_train_loader, transfer_val_loader = load_UCIHAR_all("few_samples", n_act, seed=cfg.SEED, batch_size=cfg.BATCH_SIZE, batch_size_transfer=cfg.BATCH_SIZE_TRANSFER, verbose=False)
        if cfg.VERBOSE:
            print(f"➡️  Downstream training with {n_act} activities ({len(transfer_train_loader.dataset)} samples)")
        
        model_classifier = Classifier(
            encoder=encoder,
            num_classes=cfg.UCIHAR_NUM_CLASSES
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
                loader=transfer_train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=cfg.DEVICE,
                is_training=True
            )

            val_loss, val_accuracy, val_f1 = model_classifier.run_epoch(
                loader=transfer_val_loader,
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

        # if cfg.VERBOSE:
        #     print_downstream_regime_summary(n_act, best_epoch, best_val_loss, best_val_accuracy, test_loss, test_accuracy, test_f1, cfg)

        new_row = pd.DataFrame([{
            "n_act": n_act,
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