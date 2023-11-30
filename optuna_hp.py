import warnings
import os
import glob
import pickle
import torch
import pytorch_lightning as pl
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

import args_processing
from utils import *
from callbacks import *
from run_model import *
from get_sets import get_sets, get_class_weights

import optuna
from optuna.integration import PyTorchLightningPruningCallback


args = args_processing.parse_arguments()

print("\n" + "-"*80 + "\n")
pl.seed_everything(args.rseed)
print("\n" + "-"*80 + "\n")


# ------------------------------ Warnings config ----------------------------- #
if args.disable_warnings: 
    print("Warnings are DISABLED!\n\n")
    warnings.filterwarnings("ignore")
else:
    warnings.filterwarnings("default")
# ----------------------------------- Paths ---------------------------------- #
working_dir = args.working_dir_path
data_file = args.dataset_h5_path
libraries_dir = working_dir + "/libraries"

# ---------------------------- Import custom libs ---------------------------- #
import sys
sys.path.append(working_dir)
from data_setup import HDF5Dataset, FrameTargetDataset, split_dataset, reduce_sets
from lightning_modules.LUSModelLightningModule import LUSModelLightningModule
from lightning_modules.LUSDataModule import LUSDataModule

# ---------------------------------- Dataset --------------------------------- #

sets, split_info = get_sets(args)
lus_data_module = LUSDataModule(sets["train"], 
                                sets["test"],
                                sets["val"],
                                args.num_workers, 
                                args.batch_size,
                                args.mixup)

print("- Train set class weights: ")
train_weight_tensor = get_class_weights(sets["train_indices"], split_info)
print("\n- Val set class weights: ")
get_class_weights(sets["val_indices"], split_info)
print("\n- Test set class weights: ")
get_class_weights(sets["test_indices"], split_info)


# ------------------- Model hyperparameters & instantiation ------------------ #
def objective(trial: optuna.trial.Trial) -> float:
    # We optimize the number of layers, hidden units in each layer and dropouts.
    lr = trial.suggest_float("lr", 1e-5, 1e-2)
    drop_rate = trial.suggest_float("drop_rate", 0.1, 0.5)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1)
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.001)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "adamw"])
    
    hyperparameters = {
        "num_classes": 4,
        "optimizer": optimizer,
        "lr": lr,
        "batch_size": 16,
        "weight_decay": weight_decay,    
        "momentum": 0.9,
        "label_smoothing": label_smoothing,
        "drop_rate":drop_rate
    }
    

    freeze_layers = None
    if args.pretrained:
        if args.freeze_layers:
            freeze_layers = args.freeze_layers
            
    model = LUSModelLightningModule(model_name=args.model, 
                                    hparams=hyperparameters,
                                    class_weights=train_weight_tensor,
                                    pretrained=args.pretrained,
                                    freeze_layers=freeze_layers,
                                    show_model_summary=args.summary)
    
    model_name, version = get_model_name(args)
    logger = TensorBoardLogger("tb_logs", name=model_name, version=version)
    
    trainer = pl.Trainer(
        limit_val_batches=0.3,
        enable_checkpointing=False,
        max_epochs=10,
        accelerator="gpu",
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
        logger=logger
    )
    
    hparams = dict(optimizer=optimizer,
                    lr=lr,
                    batch_size=args.batch_size,
                    weight_decay=weight_decay,    
                    momentum=args.momentum,
                    label_smoothing=label_smoothing,
                    drop_rate=drop_rate
                    )
    trainer.logger.log_hyperparams(hparams)
    trainer.fit(model, datamodule=lus_data_module)
    return trainer.callback_metrics["val_acc"].item()

if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
