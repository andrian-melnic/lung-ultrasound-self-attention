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
import sys
from optuna.integration import PyTorchLightningPruningCallback
# Import the logging module
import logging
from datetime import datetime
args = args_processing.parse_arguments()

print("\n" + "-"*80 + "\n")
pl.seed_everything(args.rseed)
print("\n" + "-"*80 + "\n")


# ---------------------------------------------------------------------------- #
#                              Optuna logging config                           #
# ---------------------------------------------------------------------------- #
current_time = datetime.now().strftime("%d-%m_%H:%M")
study_name = f"study_{current_time}"
storage_name = "sqlite:///{}.db".format(f"optuna_dbs/{study_name}")

log_file_path = f"optuna_logs/{study_name}.txt"
logging.basicConfig(filename=log_file_path, level=logging.INFO)

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

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



# ------------------- Model hyperparameters & instantiation ------------------ #
def objective(trial: optuna.trial.Trial) -> float:
    
    
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    lr = trial.suggest_categorical("lr", [1e-3, 2e-4, 1e-4, 2e-5])
    drop_rate = trial.suggest_categorical("drop_rate", [0, 0.1, 0.2, 0.3])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-1, 1e-2, 1e-3, 1e-4])
    
    # ---------------------------------- Dataset --------------------------------- #

    sets, split_info = get_sets(args)
    lus_data_module = LUSDataModule(sets["train"], 
                                    sets["test"],
                                    sets["val"],
                                    args.num_workers,
                                    batch_size,
                                    args.mixup)

    print("- Train set class weights: ")
    train_weight_tensor = get_class_weights(sets["train_indices"], split_info)

    print("\n- Val set class weights: ")
    get_class_weights(sets["val_indices"], split_info)

    print("\n- Test set class weights: ")
    get_class_weights(sets["test_indices"], split_info)
    
    hparams = {
        "batch_size": batch_size,
        "lr": lr,
        "drop_rate":drop_rate,
        "weight_decay": weight_decay,    
        "momentum": 0.9,
        "num_classes": 4,
        "optimizer": args.optimizer,
        "label_smoothing": args.label_smoothing,
    }
    
    

    freeze_layers = None
    if args.pretrained:
        if args.freeze_layers:
            freeze_layers = args.freeze_layers
            
            
    # Print info about the params and the metric to the console
    print("\nTrial {}: {}".format(trial.number, "-" * 40))
    print("  Params: {}".format(trial.params))

    # Log the same information to the file
    logging.info("\nTrial {}: {}".format(trial.number, "-" * 40))
    logging.info("  Params: {}".format(trial.params))
    
    model = LUSModelLightningModule(model_name=args.model, 
                                    hparams=hparams,
                                    class_weights=train_weight_tensor,
                                    pretrained=args.pretrained,
                                    freeze_layers=freeze_layers,
                                    show_model_summary=args.summary,
                                    augmentation=args.augmentation)
    
    # args.lr = hparams["lr"]
    # args.batch_size = hparams["batch_size"]
    # args.weight_decay = hparams["weight_decay"]
    # args.label_smoothing = hparams["label_smoothing"]
    # args.drop_rate = hparams["drop_rate"]
    
    # model_name, version = get_model_name(args)
    logger = TensorBoardLogger(f"tb_logs/optuna/{args.model}_{args.optimizer}", name=study_name, version=f"trial_{trial.number}")
    early_stop_callback = early_stopper()
    
    checkpoint_dir = f"checkpoints/optuna/{args.model}_{args.optimizer}/{study_name}"
    checkpoint_trial =  f"trial_{trial.number}"
    checkpoint_callback = checkpoint_saver_optuna(checkpoint_dir, checkpoint_trial)
    
    trainer = pl.Trainer(
        enable_checkpointing=True,
        max_epochs=30,
        accelerator="gpu",
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_f1"), 
                   early_stop_callback,
                   checkpoint_callback],
        logger=logger
    )
    
    hyperparameters = dict(
                    optimizer=args.optimizer,
                    lr=lr,
                    batch_size=batch_size,
                    weight_decay=weight_decay,    
                    momentum=args.momentum,
                    label_smoothing=args.label_smoothing,
                    drop_rate=drop_rate
                    )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=lus_data_module)
    
    return trainer.callback_metrics["val_f1"].item()

def main():
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(study_name=study_name, 
                                storage=storage_name,
                                direction="maximize",
                                pruner=pruner)
    
    study.optimize(objective, n_trials=20)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    # Log the final results to the file
    logging.info("\nNumber of finished trials: {}".format(len(study.trials)))
    logging.info("\nBest trial:")
    logging.info("  Value: {}".format(trial.value))
    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info("    {}: {}".format(key, value))
        
if __name__ == "__main__":
    main()