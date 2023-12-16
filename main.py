import warnings
import signal
import os
import glob
import pickle
import torch
import lightning.pytorch as pl
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner

import args_processing
from utils import *
from callbacks import *
from run_model import *
from get_sets import get_sets, get_class_weights

def main():
    
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

    print("\n- Train set class weights: ")
    train_weight_tensor = get_class_weights(sets["train_indices"], split_info)
    print("\n- Val set class weights: ")
    get_class_weights(sets["val_indices"], split_info)
    print("\n- Test set class weights: ")
    get_class_weights(sets["test_indices"], split_info)
# ---------------------------------------------------------------------------- #
#                         Model & trainer configuration                        #
# ---------------------------------------------------------------------------- #

# ------------------- Model hyperparameters & instantiation ------------------ #

    print("\n\nModel configuration...")
    print('=' * 80)

    hyperparameters = {
    "num_classes": 4,
    "optimizer": args.optimizer,
    "lr": args.lr,
    "batch_size": args.batch_size,
    "weight_decay": args.weight_decay,    
    "momentum": args.momentum,
    "label_smoothing": args.label_smoothing,
    "drop_rate":args.drop_rate,
    "max_epochs":args.max_epochs,
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
                                    show_model_summary=args.summary,
                                    augmentation=args.augmentation)

    generate_table(f"{args.model} Hyperparameters", hyperparameters, ["train_dataset", "test_dataset"])
# --------------------------- Trainer configuration -------------------------- #
    print("\n\nTrainer configuration...")
    print('=' * 80)

    # Callbacks
    early_stop_callback = early_stopper()
    model_name, version = get_model_name(args)
    logger = TensorBoardLogger("tb_logs", name=model_name, version=version)
    checkpoint_dir = f"{working_dir}/checkpoints/{model_name}"
    checkpoint_callback = checkpoint_saver(checkpoint_dir)
    callbacks = [
                RichProgressBar(),
                LearningRateMonitor(),
                early_stop_callback,
                checkpoint_callback
                ]
    # Trainer args
    trainer_args = {
        "max_epochs": args.max_epochs,
        "callbacks": callbacks,
        "precision": args.precision,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "logger": logger
    }
    # Trainer 
    trainer = Trainer(**trainer_args,
                        accelerator="gpu",
                        default_root_dir = checkpoint_dir,
                        devices=1)

    generate_table("Trainer args", trainer_args, ["callbacks", "logger"])
    print("\n\n" + "-" * 20)
    print("Trainer Callbacks:")
    print("-" * 20 + "\n\n")
    for callback in trainer.callbacks:
        print(f"- {type(callback).__name__}")
    print(f"Model checkpoints directory is {checkpoint_dir}\n\n")

        
    # ---------------------------- Model fit and test ---------------------------- #
    if args.mode == "train":
        fit_model(model, trainer, lus_data_module, args.chkp)
        
    if args.mode == "test":
        test_model(model, trainer, lus_data_module, args.chkp)
        
    if args.mode == "tune":
        tuner = Tuner(trainer)
        tuner.lr_find(model=model, 
                    datamodule=lus_data_module, 
                    method='fit',
                    min_lr=1e-05,
                    max_lr=0.1,
                    mode="exponential",
                    num_training=1000)

if __name__ == "__main__":
    main()


