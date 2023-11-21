import warnings
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

from tuner import tune_model, train_function
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer


if __name__ == "__main__":
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

    sets, split_info = get_sets(
        args.rseed,
        data_file,
        args.hospitaldict_path,
        args.train_ratio,
        args.trim_data,
        args.pretrained
    )

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
        "drop_rate":args.drop_rate
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
                                    augmentation=args.augmentation)

    generate_table(f"{args.model} Hyperparameters", hyperparameters, ["train_dataset", "test_dataset"])


# ---------------------------- Model fit and test ---------------------------- #
    if args.mode == "train":
        fit_model(model, trainer, lus_data_module, args.chkp)
    if args.mode == "test":
        test_model(model, trainer, lus_data_module, args.chkp)
        
    if args.mode == "tune":
        
        scaling_config = ScalingConfig(
            num_workers=args.num_workers, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
        )

        run_config = RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
        )
        # Define a TorchTrainer without hyper-parameters for Tuner
        ray_trainer = TorchTrainer(
            train_function(model, lus_data_module),
            scaling_config=scaling_config,
            run_config=run_config,
        )
        results = tune_model(ray_trainer, args.max_epochs, num_samples=10)



