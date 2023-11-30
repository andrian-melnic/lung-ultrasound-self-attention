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

import args_processing
from utils import *
from callbacks import *
from run_model import *
from get_sets import get_sets, get_class_weights
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
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


# ---------------------------------------------------------------------------- #
#                                    RayTune                                   #
# ---------------------------------------------------------------------------- #
# ------------------- Model hyperparameters & instantiation ------------------ #
def train_func(config):

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
                                    show_model_summary=args.summary)

    generate_table(f"{args.model} Hyperparameters", hyperparameters, ["train_dataset", "test_dataset"])


    # Callbacks
    ckpt_report_callback = RayTrainReportCallback()
    early_stop_callback = early_stopper()
    model_name, version = get_model_name(args)
    logger = TensorBoardLogger("tb_logs", name=model_name, version=version)
    checkpoint_dir = f"{working_dir}/checkpoints/{model_name}"
    checkpoint_callback = checkpoint_saver(checkpoint_dir)
    callbacks = [RichProgressBar(),
                ckpt_report_callback]
    
    # trainer
    trainer = pl.Trainer(
        max_epochs=10,
        devices="auto",
        accelerator="gpu",
        callbacks=callbacks,
        logger=logger,
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=data_module)


search_space = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "optimizer": tune.choice(["adam", "sgd", "adamw"]),
    "weight_decay": tune.choice([0.0, 0.005, 0.001, 0.0005]),
    "drop_rate": tune.choice([0.0, 0.1, 0.2]),
    "label_smoothing": tune.choice([0.0, 0.1])
}


scheduler = ASHAScheduler(max_t=5, grace_period=1, reduction_factor=2)
# scaling_config = ScalingConfig(num_workers=1, use_gpu=True)
run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="ptl/val_loss",
        checkpoint_score_order="min",
    ),
)
ray_trainer = TorchTrainer(
    train_func,
    # scaling_config=scaling_config,
    run_config=run_config,
)
tuner = tune.Tuner(
    ray_trainer,
    param_space={"train_loop_config": search_space},
    tune_config=tune.TuneConfig(
        metric="ptl/val_loss",
        mode="min",
        num_samples=10,
        scheduler=scheduler,
    ),
)
results = tuner.fit()
print(f"RayTuner results: {results}")