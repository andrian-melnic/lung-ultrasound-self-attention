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
from lightning.pytorch.callbacks import EarlyStopping, DeviceStatsMonitor, ModelCheckpoint, LearningRateMonitor
from tabulate import tabulate

from lightning.pytorch.tuner import Tuner

from args_processing import parse_arguments
from get_sets import get_sets, get_class_weights

args = parse_arguments()

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
# from dataset import RichHDF5Dataset, HDF5Dataset, split_strategy, plot_split_graphs, reduce_sets
# from lightning_modules.ViTLightningModule import ViTLightningModule
# from lightning_modules.ResNet18LightningModule import ResNet18LightningModule
# from lightning_modules.BEiTLightningModule import BEiTLightningModule
from lightning_modules.LUSModelLightningModule import LUSModelLightningModule
from lightning_modules.LUSDataModule import LUSDataModule
# from lightning_modules.ConfusionMatrixCallback import ConfusionMatrixCallback

# ---------------------------------- Dataset --------------------------------- #

sets, split_info = get_sets(
    args.rseed,
    data_file,
    args.hospitaldict_path,
    args.train_ratio,
    args.trim_data
)

lus_data_module = LUSDataModule(sets["train"], 
                                sets["test"],
                                sets["val"],
                                args.num_workers, 
                                args.batch_size,
                                args.mixup)


train_weight_tensor = get_class_weights(sets["train_indices"], split_info)
# ---------------------------------------------------------------------------- #
#                         Model & trainer configuration                        #
# ---------------------------------------------------------------------------- #

# ------------------- Model hyperparameters & instantiation ------------------ #

print("\n\nModel configuration...")
print('=' * 80)

# configuration = {
#     "num_labels": 4,
#     "num_attention_heads": 4,
#     "num_hidden_layers":4
# }
hyperparameters = {
  "num_classes": 4,
  "optimizer": args.optimizer,
  "lr": args.lr,
  "batch_size": args.batch_size,
  "weight_decay": args.weight_decay,    
  "momentum": args.momentum,
  "label_smoothing": args.label_smoothing,
#   "class_weights": class_weights
#   "configuration": configuration
}
# Instantiate lightning model

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


table_data = []
table_data.append(["MODEL HYPERPARAMETERS"])
table_data.append(["model", args.model])
for key, value in hyperparameters.items():
    if key not in ["train_dataset", "test_dataset"]:
      table_data.append([key, value])

table = tabulate(table_data, headers="firstrow", tablefmt="fancy_grid")
print(table)
# model.to('cuda')

# print(f"\n\n{model.config}\n")

# --------------------------- Trainer configuration -------------------------- #

print("\n\nTrainer configuration...")
print('=' * 80)
# Callbacks
# -EarlyStopping
early_stop_callback = EarlyStopping(
    monitor='validation_loss',
    patience=20,
    strict=False,
    verbose=False,
    mode='min'
)

# -Logger configuration
version = f"V{args.version}" if args.version else "V1"
version = version.strip()

version = f"V{args.version}" if args.version else "V1"

name_version = f"_{version}"
name_trained = "_pretrained" if args.pretrained==True else ""
name_layer = f"_{args.freeze_layers}" if args.freeze_layers else ""
name_trimmed = "_trimmed" if args.trim_data else ""

model_name = f"{args.model}{name_version}{name_trained}{name_layer}{name_trimmed}/{args.optimizer}/ds_{args.train_ratio}_lr{args.lr}_bs{args.batch_size}"
logger = TensorBoardLogger("tb_logs", name=model_name, version=version)
# -Checkpointing
#   Checkpoints directory
checkpoint_dir = f"{working_dir}/checkpoints/{model_name}"
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, 
                                      save_top_k=1,
                                      mode="min",
                                      monitor="validation_loss",
                                      save_last=True,
                                      save_on_train_epoch_end=False,
                                      verbose=True,
                                      filename="{epoch}-{validation_loss:.4f}")

callbacks = [
            # DeviceStatsMonitor(),
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
table_data = []
table_data.append(["TRAINER ARGUMENTS"])
for key, value in trainer_args.items():
    if key not in ["callbacks", "logger"]:
        table_data.append([key, value])

table = tabulate(table_data, headers="firstrow", tablefmt="fancy_grid")
print("\n\n" + table)
print(f"Model checkpoints directory is {checkpoint_dir}")
print("\n\n")

# Trainer 
trainer = Trainer(**trainer_args,
                #   detect_anomaly=True,
                #   overfit_batches=0.01,
                #   val_check_interval=0.25,
                #   gradient_clip_val=0.1,
                    # benchmark=True,
                    accelerator="gpu",
                    default_root_dir = checkpoint_dir)

# Trainer tuner
# tuner = Tuner(trainer)
# tuner.lr_find(model)
# Print the information of each callback
print("\n\n" + "-" * 20)
print("Trainer Callbacks:")
print("-" * 20 + "\n\n")
for callback in trainer.callbacks:
    print(f"- {type(callback).__name__}")


# ---------------------------- Model fit and test ---------------------------- #
def check_checkpoint(chkp):

    print("Checkpoint mode activated...\n")

    if (chkp == "best"):
        print("Loading BEST checkpoint...\n")

    if (chkp == "last"):
        print("Loading LAST checkpoint...\n")

    else:
    # Check if checkpoint file exists
        if not os.path.isfile(chkp):
            print(f"Checkpoint file '{chkp}' does not exist. Exiting...")
            exit()

    print(f"Loading checkpoint from PATH: '{chkp}'...\n")

# model = torch.compile(model, mode="reduce-overhead")

if args.mode == "train":
    print("\n\nTRAINING MODEL...")
    print('=' * 80 + "\n")
    if args.chkp:
        check_checkpoint(args.chkp)
        trainer.fit(model, lus_data_module, ckpt_path=args.chkp)
        
    else:
        print("Instantiating trainer without checkpoint...")
        trainer.fit(model, lus_data_module)
        
if args.mode == "test":
    print("\n\nTESTING MODEL...")
    print('=' * 80 + "\n")
    if args.chkp:
        check_checkpoint(args.chkp)
        trainer.test(model, lus_data_module, ckpt_path=args.chkp)
    else:
        print("No checkpoint provided, testing from scratch...")
        trainer.test(model, lus_data_module)

