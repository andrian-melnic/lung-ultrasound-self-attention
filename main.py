import warnings
import os
import glob
import pickle
import torch
import lightning as pl
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.utils.class_weight import compute_class_weight
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, DeviceStatsMonitor, ModelCheckpoint, LearningRateMonitor
from tabulate import tabulate
from torch.utils.data import Subset
from lightning.pytorch.tuner import Tuner

import json

# ------------------------------ Parse arguments ----------------------------- #

# Parse command-line arguments
parser = ArgumentParser()

allowed_models = ["google_vit", 
                  "resnet18",
                  "resnet50",
                  "beit", 
                  'timm_bot', 
                  "botnet18", 
                  "botnet50",
                  "vit",
                  "swin_vit",
                  "simple_vit"]

allowed_modes = ["train", "test", "train_test"]
parser.add_argument("--model", type=str, choices=allowed_models)
parser.add_argument("--mode", type=str, choices=allowed_modes)
parser.add_argument("--version", type=str)
parser.add_argument("--working_dir_path", type=str)
parser.add_argument("--dataset_h5_path", type=str)
parser.add_argument("--hospitaldict_path", type=str)
parser.add_argument("--trim_data", type=float)
parser.add_argument("--chkp", type=str)
parser.add_argument("--rseed", type=int)
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--batch_size", type=int, default=90)
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--weight_decay", type=float, default=0.001)
parser.add_argument("--momentum", type=float, default=0.001)
parser.add_argument("--label_smoothing", type=float, default=0.1)
parser.add_argument("--max_epochs", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--accumulate_grad_batches", type=int, default=4)
parser.add_argument("--precision", default=32)
parser.add_argument("--disable_warnings", dest="disable_warnings", action='store_true')
parser.add_argument("--pretrained", dest="pretrained", action='store_true')
parser.add_argument("--freeze_layers", type=str)
parser.add_argument("--test", dest="test", action='store_true')
parser.add_argument("--mixup", dest="mixup", action='store_true')

# Add an argument for the configuration file
parser.add_argument('--config', type=str, help='Path to JSON configuration file')

args = parser.parse_args()

# -------------------------------- json config ------------------------------- #

config_path = 'configs/configs.json'
selected_config = None
# If a configuration file was provided, load it
if args.config:
    with open(config_path, 'r') as f:
        configurations = json.load(f)
    for config in configurations:
        if config['model'] == args.config:
            selected_config = config
            break

    # Override the command-line arguments with the configuration file
    for key, value in selected_config.items():
        if hasattr(args, key):
            setattr(args, key, value)

print(f"args are: {args}")

print("\n" + "-"*80 + "\n")
pl.seed_everything(args.rseed)
print("\n" + "-"*80 + "\n")

if torch.cuda.is_available():
    dev = torch.cuda.get_device_name()
    accelerator = "gpu"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
elif torch.backends.mps.is_built():
    accelerator="mps"
    dev = "mps"  
    torch.set_default_device(f"{dev}")
else:
    dev = "cpu"

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

dataset = HDF5Dataset(args.dataset_h5_path)

train_indices = []
val_indices = []
test_indices = []

train_ratio = args.train_ratio
test_ratio = (1 - train_ratio)/2
val_ratio = test_ratio
ratios = [train_ratio, test_ratio, val_ratio]


def create_default_dict():
    return defaultdict(float)
def initialize_inner_defaultdict():
    return defaultdict(int)

print(f"Split ratios: {ratios}")
train_indices, val_indices, test_indices, split_info = split_dataset(
    rseed=args.rseed,
    dataset=dataset,
    pkl_file=args.hospitaldict_path,
    ratios=ratios)

# Create training and test subsets
train_subset = Subset(dataset, train_indices)
test_subset = Subset(dataset, test_indices)  
val_subset = Subset(dataset, val_indices)  


if args.trim_data:
    train_indices_trimmed, \
    val_indices_trimmed, \
    test_indices_trimmed = reduce_sets(args.rseed,
                                       train_subset,
                                       val_subset,
                                       test_subset,
                                       args.trim_data)
    
    train_subset = Subset(dataset, train_indices_trimmed)
    test_subset = Subset(dataset, test_indices_trimmed)
    val_subset = Subset(dataset, val_indices_trimmed)
    
    train_indices = train_indices_trimmed
    val_indices = val_indices_trimmed
    test_indices = test_indices_trimmed


train_dataset = FrameTargetDataset(train_subset)
test_dataset = FrameTargetDataset(test_subset)
val_dataset = FrameTargetDataset(val_subset)
 
print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")    
print(f"Validation size: {len(val_dataset)}")    

lus_data_module = LUSDataModule(train_dataset, 
                                test_dataset,
                                val_dataset,
                                args.num_workers, 
                                args.batch_size,
                                args.mixup)
# ---------------------------------------------------------------------------- #
#                                 Class Weights                                #
# ---------------------------------------------------------------------------- #
# Retrieves the dataset's labels
ds_labels = split_info['labels']

# Extract the train and test set labels
y_train_labels = np.array(ds_labels)[train_indices]
# y_test_labels = np.array(ds_labels)[test_indices]

# Calculate class balance using 'compute_class_weight'
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train_labels), 
                                     y=y_train_labels)

weights_tensor = torch.Tensor(class_weights)
print("Class Weights: ", class_weights)


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
                                class_weights=weights_tensor,
                                pretrained=args.pretrained,
                                freeze_layers=freeze_layers)


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
    patience=10,
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
                    accelerator=accelerator,
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

