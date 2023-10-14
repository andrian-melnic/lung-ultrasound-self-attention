from argparse import ArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger
import warnings
import os
import glob
import pickle
import lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, DeviceStatsMonitor, ModelCheckpoint
from tabulate import tabulate
from torch.utils.data import Subset
from lightning.pytorch.tuner import Tuner
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 


# ------------------------------ Parse arguments ----------------------------- #
parser = ArgumentParser()

parser.add_argument("--model", type=str)
parser.add_argument("--working_dir_path", type=str)
parser.add_argument("--dataset_h5_path", type=str)
parser.add_argument("--hospitaldict_path", type=str)
parser.add_argument("--checkpoint_path", type=str)
parser.add_argument("--rseed", type=int)
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--batch_size", type=int, default=90)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--max_epochs", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--accumulate_grad_batches", type=int, default=4)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--precision", default=32)
parser.add_argument("--disable_warnings", default=True)
parser.add_argument("--pretrained", default=True)

# Parse the user inputs and defaults (returns a argparse.Namespace)

print("\n" + "-"*80 + "\n")
args = parser.parse_args()
pl.seed_everything(args.rseed)
print("\n" + "-"*80 + "\n")


# ------------------------------ Warnings config ----------------------------- #
if args.disable_warnings == True: 
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
from data_setup import HDF5Dataset, FrameTargetDataset
from lightning_modules.ViTLightningModule import ViTLightningModule
from lightning_modules.ResNet18LightningModule import ResNet18LightningModule
from lightning_modules.BEiTLightningModule import BEiTLightningModule

# ---------------------------------- Dataset --------------------------------- #
dataset = HDF5Dataset(args.dataset_h5_path)

train_indices_path = os.path.dirname(args.dataset_h5_path) + f"/train_indices_{args.train_ratio}.pkl"
test_indices_path = os.path.dirname(args.dataset_h5_path) + f"/test_indices_{args.train_ratio}.pkl"


if os.path.exists(train_indices_path) and os.path.exists(test_indices_path):
    print("Loading pickled indices")
    with open(train_indices_path, 'rb') as train_pickle_file:
        train_indices = pickle.load(train_pickle_file)
    with open(test_indices_path, 'rb') as test_pickle_file:
        test_indices = pickle.load(test_pickle_file)
    # Create training and test subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)  
else:
    train_subset, test_subset, split_info, train_indices, test_indices = dataset.split_dataset(args.hospitaldict_path, 
                                                              args.rseed, 
                                                              args.train_ratio)
    print("Pickling sets...")
    
    # Pickle the indices
    with open(train_indices_path, 'wb') as train_pickle_file:
        pickle.dump(train_indices, train_pickle_file)
    with open(test_indices_path, 'wb') as test_pickle_file:
        pickle.dump(test_indices, test_pickle_file)

# test_subset_size = args.train_ratio/2
# test_subset = Subset(test_subset, range(int(test_subset_size * len(test_indices))))


train_dataset = FrameTargetDataset(train_subset)
test_dataset = FrameTargetDataset(test_subset)

print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")


# ---------------------------------------------------------------------------- #
#                         Model & trainer configuration                        #
# ---------------------------------------------------------------------------- #

# ------------------- Model hyperparameters & instantiation ------------------ #

print("\n\nModel configuration...")
print('=' * 80)

configuration = {
    "num_labels": 4,
    "num_attention_heads": 4,
    "num_hidden_layers":4
}
hyperparameters = {
  "train_dataset": train_dataset,
  "test_dataset": test_dataset,
  "batch_size": args.batch_size,
  "lr": args.lr,
  "optimizer": args.optimizer,
  "num_workers": args.num_workers if args.accelerator != "mps" else 0,
  "pretrained": args.pretrained
#   "configuration": configuration
}
# Instantiate lightning model
if args.model == "google_vit":
  model = ViTLightningModule(**hyperparameters)
elif args.model == "resnet18":
  model =  ResNet18LightningModule(**hyperparameters)
elif args.model == "beit": 
  model =  BEiTLightningModule(**hyperparameters)
else:
  raise ValueError("Invalid model name. Please choose either 'google_vit' or 'resnet18'.")


table_data = []
table_data.append(["MODEL HYPERPARAMETERS"])
table_data.append(["model", args.model])
for key, value in hyperparameters.items():
    if key not in ["train_dataset", "test_dataset"]:
      table_data.append([key, value])

table = tabulate(table_data, headers="firstrow", tablefmt="fancy_grid")
print(table)

# print(f"\n\n{model.config}\n")

# --------------------------- Trainer configuration -------------------------- #

print("\n\nTrainer configuration...")
print('=' * 80)
# Callbacks
# -EarlyStopping
early_stop_callback = EarlyStopping(
    monitor='validation_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)

# -Logger configuration
name_trained = "pretrained_" if args.pretrained==True else ""
model_name = f"{name_trained}{args.model}/{args.optimizer}/{args.lr}_{args.batch_size}"
logger = TensorBoardLogger("tb_logs", name=model_name)

# -Checkpointing
#   Checkpoints directory
checkpoint_dir = f"{working_dir}/checkpoints/{model_name}"
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, 
                                      save_top_k=3,
                                      mode="min",
                                      monitor="training_loss",
                                      save_last=True,
                                      verbose=True)

callbacks=[early_stop_callback, 
           DeviceStatsMonitor(), 
           checkpoint_callback]



print("\n\nTRAINING MODEL...")
print('=' * 80 + "\n")

# Trainer args
trainer_args = {
    "accelerator": args.accelerator,
    "strategy": "ddp" if args.accelerator == "gpu" else "auto",
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
                  default_root_dir = checkpoint_dir)
# Create a Tuner
# tuner = Tuner(trainer)
# tuner.lr_find(model)
# Print the information of each callback
print("\n\n" + "-" * 20)
print("Trainer Callbacks:")
print("-" * 20 + "\n\n")
for callback in trainer.callbacks:
    print(f"- {type(callback).__name__}")
    
# ---------------------------- Model fit and test ---------------------------- #
# Check if checkpoint path is provided
if args.checkpoint_path:
  
    checkpoint_path = args.checkpoint_path
    print("Checkpoint mode activated...\n")
    
    if (checkpoint_path == "best"):
      print("Loading BEST checkpoint...\n")

    if (checkpoint_path == "last"):
      print("Loading LAST checkpoint...\n")

    else:
      # Check if checkpoint file exists
      if not os.path.isfile(checkpoint_path):
          print(f"Checkpoint file '{checkpoint_path}' does not exist. Exiting...")
          exit()
    
    print(f"Loading checkpoint from PATH: '{checkpoint_path}'...\n")
    trainer.fit(model, ckpt_path=checkpoint_path)
else:
    # Instantiate trainer without checkpoint
    print("Instantiating trainer without checkpoint...")
    trainer.fit(model)


print("\n\nTESTING MODEL...")
print('=' * 80 + "\n")
trainer.test()