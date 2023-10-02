from argparse import ArgumentParser
import lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from tabulate import tabulate
import warnings



# ------------------------------ Parse arguments ----------------------------- #
parser = ArgumentParser()

parser.add_argument("--model", type=str)
parser.add_argument("--working_dir_path", type=str)
parser.add_argument("--dataset_h5_path", type=str)
parser.add_argument("--hospitaldict_path", type=str)
parser.add_argument("--rseed", type=int)
parser.add_argument("--train_ratio", type=int, default=0.7)
parser.add_argument("--batch_size", type=int, default=90)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--pretrained", type=str, default=True)
parser.add_argument("--max_epochs", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--accumulate_grad_batches", type=int, default=4)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--precision", default=16)
parser.add_argument("--disable_warnings", type=bool, default=False)

# Parse the user inputs and defaults (returns a argparse.Namespace)

print("\n" + "-"*80 + "\n")
args = parser.parse_args()
pl.seed_everything(args.rseed)

# ------------------------------ Warnings config ----------------------------- #
print(args.disable_warnings)
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
from data_setup import HDF5Dataset, splitting_strategy, FrameTargetDataset
from ViTLightningModule import ViTLightningModule
from ResNet18LightningModule import ResNet18LightningModule

# ---------------------------------- Dataset --------------------------------- #
dataset = HDF5Dataset(args.dataset_h5_path)

train_subset, test_subset, split_info = splitting_strategy(dataset, 
                                                           args.hospitaldict_path, 
                                                           args.rseed, 
                                                           args.train_ratio)

train_dataset = FrameTargetDataset(train_subset)
test_dataset = FrameTargetDataset(test_subset)

print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")


# ---------------------------------------------------------------------------- #
#                         Model & trainer configuration                        #
# ---------------------------------------------------------------------------- #

# ------------------- Model hyperparameters & instantiation ------------------ #
hyperparameters = {
  "train_dataset": train_dataset,
  "test_dataset": test_dataset,
  "batch_size": args.batch_size,
  "lr": args.lr,
  "optimizer": args.optimizer,
  "num_workers": args.num_workers if args.accelerator != "mps" else 0,
  "pretrained": args.pretrained
}
# Instantiate lightning model
if args.model == "google_vit":
  model = ViTLightningModule(**hyperparameters)
elif args.model == "resnet18":
  model =  ResNet18LightningModule(**hyperparameters)
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


# --------------------------- Trainer configuration -------------------------- #
# Callbacks
early_stop_callback = EarlyStopping(
    monitor='validation_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)
# callbacks=[early_stop_callback, DeviceStatsMonitor()]
callbacks = []

# Logger onfiguration
name_trained = "pretrained_" if args.pretrained==True else ""
model_name = f"{name_trained}resnet18_{args.optimizer}_{args.lr}_{args.batch_size}"
logger = TensorBoardLogger("tb_logs", name=model_name)

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
print("\n\n" + table + "\n\n")

# Trainer 
trainer = Trainer(**trainer_args)

# Print the information of each callback
print("\n\n" + "-" * 20)
print("Trainer Callbacks:")
print("-" * 20)
for callback in trainer.callbacks:
    print(f"- {type(callback).__name__}")
    
# ---------------------------- Model fit and test ---------------------------- #
trainer.fit(model)
trainer.test()