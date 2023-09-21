from argparse import ArgumentParser
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, DeviceStatsMonitor

# ------------------------------ Parse arguments ----------------------------- #
parser = ArgumentParser()

parser.add_argument("--working_dir_path", type=str)
parser.add_argument("--dataset_h5_path", type=str)
parser.add_argument("--hospitaldict_path", type=str)
parser.add_argument("--rseed", type=int)
parser.add_argument("--batch_size", type=int, default=90)
parser.add_argument("--max_epochs", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--accumulate_grad_batches", type=int, default=4)
parser.add_argument("--accelerator", type=str, default="gpu")

# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()
pl.seed_everything(args.rseed)

# ----------------------------------- Paths ---------------------------------- #
working_dir = args.working_dir_path
data_file = args.dataset_h5_path
libraries_dir = working_dir + "/libraries"
#"/Users/andry/Documents/GitHub/lus-dl-framework/data/test_dataset_clinic_eval.h5"
# data_file = "/content/drive/MyDrive/Tesi/dataset/dataset_full.h5"

# ---------------------------- Import custom libs ---------------------------- #
import sys
sys.path.append(working_dir)
from data_setup import HDF5Dataset, splitting_strategy, FrameTargetDataset
from ViTLightningModule import ViTLightningModule
from ResNet18LightningModule import ResNet18LightningModule

# ---------------------------------- Dataset --------------------------------- #
dataset = HDF5Dataset(args.dataset_h5_path)

train_subset, test_subset, split_info = splitting_strategy(dataset, args.hospitaldict_path, args.rseed, 0.7)

train_dataset = FrameTargetDataset(train_subset)
test_dataset = FrameTargetDataset(test_subset)

print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

# ----------------------------------- Main ----------------------------------- #

early_stop_callback = EarlyStopping(
    monitor='validation_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)
callbacks=[early_stop_callback, DeviceStatsMonitor()]
#model = ViTLightningModule(train_dataset, test_dataset, 
#                          args.batch_size, 
#                          args.num_workers)

model = ResNet18LightningModule(train_dataset, test_dataset, 
                                args.batch_size, 
                                args.num_workers,
                                "sgd")
trainer = Trainer(precision=16,
                  accelerator=args.accelerator,
                  accumulate_grad_batches=args.accumulate_grad_batches,
                  max_epochs=args.max_epochs,
                  callbacks=callbacks,
                  devices=1)
                  # strategy="ddp"
trainer.fit(model)
trainer.test()