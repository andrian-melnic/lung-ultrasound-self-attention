from argparse import ArgumentParser
from torch.utils.data import random_split

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, DeviceStatsMonitor
import h5py

working_dir = "/content/drive/MyDrive/Tesi/lus-dl-framework"
libraries_dir = working_dir + "/libraries"
data_file = working_dir + "/data/dataset_full.h5"

# Import custom libs
# import sys
# sys.path.append(working_dir)
from libraries.HDF5Dataset import HDF5Dataset
from libraries.ViTLightningModule import ViTLightningModule, pp_frames

# Dataset ---------------------------------------------------------------


print("Setting up the Dataset")
dataset = HDF5Dataset(data_file, working_dir, pp_frames)
dataset_size = len(dataset)

train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

# Parse arguments---------------------------------------------------------------
parser = ArgumentParser()

parser.add_argument("--batch_size", type=int, default=90)
parser.add_argument("--max_epochs", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--accumulate_grad_batches", type=int, default=4)
parser.add_argument("--accelerator", type=str, default="gpu")


# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()



batch_size = args.batch_size
num_workers = args.num_workers

# MAIN -------------------------------------------------------------------------

early_stop_callback = EarlyStopping(
    monitor='validation_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)
callbacks=[early_stop_callback, DeviceStatsMonitor()]
model = ViTLightningModule(train_dataset, test_dataset, val_dataset, 
                          args.batch_size, 
                          args.num_workers)
trainer = Trainer(precision=16,
                  accelerator=args.accelerator,
                  accumulate_grad_batches=args.accumulate_grad_batches,
                  max_epochs=args.max_epochs,
                  callbacks=callbacks,
                  strategy="ddp")
trainer.fit(model)
trainer.test()