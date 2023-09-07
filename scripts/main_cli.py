import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import h5py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, DeviceStatsMonitor
import torchmetrics
from transformers import ViTForImageClassification
import h5py


# Import custom libs
import sys
sys.path.append('/content/drive/MyDrive/Tesi/Transformer/Testing/lib')

import dataset_utility as util
from LazyLoadingDataset import LazyLoadingDataset
from HDF5Dataset import HDF5Dataset

# Dataset ---------------------------------------------------------------
input_file = "/content/drive/MyDrive/Tesi/dataset/dataset_full.h5"
def collate_fn(examples):
    frames = torch.stack([example[0] for example in examples])  # Extract the preprocessed frames
    scores = torch.tensor([example[1] for example in examples])  # Extract the scores
    return {"pixel_values": frames, "labels": scores}
batch_size = 90
num_workers = 4

print("Setting up the Dataset")
dataset = HDF5Dataset(input_file)
dataset_size = len(dataset)

train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

# Parse arguments---------------------------------------------------------------
"""parser = ArgumentParser()

parser.add_argument("--batch_size", type=int, default=90)
parser.add_argument("--max_epochs", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--accumulate_grad_batches", type=int, default=4)
parser.add_argument("--accelerator", type=str, default="gpu")


# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()"""

id2label = {0: 'no', 1: 'yellow', 2: 'orange', 3: 'red'}
label2id = {"no": 0, "yellow": 1, "orange": 2, "red": 3}


# Model class ------------------------------------------------------------
class ViTLightningModule(pl.LightningModule):
    def __init__(self, num_labels=4):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                              num_labels=4,
                                                              id2label=id2label,
                                                              label2id=label2id)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits
        
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]
        #accuracy = torchmetrics.functional.accuracy(predictions, labels, task="multiclass", num_classes=4)

        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        print(f"val_acc: {accuracy}")
        print(f"val_loss: {loss}")     
        self.log("validation_loss", loss, on_epoch=True, sync_dist=True)
        self.log("validation_accuracy", accuracy, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-5)

    def train_dataloader(self):
      return DataLoader(train_dataset,
                            shuffle=True,
                            collate_fn=collate_fn,
                            batch_size=batch_size,
                            num_workers=num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(val_dataset,
                            collate_fn=collate_fn,
                            batch_size=batch_size,
                            num_workers=num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(test_dataset,
                            collate_fn=collate_fn,
                            batch_size=batch_size,
                            num_workers=num_workers, persistent_workers=True)

print("-"*20)
# MAIN -------------------------------------------------------------------------
def cli_main():
    """early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )
    callbacks=[early_stop_callback, DeviceStatsMonitor()]
    trainer = Trainer(precision=16,
                      accelerator=args.accelerator,
                      accumulate_grad_batches=args.accumulate_grad_batches,
                      max_epochs=args.max_epochs,
                      callbacks=callbacks,
                      strategy="ddp")
    model = ViTLightningModule()
    us_dm = USDataModule()"""
    cli = LightningCLI(ViTLightningModule)
                        #seed_everything_default = 11)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block