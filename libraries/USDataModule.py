import lightning.pytorch_pytorch as pl
import torch
from torch.utils.data import random_split, DataLoader
import h5py


# Import custom libs
import sys
sys.path.append('/content/drive/MyDrive/Tesi/lus-dl-framework/libraries')

import dataset_utility as util
from LazyLoadingDataset import LazyLoadingDataset
from HDF5Dataset import HDF5Dataset


def collate_fn(examples):
    frames = torch.stack([example[0] for example in examples])  # Extract the preprocessed frames
    scores = torch.tensor([example[1] for example in examples])  # Extract the scores
    return {"pixel_values": frames, "labels": scores}


class USDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/content/drive/MyDrive/Tesi/dataset/dataset_full.h5"):
        super().__init__()
        self.input_file = data_dir
        self.batch_size = 90
        self.num_workers = 4
        #self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    #def prepare_data(self):

    def setup(self, stage: str):
        dataset = HDF5Dataset(self.input_file)
        dataset_size = len(dataset)

        train_size = int(0.7 * dataset_size)
        val_size = int(0.15 * dataset_size)
        test_size = len(dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])
        print(f"Train size: {len(self.train_dataset)}")
        print(f"Val size: {len(self.val_dataset)}")
        print(f"Test size: {len(self.test_dataset)}")


   