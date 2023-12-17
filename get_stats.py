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

device = torch.device('cuda')
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

    sets, split_info = get_sets(args)
    lus_data_module = LUSDataModule(sets["train"], 
                                    sets["test"],
                                    sets["val"],
                                    args.num_workers, 
                                    args.batch_size,
                                    args.mixup)

    print("\n- Train set class weights: ")
    train_weight_tensor = get_class_weights(sets["train_indices"], split_info)
    print("\n- Val set class weights: ")
    get_class_weights(sets["val_indices"], split_info)
    print("\n- Test set class weights: ")
    get_class_weights(sets["test_indices"], split_info)
    
    import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

def calculate_mean_std(dataset):
    """
    Calculate the mean and standard deviation for each channel of the dataset.

    Args:
        dataset (Dataset): The dataset for which to calculate mean and std.

    Returns:
        tuple: A tuple containing mean and std for each channel.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    channel_sum = 0.0
    channel_sum_squared = 0.0
    num_batches = 0

    for data, _ in tqdm(loader, desc="Computing mean and std", leave=False):
        data=data.to(device)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        channel_sum += data.mean(2).sum(0)
        channel_sum_squared += (data ** 2).mean(2).sum(0)
        num_batches += batch_samples

    mean = channel_sum / num_batches
    std = torch.sqrt(channel_sum_squared / num_batches - mean ** 2)

    return mean.tolist(), std.tolist()

# Assuming you have created an instance of your dataset
# For example:
# my_dataset = FrameTargetDataset(hdf5_dataset, pretrained=False)

# Calculate mean and std
mean, std = calculate_mean_std(sets["train"])

# Round the values to four decimal places
rounded_mean = [round(value, 5) for value in mean]
rounded_std = [round(value, 5) for value in std]

print(f"\nRounded Mean:", rounded_mean)
print(f"Rounded Std:", rounded_std)