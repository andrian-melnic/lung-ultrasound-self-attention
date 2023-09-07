import h5py
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms as transforms

class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.h5file = h5py.File(file_path, 'r')
        self.group_names = list(self.h5file.keys())
        self.total_frames = sum(len(self.h5file[group_name][video_name]['frames']) for group_name in self.group_names for video_name in self.h5file[group_name])

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):
        if index < 0 or index >= self.total_frames:
            raise IndexError("Index out of range")

        for group_name in self.group_names:
            for video_name in self.h5file[group_name]:
                video_group = self.h5file[group_name][video_name]
                frame_idx_start = video_group.attrs['frame_idx_start']
                frame_idx_end = video_group.attrs['frame_idx_end']
                if frame_idx_start <= index <= frame_idx_end:
                    frame_data = video_group['frames'][f'frame_{index}'][:]
                    target_data = video_group['targets'][f'target_{index}'][:]
                    return index, frame_data, target_data

        return None