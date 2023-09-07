import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from transformers import ViTImageProcessor
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)



class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.h5file = h5py.File(file_path, 'r')
        self.group_names = list(self.h5file.keys())
        self.total_videos = sum(len(self.h5file[group_name]) for group_name in self.group_names)
        self.resize_size = (224, 224)
        self.frame_info_path = "/content/drive/MyDrive/Tesi/Transformer/Testing/lib/frame_info.pkl"
         # Try to load serialized data
        try:
            with open(self.frame_info_path, 'rb') as f:
                print("Serialized frame index map FOUND.")
                saved_data = pickle.load(f)
                self.total_frames = saved_data['total_frames']
                self.frame_index_map = saved_data['frame_index_map']
                print("Loaded serialized data.")
        except FileNotFoundError:
            print("Serialized frame index map NOT FOUND.")
            self.total_frames, self.frame_index_map = self.calculate_total_frames_and_index_map()
            # Save calculated data to a pickle file
            with open(self.frame_info_path, 'wb') as f:
                saved_data = {'total_frames': self.total_frames, 'frame_index_map': self.frame_index_map}
                pickle.dump(saved_data, f)
            print("Calculated and saved data.")
        
        print(f"\n{self.total_videos} videos ({self.total_frames} frames).\n\n")

    def calculate_total_frames_and_index_map(self):
        max_frame_idx_end = 0
        frame_index_map = {}

        # Create tqdm progress bar
        with tqdm(total=self.total_videos, desc="Calculating frames and index map", unit='video', dynamic_ncols=True) as pbar:
            for group_name in self.group_names:
                for video_name in self.h5file[group_name]:
                    video_group = self.h5file[group_name][video_name]
                    frame_idx_start = video_group.attrs['frame_idx_start']
                    frame_idx_end = video_group.attrs['frame_idx_end']
                    max_frame_idx_end = max(max_frame_idx_end, frame_idx_end)
                    for i in range(frame_idx_start, frame_idx_end + 1):
                        frame_index_map[i] = (group_name, video_name)
                    pbar.update(1)  # Update progress bar for each video

        total_frames = max_frame_idx_end + 1

        return total_frames, frame_index_map

    def __len__(self):
        return self.total_frames

    # This function transforms the images with the same pre processing operations
    # used for training the ViT
    def pp_frames(self, frame_data):
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        image_mean = processor.image_mean
        image_std = processor.image_std
        size = processor.size["height"]
        normalize = Normalize(mean=image_mean, std=image_std)


        frame_tensor = transforms.ToTensor()(frame_data)
        frame_tensor = transforms.Resize(size, antialias=True)(frame_tensor)
        frame_tensor = transforms.CenterCrop(size)(frame_tensor)
        frame_tensor = transforms.Normalize(mean=image_mean, std=image_std)(frame_tensor)

        return frame_tensor.permute(0, 2, 1)



    def __getitem__(self, index):
        if index < 0 or index >= self.total_frames:
            raise IndexError("Index out of range")

        group_name, video_name = self.frame_index_map[index]
        video_group = self.h5file[group_name][video_name]
        frame_data = video_group['frames'][f'frame_{index}'][:]
        target_data = video_group['targets'][f'target_{index}'][:]

        # Apply ViT Pre Processing
        frame_tensor = self.pp_frames(frame_data)

        # Target data to integer scores
        target_data = torch.tensor(sum(target_data))

        return frame_tensor, target_data


