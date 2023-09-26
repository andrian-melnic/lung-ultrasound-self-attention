from torch.utils.data import Dataset, Subset
import os
import h5py
from tqdm import tqdm
import pickle
import random
import torchvision.transforms as transforms
from collections import defaultdict
from transformers import ViTImageProcessor
import torch
import torch.nn as nn
import kornia.augmentation as K



class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.index_map_path = os.path.dirname(file_path) + "/index_map_" + os.path.splitext(os.path.basename(file_path))[0] + ".pkl"
        self.h5file = h5py.File(file_path, 'r')
        self.group_names = list(self.h5file.keys())
        self.total_videos = sum(len(self.h5file[group_name]) for group_name in self.group_names)
        self.check_for_index_map()
        #self.total_frames, self.frame_index_map = self.calculate_total_frames_and_index_map()

        print(f"\n{self.total_videos} videos ({self.total_frames} frames) loaded.")


        
    def check_for_index_map(self):
      try:
          with open(self.index_map_path, 'rb') as f:
              print("Serialized frame index map FOUND.\n")
              saved_data = pickle.load(f)
              self.total_frames = saved_data['total_frames']
              self.frame_index_map = saved_data['frame_index_map']
              print("Loaded serialized data.\n")
      except FileNotFoundError:
          print("Serialized frame index map NOT FOUND\n")
          self.total_frames, self.frame_index_map = self.calculate_total_frames_and_index_map()
          # Save calculated data to a pickle file
          with open(self.index_map_path, 'wb') as f:
              saved_data = {'total_frames': self.total_frames, 'frame_index_map': self.frame_index_map}
              pickle.dump(saved_data, f)
          print("\nIndex map calculated and saved")


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

    def __getitem__(self, index):
        if index < 0 or index >= self.total_frames:
            raise IndexError("Index out of range")

        group_name, video_name = self.frame_index_map[index]
        video_group = self.h5file[group_name][video_name]
        frame_data = video_group['frames'][f'frame_{index}'][:]
        target_data = video_group['targets'][f'target_{index}'][:]

        # get metadata
        patient = video_group.attrs['patient']
        medical_center = video_group.attrs['medical_center']

        #return index, frame_tensor, target_data
        return index, frame_data, target_data, patient, medical_center

def splitting_strategy(dataset, pkl_file, rseed, train_ratio=0.7):
    # iteration seed
    random.seed(rseed)

    # Check if the pickle file exists
    if os.path.exists(pkl_file):
        # If the pickle file exists, load the data from it
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            medical_center_patients = data['medical_center_patients']
            data_index = data['data_index']
    else:
        # If the pickle file doesn't exist, create the data
        medical_center_patients = defaultdict(set)
        data_index = {}
        for index, (_, _, _, patient, medical_center) in enumerate(dataset):
            medical_center_patients[medical_center].add(patient)
            data_index[index] = (patient, medical_center)

        # Save the data to a pickle file
        data = {
            'medical_center_patients': medical_center_patients,
            'data_index': data_index
        }

        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)

    # Split the patients for each medical center
    train_indices = []
    test_indices = []

    # Lists to store statistics about medical centers and patients
    train_patients_by_center = defaultdict(set)
    test_patients_by_center = defaultdict(set)
    frame_counts_by_center = defaultdict(int)
    frame_counts_by_center_patient = defaultdict(lambda: defaultdict(int))

    for medical_center, patients in medical_center_patients.items():
        patients = list(patients)
        random.shuffle(patients)
        split_index = int(train_ratio * len(patients))

        for index, (patient, center) in data_index.items():
            if center == medical_center:
                if patient in patients[:split_index]:
                    train_indices.append(index)
                    train_patients_by_center[medical_center].add(patient)
                else:
                    test_indices.append(index)
                    test_patients_by_center[medical_center].add(patient)

                frame_counts_by_center[medical_center] += 1
                frame_counts_by_center_patient[medical_center][patient] += 1

    # Create training and test subsets
    train_dataset_subset = Subset(dataset, train_indices)
    test_dataset_subset = Subset(dataset, test_indices)

    # Sum up statistics info
    split_info = {
        'medical_center_patients': medical_center_patients,
        'frame_counts_by_center': frame_counts_by_center,
        'train_patients_by_center': train_patients_by_center,
        'test_patients_by_center': test_patients_by_center,
        'frame_counts_by_center_patient': frame_counts_by_center_patient,
        'total_train_frames': len(train_indices),
        'total_test_frames': len(test_indices)
    }

    return train_dataset_subset, test_dataset_subset, split_info

# Custom replica class of the dataset to train the neural network (return -> [frame,target])
class FrameTargetDataset(Dataset):
    def __init__(self, hdf5_dataset):
        self.hdf5_dataset = hdf5_dataset
        # self.resize_size = (100, 150)

    def __len__(self):
        return len(self.hdf5_dataset)

    def __getitem__(self, index):
        _, frame_data, target_data, _, _ = self.hdf5_dataset[index]

        frame_tensor = self.pp_frames(frame_data)
        # Target data to integer scores
        target_data = torch.tensor(sum(target_data))
        # Apply Resize transformation
        # frame_tensor = transforms.ToTensor()(frame_data)
        # frame_tensor = transforms.Resize(self.resize_size, antialias=True)(frame_tensor)
        # frame_tensor = frame_tensor.permute(0, 2, 1) # Move channels to the last dimension (needed after resize)

        return frame_tensor, target_data
    def pp_frames(self, frame_data):
        # processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        # image_mean = processor.image_mean
        # image_std = processor.image_std
        # size = processor.size["height"]
        size = (224, 224)
        image_mean = frame_data.mean()
        image_std = frame_data.std()

        frame_tensor = transforms.ToTensor()(frame_data)
        frame_tensor = transforms.CenterCrop(size)(frame_tensor)
        frame_tensor = transforms.Normalize(mean=image_mean, std=image_std)(frame_tensor)

        return frame_tensor.permute(0, 2, 1)  
      
      
      
class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self):
        super().__init__()

        self.transforms = torch.nn.Sequential(
            K.RandomRotation(degrees=(0, 20)),
            K.RandomAffine(degrees=(-10, 10), scale=(0.8, 1.2))
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x_out = self.transforms(x)  # BxCxHxW
        return x_out

