from torch.utils.data import Dataset, Subset
import os
import h5py
from tqdm import tqdm
import pickle
import random
import torchvision.transforms as transforms
from collections import defaultdict
import torch
import torch.nn as nn
import kornia.augmentation as K


# ---------------------------------------------------------------------------- #
#                                  HDF5Dataset                                 #
# ---------------------------------------------------------------------------- #


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
      """
      Check if the index map file exists and load it if found. 
      If not found, calculate the index map and save it to a pickle file.

      Parameters:
          None

      Returns:
          None
      """        
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
        """
        Calculates the total number of frames and creates an index map for each frame.

        Returns:
            total_frames (int): The total number of frames.
            frame_index_map (dict): A dictionary mapping frame indices to their corresponding group and video names.
        """
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
      
      
    def split_dataset(self, pkl_file, rseed, train_ratio=0.7):
      """
      Split the dataset into training and test subsets based on a given pickle file.

      Parameters:
          pkl_file (str): The path to the pickle file.
          rseed (int): The seed for random number generation.
          train_ratio (float, optional): The ratio of data to be assigned to the training subset. Defaults to 0.7.

      Returns:
          train_dataset_subset (Subset): The training subset of the dataset.
          test_dataset_subset (Subset): The test subset of the dataset.
          split_info (dict): A dictionary containing various statistics and information about the split.

      Raises:
          FileNotFoundError: If the pickle file does not exist.

      """

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
          for index, (_, _, _, patient, medical_center) in enumerate(tqdm(self)):
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
      train_dataset_subset = Subset(self, train_indices)
      test_dataset_subset = Subset(self, test_indices)

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

      return train_dataset_subset, test_dataset_subset, split_info, train_indices, test_indices
    
    
    def __len__(self):
        """
        Returns the total number of frames in the dataset.
        """
        return self.total_frames
    
    
    def __getitem__(self, index):
        """
        Retrieves the data for a specific frame at the given index.
    
        Args:
            index (int): The index of the frame to retrieve.
    
        Returns:
            tuple: A tuple containing the index, frame data, target data, patient, and medical center.
        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= self.total_frames:
            raise IndexError("Index out of range")
    
        group_name, video_name = self.frame_index_map[index]
        video_group = self.h5file[group_name][video_name]
        frame_data = video_group['frames'][f'frame_{index}'][:]
        target_data = video_group['targets'][f'target_{index}']
    
        # Get metadata
        patient = video_group.attrs['patient']
        medical_center = video_group.attrs['medical_center']
    
        return index, frame_data, target_data, patient, medical_center


# ---------------------------------------------------------------------------- #
#                              FrameTargetDataset                              #
# ---------------------------------------------------------------------------- #

# Custom replica class of the dataset to train the neural network (return -> [frame,target])
class FrameTargetDataset(Dataset):
    def __init__(self, hdf5_dataset, transform=None):
        """
        Initialize the dataset.

        Args:
            hdf5_dataset (h5py.Dataset): The HDF5 dataset.
        """
        self.hdf5_dataset = hdf5_dataset
        self.transform = transform
        self.resize_size = (224, 224)

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.hdf5_dataset)

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): The index of the item.

        Returns:
            tuple: A tuple containing the frame tensor and the target data.
        """
        _, frame_data, target_data, _, _ = self.hdf5_dataset[index]

        # frame_tensor = self.pp_frames(frame_data)
        frame_tensor = transforms.ToTensor()(frame_data)
        frame_tensor = transforms.Resize(self.resize_size, antialias=True)(frame_tensor)
        if self.transform is not None:
            frame_tensor = transforms.CenterCrop(self.transform.size["height"])(frame_tensor)
            frame_tensor = transforms.Normalize(mean=self.transform.image_mean, std=self.transform.image_std)(frame_tensor)
        frame_tensor = frame_tensor.permute(0, 1, 2) # Move channels to the last dimension (needed after resize)
            
        # Target data to integer scores
        # target_data = torch.tensor(sum(target_data))
        target_data = int(target_data[()])

        

        return frame_tensor, target_data
    
    def set_transform(self, transform):
        self.transform = transform


    def pp_frames(self, frame_data):
        """
        Preprocess the frame data.

        Args:
            frame_data: The frame data.

        Returns:
            torch.Tensor: The preprocessed frame tensor.
        """

        size = (224, 224)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        frame_tensor = transforms.ToTensor()(frame_data)
        frame_tensor = transforms.Resize(size)(frame_tensor)
        frame_tensor = transforms.Normalize(mean=image_mean, std=image_std)(frame_tensor)

        return frame_tensor.permute(0, 2, 1)
      
      
# ---------------------------------------------------------------------------- #
#                               DataAugmentation                               #
# ---------------------------------------------------------------------------- #

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            K.RandomRotation(degrees=(-20, 20)),  # random rotation between -20 to 20 degrees
            K.RandomAffine(degrees=(-10, 10), scale=(0.8, 1.2))  # random affine transformation with rotation between -10 to 10 degrees and scale between 0.8 to 1.2
        )

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x):
        """Perform data augmentation on input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape BxCxHxW.

        Returns:
            torch.Tensor: Augmented tensor of shape BxCxHxW.
        """
        x_out = self.transforms(x)
        return x_out