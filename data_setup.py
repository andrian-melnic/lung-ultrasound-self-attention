from torch.utils.data import Dataset, Subset
import os
import h5py
from tqdm import tqdm
import pickle
import random
from collections import defaultdict
import torch
import torch.nn as nn
from torchvision import transforms
from kornia import image_to_tensor, tensor_to_image


# ---------------------------------------------------------------------------- #
#                                  HDF5Dataset                                 #
# ---------------------------------------------------------------------------- #


class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.index_map_path = os.path.dirname(file_path) + "/frame_index_map.pkl"
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
    def __init__(self, hdf5_dataset, pretrained=False, trainset=False):
        """
        Initialize the dataset.

        Args:
            hdf5_dataset (h5py.Dataset): The HDF5 dataset.
        """
        
        self.hdf5_dataset = hdf5_dataset
        self.trainset = trainset
        self.resize_size = (224, 224)
        self.pretrained = pretrained
        self.image_mean = [0.124, 0.1274, 0.131]
        self.image_std = [0.1621, 0.1658, 0.1717]
        # self.image_mean = [0.485, 0.456, 0.406] if self.pretrained else [0.1236, 0.1268, 0.1301]
        # self.image_std = [0.229, 0.224, 0.225] if self.pretrained else [0.1520, 0.1556, 0.1610]

        print(f"\nimage_mean: {self.image_mean}\nimage_std: {self.image_std}\n")
        

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
        
        self.test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_mean, std=self.image_std)
        ])
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize_size),
            transforms.ToTensor(),
        ])
        if self.trainset:
            # frame_tensor = image_to_tensor(frame_data, keepdim=True).float()
            # frame_tensor = transforms.Resize(self.resize_size)(frame_tensor)
            frame_tensor = self.train_transforms(frame_data)
        else:
            frame_tensor = self.test_transforms(frame_data)
            
        # Target data to integer scores
        # target_data = torch.tensor(sum(target_data))
        target_data = int(target_data[()])

        return frame_tensor, target_data


def _load_dsdata_pickle(dataset, pkl_file):
    # Check if the pickle file exists
    if pkl_file and os.path.exists(pkl_file):
        # If the pickle file exists, load the data from it
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            medical_center_patients = data['medical_center_patients']
            data_index = data['data_index']
            data_map_idxs_pcm = data['data_map_idxs_pcm']
            score_counts = data['score_counts']
            labels = data['labels']
    else:
        # If the pickle file doesn't exist, create the data
        medical_center_patients = defaultdict(set)
        data_index = {}
        data_map_idxs_pcm = defaultdict(list)
        score_counts = defaultdict(int)
        labels = []  # List to store target labels

        for index, (_, _, target_data, patient, medical_center) in enumerate(tqdm(dataset)):
            medical_center_patients[medical_center].add(patient)
            data_index[index] = (patient, medical_center)
            data_map_idxs_pcm[(patient, medical_center)].append(index)
            score_counts[int(target_data[()])] += 1
            labels.append(int(target_data[()]))
        
        # Save the data to a pickle file if pkl_file is provided
        if pkl_file:
            data = {
                'medical_center_patients': medical_center_patients,
                'data_index': data_index,
                'data_map_idxs_pcm': data_map_idxs_pcm,
                'score_counts': score_counts,
                'labels': labels
            }
            
            with open(pkl_file, 'wb') as f:
                pickle.dump(data, f)
    
    return medical_center_patients, data_index, data_map_idxs_pcm, score_counts, labels 
   
def create_default_dict():
    return defaultdict(float)
def initialize_inner_defaultdict():
    return defaultdict(int)

def split_dataset(rseed, dataset, pkl_file, ratios=[0.6, 0.2, 0.2]):
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
        
        split_info_filename = os.path.dirname(pkl_file) + f"/_split_info_{ratios[0]}.pkl"
        train_indices_filename = os.path.dirname(pkl_file) + f"/_train_indices_{ratios[0]}.pkl"
        val_indices_filename = os.path.dirname(pkl_file) + f"/_val_indices_{ratios[1]}.pkl"
        test_indices_filename = os.path.dirname(pkl_file) + f"/_test_indices_{ratios[2]}.pkl"

        if os.path.exists(split_info_filename) and os.path.exists(train_indices_filename) and os.path.exists(val_indices_filename) and os.path.exists(test_indices_filename):
            print("\nSerialized splits found, loading ...\n")
            # Load existing split data
            with open(split_info_filename, 'rb') as split_info_file:
                split_info = pickle.load(split_info_file)
            with open(train_indices_filename, 'rb') as train_indices_file:
                train_indices = pickle.load(train_indices_file)
            with open(val_indices_filename, 'rb') as val_indices_file:
                val_indices = pickle.load(val_indices_file)
            with open(test_indices_filename, 'rb') as test_indices_file:
                test_indices = pickle.load(test_indices_file)
            return train_indices, val_indices, test_indices, split_info
        random.seed(rseed)
        
        if len(ratios) == 2:
            train_ratio, _ = ratios
            val_ratio = 0.0
        elif len(ratios) == 3:
            train_ratio, val_ratio, _ = ratios
        else:
            raise ValueError("Ratios list must have 1, 2, or 3 values that sum to 1.0")
        
        # 0. Gather the metadata
        medical_center_patients, data_index, data_map_idxs_pcm, score_counts, labels = _load_dsdata_pickle(dataset, pkl_file)

        
         # 1. calculate the target number of frames for each split
        total_frames = len(labels)
        train_frames = int(total_frames * train_ratio)
        val_frames = int(total_frames * val_ratio)
        test_frames = total_frames - train_frames - val_frames
        
        # 2. Splitting the dataset by patients taking into account frames ratio
        # lists
        train_indices = []
        val_indices = []
        test_indices = []

        # sets to store statistics about medical centers and patients
        train_patients_by_center = defaultdict(set)
        val_patients_by_center = defaultdict(set)
        test_patients_by_center = defaultdict(set)

    # 2.1 test set
        while (len(test_indices) < test_frames):
            center = random.choice(list(medical_center_patients.keys()))
            patients = medical_center_patients[center]
            try:
                patient = patients.pop()
                test_indices.extend(data_map_idxs_pcm[(patient, center)])
                test_patients_by_center[center].add(patient)
            except:
                del medical_center_patients[center]
            
        # 2.2 validation set
        while (len(val_indices) < val_frames):
            center = random.choice(list(medical_center_patients.keys()))
            patients = medical_center_patients[center]
            try:
                patient = patients.pop()
                val_indices.extend(data_map_idxs_pcm[(patient, center)])
                val_patients_by_center[center].add(patient)
            except:
                    del medical_center_patients[center]

        # 2.3 training set
        for center in list(medical_center_patients.keys()):
            for patient in list(medical_center_patients[center]):
                train_indices.extend(data_map_idxs_pcm[(patient, center)])
                train_patients_by_center[center].add(patient)
        
        # 3. Diagnostic checks and return values
        total_frames_calc = len(train_indices) + len(val_indices) + len(test_indices)
        if total_frames != total_frames_calc:
            print(f"dataset splitting gone wrong (expected: {total_frames}, got:{total_frames_calc})")
        
        # Sum up statistics info
        split_info = {
                'medical_center_patients': medical_center_patients,
                'train_patients_by_center': train_patients_by_center,
                'val_patients_by_center': val_patients_by_center,
                'test_patients_by_center': test_patients_by_center,
                'score_counts': score_counts,
                'labels': labels
            }

        if val_ratio == 0.0:
            print(f"Train size:{len(train_indices)}, Test Size:{len(test_indices)}")
            return train_indices, test_indices, split_info
        
        print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}, Test size: {len(test_indices)}")
        # Serialize the split data for future use
        print(f"\nSerializing splits...\n") 
        with open(split_info_filename, 'wb') as split_info_file:
            pickle.dump(split_info, split_info_file)
        with open(train_indices_filename, 'wb') as train_indices_file:
            pickle.dump(train_indices, train_indices_file)
        with open(val_indices_filename, 'wb') as val_indices_file:
            pickle.dump(val_indices, val_indices_file)
        with open(test_indices_filename, 'wb') as test_indices_file:
            pickle.dump(test_indices, test_indices_file)
            
            
        return train_indices, val_indices, test_indices, split_info   
            
 
 
            
def reduce_sets(seed, train, val=[], test=[], perc=1.0):
    random.seed(seed)
# Compute length of subsets
    num_train_samples = int(len(train) * perc)
    num_test_samples = int(len(test) * perc)

    # Create random subsets
    train_indices = random.sample(range(len(train)), num_train_samples)
    test_indices = random.sample(range(len(test)), num_test_samples)
    
    if val:
        num_val_samples = int(len(val) * perc)
        val_indices = random.sample(range(len(val)), num_val_samples)
        print(f"dataset reduction: {int(perc*100)}% (train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)})")
        return train_indices, val_indices, test_indices
    
    print(f"dataset reduction: {int(perc*100)}% (train={len(train_indices)}, test={len(test_indices)})")
    return train_indices, test_indices