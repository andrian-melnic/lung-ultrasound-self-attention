from torch.utils.data import Dataset, Subset
import os
import h5py
from tqdm import tqdm
import pickle
import random
from torchvision.transforms import v2
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
        # self.resize_size = (256, 256)

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
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        frame_tensor = v2.ToTensor()(frame_data)
        frame_tensor = v2.Resize(self.resize_size)(frame_tensor)
        # frame_tensor = v2.Normalize(mean=image_mean, std=image_std)(frame_tensor)
        # frame_tensor = frame_tensor.float() / 255.0
        frame_tensor = frame_tensor.permute(0, 1, 2)
            
        # Target data to integer scores
        # target_data = torch.tensor(sum(target_data))
        target_data = int(target_data[()])

        

        return frame_tensor, target_data
    
    def set_transform(self, transform):
        self.transform = transform


    # def pp_frames(self, frame_data):
    #     """
    #     Preprocess the frame data.

    #     Args:
    #         frame_data: The frame data.

    #     Returns:
    #         torch.Tensor: The preprocessed frame tensor.
    #     """

    #     size = (224, 224)
    #     image_mean = [0.485, 0.456, 0.406]
    #     image_std = [0.229, 0.224, 0.225]

    #     frame_tensor = v2.ToTensor()(frame_data)
    #     frame_tensor = v2.Resize(size)(frame_tensor)
    #     frame_tensor = v2.Normalize(mean=image_mean, std=image_std)(frame_tensor)

    #     return 
      
      
# ---------------------------------------------------------------------------- #
#                               DataAugmentation                               #
# ---------------------------------------------------------------------------- #

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self):
        super().__init__()
        # self.transforms = torch.nn.Sequential(
        #     K.RandomRotation(degrees=(-20, 20)),  # random rotation between -20 to 20 degrees
        #     K.RandomAffine(degrees=(-10, 10), scale=(0.8, 1.2))  # random affine transformation with rotation between -10 to 10 degrees and scale between 0.8 to 1.2
        # )
        
        self.transforms = torch.nn.Sequential(
            K.RandomAffine(degrees=(-23, 23), scale=(1.1, 1.25), p=0.5),
            K.RandomElasticTransform(alpha=(0.01,0.01), sigma=(0.01,0.01), p=0.3),
            K.RandomResizedCrop(size=(224,224), scale=(0.7, 1.0), p=0.3),
            K.RandomContrast(contrast=(0.5, 1), p=0.5),
            K.RandomGaussianBlur((3, 3), (0.5, 1.5), p=0.3)
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

        # 1. Calculate the number of patients and frames for each medical center
        frames_by_center = defaultdict(int)
        frames_by_center_patient = defaultdict(initialize_inner_defaultdict)

        for (patient, center) in data_index.values():
            frames_by_center[center] += 1
            frames_by_center_patient[center][patient] += 1
        
        # 2. Calculate the target number of frames for each split
        total_frames = sum(frames_by_center.values())
        train_frames = int(total_frames * train_ratio)
        val_frames = int(total_frames * val_ratio)
        test_frames = total_frames - train_frames - val_frames

        # 3. Create a dictionary to track patient percentages for each center
        patient_perc_by_center = defaultdict(create_default_dict)
        for center, patients in medical_center_patients.items():
            patients = list(patients)

            for patient in patients:
                patient_frames = frames_by_center_patient[center][patient]
                patient_percentage = patient_frames / total_frames
                patient_perc_by_center[center][patient] = patient_percentage
        
        # 4. Splitting the dataset by patients taking into account frames ratio
        # lists
        train_indices = []
        val_indices = []
        test_indices = []

        # sets to store statistics about medical centers and patients
        train_patients_by_center = defaultdict(set)
        val_patients_by_center = defaultdict(set)
        test_patients_by_center = defaultdict(set)

        # 4.1 Test set
        while len(test_indices) < test_frames:
            center = random.choice(list(patient_perc_by_center.keys()))
            patients = list(patient_perc_by_center[center].keys())
            if patients:
                patient = random.choice(patients)
                if center in patient_perc_by_center and patient in patient_perc_by_center[center]:
                    if len(test_indices) + patient_perc_by_center[center][patient] * total_frames <= test_frames:
                        test_indices.extend(data_map_idxs_pcm[(patient, center)])
                        test_patients_by_center[center].add(patient)
                        del patient_perc_by_center[center][patient]
                    else:
                        # Se supera test_frames, cerca i pazienti rimasti che possono essere aggiunti per avvicinare il rapporto
                        remaining_frames = test_frames - len(test_indices)
                        candidates = [p for p in patients if patient_perc_by_center[center][p] * total_frames <= remaining_frames]
                        if candidates:
                            # Ordina i candidati in base a quanto si avvicinano al rapporto desiderato
                            candidates = sorted(candidates, key=lambda p: abs((len(test_indices) + patient_perc_by_center[center][p] * total_frames) / test_frames - 1))
                            
                            for best_candidate in candidates:
                                if len(test_indices) + patient_perc_by_center[center][best_candidate] * total_frames <= test_frames:
                                    test_indices.extend(data_map_idxs_pcm[(best_candidate, center)])
                                    test_patients_by_center[center].add(best_candidate)
                                    del patient_perc_by_center[center][best_candidate]
                        else:
                            break

        # 4.2 Validation set
        while len(val_indices) < val_frames:
            center = random.choice(list(patient_perc_by_center.keys()))
            patients = list(patient_perc_by_center[center].keys())
            if patients:
                patient = random.choice(patients)
                if center in patient_perc_by_center and patient in patient_perc_by_center[center]:
                    if len(val_indices) + patient_perc_by_center[center][patient] * total_frames <= val_frames:
                        val_indices.extend(data_map_idxs_pcm[(patient, center)])
                        val_patients_by_center[center].add(patient)
                        del patient_perc_by_center[center][patient]
                    else:
                        # Se supera train_frames, cerca i pazienti rimasti che possono essere aggiunti per avvicinare il rapporto
                        remaining_frames = val_frames - len(val_indices)
                        candidates = [p for p in patients if patient_perc_by_center[center][p] * total_frames <= remaining_frames]
                        if candidates:
                            # Ordina i candidati in base a quanto si avvicinano al rapporto desiderato
                            candidates = sorted(candidates, key=lambda p: abs((len(val_indices) + patient_perc_by_center[center][p] * total_frames) / val_frames - 1))
                            
                            for best_candidate in candidates:
                                if len(val_indices) + patient_perc_by_center[center][best_candidate] * total_frames <= val_frames:
                                    val_indices.extend(data_map_idxs_pcm[(best_candidate, center)])
                                    val_patients_by_center[center].add(best_candidate)
                                    del patient_perc_by_center[center][best_candidate]
                        else:
                            break
        
        # 4.3 Train set
        for center in patient_perc_by_center:
            for patient in patient_perc_by_center[center]:
                train_indices.extend(data_map_idxs_pcm[(patient, center)])
                train_patients_by_center[center].add(patient)
        
        # 5. Diagnostic checks and return values
        total_frames_calc = len(train_indices) + len(val_indices) + len(test_indices)
        if total_frames != total_frames_calc:
            print(f"dataset splitting gone wrong (expected: {total_frames}, got:{total_frames_calc})")
        
        # Sum up statistics info
        split_info = {
            'medical_center_patients': medical_center_patients,
            'frames_by_center': frames_by_center,
            'train_patients_by_center': train_patients_by_center,
            'val_patients_by_center': val_patients_by_center,
            'test_patients_by_center': test_patients_by_center,
            'frames_by_center_patient': frames_by_center_patient,
            'score_counts': score_counts,
            'labels': labels
        }

        train_idxs_p = round((len(train_indices) / len(dataset)) * 100)
        val_idxs_p = round((len(val_indices) / len(dataset)) * 100)
        test_idxs_p = 100 - (train_idxs_p + val_idxs_p)

        if val_ratio == 0.0:
            print(f"dataset split: train={len(train_indices)}({train_idxs_p}%), test={len(test_indices)}({test_idxs_p}%)")
            return train_indices, test_indices, split_info
        
        print(f"dataset split: train={len(train_indices)}({train_idxs_p}%), val={len(val_indices)}({val_idxs_p}%), test={len(test_indices)}({test_idxs_p}%)")

        
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