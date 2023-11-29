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
import albumentations as A
from albumentations.pytorch import ToTensorV2

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



class HDF5ConvexDataset(HDF5Dataset):
    def __init__(self, file_path):
        super().__init__(file_path)
        # Override the group_names attribute to only include 'convex'
        self.group_names = ['convex']
        # Recalculate the total number of videos and frames
        self.total_videos, self.total_frames, self.frame_index_map = self.calculate_total_videos_and_frames()

    def calculate_total_videos_and_frames(self):
        """
        Calculates the total number of videos and creates an index map for each frame.

        Returns:
            total_videos (int): The total number of videos.
            total_frames (int): The total number of frames.
            frame_index_map (dict): A dictionary mapping frame indices to their corresponding group and video names.
        """
        convex_index_map_path = os.path.dirname(self.file_path) + "/convex_index_map.pkl"

        # Check if the convex index map file exists and load it if found
        try:
            with open(convex_index_map_path, 'rb') as f:
                print("\n\nConvex index map found, loading ...\n")
                saved_data = pickle.load(f)
                total_frames = saved_data['total_frames']
                frame_index_map = saved_data['frame_index_map']
                print(f"Loaded convex index map: {len(frame_index_map)} frames\n")
        except FileNotFoundError:
            print("Convex index map NOT FOUND\n")
            max_frame_idx_end = 0
            frame_index_map = {}

            # Create tqdm progress bar
            with tqdm(total=len(self.h5file['convex']), desc="Calculating frames and index map", unit='video', dynamic_ncols=True) as pbar:
                for video_name in self.h5file['convex']:
                    video_group = self.h5file['convex'][video_name]
                    frame_idx_start = video_group.attrs['frame_idx_start']
                    frame_idx_end = video_group.attrs['frame_idx_end']
                    max_frame_idx_end = max(max_frame_idx_end, frame_idx_end)
                    for i in range(frame_idx_start, frame_idx_end + 1):
                        frame_index_map[i] = ('convex', video_name)
                    pbar.update(1)  # Update progress bar for each video

            total_frames = max_frame_idx_end + 1

            # Save convex index map to a pickle file
            with open(convex_index_map_path, 'wb') as f:
                pickle.dump({'total_frames': total_frames, 'frame_index_map': frame_index_map}, f)

            print(f"\n\nConvex index map calculated and saved: {len(frame_index_map)} frames.")

        total_videos = len(self.h5file['convex'])
        return total_videos, total_frames, frame_index_map



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
        self.resize_size = (224, 224)
        self.transform = transform
        print(f"\nTransforms:\n{transform}\n")
    

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

        # Transform without normalization to get set stats
        norm_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize_size),
            transforms.ToTensor()
        ])
        if self.transform:
            frame_tensor = self.transform(image=frame_data)
            frame_tensor = frame_tensor["image"]    
        else:
            frame_tensor = norm_transforms(frame_data)
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
        combined_filename = os.path.dirname(pkl_file) + f"/_patinets_combined_sets_info_{round(ratios[0], 1)}_{round(ratios[1], 1)}_{round(ratios[2], 1)}.pkl"


        if os.path.exists(combined_filename):
            print("\nSerialized splits found, loading ...\n")
            # Load existing split data
            with open(combined_filename, 'rb') as combined_file:
                all_sets_info = pickle.load(combined_file)
            # Extract individual sets from the combined data
            split_info = all_sets_info['split_info']
            train_indices = all_sets_info['train_indices']
            val_indices = all_sets_info['val_indices']
            test_indices = all_sets_info['test_indices']
            print(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}, Val size: {len(val_indices)}")
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
        all_sets_info = {
            'split_info': split_info,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        }
        # Serialize the combined information
        with open(combined_filename, 'wb') as combined_file:
            pickle.dump(all_sets_info, combined_file)

            
            
        return train_indices, val_indices, test_indices, split_info   
            
def reduce_set(seed, indices_set, perc=1.0):
    random.seed(seed)
    num_samples = int(len(indices_set) * perc)
    indices = random.sample(range(len(indices_set)), num_samples)
    print(f"set reduction: {int(perc*100)}% (indices={len(indices)})")
    return indices
            
def reduce_sets(seed, train=[], val=[], test=[], perc=1.0):
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

def split_dataset_videos(rseed, dataset, pkl_file, ratios=[0.8, 0.1, 0.2]):
    """
    Split the dataset into training and test subsets based on a given pickle file, considering videos at the patient level.

    Parameters:
        rseed (int): The seed for random number generation.
        dataset (HDF5Dataset): The HDF5 dataset.
        pkl_file (str): The path to the pickle file.
        ratios (list): A list of ratios for the train and test sets. The sum of ratios should be 1.0.

    Returns:
        train_indices (list): The indices of the training set.
        test_indices (list): The indices of the test set.
        split_info (dict): A dictionary containing various statistics and information about the split.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
    """
    # Adjust the filenames for combined pickle file
    combined_filename = os.path.dirname(pkl_file) + f"/_combined_sets_info_{round(ratios[0], 1)}_{round(ratios[1], 1)}_{round(ratios[2], 1)}.pkl"

    if os.path.exists(combined_filename):
        print("\nSerialized splits found, loading ...\n")
        # Load existing combined data
        with open(combined_filename, 'rb') as combined_file:
            all_sets_info = pickle.load(combined_file)

        # Extract individual sets from the combined data
        split_info = all_sets_info['split_info']
        train_indices = all_sets_info['train_indices']
        val_indices = all_sets_info['val_indices']
        test_indices = all_sets_info['test_indices']
        print(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}, Val size: {len(val_indices)}")
            
        return train_indices, val_indices, test_indices, split_info

    random.seed(rseed)

    # 0. Gather the metadata
    medical_center_patients, data_index, data_map_idxs_pcm, score_counts, labels = _load_dsdata_pickle(dataset, pkl_file)

    # 1. Calculate the target number of videos for each split
    total_videos = len(set(video_info[1] for video_info in dataset.frame_index_map.values()))
    train_videos_count = int(total_videos * ratios[0])
    test_videos_count = total_videos - train_videos_count
    val_videos_count = int(train_videos_count * ratios[1])
    train_videos_count = train_videos_count - val_videos_count

    dataset_videos = set(video_info for video_info in dataset.frame_index_map.values())
    
    # 2. Splitting the dataset by patients taking into account video ratio
    train_videos = set()
    test_videos = set()
    val_videos = set()
    

    # 2.1 Test set
    while len(test_videos) < test_videos_count:
        # pick randomly a video from the dataset and add it to the set
        video_info = random.choice(list(dataset_videos - test_videos))
        test_videos.add(video_info)
        # get the video group name and video name and use them to retrieve the patient and medical center
        group_name, video_name = video_info
        video_patient = dataset.h5file[group_name][video_name].attrs["patient"]
        video_cetner = dataset.h5file[group_name][video_name].attrs["medical_center"]

        # check if the patient in that medical center has other videos
        for other_video in list(dataset_videos - test_videos):
            other_group_name, other_video_name = other_video
            patient = dataset.h5file[other_group_name][other_video_name].attrs["patient"]
            center = dataset.h5file[other_group_name][other_video_name].attrs["medical_center"]
            if(patient == video_patient and center == video_cetner):
                test_videos.add(other_video)
    
    train_val_videos = dataset_videos - test_videos
    # 2.2 Val set
    while len(val_videos) < val_videos_count:
        # pick randomly a video from the dataset and add it to the set
        video_info = random.choice(list(train_val_videos - val_videos))
        val_videos.add(video_info)
        # get the video group name and video name and use them to retrieve the patient and medical center
        group_name, video_name = video_info
        video_patient = dataset.h5file[group_name][video_name].attrs["patient"]
        video_cetner = dataset.h5file[group_name][video_name].attrs["medical_center"]

        # check if the patient in that medical center has other videos
        for other_video in list(train_val_videos - val_videos):
            other_group_name, other_video_name = other_video
            patient = dataset.h5file[other_group_name][other_video_name].attrs["patient"]
            center = dataset.h5file[other_group_name][other_video_name].attrs["medical_center"]
            if(patient == video_patient and center == video_cetner):
                val_videos.add(other_video)

    # 2.3 Training set
    train_videos.update(video_info for video_info in train_val_videos-val_videos)

    # 3. Create indices
    train_indices = [index for index, video_info in dataset.frame_index_map.items() if video_info in train_videos]
    test_indices = [index for index, video_info in dataset.frame_index_map.items() if video_info in test_videos]
    val_indices = [index for index, video_info in dataset.frame_index_map.items() if video_info in val_videos]
    
    
    # 4. Get the number of videos and frames for each patient-center combination
    test_videos_per_patient = defaultdict(int)
    test_frames_per_patient = defaultdict(int)

    for video_info in test_videos:
        group_name, video_name = video_info
        patient = dataset.h5file[group_name][video_name].attrs["patient"]
        center = dataset.h5file[group_name][video_name].attrs["medical_center"]
        test_videos_per_patient[(center, patient)] += 1
        test_frames_per_patient[(center, patient)] += len(dataset.h5file[group_name][video_name]['frames'])

    train_videos_per_patient = defaultdict(int)
    train_frames_per_patient = defaultdict(int)

    for video_info in train_videos:
        group_name, video_name = video_info
        patient = dataset.h5file[group_name][video_name].attrs["patient"]
        center = dataset.h5file[group_name][video_name].attrs["medical_center"]
        train_videos_per_patient[(center, patient)] += 1
        train_frames_per_patient[(center, patient)] += len(dataset.h5file[group_name][video_name]['frames'])
        
    val_videos_per_patient = defaultdict(int)
    val_frames_per_patient = defaultdict(int)

    for video_info in val_videos:
        group_name, video_name = video_info
        patient = dataset.h5file[group_name][video_name].attrs["patient"]
        center = dataset.h5file[group_name][video_name].attrs["medical_center"]
        val_videos_per_patient[(center, patient)] += 1
        val_frames_per_patient[(center, patient)] += len(dataset.h5file[group_name][video_name]['frames'])

        
    # 4. Diagnostic checks and return values
    total_videos_calc = len(set(video_info[1] for video_info in train_videos)) + \
                        len(set(video_info[1] for video_info in test_videos)) +\
                        len(set(video_info[1] for video_info in val_videos))
    if total_videos != total_videos_calc:
        print(f"dataset splitting gone wrong (expected: {total_videos}, got:{total_videos_calc})")

    # Sum up statistics info
    split_info = {
        'train_indices': train_indices,
        'train_videos': train_videos,
        'test_indices': test_indices,
        'test_videos': test_videos,
        'val_indices': val_indices,
        'val_videos': val_videos,
        
        'train_videos_per_patient': train_videos_per_patient,
        'train_frames_per_patient': train_frames_per_patient,
        
        'test_videos_per_patient': test_videos_per_patient,
        'test_frames_per_patient': test_frames_per_patient, 
        
        'val_videos_per_patient': val_videos_per_patient,
        'val_frames_per_patient': val_frames_per_patient,

        'data_map_idxs_pcm': data_map_idxs_pcm,
        'medical_center_patients': medical_center_patients,

        'score_counts': score_counts,
        'labels': labels,
        'train_videos_count': train_videos_count,
        'test_videos_count': test_videos_count
    }

    print(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}, Val size: {len(val_indices)}")
    # Serialize the split data for future use
    print(f"\nSerializing splits...\n")
    # Create a dictionary to store all information
    all_sets_info = {
        'split_info': split_info,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }

    # Serialize the combined information
    with open(combined_filename, 'wb') as combined_file:
        pickle.dump(all_sets_info, combined_file)

    return train_indices, val_indices, test_indices, split_info
