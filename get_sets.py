import numpy as np
import torch
from data_setup import (HDF5Dataset, 
                        HDF5ConvexDataset, 
                        FrameTargetDataset,
                        split_dataset,
                        split_dataset_videos,
                        reduce_sets,
                        reduce_set)
from torch.utils.data import Subset   
from sklearn.utils.class_weight import compute_class_weight
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from DataAugmentation import Preprocess, TrainPreprocess


def create_default_dict():
    return defaultdict(float)
def initialize_inner_defaultdict():
    return defaultdict(int)

def get_sets(args):
    
    rseed = args.rseed
    dataset_h5_path = args.dataset_h5_path
    hospitaldict_path = args.hospitaldict_path
    train_ration = args.train_ratio
    
    
    image_mean = [0.12516, 0.12867, 0.13234]
    image_std = [0.16526, 0.16911, 0.1752]
    print(f"\nimage_mean: {image_mean}\nimage_std: {image_std}\n")
    
    test_transforms = Preprocess()
    train_transforms = TrainPreprocess()
    dataset = HDF5Dataset(dataset_h5_path)

    train_indices = []
    val_indices = []
    test_indices = []

    ratios = args.ratios


    print(f"Split ratios: {ratios}")
    train_indices, val_indices, test_indices, split_info = split_dataset(
        rseed=rseed,
        dataset=dataset,
        pkl_file=hospitaldict_path,
        ratios=ratios)

    if args.trim_data:
        print("\nTriming train:")
        train_indices_trimmed = reduce_set(rseed, train_indices, args.trim_data)
        train_indices = train_indices_trimmed
        
        print("\nTriming test:")
        test_indices_trimmed = reduce_set(rseed, test_indices, args.trim_data)
        test_indices = test_indices_trimmed
        
        print("\nTriming val:")
        val_indices_trimmed = reduce_set(rseed, val_indices, args.trim_data)
        val_indices = val_indices_trimmed
    
    if args.trim_train:
        print("\nTriming train:")
        train_indices_trimmed = reduce_set(rseed, train_indices, args.trim_train)
        train_indices = train_indices_trimmed
    
    if args.trim_val:
        print("\nTriming val:")
        val_indices_trimmed = reduce_set(rseed, val_indices, args.trim_val)
        val_indices = val_indices_trimmed
    
    if args.trim_test:
        print("\nTriming test:")
        test_indices_trimmed = reduce_set(rseed, test_indices, args.trim_test)
        test_indices = test_indices_trimmed

     # Create training and test subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)  
    val_subset = Subset(dataset, val_indices)  
    
    train_dataset = FrameTargetDataset(train_subset, transform=train_transforms)
    test_dataset = FrameTargetDataset(test_subset, transform=test_transforms)
    val_dataset = FrameTargetDataset(val_subset, transform=test_transforms)
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")    
    print(f"Validation size: {len(val_dataset)}")
    
    sets = {
        "train_indices": train_indices,
        "train": train_dataset,
        "test_indices": test_indices,
        "test": test_dataset,
        "val_indices": val_indices,
        "val": val_dataset
    }

    return sets, split_info

# def get_class_weights(indices, split_info):
#     # Retrieves the dataset's labels
#     ds_labels = split_info['labels']

#     # Extract the train and test set labels
#     y_labels = np.array(ds_labels)[indices]
#     # y_test_labels = np.array(ds_labels)[test_indices]

#     # Calculate class balance using 'compute_class_weight'
#     class_weights = compute_class_weight('balanced', 
#                                         classes=[0, 1, 2, 3], 
#                                         y=y_labels)
#     class_weights[0] = 1.
#     weights_tensor = torch.Tensor(class_weights)
#     print(class_weights)
#     return weights_tensor

def get_class_weights(indices, split_info):
    # Retrieves the dataset's labels
    ds_labels = split_info['labels']

    # Extract the train and test set labels
    y_labels = np.array(ds_labels)[indices]

    # Calculate the count of samples for each class
    class_counts = np.bincount(y_labels)

    # Find the majority class
    majority_class = np.argmax(class_counts)

    # Calculate custom class weights based on the relative difference from the majority class
    class_weights = class_counts[majority_class] / class_counts

    # Set a weight of 1 for the majority class
    class_weights[majority_class] = 1.0

    # Convert to PyTorch tensor
    weights_tensor = torch.Tensor(class_weights)

    print(class_weights)
    return weights_tensor