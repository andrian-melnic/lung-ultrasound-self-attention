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


def create_default_dict():
    return defaultdict(float)
def initialize_inner_defaultdict():
    return defaultdict(int)

def get_sets(args):
    
    rseed = args.rseed
    dataset_h5_path = args.dataset_h5_path
    hospitaldict_path = args.hospitaldict_path
    train_ration = args.train_ratio
    
    
    image_mean = (0.12768, 0.13132, 0.13534)
    image_std = (0.1629, 0.16679, 0.17305)
    print(f"\nimage_mean: {image_mean}\nimage_std: {image_std}\n")
    
    test_transforms = A.Compose([
        A.Resize(width=224, height=224, always_apply=True, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=image_mean, std=image_std),
        ToTensorV2(),
    ])
    
    if args.augmentation:
        train_transforms = A.Compose([
            A.Resize(width=224, height=224, always_apply=True, interpolation=cv2.INTER_CUBIC),
            
            A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=23, scale_limit=(0.1, 0.5), p=0.5, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT),
            A.ElasticTransform(alpha=0.5, sigma=25, alpha_affine=15, interpolation=cv2.INTER_CUBIC, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            
            A.GaussianBlur(blur_limit=(3,3), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.5),
            A.RandomGamma(gamma_limit=(90, 110), p=0.5),
            
            A.Normalize(mean=image_mean, std=image_std),
            ToTensorV2(),
        ])
        print(f"Using Augmentations: {augmentation}")
        
    else:
        # Use test_transforms if augmentation is not enabled
        train_transforms = test_transforms
    
    dataset = HDF5Dataset(dataset_h5_path)

    train_indices = []
    val_indices = []
    test_indices = []

    # train_ratio = train_ratio
    # test_ratio = round(1 - train_ratio, 1)
    # val_ratio = 0.2

    # ratios = [train_ratio, val_ratio, test_ratio]
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

def get_class_weights(indices, split_info):
    # Retrieves the dataset's labels
    ds_labels = split_info['labels']

    # Extract the train and test set labels
    y_labels = np.array(ds_labels)[indices]
    # y_test_labels = np.array(ds_labels)[test_indices]

    # Calculate class balance using 'compute_class_weight'
    class_weights = compute_class_weight('balanced', 
                                        classes=np.unique(y_labels), 
                                        y=y_labels)

    weights_tensor = torch.Tensor(class_weights)
    print(class_weights)
    return weights_tensor

