import numpy as np
import torch
from data_setup import HDF5Dataset, FrameTargetDataset, split_dataset, reduce_sets 
from torch.utils.data import Subset   
from sklearn.utils.class_weight import compute_class_weight


def create_default_dict():
    return defaultdict(float)
def initialize_inner_defaultdict():
    return defaultdict(int)

def get_sets(rseed, 
             dataset_h5_path,
             hospitaldict_path,
             train_ratio,
             trim_data,
             pretrained
             ):
    dataset = HDF5Dataset(dataset_h5_path)

    train_indices = []
    val_indices = []
    test_indices = []

    train_ratio = train_ratio
    test_ratio = (1 - train_ratio)/2
    val_ratio = test_ratio
    ratios = [train_ratio, test_ratio, val_ratio]


    print(f"Split ratios: {ratios}")
    train_indices, val_indices, test_indices, split_info = split_dataset(
        rseed=rseed,
        dataset=dataset,
        pkl_file=hospitaldict_path,
        ratios=ratios)

    # Create training and test subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)  
    val_subset = Subset(dataset, val_indices)  


    if trim_data:
        train_indices_trimmed, \
        val_indices_trimmed, \
        test_indices_trimmed = reduce_sets(rseed,
                                        train_subset,
                                        val_subset,
                                        test_subset,
                                        trim_data)
        
        train_subset = Subset(dataset, train_indices_trimmed)
        test_subset = Subset(dataset, test_indices_trimmed)
        val_subset = Subset(dataset, val_indices_trimmed)
        
        train_indices = train_indices_trimmed
        val_indices = val_indices_trimmed
        test_indices = test_indices_trimmed


    train_dataset = FrameTargetDataset(train_subset, pretrained=pretrained, trainset=True)
    test_dataset = FrameTargetDataset(test_subset, pretrained=pretrained)
    val_dataset = FrameTargetDataset(val_subset, pretrained=pretrained)
    
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

