import kornia.augmentation as K
import kornia.geometry as KG
import kornia
from kornia import image_to_tensor, tensor_to_image
import math
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F
from torch import Tensor

class DataAugmentation(nn.Module):

    def __init__(self):
        super().__init__()
        self.image_mean = [0.1154, 0.11836, 0.12134]
        self.image_std = [0.15844, 0.16195, 0.16743]
        self.transforms = nn.Sequential(
            transforms.RandomAffine(degrees=(-15, 15), scale=(1., 1.15), translate=(0.15, 0.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.Normalize(self.image_mean, self.image_std)
        )
        print(self.transforms)

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x):
        transformed_images = []

        for i in range(x.size(0)):  # iterate through the batch dimension
            img = x[i]
            img = self.transforms(img)
            transformed_images.append(img)

        x_out = torch.stack(transformed_images, dim=0)
        return x_out
    
class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self):
        super().__init__()
        self.image_mean = [0.1154, 0.11836, 0.12134]
        self.image_std = [0.15844, 0.16195, 0.16743]
        
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: Tensor = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        x_out = transforms.Resize((224, 224))(x_out)
        x_out = transforms.ConvertImageDtype(dtype=torch.float32)(x_out)
        x_out = transforms.Normalize(self.image_mean, self.image_std)(x_out)

        return x_out
    
class TrainPreprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: Tensor = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        x_out = transforms.Resize((224, 224))(x_out)
        x_out = transforms.ConvertImageDtype(dtype=torch.float32)(x_out)
        
        return x_out