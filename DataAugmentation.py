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
        # self.transforms = torch.nn.Sequential(
        #     K.RandomAffine(degrees=(-23, 23), scale=(1, 1.5), p=0.5),
        #     K.RandomRotation(degrees=(-23, 23), p=0.5),
        #     # K.RandomElasticTransform(alpha=(0.01,0.01), sigma=(0.01,0.01), p=0.5),
        #     # K.RandomResizedCrop(size=(224,224), scale=(0.5, 1.0), p=0.3),
        #     K.RandomContrast(contrast=(0.7, 1.8), p=0.5),
        #     K.RandomGamma(gamma=(0.9, 1.8), gain=(0.9, 1.8), p=0.5),
        #     K.RandomGaussianBlur((3, 3), (1, 1.5), p=0.3),
        #     K.RandomHorizontalFlip(p=0.5),
        #     # K.RandomVerticalFlip(p=0.3)
        # )
        
        self.image_mean = [0.12768, 0.13132, 0.13534]
        self.image_std = [0.1629, 0.16679, 0.17305]
        self.transforms = nn.Sequential(
            K.RandomAffine(degrees=(-25, 25), scale=(1.1, 1.5), p=1),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomBrightness(brightness=(0.7,1.3), p=0.5),
            K.RandomContrast(contrast=(0.7, 1.3), p=0.5),
            K.RandomGamma(gamma=(0.7, 1.3), gain=(1., 1.), p=0.5),
            K.Normalize(mean=self.image_mean, std=self.image_std, p=1)
        )
        print(self.transforms)

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x):
        x_out = self.transforms(x)
        # x_out = self.us_classification_augmentation(x)
        return x_out

class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self):
        super().__init__()
        # self.image_mean = torch.tensor([0.12768, 0.13132, 0.13534])
        # self.image_std = torch.tensor([0.1629, 0.16679, 0.17305])
        self.image_mean = [0.12768, 0.13132, 0.13534]
        self.image_std = [0.1629, 0.16679, 0.17305]
        
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: Tensor = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        x_out = transforms.Resize((224, 224))(x_out)
        # x_out = K.Normalize(mean=self.image_mean, std=self.image_std, p=1, keepdim=True)(x_out.float() / 255.0)
        return x_out.float() / 255.0
    
class TrainPreprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: Tensor = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        x_out = transforms.Resize((224, 224))(x_out)
        return x_out.float() / 255.0