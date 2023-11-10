import kornia.augmentation as K
import kornia.geometry as KG
import math
import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image


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
        # x_out = self.us_classification_augmentation(x)
        return x_out

    def _do_nothing(self, image):
        return image

    def _random_true_false(self):
        prob = torch.rand(1)
        predicate = prob < 0.5
        return predicate

    def _image_and_label_flip_up_down(self, image):
        image_flip = KG.vflip(image)
        return image_flip

    def _image_and_label_flip_left_right(self, image):
        image_flip = KG.hflip(image)
        return image_flip

    def _image_random_flip_left_right(self, image):
        predicate = self._random_true_false()
        image_aug = torch.where(predicate, self._image_and_label_flip_left_right(image), self._do_nothing(image))
        return image_aug

    def _image_random_flip_up_down(self, image):
        predicate = self._random_true_false()
        image_aug = torch.where(predicate, self._image_and_label_flip_up_down(image), self._do_nothing(image))
        return image_aug

    def us_classification_augmentation(self, image):
        img = self._image_random_flip_left_right(image)
        img = self._image_random_flip_up_down(img)
        gamma = torch.rand(1) * 1.4 + 0.3
        img = torch.pow(img, gamma)
        img /= torch.max(img)
        size_r = int(torch.rand(1) * 0.5 + 1.5) * 320
        angle_r = (torch.rand(1) * 60 - 30) * math.pi / 180
        img = F.resize(img, size_r, interpolation=Image.BILINEAR)
        img = F.resize(img, 224, interpolation=Image.BILINEAR)
        img = KG.rotate(img, angle_r, mode='bilinear')
        return img