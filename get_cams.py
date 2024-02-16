import warnings
import signal
import os
import glob
import pickle
import torch
import lightning.pytorch as pl
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer

from pytorch_grad_cam import GradCAMPlusPlus, ScoreCAM, AblationCAM, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import args_processing
from utils import *
from callbacks import *
from run_model import *
from get_sets import get_sets, get_class_weights

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


#SwinViT
def swin_reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def denormalize_tensor(image_tensor):
    
    mean=torch.tensor([0.1154, 0.11836, 0.12134])
    std=torch.tensor([0.15844, 0.16195, 0.16743])
    
    """Denormalize a normalized image tensor."""
    denormalize_transform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=1 / std),
        transforms.Normalize(mean=-mean, std=[1, 1, 1])
    ])
    denormalized_tensor = denormalize_transform(image_tensor)
    
    return denormalized_tensor


def generate_and_display_CAM(args, image_tensor, cam_model, target_layers, cam_method="gradcamplusplus", target_class=None):
    
    if cam_method == "scorecam":
        cam = ScoreCAM(model=cam_model, target_layers=target_layers)
    elif cam_method == "ablationcam":
        cam = AblationCAM(model=cam_model, target_layers=target_layers)
    else:
        if "resnet" in args.model or "botnet" in args.model:
            cam = GradCAMPlusPlus(model=cam_model, 
                                target_layers=target_layers)
        elif "swin" in args.model:
            cam = GradCAMPlusPlus(model=cam_model, 
                                target_layers=target_layers, 
                                reshape_transform=swin_reshape_transform)
    # Prepare the input tensor
    cam_input_tensor = image_tensor.unsqueeze(0)
    
    targets = None
    if target_class is not None:
        targets=[ClassifierOutputTarget(target_class)]

    # targets = [0, 1, 2, 3]
    # Generate CAM
    grayscale_cams = cam(input_tensor=cam_input_tensor, 
                         aug_smooth=True,
                         eigen_smooth=True,
                         targets=targets)
    
    # Convert the input tensor to a numpy image
    image_tensor = denormalize_tensor(image_tensor)
    image = np.float32(transforms.ToPILImage()(image_tensor)) / 255
    
    # Show CAM on the image
    cam_image = show_cam_on_image(image, grayscale_cams[0, :], use_rgb=True, image_weight=0.85)
    
    # Convert CAM to BGR format for display
    cam = np.uint8(255 * grayscale_cams[0, :])
    cam = cv2.merge([cam, cam, cam])
    
    # Display the original image and the associated CAM
    images = np.hstack((np.uint8(255 * image), cam_image))
    return Image.fromarray(images)

def main():
    args = args_processing.parse_arguments()
    
    print("\n" + "-"*80 + "\n")
    pl.seed_everything(args.rseed)
    print("\n" + "-"*80 + "\n")


# ------------------------------ Warnings config ----------------------------- #
    if args.disable_warnings: 
        print("Warnings are DISABLED!\n\n")
        warnings.filterwarnings("ignore")
    else:
        warnings.filterwarnings("default")
# ----------------------------------- Paths ---------------------------------- #
    working_dir = args.working_dir_path
    data_file = args.dataset_h5_path
    libraries_dir = working_dir + "/libraries"

# ---------------------------- Import custom libs ---------------------------- #
    import sys
    sys.path.append(working_dir)
    from data_setup import HDF5Dataset, FrameTargetDataset, split_dataset, reduce_sets
    from lightning_modules.LUSModelLightningModule import LUSModelLightningModule
    from lightning_modules.LUSDataModule import LUSDataModule

# ---------------------------------- Dataset --------------------------------- #

    sets, split_info = get_sets(args)
    lus_data_module = LUSDataModule(sets["train"], 
                                    sets["test"],
                                    sets["val"],
                                    args.num_workers, 
                                    args.batch_size,
                                    args.mixup)

    print("\n- Train set class weights: ")
    train_weight_tensor = get_class_weights(sets["train_indices"], split_info)
    print("\n- Val set class weights: ")
    get_class_weights(sets["val_indices"], split_info)
    print("\n- Test set class weights: ")
    get_class_weights(sets["test_indices"], split_info)
# ---------------------------------------------------------------------------- #
#                         Model & trainer configuration                        #
# ---------------------------------------------------------------------------- #

    model = LUSModelLightningModule.load_from_checkpoint(args.chkp, 
                                                          strict=False,
                                                          map_location=torch.device('mps'))
    model.eval()
    
    model_name, version = get_model_name(args)
    logger = TensorBoardLogger("tb_logs", name=model_name, version=version)
    
# ------------------------------- get the cams ------------------------------- #
    test_dataset = sets["test"]
    
    # List of image indices you want to display
    image_indices_to_plot = list(range(0, len(test_dataset), 1000))

    # Class to target
    target_class = None

    cam_method = "gradcamplusplus"
    # cam_method = "scorecam"
    # cam_method = "ablationcam"

    # Specify the target layers for CAM
    if "resnet" in args.model or "botnet" in args.model:
        target_layers = [model.model.layer4[-1]]
    elif "swin" in args.model:
        target_layers = [model.model.layers[-1].blocks[-1].norm2]
    # target_layers = [model.model.layer[1]]
    # target_layers = [model.model.transformer.layers[5][0].dropout]


    # Create subplots for the selected images with a larger figsize
    # num_maps = activation_maps.size(1)  # Get the total number of activation maps
    num_rows = len(image_indices_to_plot)  # Calculate the number of rows needed
    num_images = len(image_indices_to_plot)
    fig, axes = plt.subplots(num_rows, 1, figsize=(20, 4 * num_rows))  # Adjust the figsize as per your preference

    for i, image_idx in enumerate(image_indices_to_plot):
        image_tensor = test_dataset[image_idx][0].to(device)
        displayed_image = generate_and_display_CAM(args, 
                                                   image_tensor, 
                                                   model, 
                                                   target_layers, 
                                                   cam_method=cam_method, 
                                                   target_class=target_class)
        
        # Convert PIL Image to numpy array
        np_image = np.array(displayed_image)
        title = f"Idx: {image_idx}, Target: {test_dataset[image_idx][1]}, Predicted: {model(image_tensor.unsqueeze(0))[0].argmax()}"
                    
        logger.experiment.add_image(
            title,  # Choose a unique tag for each image
            np_image,
            global_step=i,
            dataformats="HWC",  # Height, Width, Channels
        )
        
        axes[i].imshow(displayed_image)
        axes[i].set_title(title)
        axes[i].axis('off')
        
    # plt.savefig(f"cams/{version}.png")
    # Close the figure
    plt.close()
    
    
if __name__ == "__main__":
    main()