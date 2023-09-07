from PIL import Image
import pickle
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from tqdm import tqdm
from itertools import islice


# Function to print diagnostic information about the dataset
def print_dataset_info(h5file):
    for group_name in h5file:
        group = h5file[group_name]
        print(f"Group: {group_name}")

        num_frames_counter = 0

        for video_name in group:
            video_group = group[video_name]
            print(f"  Video: {video_name}")

            frames_group = video_group['frames']
            num_frames = len(frames_group)
            print(f"    Number of frames: {num_frames}")

            targets_group = video_group['targets']
            num_targets = len(targets_group)
            print(f"    Number of targets: {num_targets}")

            # Get patient_reference and medical_center attributes
            patient_reference = video_group.attrs['patient_reference']
            medical_center = video_group.attrs['medical_center']
            print(f"    Patient Reference: {patient_reference}")
            print(f"    Medical Center: {medical_center}")

            # Get idx attributes
            fis = video_group.attrs['frame_idx_start']
            fie = video_group.attrs['frame_idx_end']
            print(f"    Frame idx Range: {fis} - {fie}")

            continue

            print("    Frame and Target information:")
            for i in range(num_frames):
                frame_data = frames_group[f'frame_{num_frames_counter+i}']
                target_data = targets_group[f'target_{num_frames_counter+i}']
                print(f"      Frame {num_frames_counter+i}: Shape = {frame_data.shape}, Target = {target_data.shape}")

            num_frames_counter += num_frames
# Function to resize the image to 5% of the original size
def resize_image(image_array):
    image = Image.fromarray(image_array)
    width, height = image.size
    new_width = int(width * 0.3)
    new_height = int(height * 0.3)
    resized_image = image.resize((new_width, new_height))
    return resized_image

def get_label_color(target_array):
    color_mapping = {
        (0, 0, 0): 'lightgreen',
        (1, 0, 0): 'gold',
        (1, 1, 0): 'orange',
        (1, 1, 1): 'red',
    }
    return color_mapping.get(tuple(target_array), 'black')

def get_frames_resolutions(dataset, num_videos=None):
    video_resolutions = {}
    video_patients = {}
    nframes = 0
    nframes_linear = 0
    nframes_convex = 0
    nframes_unclassified = 0
    total_videos = len(dataset)
    num_videos = num_videos if num_videos is not None else total_videos
    broken_videos = []

    with tqdm(total=num_videos, desc="Getting frame resolutions", unit='video', dynamic_ncols=True) as pbar:
        for video_data, _ in dataset:
            resolutions = set()
            num_frames = video_data.get_num_frames()
            if num_frames == -1:
                print(f"\nError: Video '{video_data.get_video_name()}' has an invalid number of frames (-1). Skipping this video.")
                broken_videos.append({
                    'video_name': video_data.get_video_name(),
                    'patient': video_data.get_patient(),
                    'center': video_data.get_medical_center()
                })
                pbar.update(1)  # Increase the count even for broken videos
                continue

            nframes += num_frames
            pbar.set_description(f"Getting frame resolutions (frames: {nframes})")
            video_name = video_data.get_video_name()
            patient = video_data.get_patient()
            center = video_data.get_medical_center()

            if 'linear' in video_name:
                nframes_linear += num_frames
            elif 'convex' in video_name:
                nframes_convex += num_frames
            else:
                nframes_unclassified += num_frames

            for i in range(num_frames):
                frame_data = video_data.get_frame(i)
                resolution = frame_data.shape[:2]
                resolutions.add(resolution)
            video_resolutions[video_name] = resolutions
            video_patients[video_name] = (patient, center)
            pbar.update(1)
            if pbar.n == num_videos:
                break

    return video_resolutions, video_patients, nframes, nframes_linear, nframes_convex, nframes_unclassified, broken_videos

def print_frame(frame_idx, dataset_idx, dataset):
    if dataset_idx < 0 or dataset_idx >= len(dataset):
        print(f"Error: Invalid dataset index {dataset_idx}. Dataset index must be between 0 and {len(dataset) - 1}.")
        dataset_idx = len(dataset) - 1

    video_data, target_data = dataset[dataset_idx]
    num_frames = video_data.get_num_frames()

    if frame_idx < 0 or frame_idx >= num_frames:
        print(f"Error: Invalid frame index {frame_idx}. Frame index must be between 0 and {num_frames - 1}.")
        frame_idx = num_frames - 1

    frame = video_data.get_frame(frame_idx)
    score = target_data.get_score(frame_idx)
    video_name = video_data.get_video_name()
    patient = video_data.get_patient()
    medical_center = video_data.get_medical_center()

    # Create an image from RGB
    img = np.zeros_like(frame, dtype=np.uint8)
    img[:, :, 0] = frame[:, :, 0]  # Red channel
    img[:, :, 1] = frame[:, :, 1]  # Green channel
    img[:, :, 2] = frame[:, :, 2]  # Blue channel

    # Show the new image
    plt.imshow(img)
    plt.axis('off')

    # Add annotations to the image
    plt.annotate(f"Score: {score}", (20, img.shape[0] - 20), color='white', fontsize=10, ha='left', va='bottom')
    plt.annotate(f"Frame shape: {frame.shape}", (20, img.shape[0] - 45), color='white', fontsize=10, ha='left', va='bottom')
    plt.annotate(f"Patient: {patient}", (20, img.shape[0] - 70), color='white', fontsize=10, ha='left', va='bottom')
    plt.annotate(f"Medical center: {medical_center}", (20, img.shape[0] - 95), color='white', fontsize=10, ha='left', va='bottom')
    plt.annotate(f"Video name: {video_name}", (20, img.shape[0] - 120), color='white', fontsize=10, ha='left', va='bottom')

    plt.title(f"Frame {frame_idx}, Dataset {dataset_idx}")
    plt.show()

# Function to save a single video data to the HDF5 file
def save_video_data(h5file, group_name, video_name, video_data, target_data, patient_reference, medical_center, start_index):
    num_frames = video_data.get_num_frames()
    if num_frames == -1:
        print(f"\rError: Video '{medical_center}/{patient_reference}/{video_name}' has an invalid number of frames (-1). Skipping this video.")
        return start_index

    group = h5file.require_group(group_name)

    # Check if the video_name already exists in the group
    count = 2
    new_video_name = video_name
    while new_video_name in group:
        new_video_name = f"{video_name}_{count}"
        count += 1

    # If the video_name is changed, print a warning message
    if new_video_name != video_name:
        print(f"\rWarning: Video group '{medical_center}/{patient_reference}/{video_name}' already exists. Renaming the new video to '{new_video_name}'.")

    video_group = group.require_group(new_video_name)

    frames_group = video_group.require_group('frames')
    targets_group = video_group.require_group('targets')

    # Add patient_reference and medical_center attributes to the video_group
    video_group.attrs['patient_reference'] = patient_reference
    video_group.attrs['medical_center'] = medical_center

    for i in range(num_frames):
        frame_data = video_data.get_frame(i)
        frames_group.create_dataset(f'frame_{start_index + i}', data=frame_data, compression='gzip')

        target_data_i = target_data.get_score(i)
        targets_group.create_dataset(f'target_{start_index + i}', data=target_data_i, compression='gzip')

    # Update the 'idx_start' and 'idx_end' attributes
    video_group.attrs['frame_idx_start'] = start_index
    video_group.attrs['frame_idx_end'] = start_index + num_frames - 1

    return start_index + num_frames

# Create the HDF5 file and save the dataset
def convert_dataset_to_h5(dataset, output_file, num_videos=None):
    with h5py.File(output_file, 'w') as h5file:
        convex_group = h5file.create_group('convex')
        linear_group = h5file.create_group('linear')

        current_index_convex = 0
        current_index_linear = 0

        dataset_subset = islice(dataset, num_videos) if num_videos is not None else dataset

        with tqdm(total=num_videos if num_videos is not None else len(dataset), desc="Converting dataset to HDF5", dynamic_ncols=True, unit="video") as pbar_outer:
            for video_data, target_data in dataset_subset:
                video_name = video_data.get_video_name()
                patient = video_data.get_patient()
                medical_center = video_data.get_medical_center()

                if 'convex' in video_name:
                    current_index_convex = save_video_data(h5file, 'convex', video_name, video_data, target_data, patient, medical_center, current_index_convex)
                elif 'linear' in video_name:
                    current_index_linear = save_video_data(h5file, 'linear', video_name, video_data, target_data, patient, medical_center, current_index_linear)
                else:
                    print(f"\rWarning: Video '{medical_center}/{patient}/{video_name}' has an unclassified probe. Skipping this video.")

                pbar_outer.update(1)

                # Monitor file size
                file_size_gb = os.path.getsize(output_file) / (1024.0 ** 3)  # Convert to GB
                pbar_outer.set_postfix(file_size=f"{file_size_gb:.2f} GB")


# Function to create a flexible grid for displaying frames
def create_flexible_grid(batch_size):
    num_rows = 1
    num_cols = 1
    while num_rows * num_cols < batch_size:
        if num_rows == num_cols:
            num_cols *= 2
        else:
            num_rows *= 2

    return num_rows, num_cols

def get_label_color(target_array):
    color_mapping = {
        (0, 0, 0): 'lightgreen',
        (1, 0, 0): 'gold',
        (1, 1, 0): 'orange',
        (1, 1, 1): 'red',
    }
    return color_mapping.get(tuple(target_array), 'black')