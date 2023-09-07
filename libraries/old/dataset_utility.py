from PIL import Image

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
