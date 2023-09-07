import os
from pathlib import Path
from LazyMatLoader import LazyMatLoader
from torch.utils.data import Dataset, DataLoader

class LazyLoadingDataset(Dataset):
    """Custom dataset for lazy-loading .mat images and corresponding targets."""

    def __init__(self, folder_names, base_path=None, transform=None):
        self.base_path = base_path if base_path is not None else ''
        self.folder_names = folder_names
        self.file_list = self.get_file_list()

    def get_file_list(self):
        file_list = []
        for folder_name in self.folder_names:
            root_folder = Path(os.path.join(self.base_path, folder_name))
            files = [f for f in root_folder.rglob('*.mat') if not f.name.endswith('_score.mat')]
            file_list.extend(files)
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        target_path = file_path.parent / f"{file_path.stem}_score.mat"

        # Load the video data using lazy-loading
        video_data = LazyMatLoader(file_path)
        # Load the target data using lazy-loading
        target_data = LazyMatLoader(target_path)

        # return video_data, target_data
        return video_data, target_data