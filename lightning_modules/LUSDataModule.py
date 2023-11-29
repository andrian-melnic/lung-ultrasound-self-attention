import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from timm.data.mixup import Mixup
mixup_args = {
    'mixup_alpha': 0.8,
    'cutmix_alpha': 0,
    'prob': 1,
    'switch_prob': 0.,
    'mode': 'batch',
    'label_smoothing': 0,
    'num_classes': 4
}
mixup_fn = Mixup(**mixup_args)
def mixup_collate_fn(examples):
    frames = torch.stack([example[0] for example in examples])  # Extract the preprocessed frames
    scores = torch.tensor([example[1] for example in examples])  # Extract the scores
    x, y = mixup_fn(frames, scores)
    return (x, y)

def collate_fn(examples):
    frames = torch.stack([example[0] for example in examples])  # Extract the preprocessed frames
    scores = torch.tensor([example[1] for example in examples])  # Extract the scores
    return (frames, scores)

class LUSDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, test_dataset, val_dataset, num_workers, batch_size, mixup):
        super().__init__()
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.mixup = mixup
        self.persistent_workers = True if num_workers > 0 else False
        
    def train_dataloader(self):
        print(f"Use MixUp augmentation: {self.mixup}")
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=self.persistent_workers,
                          collate_fn=mixup_collate_fn if self.mixup==True else collate_fn,
                          drop_last=True,
                          shuffle=True
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=self.persistent_workers,
                          collate_fn=collate_fn)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=self.persistent_workers,
                          collate_fn=collate_fn,
                          shuffle=True)
