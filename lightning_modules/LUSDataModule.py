import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from timm.data.mixup import Mixup
mixup_args = {
    'mixup_alpha': 1.,
    'cutmix_alpha': 1.,
    'prob': 1,
    'switch_prob': 0.5,
    'mode': 'batch',
    'label_smoothing': 0.1,
    'num_classes': 4
}
mixup_fn = Mixup(**mixup_args)
def train_collate_fn(examples):
    frames = torch.stack([example[0] for example in examples])  # Extract the preprocessed frames
    scores = torch.tensor([example[1] for example in examples])  # Extract the scores
    x, y = mixup_fn(frames, scores)
    
    return (x, y)
def collate_fn(examples):
    frames = torch.stack([example[0] for example in examples])  # Extract the preprocessed frames
    scores = torch.tensor([example[1] for example in examples])  # Extract the scores
    
    return (frames, scores)

class LUSDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, test_dataset, val_dataset, num_workers, batch_size):
        super().__init__()
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        
        self.num_workers = num_workers
        self.batch_size = batch_size
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          collate_fn=train_collate_fn,
                          drop_last=True,
                        #   shuffle=True
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          collate_fn=collate_fn)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          collate_fn=collate_fn)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=32)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=32)