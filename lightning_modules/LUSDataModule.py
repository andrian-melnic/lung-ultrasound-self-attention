import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


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
                          collate_fn=collate_fn
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