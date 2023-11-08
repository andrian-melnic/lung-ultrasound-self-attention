import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_lightning import pytorch_lightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import timm
from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10
import os

def collate_fn(examples):
    frames = torch.stack([example[0] for example in examples])  # Extract the preprocessed frames
    scores = torch.tensor([example[1] for example in examples])  # Extract the scores
    return (frames, scores)
  
id2label = {0: 'no', 1: 'yellow', 2: 'orange', 3: 'red'}
label2id = {"no": 0, "yellow": 1, "orange": 2, "red": 3}

class BeitTimmLightningModule(LightningModule):
    def __init__(self, model_name, num_classes, learning_rate=1e-4):
        super(BeitTimmLightningModule, self).__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.hparams.learning_rate)


    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.05)
        elif self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Invalid optimizer name. Please choose either 'adam' or 'sgd'.")

        return self.optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          collate_fn=collate_fn, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          collate_fn=collate_fn)


    def show_batch(self, win_size=(10, 10)):
        def _to_vis(data):
            # Ensure that pixel values are in the valid range [0, 1]
            data = torch.clamp(data, 0, 1)
            return tensor_to_image(torchvision.utils.make_grid(data, nrow=8))

        # Get a batch from the training set
        imgs, labels = next(iter(self.train_dataloader()))

        # Apply data augmentation to the batch
        imgs_aug = self.transform(imgs)

        # Use matplotlib to visualize the original and augmented images
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs))
        plt.title("Original Images")

        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs_aug))
        plt.title("Augmented Images")