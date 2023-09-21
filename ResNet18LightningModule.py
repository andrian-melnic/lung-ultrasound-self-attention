# Model class ------------------------------------------------------------
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

def collate_fn(examples):
    frames = torch.stack([example[0] for example in examples])  # Extract the preprocessed frames
    scores = torch.tensor([example[1] for example in examples])  # Extract the scores
    return {"pixel_values": frames, "labels": scores}
  
id2label = {0: 'no', 1: 'yellow', 2: 'orange', 3: 'red'}
label2id = {"no": 0, "yellow": 1, "orange": 2, "red": 3}

class ResNet18LightningModule(pl.LightningModule):
    def __init__(self, train, test, batch_size, num_workers, optimizer, num_classes=4, lr=1e-3):
        super(ResNet18LightningModule, self).__init__()
        self.train_dataset = train
        self.test_dataset = test
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.resnet_model = models.resnet18(pretrained=True)
        num_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(num_features, num_classes)

        self.optimizer_name = str(optimizer).lower()
        self.optimizer = None

    def forward(self, x):
        return self.resnet_model(x)

    def common_step(self, batch, batch_idx):
        
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        correct = (predictions == labels).sum().item()
        accuracy = correct / labels.size(0)

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

        return loss

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Invalid optimizer name. Please choose either 'adam' or 'sgd'.")

        return self.optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_fn, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_fn)