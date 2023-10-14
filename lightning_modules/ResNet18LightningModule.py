from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch
from kornia import tensor_to_image
import torchvision
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassF1Score, Accuracy
import torchmetrics.functional as metrics
from torch import mps


from data_setup import DataAugmentation

def collate_fn(examples):
    frames = torch.stack([example[0] for example in examples])  # Extract the preprocessed frames
    scores = torch.tensor([example[1] for example in examples])  # Extract the scores
    return (frames, scores)
  
id2label = {0: 'no', 1: 'yellow', 2: 'orange', 3: 'red'}
label2id = {"no": 0, "yellow": 1, "orange": 2, "red": 3}

class ResNet18LightningModule(pl.LightningModule):
    def __init__(self, train_dataset, test_dataset, batch_size, num_workers, optimizer, num_classes=4, lr=1e-3, pretrained=True):
        super(ResNet18LightningModule, self).__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers

        if pretrained:
            print("\n\nUsing pretrained weights\n\n")
            self.resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            print("\n\nNo pretrained weights\n\n")
            self.resnet_model = resnet18(weights=None)
            
        self.resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(num_features, num_classes)

        self.optimizer_name = str(optimizer).lower()
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        self.transform = DataAugmentation()
        
        self.f1_score_metric = MulticlassF1Score(num_classes=num_classes)
        self.accuracy_metric = Accuracy(task='multiclass', num_classes=num_classes)
    def forward(self, x):
        return self.resnet_model(x)


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


    def on_after_batch_transfer(self, batch, dataloader_idx):
        pixel_values, labels = batch
        x = self.transform(pixel_values)  # => we perform GPU/Batched data augmentation
        return x, labels
      
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy_metric(logits, y)
        self.log('training_loss', loss, prog_bar=True)
        self.log('training_accuracy', acc, prog_bar=True)
        mps.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy_metric(logits, y)
        self.log('validation_loss', loss, prog_bar=True)
        self.log('validation_acc', acc, prog_bar=True)
        mps.empty_cache()

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.005)
        elif self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Invalid optimizer name. Please choose either 'adam' or 'sgd'.")

        return self.optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                        #   num_workers=self.num_workers,
                        #   pin_memory=True,
                          collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=collate_fn)