# Model class ------------------------------------------------------------
from transformers import ViTForImageClassification
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning.pytorch as pl
import torch
import torchvision
from transformers import ViTImageProcessor
from torchmetrics.classification import MulticlassF1Score, Accuracy
from kornia import tensor_to_image
import matplotlib.pyplot as plt
from data_setup import DataAugmentation
from transformers import ViTConfig


def collate_fn(examples):
    frames = torch.stack([example[0] for example in examples])  # Extract the preprocessed frames
    scores = torch.tensor([example[1] for example in examples])  # Extract the scores
    return (frames, scores)
  
id2label = {0: 'no', 1: 'yellow', 2: 'orange', 3: 'red'}
label2id = {"no": 0, "yellow": 1, "orange": 2, "red": 3}

class ViTLightningModule(pl.LightningModule):
    def __init__(self, 
                 train_dataset,
                 test_dataset,
                 batch_size,
                 num_workers,
                 optimizer,
                 num_classes=4,
                 lr=1e-3,
                 pretrained=True,
                 configuration=None):
        
        super(ViTLightningModule, self).__init__()
        
        if pretrained == True:
            self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                                  num_labels=4,
                                                                  id2label=id2label,
                                                                  label2id=label2id)
        else:
            self.config = ViTConfig(**configuration)
            self.vit = ViTForImageClassification(config=self.config)
        
        self.preprocess = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', do_rescale=False)
        self.transform = DataAugmentation()
        
        self.train_dataset = train_dataset
        self.train_dataset.set_transform(self.preprocess)
        self.test_dataset = test_dataset
        self.test_dataset.set_transform(self.preprocess)
        
        
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.optimizer_name = str(optimizer).lower()
        self.optimizer = None
        self.f1_score_metric = MulticlassF1Score(num_classes=num_classes)
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        
        
    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     x, y = batch
    #     if self.trainer.training:
    #         x = self.transform(x)  # => we perform GPU/Batched data augmentation
    #     return x, y
      
      
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits
      
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log('training_loss', loss, prog_bar=True)
        self.log('training_accuracy', self.train_acc, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.log('validation_loss', loss, prog_bar=True)
        self.log('validation_acc', self.val_acc, prog_bar=True)
        


    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.005)
        elif self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.4, weight_decay=0.005)
        else:
            raise ValueError("Invalid optimizer name. Please choose either 'adam' or 'sgd'.")

        return self.optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          collate_fn=collate_fn, shuffle=True)

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