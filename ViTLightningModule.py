# Model class ------------------------------------------------------------
from transformers import ViTForImageClassification
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch
from transformers import ViTImageProcessor
import torchvision.transforms as transforms
from torchmetrics.classification import MulticlassF1Score
import torchmetrics.functional as metrics
from kornia import tensor_to_image
import matplotlib.pyplot as plt
from data_setup import DataAugmentation





def collate_fn(examples):
    frames = torch.stack([example[0] for example in examples])  # Extract the preprocessed frames
    scores = torch.tensor([example[1] for example in examples])  # Extract the scores
    return {"pixel_values": frames, "labels": scores}
  
id2label = {0: 'no', 1: 'yellow', 2: 'orange', 3: 'red'}
label2id = {"no": 0, "yellow": 1, "orange": 2, "red": 3}

class ViTLightningModule(pl.LightningModule):
    def __init__(self, train_dataset, test_dataset, batch_size, num_workers, optimizer, num_classes=4, lr=1e-3):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                              num_labels=4,
                                                              id2label=id2label,
                                                              label2id=label2id)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.transform = DataAugmentation()
        
        self.f1_score_metric = MulticlassF1Score(num_classes=num_classes)
        
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits
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
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        x = self.transform(pixel_values)  # => we perform GPU/Batched data augmentation
        return {"pixel_values": x, "labels": labels}
      
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        image_mean = processor.image_mean
        image_std = processor.image_std
        frame_tensor = transforms.Normalize(mean=image_mean, std=image_std)(pixel_values)
        
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]
        #accuracy = torchmetrics.functional.accuracy(predictions, labels, task="multiclass", num_classes=4)
        f1 = self.f1_score_metric(logits, labels)

        return loss, accuracy, f1
      
    def training_step(self, batch, batch_idx):
        loss, accuracy, f1 = self.common_step(batch, batch_idx)
        self.log("training_loss", loss, on_epoch=True, prog_bar=True)
        self.log("training_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("training_f1", f1, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, f1 = self.common_step(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("test_f1", f1, on_epoch=True, prog_bar=True)

    
    # def validation_step(self, batch, batch_idx):
    #     loss, accuracy = self.common_step(batch, batch_idx)
    #     print(f"val_acc: {accuracy}")
    #     print(f"val_loss: {loss}")     
    #     #self.log("validation_loss", loss, on_epoch=True, sync_dist=True)
    #     #self.log("validation_accuracy", accuracy, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-5)

    def train_dataloader(self): 
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          collate_fn=collate_fn, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          collate_fn=collate_fn)

