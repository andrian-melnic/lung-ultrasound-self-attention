# Model class ------------------------------------------------------------
from transformers import ViTForImageClassification
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch
from transformers import ViTImageProcessor
import torchvision.transforms as transforms



def collate_fn(examples):
    frames = torch.stack([example[0] for example in examples])  # Extract the preprocessed frames
    scores = torch.tensor([example[1] for example in examples])  # Extract the scores
    return {"pixel_values": frames, "labels": scores}
  
id2label = {0: 'no', 1: 'yellow', 2: 'orange', 3: 'red'}
label2id = {"no": 0, "yellow": 1, "orange": 2, "red": 3}

# This function transforms the images with the same pre processing operations
# used for training the ViT
def pp_frames(frame_data):
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = processor.size["height"]
    normalize = transforms.Normalize(mean=image_mean, std=image_std)


    frame_tensor = transforms.ToTensor()(frame_data)
    frame_tensor = transforms.Resize(size, antialias=True)(frame_tensor)
    frame_tensor = transforms.CenterCrop(size)(frame_tensor)
    frame_tensor = transforms.Normalize(mean=image_mean, std=image_std)(frame_tensor)

    return frame_tensor.permute(0, 2, 1)

class ViTLightningModule(pl.LightningModule):
    def __init__(self, train, test, val, batch_size=90, num_workers=4):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                              num_labels=4,
                                                              id2label=id2label,
                                                              label2id=label2id)
        self.train_dataset = train
        self.test_dataset = test
        self.val_dataset = val
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits
        
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]
        #accuracy = torchmetrics.functional.accuracy(predictions, labels, task="multiclass", num_classes=4)

        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        print(f"val_acc: {accuracy}")
        print(f"val_loss: {loss}")     
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log("validation_accuracy", accuracy, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-5)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                            shuffle=True,
                            collate_fn=collate_fn,
                            batch_size=self.batch_size)
                            # num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                            collate_fn=collate_fn,
                            batch_size=self.batch_size)
                            # num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                            collate_fn=collate_fn,
                            batch_size=self.batch_size)
                            # num_workers=self.num_workers, persistent_workers=True)


print("-"*20)
