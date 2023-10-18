import torch
import torchvision
import torch.nn as nn
import timm
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from kornia import tensor_to_image
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassF1Score, Accuracy
from torchvision.models import resnet18, ResNet18_Weights
from transformers import ViTForImageClassification

from data_setup import DataAugmentation

def collate_fn(examples):
    frames = torch.stack([example[0] for example in examples])  # Extract the preprocessed frames
    scores = torch.tensor([example[1] for example in examples])  # Extract the scores
    return (frames, scores)
  
id2label = {0: 'no', 1: 'yellow', 2: 'orange', 3: 'red'}
label2id = {"no": 0, "yellow": 1, "orange": 2, "red": 3}

class LUSModelLightningModule(pl.LightningModule):
    def __init__(self, 
                 model_name, 
                 hparams,
                 train_dataset,
                 test_dataset,
                 num_workers,
                 pretrained=True):
        
        super(LUSModelLightningModule, self).__init__()
        
# -------------------------------- Params init ------------------------------- #

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        self.num_classes = hparams['num_classes']
        self.batch_size = hparams['batch_size']
        
        self.optim = {
            "lr": hparams['lr'],
            "weight_decay": hparams['weight_decay'],
            "momentum": hparams['momentum']
        }
        
        self.num_workers = num_workers
        self.pretrained = pretrained
        
# ----------------------------------- Model ---------------------------------- #

        if model_name == "resnet18":
            
            if self.pretrained:
                print("\n\nUsing pretrained weights\n\n")
                self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                print("\n\nNo pretrained weights\n\n")
                self.model = resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
            
        elif model_name =="timm_bot":
            print(f"\n\nUsing pretrained weights: {pretrained}\n\n")
            self.model = timm.create_model('botnet26t_256.c1_in1k', 
                         pretrained=self.pretrained, 
                         num_classes=self.num_classes,
                         )


        self.optimizer_name = str(hparams['optimizer']).lower()
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        
# ------------------------------ Data processing ----------------------------- #

        self.transform = DataAugmentation()
        
# ---------------------------------- Metrics --------------------------------- #

        self.f1_score_metric = MulticlassF1Score(num_classes=self.num_classes)
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)

# ------------------------------ Methods & Hooks ----------------------------- #

    def forward(self, x):
        """
        Forward pass of the model.
        Parameters:
            x (Tensor): The input tensor.
        Returns:
            Tensor: The output tensor.
        """
        return self.model(x)


    def show_batch(self, win_size=(10, 10)):
      """
      Displays a batch of images along with their augmented versions.

      Parameters:
          win_size (tuple): The size of the window to display the images. Defaults to (10, 10).

      Returns:
          None
      """
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
        """
        This function is called after a batch transfer in the data loader.
        It takes in two parameters:
            - batch: the batch of data
            - dataloader_idx: the index of the data loader
        
        The function performs GPU/batched data augmentation on the pixel values of the batch.
        It then returns the augmented pixel values (x) and the labels.
        """
        x, y = batch
        if self.trainer.training:
            x = self.transform(x)  # => we perform GPU/Batched data augmentation
        return x, y
      
    
    def training_step(self, batch, batch_idx):
        """
        Executes a single training step.

        Args:
            batch (Tuple): A tuple containing the input data and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            float: The computed loss value.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log('training_loss', loss, prog_bar=True)
        self.log('training_accuracy', self.train_acc(logits, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step on a batch of data.

        Args:
            batch (tuple): A tuple containing the input data and the corresponding labels.
            batch_idx (int): The index of the current batch.

        Returns:
            float: The loss calculated during the validation step.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits, y)
        self.log('validation_loss', loss, prog_bar=True)
        self.log('validation_acc', self.val_acc(logits, y), prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure and return the optimizer based on the selected optimizer name.

        Returns:
            optimizer (torch.optim.Optimizer): The configured optimizer.

        Raises:
            ValueError: If the optimizer name is invalid.
        """
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(),
                                              lr=self.optim["lr"],
                                              weight_decay=self.optim["weight_decay"])
        elif self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(),
                                             lr=self.optim["lr"],
                                             momentum=self.optim["momentum"],
                                             weight_decay=self.optim["weight_decay"])
        else:
            raise ValueError("Invalid optimizer name. Please choose either 'adam' or 'sgd'.")
        print(self.optimizer)
        return self.optimizer

    def train_dataloader(self):
        """
        Returns a `DataLoader` object for training the model.

        This function creates a `DataLoader` object using the `train_dataset` as the dataset to be loaded. The `batch_size` parameter specifies the number of samples per batch. The `collate_fn` parameter is used to collate the samples in each batch. The `shuffle` parameter indicates whether to shuffle the samples before each epoch.

        Returns:
            A `DataLoader` object for training the model.
        """
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self):
        """
        Returns a DataLoader object for validation data.
        
        Parameters:
            self (object): The current instance of the class.
        
        Returns:
            DataLoader: A DataLoader object that loads the validation dataset in batches.
        """
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          collate_fn=collate_fn)
