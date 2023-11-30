# Standard libraries
from PIL import Image
import numpy as np

# PyTorch and related libraries
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms.functional as TF

# PyTorch Lightning
import pytorch_lightning as pl

# Third-party libraries
from kornia import tensor_to_image
import timm
import matplotlib.pyplot as plt
import seaborn as sns

# Model-related imports
from torchvision.models import resnet18, resnet50
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from transformers import ViTForImageClassification
from lightning_modules.BotNet18LightningModule import BotNet
from vit_pytorch import ViT, SimpleViT

# Metrics and evaluation
from torchmetrics.classification import MulticlassF1Score, Accuracy
from sklearn.metrics import confusion_matrix


id2label = {0: 'no', 1: 'yellow', 2: 'orange', 3: 'red'}
label2id = {"no": 0, "yellow": 1, "orange": 2, "red": 3}

class LUSModelLightningModule(pl.LightningModule):
    def __init__(self, 
                 model_name, 
                 hparams,
                 class_weights=None,
                 freeze_layers=None,
                 pretrained=False,
                 show_model_summary=False):
        
        super(LUSModelLightningModule, self).__init__()
        
# -------------------------------- Params init ------------------------------- #

        
        self.num_classes = hparams['num_classes']
    
        self.lr = hparams['lr']
        self.weight_decay = hparams['weight_decay']
        self.momentum = hparams['momentum']
        self.label_smoothing = hparams['label_smoothing']
        self.drop_rate = hparams['drop_rate']
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
        self.show_model_summary = show_model_summary
        
# ----------------------------------- Model ---------------------------------- #

# --------------------------------- BotNet18 --------------------------------- #
        if model_name == "botnet50":
            self.model = BotNet("bottleneck",
                                 [3, 4, 6, 3], 
                                  num_classes=self.num_classes, 
                                  resolution=(224, 224), 
                                  heads=4)
            
        elif model_name == "botnet18":
            self.model = BotNet("basic",
                                  [2, 2, 2, 2], 
                                  num_classes=self.num_classes, 
                                  resolution=(224, 224), 
                                  heads=4)

# --------------------------------- resnet --------------------------------- #
        # if "resnet18" in model_name:
        #     self.model = models.resnet18(pretrained=pretrained)
        #     self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        #     print(f"\nUsing pretrained weights {self.pretrained}\n")
        #     if self.pretrained:
        #         # List of layers to exclude from freezing
        #         excluded_layers = ['fc', 'layer3', 'layer4']
        #         self.freeze_layers_with_exclusion(excluded_layers)
        # if self.show_model_summary:
        #     self.print_layers_req_grad()
                
        # elif "resnet50" in model_name:
        #     self.model = models.resnet50(pretrained=pretrained)
        #     self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        #     print(f"\nUsing pretrained weights {self.pretrained}\n")
        #     if self.pretrained:
        #         # List of layers to exclude from freezing
        #         excluded_layers = ['fc', 'layer3', 'layer4']
        #         self.freeze_layers_with_exclusion(excluded_layers)
        # if self.show_model_summary:
        #     self.print_layers_req_grad()
                
                
        if "resnet" in model_name :   
            # torch image models resnet18/50
            if "resnet10t" in model_name:
                self.model = timm.create_model(f"resnet10t.c3_in1k",
                                                pretrained=self.pretrained,
                                                num_classes=self.num_classes,
                                                drop_rate=self.drop_rate)
            else:
                self.model = timm.create_model(f"{model_name}.a1_in1k",
                                                pretrained=self.pretrained,
                                                num_classes=self.num_classes,
                                                drop_rate=self.drop_rate)
            print(f"\nUsing pretrained weights {self.pretrained}\n")
            if self.pretrained:
                # List of layers to exclude from freezing
                excluded_layers = ['fc', 'layer3', 'layer4']
                self.freeze_layers_with_exclusion(excluded_layers)
                if self.show_model_summary:
                    self.print_layers_req_grad()
            
# -------------------------------- timm_botnet ------------------------------- #
        elif model_name == "timm_bot":
            print(f"\nUsing pretrained weights {self.pretrained}\n")

            self.model = timm.create_model('botnet26t_256.c1_in1k',
                                           pretrained=self.pretrained,
                                           num_classes=self.num_classes,
                                           )
            if self.pretrained:
                # List of layers to exclude from freezing
                excluded_layers = ['fc', 'layer3', 'layer4']
                self.freeze_layers_with_exclusion(excluded_layers)
                if self.show_model_summary:
                    self.print_layers_req_grad()

# -------------------------------- swin_vit ---------------------------------- #
        if model_name == 'swin_vit':
            
            print(f"\nUsing pretrained weights: {pretrained}\n")
            self.model = timm.create_model('swin_base_patch4_window7_224.ms_in1k', 
                                           pretrained=pretrained, 
                                           num_classes=self.num_classes,
                                           drop_rate=self.drop_rate)
            
            if self.pretrained:
                if self.freeze_layers is not None:
                    self.freeze_layers_with_name()
                else:    
                    excluded_layers = ['head']
                    self.freeze_layers_with_exclusion(excluded_layers)
                    
                if self.show_model_summary:
                    self.print_layers_req_grad()
                
# ------------------------------------ vit ----------------------------------- #
            # self.model = ViT(
            #         image_size = 224,
            #         patch_size = 32,
            #         num_classes = self.num_classes,
            #         dim = 1024,
            #         depth = 6,
            #         heads = 16,
            #         mlp_dim = 2048,
            #         dropout = 0.1,
            #         emb_dropout = 0.1
            #     )
# ------------------------------------ HP ------------------------------------ #
        if show_model_summary:
            print(f"\nModel summary:\n{self.model}")
        
        self.optimizer_name = str(hparams['optimizer']).lower()
        
        self.weighted_cross_entropy = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=self.label_smoothing)
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        self.save_hyperparameters(ignore=['class_weights'])
        
# ---------------------------------- Metrics --------------------------------- #

        
        self.train_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
        self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
        
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        
        # Your model initialization here
        self.train_losses = []
        self.val_losses = []

        
# ------------------------------ Methods & Hooks ----------------------------- #
    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                              lr=self.lr,
                                              weight_decay=self.weight_decay)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(),
                                              lr=self.lr,
                                              weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(),
                                             lr=self.lr,
                                             momentum=self.momentum,
                                             weight_decay=self.weight_decay)
        else:
            raise ValueError("Invalid optimizer name. Please choose either 'adam' or 'sgd'.")
        
        
        scheduler = {
            # 'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
            #                                                         T_max=5,
            #                                                         verbose=True),
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                    mode='min', 
                                                                    patience=10, 
                                                                    factor=0.5,
                                                                    verbose=True),
            'monitor': 'val_loss',  # Monitor validation loss
            # 'interval': 'epoch',  # Adjust the LR on every step
            # 'frequency': 1,  # Frequency of the adjustment
            # 'warm_up_start_lr': self.lr,  # Initial LR during warm-up
            'warm_up_epochs': 5,  # Number of warm-up epochs
            'verbose': True
        }
        
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def on_epoch_end(self):
        # Logging train and validation losses to TensorBoard
        self.logger.experiment.add_scalar('train_loss', torch.tensor(self.train_losses).mean(), self.current_epoch)
        self.logger.experiment.add_scalar('val_loss', torch.tensor(self.val_losses).mean(), self.current_epoch)
        
        self.train_losses = []
        self.val_losses = []

    def training_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)
        
        loss = self.weighted_cross_entropy(logits, y)
        
        self.train_acc(logits, y)
        self.log_dict({'train_loss': loss, 
                       'train_acc': self.train_acc}, prog_bar=True, on_epoch=True)
        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)
        
        loss = self.cross_entropy(logits, y)
        
        self.val_acc(logits, y)
        self.log_dict({'val_loss': loss, 
                       'val_acc': self.val_acc}, prog_bar=True, on_epoch=True)
        self.val_losses.append(loss.item())
        return loss

    def test_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)
        
        loss = self.cross_entropy(logits, y)
        
        # Convert logits to predicted labels
        preds = torch.argmax(logits, dim=1)
        # Compute confusion matrix
        cm = confusion_matrix(y.cpu().numpy(), preds.cpu().numpy())
        
        # Log confusion matrix to TensorBoard
        self.logger.experiment.add_figure('Confusion Matrix', self.plot_confusion_matrix(cm), self.current_epoch)
        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.log_dict({'test_loss': loss, 
                       'test_acc': self.test_acc, 
                       'test_f1': self.test_f1}, prog_bar=True, on_epoch=True)
        return loss
    
    def plot_confusion_matrix(self, cm):
        class_names = [str(i) for i in range(len(cm))]
        fig, ax = plt.subplots(figsize=(len(class_names), len(class_names)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.close(fig)
        return fig
    
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
      
      
    def freeze_layers_with_exclusion(self, excluded_layers):
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the layers in the excluded list
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in excluded_layers):
                param.requires_grad = True  
                
    def freeze_layers_with_name(self):
        print(f"Freezing all layers with {self.freeze_layers} in name")
        for name, param in self.model.named_parameters():
            if self.freeze_layers in name:
                param.requires_grad = False
                
    def print_layers_req_grad(self):
        # Print all layers and their requires_grad status
        for name, param in self.model.named_parameters():
            print(f'Parameter: {name}, Requires Gradient: {param.requires_grad}')