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
import lightning.pytorch as pl

# Third-party libraries
import timm
import matplotlib.pyplot as plt
import seaborn as sns

# Model-related imports
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchvision.models import resnet18, resnet50
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from transformers import ViTForImageClassification
from lightning_modules.BotNet18LightningModule import BotNet
from vit_pytorch import ViT, SimpleViT
from DataAugmentation import DataAugmentation
from sam import SAM
# Metrics and evaluation
from torchmetrics.classification import (MulticlassF1Score, 
                                         Accuracy,
                                         MulticlassConfusionMatrix,
                                         MulticlassAUROC,
                                         MulticlassROC)


id2label = {0: 'no', 1: 'yellow', 2: 'orange', 3: 'red'}
label2id = {"no": 0, "yellow": 1, "orange": 2, "red": 3}

class LUSModelLightningModule(pl.LightningModule):
    def __init__(self, 
                 model_name, 
                 hparams,
                 class_weights=None,
                 freeze_layers=None,
                 pretrained=False,
                 show_model_summary=False,
                 augmentation=False):
        
        super(LUSModelLightningModule, self).__init__()
        
# -------------------------------- Params init ------------------------------- #

        
        self.num_classes = hparams['num_classes']
    
        self.lr = hparams['lr']
        self.batch_size = hparams['batch_size']
        self.max_epochs = hparams['max_epochs']
        self.weight_decay = hparams['weight_decay']
        self.momentum = hparams['momentum']
        self.label_smoothing = hparams['label_smoothing']
        self.drop_rate = hparams['drop_rate']
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
        self.show_model_summary = show_model_summary
        self.augmentation = augmentation
        
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
            # self.model = timm.create_model('swin_tiny_patch4_window7_224.ms_in22k', 
            self.model = timm.create_model('swin_tiny_patch4_window7_224.ms_in22k', 
                                           pretrained=pretrained, 
                                           num_classes=self.num_classes,
                                           drop_rate=self.drop_rate)
            
            if self.pretrained:
                if self.freeze_layers is not None:
                    self.freeze_layers_with_name()
                # else:    
                #     excluded_layers = ['head']
                #     self.freeze_layers_with_exclusion(excluded_layers)
                    
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
# ------------------------------ Data processing ----------------------------- #
        print(f"Using augmentation: {self.augmentation}")
        self.transform = DataAugmentation()
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
        
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        
        self.confmat_metric = MulticlassConfusionMatrix(num_classes=self.num_classes)
        self.auroc_metric = MulticlassAUROC(num_classes=self.num_classes, average='weighted')
        self.roc_metric = MulticlassROC(num_classes=self.num_classes)
        
        
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
            # base_optimizer = torch.optim.AdamW
            optimizer = torch.optim.AdamW(self.parameters(),
                                              lr=self.lr,
                                              weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            # base_optimizer = torch.optim.SGD
            optimizer = torch.optim.SGD(self.parameters(),
                                             lr=self.lr,
                                             momentum=self.momentum,
                                             weight_decay=self.weight_decay)
        else:
            raise ValueError("Invalid optimizer name. Please choose either 'adam' or 'sgd'.")
        
        # optimizer = SAM(self.parameters(), base_optimizer, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            # 'scheduler': torch.optim.lr_scheduler.CosineLRScheduler(optimizer,
            #                                                         T_max=self.batch_size,
            #                                                         eta_min=1e-5,
            #                                                         verbose=True),
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                    mode='max', 
                                                                    patience=10, 
                                                                    factor=0.5,
                                                                    verbose=True),
            'monitor': 'val_acc',  # Monitor validation loss
            'verbose': True
            # 'interval': 'epoch',  # Adjust the LR on every step
        }
        
        # return [optimizer]
        return [optimizer], [scheduler]

    def on_fit_start(self):
        self.warmup_epochs = int(self.max_epochs * 0.1)
        self.warmup_steps = int((self.trainer.estimated_stepping_batches/self.max_epochs)*self.warmup_epochs)
        print(f"\nEstimated total train steps: {self.trainer.estimated_stepping_batches}")
        print(f"\nWarmup epochs: {self.warmup_epochs}\nWarmup steps: {self.warmup_steps}")
        
    # Learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)
        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in self.trainer.optimizers[0].param_groups:
                pg["lr"] = lr_scale * self.lr
                # print(f'learning rate is {pg["lr"]}')
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.trainer.training and self.augmentation:
            x = self.transform(x)  # => we perform GPU/Batched data augmentation
        return x, y
    
    def forward(self, x):
        return self.model(x)

    def on_val_epoch_end(self):
        # Logging train and validation losses to TensorBoard
        self.logger.experiment.add_scalar('train_loss', torch.tensor(self.train_losses).mean(), self.current_epoch)
        self.logger.experiment.add_scalar('val_loss', torch.tensor(self.val_losses).mean(), self.current_epoch)
        
        self.train_losses = []
        self.val_losses = []

    def training_step(self, batch, batch_idx):
        # optimizer = self.optimizers()
        x, y = batch
        logits = self(x)
        
        # loss = self.weighted_cross_entropy(logits, y)
        loss = self.cross_entropy(logits, y)
        
        # loss_1 = self.cross_entropy(logits, y)
        # self.manual_backward(loss_1, optimizer)
        # optimizer.first_step(zero_grad=True)
        
        # second forward-backward pass
        # loss_2 = self.cross_entropy(logits, y)
        # self.manual_backward(loss_2, optimizer)
        # optimizer.second_step(zero_grad=True)
    
        self.train_acc(logits, y)
        self.log_dict({'train_loss': loss, 
                    #    'train_loss_2': loss_2, 
                       'train_acc': self.train_acc}, prog_bar=True, on_epoch=True)
        
        self.log('lr', self.trainer.optimizers[0].param_groups[0]["lr"], on_epoch=False, prog_bar=False)
        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)
        
        loss = self.cross_entropy(logits, y)
        # loss = self.weighted_cross_entropy(logits, y)
        
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.log_dict({'val_loss': loss,
                       'val_f1': self.val_f1,
                       'val_acc': self.val_acc}, prog_bar=True, on_epoch=True)
        self.val_losses.append(loss.item())
        return loss

    def test_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)
        
        loss = self.cross_entropy(logits, y)
        
        # Log confusion matrix to TensorBoard
        self.confmat_metric.update(logits, y)
        self.auroc_metric(logits, y)
        self.roc_metric.update(logits, y)
        
        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.log_dict({'test_loss': loss, 
                       'test_acc': self.test_acc, 
                       'test_f1': self.test_f1,
                       'AUROC': self.auroc_metric}, prog_bar=True, on_epoch=True)
        return loss
    
    def on_test_end(self):
        
        self.logger.experiment.add_figure('Confusion Matrix', self.confmat_metric.plot()[0])
        self.logger.experiment.add_figure('ROC', self.roc_metric.plot(score=True)[0])
        self.confmat_metric.reset()
        self.roc_metric.reset()


    def plot_losses(self):
        num_batches_per_epoch_train = len(self.trainer.train_dataloader)
        num_batches_per_epoch_val = len(self.trainer.val_dataloaders)

        epoch_train_losses = [np.mean(self.train_losses[i:i + num_batches_per_epoch_train]) for i in range(0, len(self.train_losses), num_batches_per_epoch_train)]
        epoch_val_losses = [np.mean(self.val_losses[i:i + num_batches_per_epoch_val]) for i in range(0, len(self.val_losses), num_batches_per_epoch_val)]

        sns.lineplot(x=range(len(epoch_train_losses)), y=epoch_train_losses, label='Train Loss')
        sns.lineplot(x=range(len(epoch_val_losses)), y=epoch_val_losses, label='Validation Loss')
        
        self.train_losses = []
        self.val_losses = []

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Log the figure to the logger
        self.logger.experiment.add_figure('Losses', plt.gcf())


    def on_fit_end(self):
        self.plot_losses()
      
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