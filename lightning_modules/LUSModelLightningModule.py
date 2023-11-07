import torch
import torchvision
import torch.nn as nn
import timm
import lightning.pytorch as pl
from kornia import tensor_to_image
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassF1Score, Accuracy
from torchvision.models import resnet18, ResNet18_Weights
from transformers import ViTForImageClassification
from lightning_modules.BotNet18LightningModule import BotNet
from vit_pytorch import ViT, SimpleViT

from data_augmentation import DataAugmentation


id2label = {0: 'no', 1: 'yellow', 2: 'orange', 3: 'red'}
label2id = {"no": 0, "yellow": 1, "orange": 2, "red": 3}

class LUSModelLightningModule(pl.LightningModule):
    def __init__(self, 
                 model_name, 
                 hparams,
                 class_weights=None,
                 pretrained=True,
                 freeze_layers=None):
        
        super(LUSModelLightningModule, self).__init__()
        
# -------------------------------- Params init ------------------------------- #

        
        self.num_classes = hparams['num_classes']
        
        self.optim = {
            "lr": hparams['lr'],
            "weight_decay": hparams['weight_decay'],
            "momentum": hparams['momentum']
        }
        
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
        
# ----------------------------------- Model ---------------------------------- #

# --------------------------------- BotNet18 --------------------------------- #
        if model_name == "botnet50":
            self.model = BotNet("bottleneck",
                                 [3, 4, 6, 3], 
                                  num_classes=4, 
                                  resolution=(224, 224), 
                                  heads=4)
            
        elif model_name == "botnet18":
            self.model = BotNet("basic",
                                  [2, 2, 2, 2], 
                                  num_classes=4, 
                                  resolution=(224, 224), 
                                  heads=4)

# --------------------------------- resnet --------------------------------- #
        elif model_name == "resnet50":
            if self.pretrained:
                print("\nUsing pretrained weights\n")
                
                self.model = timm.create_model("resnet50.a1_in1k",
                                               pretrained=True,
                                               num_classes=self.num_classes)
                # List of layers to exclude from freezing
                excluded_layers = ['fc', 'layer3', 'layer4']

                # Freeze all layers
                for param in self.model.parameters():
                    param.requires_grad = False

                # Unfreeze the layers in the excluded list
                for name, param in self.model.named_parameters():
                    if any(layer in name for layer in excluded_layers):
                        param.requires_grad = True
                        
                # Print all layers and their requires_grad status
                for name, param in self.model.named_parameters():
                    print(f'Parameter: {name}, Requires Gradient: {param.requires_grad}')
                        
            #     self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            #     if self.freeze_layers is None:
            #         # Freeze all layers except the final classification layer
            #         for name, param in self.model.named_parameters():
            #             if 'fc' not in name:
            #                 param.requires_grad = False
            #     else:
            #         # Freeze layers up to the specified layer
            #         freeze = True
            #         for name, param in self.model.named_parameters():
            #             if self.freeze_layers in name:
            #                 freeze = False
            #             if freeze:
            #                 param.requires_grad = False
            #     # Replace the final classification layer with a new one for the specific number of classes
            # else:
            #     print("\nNo pretrained weights\n")
            #     self.model = resnet18(weights=None)
            # self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
# -------------------------------- timm_botnet ------------------------------- #
        elif model_name == "timm_bot":
            print(f"\nUsing pretrained weights: {pretrained}\n")
            self.model = timm.create_model('botnet26t_256.c1_in1k',
                                           pretrained=self.pretrained,
                                           num_classes=self.num_classes,
                                           )
            if self.pretrained:
                print("Freezing layers up to head")
                if self.freeze_layers is None:
                    # If no specific layer is provided, freeze all layers except 'head'
                    for name, param in self.model.named_parameters():
                        if 'head' in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                else:
                    # Freeze layers up to the specified layer
                    freeze = True
                    for name, param in self.model.named_parameters():
                        if self.freeze_layers in name:
                            freeze = False
                        if freeze:
                            param.requires_grad = False
                    for name, param in self.model.named_parameters():
                        print(f'Parameter: {name}, Requires Gradient: {param.requires_grad}')



        self.optimizer_name = str(hparams['optimizer']).lower()
        self.train_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.test_criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

# -------------------------------- vit -------------------------------------- #
        if model_name == 'swin_vit':
            
            print(f"\nUsing pretrained weights: {pretrained}\n")
            self.model = timm.create_model('swin_base_patch4_window7_224.ms_in1k', 
                                           pretrained=pretrained, 
                                           num_classes=self.num_classes)
            if self.pretrained:
                if self.freeze_layers is not None:
                    print("Freezing all layers with {self.freeze_layers} in name")
                    for name, param in self.model.named_parameters():
                        if self.freeze_layers in name:
                            param.requires_grad = False
                            
                    # if 'head' in name:
                    #     param.requires_grad = True
                        
            # printing all layers and require grads
            for name, param in self.model.named_parameters():
                print(f'Parameter: {name}, Requires Gradient: {param.requires_grad}')
                        
                # if self.freeze_layers is None:
                #     print("Freezing layers up to head")
                #     # If no specific layer is provided, freeze all layers except 'head'
                #     for name, param in self.model.named_parameters():
                #         if 'head' in name:
                #             param.requires_grad = True
                #         else:
                #             param.requires_grad = False
                # else:
                #     # Freeze layers up to the specified layer
                #     freeze = True
                #     for name, param in self.model.named_parameters():
                #         if self.freeze_layers in name:
                #             freeze = True
                #         if freeze:
                #             param.requires_grad = True
                
        if model_name == 'vit':
            
            print(f"\nUsing pretrained weights: {pretrained}\n")
            self.model = ViT(
                    image_size = 224,
                    patch_size = 32,
                    num_classes = self.num_classes,
                    dim = 1024,
                    depth = 6,
                    heads = 16,
                    mlp_dim = 2048,
                    dropout = 0.1,
                    emb_dropout = 0.1
                )
            #TODO - Pretrained version of vit
            #TODO - Freeze layers up to the specified layer
            # if self.pretrained:
            #     # self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
            #     #                                                        num_labels=self.num_classes)
                
            # else:
                
# ------------------------------ Data processing ----------------------------- #

        self.transform = DataAugmentation()
        
# ---------------------------------- Metrics --------------------------------- #

        self.train_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
        self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
        
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=self.num_classes)

# ------------------------------ Methods & Hooks ----------------------------- #
    def configure_optimizers(self):
        """
        Configure and return the optimizer based on the selected optimizer name.

        Returns:
            optimizer (torch.optim.Optimizer): The configured optimizer.

        Raises:
            ValueError: If the optimizer name is invalid.
        """
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                              lr=self.optim["lr"],
                                              weight_decay=self.optim["weight_decay"])
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(),
                                             lr=self.optim["lr"],
                                             momentum=self.optim["momentum"],
                                             weight_decay=self.optim["weight_decay"])
        else:
            raise ValueError("Invalid optimizer name. Please choose either 'adam' or 'sgd'.")
        
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1),
            'monitor': 'validation_loss',  # Monitor validation loss
            'verbose': True
        }
        
        return [optimizer], [scheduler]

    def forward(self, x):
        """
        Forward pass of the model.
        Parameters:
            x (Tensor): The input tensor.
        Returns:
            Tensor: The output tensor.
        """
        return self.model(x)

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
        loss = self.train_criterion(logits, y)
        self.train_acc(logits, y)
        self.train_f1(logits, y)
        self.log('training_loss', loss, 
                 prog_bar=True,
                 on_epoch=True,
                 logger=True,
                 on_step=False)
        self.log('training_accuracy', self.train_acc(logits, y), 
                 prog_bar=True,
                 on_epoch=True,
                 logger=True,
                 on_step=False)
        # self.log('training_f1', self.train_f1(logits, y), 
        #          prog_bar=True,
        #          on_epoch=True,
        #          logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Performs a validation step on a batch of data.

        Args:
            batch (tuple): A tuple containing the input data and the correspondPing labels.
            batch_idx (int): The index of the current batch.

        Returns:
            float: The loss calculated during the validation step.
        """
        x, y = batch
        logits = self(x)
        loss = self.test_criterion(logits, y)
        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc(logits, y), prog_bar=True)
        self.log('test_f1', self.test_f1(logits, y), prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step on a batch of data.

        Args:
            batch (tuple): A tuple containing the input data and the correspondPing labels.
            batch_idx (int): The index of the current batch.

        Returns:
            float: The loss calculated during the validation step.
        """
        x, y = batch
        logits = self(x)
        loss = self.test_criterion(logits, y)
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.log('validation_loss', loss, prog_bar=True)
        self.log('validation_acc', self.val_acc(logits, y), prog_bar=True)
        self.log('validation_f1', self.val_f1(logits, y), prog_bar=True)
        return loss

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