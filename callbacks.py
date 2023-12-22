from lightning.pytorch.callbacks import EarlyStopping, DeviceStatsMonitor, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme


def early_stopper():
    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        patience=15,
        strict=False,
        verbose=True,
        mode='max'
    )
    return early_stop_callback

def checkpoint_saver(checkpoint_dir):
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, 
                                          save_top_k=2,
                                          mode="max",
                                          monitor="val_f1",
                                          save_last=True,
                                          save_on_train_epoch_end=False,
                                          verbose=True,
                                          filename="{epoch}-{val_loss:.4f}-{val_f1:.4f}-{val_acc:.4f}")
    return checkpoint_callback
    
    
def checkpoint_saver_optuna(checkpoint_dir):
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, 
                                          save_top_k=3,
                                          mode="max",
                                          monitor="val_f1",
                                          save_on_train_epoch_end=False,
                                          verbose=True,
                                          filename="{epoch}-{val_loss:.4f}-{val_f1:.4f}-{val_acc:.4f}")
    return checkpoint_callback
