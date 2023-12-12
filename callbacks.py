from pytorch_lightning.callbacks import EarlyStopping, DeviceStatsMonitor, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme


def early_stopper():
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        strict=False,
        verbose=False,
        mode='min'
    )
    return early_stop_callback

def checkpoint_saver(checkpoint_dir):
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, 
                                          save_top_k=2,
                                          mode="max",
                                          monitor="val_acc",
                                          save_last=True,
                                          save_on_train_epoch_end=False,
                                          verbose=True,
                                          filename="{epoch}-{val_acc:.4f}-{val_loss:.4f}")
    return checkpoint_callback
