import lightning.pytorch as pl

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer
)

def train_function(model, data_module):
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=data_module)

def tune_model(ray_trainer, num_epochs, num_samples=10):
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "accumulate_grad_batches": tune.choice([1, 4, 8, 16, 32]),
        "optimizer": tune.choice(["adam", "sgd", "adamw"]),
        "weight_decay": tune.choice([0.0, 0.005, 0.001, 0.0005]),
        "drop_rate": tune.choice([0.0, 0.1, 0.2]),
        "label_smoothing": tune.choice([0.0, 0.1])
    }
    
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="ptl/val_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()

