from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from pointjepa.datasets import (  # allow shorthand notation
    ModelNet40FewShotDataModule,
    ModelNet40Ply2048DataModule,
    ScanObjectNNDataModule,
    ShapeNet55DataModule,
)
from pointjepa.models import Point2Vec

if __name__ == "__main__":
    cli = LightningCLI(
        Point2Vec,
        trainer_defaults={
            "default_root_dir": "artifacts",
            "accelerator": "gpu",
            "devices": 1,
            "precision": 16,
            "max_epochs": 800,
            "track_grad_norm": 2,
            "log_every_n_steps": 10,
            "check_val_every_n_epoch": 10,
            "callbacks": [
                LearningRateMonitor(logging_interval="epoch"),
                # ModelCheckpoint(save_on_train_epoch_end=True),
                # ModelCheckpoint(
                #     save_top_k=5,
                #     monitor="val_loss",
                #     mode="min",
                #     filename="{epoch}-{step}-{val_loss:.3f}",
                # ),
            ],
        },
        # seed_everything_default=0,
        save_config_callback=None,  # https://github.com/Lightning-AI/lightning/issues/12028#issuecomment-1088325894
    )
