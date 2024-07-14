from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.cli import LightningCLI
from pointjepa.callbacks.log_at_best_val import TrackLinearAccAtMinLossCallback
from pointjepa.models import PointJepa

from pointjepa.datasets import (  # allow shorthand notation
    ModelNet40FewShotDataModule,
    ModelNet40Ply2048DataModule,
    ScanObjectNNDataModule,
    ShapeNet55DataModule,
)
from pointjepa.callbacks.wandb_checkpoint_logger import WandbModelCheckpointLogger

if __name__ == "__main__":
    cli = LightningCLI(
        PointJepa,
        trainer_defaults={
            "default_root_dir": "artifacts",
            "accelerator": "gpu",
            "devices": 1,
            "precision": 16,
            "max_epochs": 300,
            "track_grad_norm": 2,
            "log_every_n_steps": 10,
            "check_val_every_n_epoch": 10,
            "callbacks": [
                LearningRateMonitor(logging_interval="epoch"),
                TrackLinearAccAtMinLossCallback(),
            ]
        },
        seed_everything_default=1,
        save_config_callback=None,  # https://github.com/Lightning-AI/lightning/issues/12028#issuecomment-1088325894
    )