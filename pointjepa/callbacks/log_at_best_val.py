from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule

class TrackLinearAccAtMinLossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')
        self.acc_val_acc_modelnet40_at_min_loss = 0.0
        self.acc_val_acc_scanobjectnn_at_min_loss = 0.0

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        # Assume 'val_loss' and 'svm_val_acc' are logged using `self.log` in your LightningModule
        total_epochs = trainer.max_epochs
        current_epoch = trainer.current_epoch
        should_track_best = current_epoch >= total_epochs * 0.5

        # Only track the best validation loss after the halfway point as the earlier epochs produce low validation loss
        if should_track_best:
            current_val_loss = trainer.callback_metrics.get('val_loss').item()
            svm_val_acc_modelnet40 = trainer.callback_metrics.get('svm_val_acc_modelnet40').item()
            svm_val_acc_scanobjectnn = trainer.callback_metrics.get('svm_val_acc_scanobjectnn').item()

            # Halfway through the training, start tracking the best validation loss
            if current_val_loss < self.best_val_loss: 
                self.best_val_loss = current_val_loss
                self.acc_val_acc_modelnet40_at_min_loss = svm_val_acc_modelnet40
                self.acc_val_acc_scanobjectnn_at_min_loss = svm_val_acc_scanobjectnn
                
            pl_module.log("svm_val_acc_modelnet40_at_min_loss", self.acc_val_acc_modelnet40_at_min_loss)
            pl_module.log("svm_val_acc_scanobjectnn_at_min_loss", self.acc_val_acc_scanobjectnn_at_min_loss)

            pl_module.log("best_val_loss", self.best_val_loss)
