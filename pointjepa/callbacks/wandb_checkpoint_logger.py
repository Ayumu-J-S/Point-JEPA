import pytorch_lightning as pl
import wandb
import os

class WandbModelCheckpointLogger(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        # Check if WandbLogger is being used
        if isinstance(trainer.logger, pl.loggers.WandbLogger):
            # Get the Wandb run from the logger
            wandb_run = trainer.logger.experiment

            check_logged_any = False
            # Iterate over all ModelCheckpoint callbacks
            for callback in trainer.callbacks:
                if isinstance(callback, pl.callbacks.ModelCheckpoint):
                    # Check if a checkpoint was saved during training
                    if callback.best_model_path:
                        # Get the metric name being monitored
                        monitor = callback.monitor

                        # Construct the model name using the monitor and its value
                        model_name = f"best_{monitor}"
                        model_path = callback.best_model_path

                        # Create the artifact
                        artifact = wandb.Artifact(model_name, type='model', 
                                                  description=f"Final model checkpoint for {monitor} at end of training")
                        artifact.add_file(model_path)

                        print(f"Logging {model_name} to wandb artifact: {artifact}.")
                        # Log the artifact to the current Wandb run
                        wandb_run.log_artifact(artifact)
                        check_logged_any = True
            
            # If no checkpoints were saved, log a dummy artifact
            if not check_logged_any:
                print("No checkpoints were saved during training. Logging a dummy artifact.")
