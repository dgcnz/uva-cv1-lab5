import wandb
import os
from tempfile import TemporaryDirectory
import torch

def init_wandb(
    dataset_name, scheduler, scheduler_params, optimizer, optimizer_params, num_epochs, model
):
    wandb.init(
        project=dataset_name,
        config={
            "scheduler": {
                "name": scheduler.__name__,
                "params": scheduler_params,
            },
            "optimizer": {
                "name": optimizer.__name__,
                "params": optimizer_params,
            },
            "max_epochs": num_epochs,
            "model": model.name,
            "model_architecture": str(model),
        },
    )


def save_model(model):
    with TemporaryDirectory() as tmpdir:
        torch.save(model.state_dict(), os.path.join(tmpdir, "model.pt"))
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(os.path.join(tmpdir, "model.pt"))
        wandb.run.log_artifact(artifact)

