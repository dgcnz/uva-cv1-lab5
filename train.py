import os

import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets

import wandb
from src.datasets.dataloaders import init_dataloaders
from src.datasets.trainsformations import default_transforms
from src.models.vision_model import Resnet18
from src.training.early_stopper import EarlyStopper
from src.training.trainer import train_model
from src.utils.wandb import init_wandb

if __name__ == "__main__":
    model = Resnet18(num_classes=2)

    scheduler = lr_scheduler.StepLR
    scheduler_params = {
        "step_size": 7,
        "gamma": 0.1,
    }
    optimizer = optim.SGD
    optimizer_params = {
        "lr": 0.001,
        "momentum": 0.9,
    }
    max_epochs = 2
    optimizer_params_finetune = {
        "lr": 0.0001,
        "momentum": 0.9,
    }

    # load data
    data_dir = "data/hymenoptera_data"
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), default_transforms[x])
        for x in ["train", "val"]
    }

    dataloaders = init_dataloaders(
        "hymenoptera", train_set=image_datasets["train"], val_set=image_datasets["val"]
    )

    early_stopper = EarlyStopper(patience=5)

    init_wandb(
        dataloaders["name"],
        scheduler,
        scheduler_params,
        optimizer,
        optimizer_params,
        max_epochs,
        model,
    )

    model.freeze()
    model = train_model(
        model,
        dataloaders,
        num_epochs=max_epochs,
        fine_tune=False,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        scheduler=scheduler,
        optimizer=optimizer,
        early_stopper=early_stopper,
    )
    model.unfreeze()
    model = train_model(
        model,
        dataloaders,
        num_epochs=max_epochs,
        fine_tune=True,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        scheduler=scheduler,
        optimizer=optimizer,
        early_stopper=early_stopper,
    )
    wandb.finish()
