import torch.optim as optim
import wandb
from torch.optim import lr_scheduler
from torch.utils.data import random_split

from src.datasets.dataloaders import init_dataloaders
from src.datasets.stl10 import STL10_Dataset
from src.datasets.trainsformations import default_transforms
from src.models.vision_model import Resnet18
from src.training.early_stopper import EarlyStopper
from src.training.evaluator import Evaluator
from src.training.trainer import train_model
from src.utils.wandb import init_wandb

if __name__ == "__main__":
    model = Resnet18(num_classes=5)

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

    train_dataset = STL10_Dataset(train=True)
    train_size = int(0.8 * len(train_dataset))
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, len(train_dataset) - train_size]
    )
    train_dataset.dataset.transform = default_transforms["train"]
    val_dataset.dataset.transform = default_transforms["val"]

    val_dataset.transform = default_transforms["val"]
    test_dataset = STL10_Dataset(train=False, transform=default_transforms["val"])

    dataloaders = init_dataloaders("stl10", train_set=train_dataset, val_set=val_dataset)

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
        early_stopper=EarlyStopper(patience=5),
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
        early_stopper=EarlyStopper(patience=5),
    )

    Evaluator().evaluate_and_log(test_dataset, model)

    wandb.finish()
