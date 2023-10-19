import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split

import wandb
from src.datasets.dataloaders import init_dataloaders
from src.datasets.cifar100 import CIFAR100_loader
from src.datasets.cifar10 import CIFAR10
from src.datasets.stl10 import STL10_Dataset
from src.datasets.trainsformations import default_transforms
from src.models.twolayer import TwoLayerNet
from src.training.early_stopper import EarlyStopper
from src.training.evaluator import Evaluator
from src.training.trainer import train_model
from src.utils.wandb import init_wandb
import argparse
from omegaconf import OmegaConf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_config", default="config/datasets/stl10.yaml", type=str, help="dataset"
    )
    args = parser.parse_args()

    dataset_conf = OmegaConf.load(args.dataset_config)

    model = TwoLayerNet(
        dataset_conf.dim.channels * dataset_conf.dim.width * dataset_conf.dim.height,
        hidden_size=120,
        num_classes=dataset_conf.num_classes,
    )

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
    max_epochs = 10
    optimizer_params_finetune = {
        "lr": 0.0001,
        "momentum": 0.9,
    }

    dataset_classes = {"cifar100": CIFAR100_loader, "stl10": STL10_Dataset, "cifar10": CIFAR10}
    train_dataset = dataset_classes[dataset_conf.name](train=True)
    train_size = int(0.8 * len(train_dataset))
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, len(train_dataset) - train_size]
    )
    train_dataset.dataset.transform = default_transforms["train"]
    val_dataset.dataset.transform = default_transforms["val"]

    val_dataset.transform = default_transforms["val"]
    test_dataset = dataset_classes[dataset_conf.name](train=False, transform=default_transforms["val"])

    dataloaders = init_dataloaders(dataset_conf.name, train_set=train_dataset, val_set=val_dataset)

    init_wandb(
        dataloaders["name"],
        scheduler,
        scheduler_params,
        optimizer,
        optimizer_params,
        max_epochs,
        model,
    )
    
    model = train_model(
        model,
        dataloaders,
        num_epochs=max_epochs,
        fine_tune=True,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        scheduler=scheduler,
        optimizer=optimizer,
        early_stopper=EarlyStopper(patience=3),
    )

    Evaluator().evaluate_and_log(test_dataset, model)

    wandb.finish()
