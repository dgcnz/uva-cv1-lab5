import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split

import wandb
from src.datasets.dataloaders import init_dataloaders
from src.datasets.cifar100 import CIFAR100_loader
from src.datasets.cifar10 import CIFAR10
from src.datasets.stl10 import STL10_Dataset
from src.datasets.trainsformations import sized_transform
from src.models.twolayer import TwoLayerNet, TwoLayerNetDeep
from src.models.lenet5 import LeNet5, LeNet5BaseImproved, LeNet5DeepImproved
from src.training.early_stopper import EarlyStopper
from src.training.evaluator import Evaluator
from src.training.trainer import train_model
from src.utils.wandb import init_wandb, save_model
import argparse
from omegaconf import OmegaConf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_config", default="config/datasets/stl10.yaml", type=str, help="dataset"
    )
    parser.add_argument(
        "--model", default="twolayernet", type=str, help="model", choices=["twolayernet", "lenet5", "lenet5deepimproved"]
    )
    parser.add_argument(
        "--hidden_size", default=120, type=int, help="twolayernet hidden size"
    )

    parser.add_argument(
        "--max_epochs", default=10, type=int, help="max epochs"
    )
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    dataset_conf = OmegaConf.load(args.dataset_config)

    if args.model == "twolayernet":
        model = TwoLayerNet(
            dataset_conf.dim.channels * dataset_conf.dim.size * dataset_conf.dim.size,
            hidden_size=args.hidden_size,
            num_classes=dataset_conf.num_classes,
        )
    elif args.model == "twolayernet_deep":
        model = TwoLayerNetDeep(
            dataset_conf.dim.channels * dataset_conf.dim.size * dataset_conf.dim.size,
            hidden_size=args.hidden_size,
            num_classes=dataset_conf.num_classes,
        )
    elif args.model == "lenet5":
        model = LeNet5(num_classes=dataset_conf.num_classes)
    elif args.model == "lenet5_base_improved":
        model = LeNet5BaseImproved(num_classes=dataset_conf.num_classes)
    elif args.model == "lenet5deepimproved":
        model = LeNet5DeepImproved(num_classes=dataset_conf.num_classes)
    else:
        raise NotImplementedError(f"Training model {args.model} not implemented.")

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
    if args.model in ["lenet5", "lenet5_base_improved", "lenet5deepimproved"]:
        transform = sized_transform(32)
    else:
        transform = sized_transform(dataset_conf.dim.size)
    train_dataset.dataset.transform = transform["train"]
    val_dataset.dataset.transform = transform["val"]

    val_dataset.transform = transform["val"]
    test_dataset = dataset_classes[dataset_conf.name](train=False, transform=transform["val"])

    dataloaders = init_dataloaders(dataset_conf.name, train_set=train_dataset, val_set=val_dataset)

    init_wandb(
        dataloaders["name"],
        scheduler,
        scheduler_params,
        optimizer,
        optimizer_params,
        args.max_epochs,
        model,
    )

    model = train_model(
        model,
        dataloaders,
        num_epochs=args.max_epochs,
        fine_tune=False,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        scheduler=scheduler,
        optimizer=optimizer,
        early_stopper=EarlyStopper(patience=3),
    )

    Evaluator().evaluate_and_log(test_dataset, model)
    if args.save_model:
        save_model(model)

    wandb.finish()
