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
from omegaconf import OmegaConf
import torch

if __name__ == "__main__":
    with wandb.init() as run:
        config = wandb.config
        dataset_conf = OmegaConf.load(f"config/datasets/{config.dataset}.yaml")
        source_dataset_conf = OmegaConf.load(f"config/datasets/{config.model_source_dataset}.yaml")
        assert config.dataset == dataset_conf.name, (
            "Dataset name in config does not match dataset name in config file:"
            f" {config.dataset} vs. {dataset_conf.name}"
        )

        if config.model == "twolayernet":
            model = TwoLayerNet(
                source_dataset_conf.dim.channels * source_dataset_conf.dim.size * source_dataset_conf.dim.size,
                hidden_size=config.hidden_size,
                num_classes=source_dataset_conf.num_classes,
            )
        elif config.model == "twolayernet_deep":
            model = TwoLayerNetDeep(
                source_dataset_conf.dim.channels * source_dataset_conf.dim.size * source_dataset_conf.dim.size,
                hidden_size=config.hidden_size,
                num_classes=source_dataset_conf.num_classes,
            )
        elif config.model == "lenet5":
            model = LeNet5(num_classes=source_dataset_conf.num_classes)
        elif config.model == "lenet5_base_improved":
            model = LeNet5BaseImproved(num_classes=source_dataset_conf.num_classes)
        elif config.model =="lenet5deepimproved":
            model = LeNet5DeepImproved(num_classes=source_dataset_conf.num_classes)
        else:
            raise NotImplementedError(f"Training model {config.model} not implemented.")
        
        artifact = run.use_artifact(config.checkpoint_artifact)
        artifact_path = artifact.download()

        model.load_state_dict(torch.load(f"{artifact_path}/model.pt"))

        scheduler = lr_scheduler.StepLR
        scheduler_params = {
            "step_size": config.step_size,
            "gamma": config.gamma,
        }
        optimizer = optim.SGD
        optimizer_params = {
            "lr": config.lr,
            "momentum": config.momentum,
        }
        max_epochs = config.max_epochs

        config.update(
            {
                "scheduler": {
                    "name": scheduler.__name__,
                    "params": scheduler_params,
                },
                "optimizer": {
                    "name": optimizer.__name__,
                    "params": optimizer_params,
                },
                "model_architecture": str(model),
            }
        )

        dataset_classes = {"cifar100": CIFAR100_loader, "stl10": STL10_Dataset, "cifar10": CIFAR10}
        train_dataset = dataset_classes[config.dataset](train=True)
        train_size = int(0.8 * len(train_dataset))
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, len(train_dataset) - train_size]
        )
        transform = sized_transform(source_dataset_conf.dim.size)
        train_dataset.dataset.transform = transform["train"]
        val_dataset.dataset.transform = transform["val"]

        val_dataset.transform = transform["val"]
        test_dataset = dataset_classes[config.dataset](train=False, transform=transform["val"])

        dataloaders = init_dataloaders(
            config.dataset, train_set=train_dataset, val_set=val_dataset
        )
        model.replace_fc(num_classes=dataset_conf.num_classes)
        model.freeze()
        
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
