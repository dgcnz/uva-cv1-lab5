import os
from tempfile import TemporaryDirectory

import torch
import torch.optim as optim
import wandb
from torch.optim import lr_scheduler
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def train_model(
    model,
    dataloaders,
    num_epochs=25,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"),
    optimizer=optim.SGD,
    optimizer_params={},
    scheduler=lr_scheduler.StepLR,
    scheduler_params={},
    fine_tune=False,
    early_stopper=None,
):
    if fine_tune:
        wandb.config.update({"fine_tune_learnable_params": model.learnable_parameters})
        wandb.config.update({"optimizer_params_finetune": optimizer_params})
    else:
        wandb.config.update({"learnable_params": model.learnable_parameters})

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        model = model.to(device)

        optimizer = optimizer(model.parameters(), **optimizer_params)
        scheduler = scheduler(optimizer, **scheduler_params)

        for _ in range(num_epochs):
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders["phases"][phase]["dataloader"]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs, loss = model(inputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataloaders["phases"][phase]["size"]
                epoch_acc = running_corrects.float() / dataloaders["phases"][phase]["size"]

                if not fine_tune:
                    wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_acc": epoch_acc})
                else:
                    wandb.log(
                        {f"{phase}_loss_finetune": epoch_loss, f"{phase}_acc_finetune": epoch_acc}
                    )

                if phase == "val":
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)

                    if early_stopper:
                        early_stopper.losses.append(epoch_loss)
                        if early_stopper.stop():
                            model.load_state_dict(torch.load(best_model_params_path))
                            return model

        model.load_state_dict(torch.load(best_model_params_path))
    return model
