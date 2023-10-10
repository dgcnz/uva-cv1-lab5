import wandb


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
        },
    )
