from torch.utils.data import DataLoader


def init_dataloaders(
    dataset_name, train_set=None, test_set=None, val_set=None, batch_size=4, num_workers=2
):
    dataloaders = {"name": dataset_name, "phases": {}}

    if train_set:
        dataloaders["phases"]["train"] = {
            "dataloader": DataLoader(
                train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
            ),
            "size": len(train_set),
        }

    if test_set:
        dataloaders["phases"]["test"] = {
            "dataloader": DataLoader(
                test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
            ),
            "size": len(test_set),
        }
    if val_set:
        dataloaders["phases"]["val"] = {
            "dataloader": DataLoader(
                val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
            ),
            "size": len(val_set),
        }

    return dataloaders
