import torch
import torchvision


class CIFAR100_loader(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self._data = torchvision.datasets.CIFAR100(train=train, root=root, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item]

        if self.transform:
            img = self.transform(img)

        return img, target
