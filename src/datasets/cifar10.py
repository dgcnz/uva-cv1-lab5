import numpy as np
import torchvision
from PIL import Image


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data", train=True, transform=None, download=True, N=None):
        """
        Initializes a CIFAR10_loader instance.

        Args:
            root (str): Root directory of the CIFAR-10 dataset.
            train (bool, optional): If True, loads the training data. If False, loads the test
            data. Defaults to True.
            transform (callable, optional): A transform to apply to the data. Defaults to None.
            N (int, optional): Maximum number of samples per class. Defaults to None.
        """
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.N = N
        self.data_update()

    def data_update(self):
        assert len(self.data) == len(self.targets)
        label_mapping = {0: 0, 2: 1, 8: 2, 7: 3, 1: 4}

        new_data = []
        new_targets = []
        class_counter = np.zeros(5)

        for item in range(len(self.data)):
            label = self.targets[item]
            if label in label_mapping:
                new_label_value = label_mapping[label]
                if self.N is None or class_counter[new_label_value] < self.N:
                    # Increment the class_counter and add the data and new label
                    class_counter[new_label_value] += 1
                    new_data.append(self.data[item])
                    new_targets.append(new_label_value)
            if self.N is not None and np.all(class_counter == self.N):
                break

        self.data = np.asarray(new_data)
        self.targets = np.asarray(new_targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]

        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
