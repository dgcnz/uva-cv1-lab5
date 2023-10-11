import os

from PIL import Image
from torch.utils.data import Dataset


class STL10_Dataset(Dataset):
    """
    Dataset class for STL10 dataset. This dataset
    only loads 5 classes `{1: 'car', 2:'deer', 3:'horse', 4:'monkey', 5:'truck'}`.
    """

    DATA_MAP = {"8": "monkey", "10": "truck", "7": "horse", "5": "deer", "3": "car"}
    LABELS_MAP = {"3": "0", "5": "1", "7": "2", "8": "3", "10": "4"}

    def __init__(self, root="./data", train=True, transform=None):
        split = "train" if train else "test"
        root_data_dir = f"{root}/stl10/{split}/"

        self._data = []
        for label_dir in [root_data_dir + label for label in self.DATA_MAP.keys()]:
            self._data.extend(f"{label_dir}/{file}" for file in os.listdir(label_dir))

        self.transform = transform

    def __len__(self):
        return len(self._data)

    @property
    def classes(self):
        return ["car", "deer", "horse", "monkey", "truck"]

    def __getitem__(self, item):
        img_path = self._data[item]

        image = Image.open(img_path)
        label = self.LABELS_MAP[img_path.split("/")[-2]]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)
