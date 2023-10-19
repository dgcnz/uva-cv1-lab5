import torch.nn as nn
import torch.nn.functional as F

from src.models._base import Model


class TwoLayerNet(Model):
    _name = "TwoLayerNet"

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        """
        :param input_size: 3*32*32
        :param hidden_size
        :param num_classes
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        scores = self.fc2(F.relu(self.fc1(x)))
        return scores
