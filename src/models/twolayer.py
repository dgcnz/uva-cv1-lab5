import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

from src.models.vision_model import VisionModel




class TwoLayerNet(VisionModel):
    _name = "TwoLayerNet"

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        """
        :param input_size: 3*32*32
        :param hidden_size
        :param num_classes
        """
        super().__init__()
        self._model_dict = OrderedDict([
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(input_size, hidden_size)),
            ("relu", nn.ReLU()),
            ("fc", nn.Linear(hidden_size, num_classes))
        ])
        self._model = nn.Sequential(self._model_dict)

