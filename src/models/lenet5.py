import torch.nn as nn

from src.models.vision_model import VisionModel
from collections import OrderedDict

class LeNet5(VisionModel):
    _name = "LeNet-5-Base"

    def __init__(self, num_classes: int):
        super().__init__()
        # in 32 x 32 * 3, original paper works with 32 x 32 x 3
        self._model_dict = OrderedDict([
                    ("c1", nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)),  # out 28 x 28 * 6
                    ("tanh1", nn.Tanh()),
                    ("s2", nn.AvgPool2d(kernel_size=2, stride=2)),  # out 14 x 14 * 6
                    ("c3", nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)),  # out 5 x 5 * 16
                    ("tanh2", nn.Tanh()),
                    ("s4", nn.AvgPool2d(kernel_size=2, stride=2)),  # out 5 x 5 * 16
                    ("flatten", nn.Flatten()),
                    ("f5", nn.Linear(5 * 5 * 16, 120)),  # out 1 x 1 * 120
                    ("tanh3", nn.Tanh()),
                    # ("c5",  nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)  # could also be fcn layer)
                    ("f6", nn.Linear(120, 84)),
                    ("tanh4", nn.Tanh()),
                    ("fc", nn.Linear(84, num_classes))
                ]
            )
        self._model = nn.Sequential(self._model_dict)
    
    def replace_fc(self, num_classes: int):
        self._model_dict["fc"] = nn.Linear(84, num_classes)
        self._model = nn.Sequential(self._model_dict)
        return self._model


class LeNet5BaseImproved(VisionModel):
    _name = "LeNet-5-BaseImproved"

    def __init__(self, num_classes: int):
        super().__init__()
        # in 32 x 32 * 3, original paper works with 32 x 32 x 3
        self._model_dict = OrderedDict([
                    ("c1", nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)),  # out 28 x 28 * 6
                    ("relu1", nn.ReLU()),
                    ("s2", nn.MaxPool2d(kernel_size=2, stride=2)),  # out 14 x 14 * 6
                    ("bn1", nn.BatchNorm2d(6)),
                    ("c3", nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)),  # out 5 x 5 * 16
                    ("relu2", nn.ReLU()),
                    ("s4", nn.MaxPool2d(kernel_size=2, stride=2)),  # out 5 x 5 * 16
                    ("flatten", nn.Flatten()),
                    ("f5", nn.Linear(5 * 5 * 16, 120)),  # out 1 x 1 * 120
                    ("relu3", nn.ReLU()),
                    # ("c5",  nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)  # could also be fcn layer)
                    ("f6", nn.Linear(120, 84)),
                    ("relu4", nn.ReLU()),
                    ("bn2", nn.BatchNorm1d(84)),
                    ("fc", nn.Linear(84, num_classes))
                ]
            )
        self._model = nn.Sequential(self._model_dict)
