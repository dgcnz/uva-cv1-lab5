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
                    # ("c5",  nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)  # could also be fcn layer)
                    ("f6", nn.Linear(120, 84)),
                    ("fc", nn.Linear(84, num_classes))
                ]
            )
        self._model = nn.Sequential(self._model_dict)
