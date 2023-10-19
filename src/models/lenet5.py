import torch.nn as nn
import torch.nn.functional as F

from src.models._base import Model


class LeNet5(Model):
    _name = "LeNet-5-Base"

    def __init__(self, num_classes: int):
        super().__init__()
        # in 32 x 32 * 3, original paper works with 32 x 32 x 3
        self.c1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        # in 28 x 28 * 6
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # in 14 x 14 * 6
        self.c3 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        # in 5 x 5 * 16
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        # in 5 x 5 * 16
        self.f5 = nn.Linear(5 * 5 * 16, 120)
        # self.c5 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)  # could also be fcn layer
        # in 1 x 1 * 120
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, num_classes)

    def forward(self, x):
        B, *_ = x.shape
        x = self.s2(F.tanh(self.c1(x)))
        x = self.s4(F.tanh(self.c3(x)))
        # x = F.tanh(self.c5(x))
        x = x.reshape(B, -1)
        x = F.tanh(self.f5(x))
        x = F.tanh(self.f6(x))
        x = self.f7(x)
        return x
