import torch
import torch.nn as nn
from torch.nn import functional as F


class Model(nn.Module):
    _name = "DefaultModel"

    def __init__(self):
        super().__init__()

        self._model = None

    def forward(self, x, targets=torch.tensor([])):
        x = self._model(x)

        if targets.size() == torch.Size([0]):
            return x, None
        else:
            loss = F.cross_entropy(x, targets)
            return x, loss

    @property
    def name(self):
        return self._name

    @property
    def learnable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
