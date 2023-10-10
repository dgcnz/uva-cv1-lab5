import torch.nn as nn
import torchvision

from src.models._base import Model


class VisionModel(Model):
    def __init__(self):
        super().__init__()

        self._model = None

    def freeze(self):
        for param in self._model.parameters():
            param.requires_grad = False
        for param in self._model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self._model.parameters():
            param.requires_grad = True


class Resnet18(VisionModel):
    _name = "Resnet18"

    def __init__(self, num_classes):
        super().__init__()

        self._model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self._model.fc.in_features
        self._model.fc = nn.Linear(num_ftrs, num_classes)
