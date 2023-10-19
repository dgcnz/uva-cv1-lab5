import pytest
import torch

from src.models.lenet5 import LeNet5


def test_lenet5_base():
    num_classes = 10
    model = LeNet5(num_classes)
    x = torch.randn(2, 3, 32, 32)
    try:
        y = model(x)
        assert y.shape == (2, num_classes)
    except Exception as e:
        pytest.fail(e)
