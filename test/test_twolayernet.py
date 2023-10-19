import pytest
import torch

from src.models.twolayer import TwoLayerNet


def test_lenet5_base():
    num_classes = 10
    B, C, H, W = 2, 3, 32, 32
    model = TwoLayerNet(input_size=C * H * W, hidden_size=100, num_classes=num_classes)
    x = torch.randn(B, C, H, W)
    try:
        y = model(x)
        assert y.shape == (B, num_classes)
    except Exception as e:
        pytest.fail(e)
