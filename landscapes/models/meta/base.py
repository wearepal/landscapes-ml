from abc import ABCMeta

from ranzen.decorators import implements
from torch import Tensor
import torch.nn as nn

__all__ = [
    "MetaModel",
]


class MetaModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
