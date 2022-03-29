from abc import ABCMeta, abstractmethod
import logging
from typing import Optional, Tuple

from conduit.logging import init_logger
from ranzen.decorators import implements
from torch import Tensor
import torch.nn as nn

__all__ = [
    "ClassificationModel",
    "Model",
]


class Model(nn.Module, metaclass=ABCMeta):
    _logger: Optional[logging.Logger] = None

    def __init__(self, target_dim: Optional[int]) -> None:
        assert target_dim is not None
        super().__init__()
        self.model, self.out_dim = self.build(target_dim)
        self.target_dim = target_dim

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = init_logger(self.__class__.__name__)
        return self._logger

    @abstractmethod
    def build(self, target_dim: int) -> Tuple[nn.Module, int]:
        ...

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ClassificationModel(Model, metaclass=ABCMeta):
    def __init__(self, target_dim: Optional[int], *, features_only: bool = False) -> None:
        super().__init__(target_dim=target_dim)
        self.features_only = features_only
        self.feature_dim = self.out_dim

        if self.features_only:
            self.classifier = nn.Identity()
        else:
            self.classifier = self.build_classifier(
                in_dim=self.feature_dim, num_classes=self.target_dim
            )
        self.encoder = self.model
        self.model = nn.Sequential(self.encoder, self.classifier)

    def build_classifier(self, in_dim: int, *, num_classes: int) -> nn.Module:
        return nn.Sequential(nn.Flatten(), nn.Linear(in_features=in_dim, out_features=num_classes))
