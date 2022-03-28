from abc import ABCMeta, abstractmethod
import logging
from typing import Optional, Tuple

from conduit.data import CdtDataset
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

    def __init__(self, dataset: Optional[CdtDataset]) -> None:
        super().__init__()
        assert dataset is not None
        self.model, self.out_dim = self.build(dataset)

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = init_logger(self.__class__.__name__)
        return self._logger

    @abstractmethod
    def build(self, dataset: CdtDataset) -> Tuple[nn.Module, int]:
        ...

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ClassificationModel(Model, metaclass=ABCMeta):
    def __init__(self, dataset: CdtDataset, features_only: bool = False) -> None:
        super().__init__(dataset=dataset)
        self.features_only = features_only
        self.feature_dim = self.out_dim
        self.out_dim = dataset.card_y
        if self.features_only:
            self.classifier = nn.Identity()
        else:
            self.classifier = self.build_classifier(
                in_dim=self.feature_dim, num_classes=self.out_dim
            )
        self.encoder = self.model
        self.model = nn.Sequential(self.encoder, self.classifier)

    def build_classifier(self, in_dim: int, *, num_classes: int) -> nn.Module:
        return nn.Sequential(nn.Flatten(), nn.Linear(in_features=in_dim, out_features=num_classes))
