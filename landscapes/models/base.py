from abc import ABCMeta, abstractmethod
import logging
from typing import Any, Optional, Tuple, TypeVar, runtime_checkable

from conduit.logging import init_logger
from ranzen.decorators import implements
from torch import Tensor
import torch.nn as nn
from typing_extensions import Protocol
from wilds.datasets.wilds_dataset import WILDSDataset

__all__ = [
    "ModelFactory",
    "Model",
    "ClassificationModel",
]


M_co = TypeVar("M_co", bound="Model", covariant=True)


@runtime_checkable
class ModelFactory(Protocol[M_co]):
    def __call__(self, dataset: WILDSDataset, *args: Any, **kwargs: Any) -> M_co:
        ...


class Model(nn.Module, metaclass=ABCMeta):
    _logger: Optional[logging.Logger] = None

    def __init__(self, dataset: Optional[WILDSDataset]) -> None:
        super().__init__()
        assert dataset is not None
        self.model, self.out_dim = self.build(dataset)

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = init_logger(self.__class__.__name__)
        return self._logger

    @abstractmethod
    def build(self, dataset: WILDSDataset) -> Tuple[nn.Module, int]:
        ...

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ClassificationModel(Model, metaclass=ABCMeta):
    def __init__(self, dataset: Optional[WILDSDataset], features_only: bool = False) -> None:
        assert dataset is not None
        if dataset.n_classes is None:
            raise AttributeError(
                f"Dataset must have a non-null 'n_classes' attribute to instantiate "
                f"a classification model."
            )
        super().__init__(dataset=dataset)
        self.features_only = features_only
        self.feature_dim = self.out_dim
        self.out_dim = dataset.n_classes
        if self.features_only:
            self.classifier = nn.Identity()
        else:
            self.classifier = self.build_classifier(in_dim=self.feature_dim, n_classes=self.out_dim)
        self.encoder = self.model
        self.model = nn.Sequential(self.encoder, self.classifier)

    def build_classifier(self, in_dim: int, n_classes: int) -> nn.Module:
        return nn.Sequential(nn.Flatten(), nn.Linear(in_features=in_dim, out_features=n_classes))
