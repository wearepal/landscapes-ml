from abc import ABCMeta
from typing import Any, Optional, Protocol, TypeVar, runtime_checkable

from landascapes.models.base import Model
from ranzen.decorators import implements
from torch import Tensor
import torch.nn as nn

__all__ = [
    "MMFactory",
    "MetaModel",
]


M_co = TypeVar("M_co", bound="MetaModel", covariant=True)


@runtime_checkable
class MMFactory(Protocol[M_co]):
    def __call__(self, model: Model, *args: Any, **kwargs: Any) -> M_co:
        ...


class MetaModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, model: Optional[Model]) -> None:
        super().__init__()
        assert model is not None
        self.model = model

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
