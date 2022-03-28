from typing import Optional

import torch.nn as nn

from landscapes.models.base import ClassificationModel

from .base import MetaModel

__all__ = ["LinearProbe", "BitFit"]


class LinearProbe(MetaModel):
    def __init__(self, model: Optional[ClassificationModel]) -> None:
        assert model is not None
        for param in model.encoder.parameters():
            param.requires_grad_(False)
        super().__init__(model=model)


class BitFit(MetaModel):
    def __init__(self, model: Optional[nn.Module]) -> None:
        assert model is not None
        for name, param in model.named_parameters():
            if "bias" not in name:
                param.requires_grad_(False)
        super().__init__(model=model)
