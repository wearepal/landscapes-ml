from landscapes.models.base import ClassificationModel, Model

from .base import MetaModel

__all__ = ["LinearProbe", "BitFit"]


class LinearProbe(MetaModel):
    def __init__(self, model: ClassificationModel) -> None:
        for param in model.encoder.parameters():
            param.requires_grad_(False)
        super().__init__(model=model)


class BitFit(MetaModel):
    def __init__(self, model: Model) -> None:
        for name, param in model.named_parameters():
            if "bias" not in name:
                param.requires_grad_(False)
        super().__init__(model=model)
