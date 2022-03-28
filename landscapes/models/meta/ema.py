import torch
from torch import Tensor
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel

from landscapes.models.meta.base import MetaModel

__all__ = ["EmaModel"]


class EmaModel(MetaModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model: nn.Module, *, decay: float, update_frequency: int = 1) -> None:
        self.decay = decay
        self.update_frequency = update_frequency
        self._training_iteration = 0
        self.ema_model = AveragedModel(model, avg_fn=self._ema_update)
        super().__init__(model)

    @torch.no_grad()
    def _ema_update(
        self,
        avg_model_param: Tensor,
        model_param: Tensor,
        num_averaged: int,
    ) -> Tensor:
        """
        Perform an EMA update of the model's parameters.
        """
        return self.decay * avg_model_param + (1 - self.decay * model_param)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            if (self._training_iteration % self.update_frequency) == 0:
                with torch.no_grad():
                    self.ema_model.update_parameters(self.model)
            return self.model(x)
        return self.ema_model(x)
