import torch
from torch import Tensor
import torch.nn as nn

__all__ = ["ExponentialMovingAverage"]


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model: nn.Module, *, decay: float) -> None:
        self.decay = decay
        super().__init__(model, avg_fn=self._ema_update)

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
