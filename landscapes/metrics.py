import torch
from torch import Tensor

__all__ = ["accuracy"]


@torch.no_grad()
def accuracy(logits: Tensor, targets: Tensor) -> Tensor:
    logits = torch.atleast_2d(logits.squeeze())
    targets = torch.atleast_1d(targets.squeeze()).long()
    if len(logits) != len(targets):
        raise ValueError("'logits' and 'targets' must match in size at dimension 0.")
    preds = (logits > 0).long() if logits.ndim == 1 else logits.argmax(dim=1)
    return (preds == targets).float().mean()
