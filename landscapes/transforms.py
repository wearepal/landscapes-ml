import math
from typing import Tuple

from PIL import Image
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF  # type: ignore
from typing_extensions import Protocol

__class__ = [
    "BatchTransform",
    "RandomCutmix",
    "rgba_to_rgb",
]


def rgba_to_rgb(image: Image.Image) -> Image.Image:
    """Conver an image in RGBA format to RGB."""
    return image.convert("RGB")


class BatchTransform(Protocol):
    def __call__(self, inputs: Tensor, *, targets: Tensor) -> Tuple[Tensor, Tensor]:
        ...


class RandomCutmix:
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    :param num_classes: number of classes used for one-hot encoding.
    :param p: probability of the batch being transformed. Default value is 0.5.
    :param alpha: hyperparameter of the Beta distribution used for cutmix.
    :param inplace: boolean to make this transform inplace. Default set to False.
    """

    def __init__(
        self, num_classes: int, *, p: float = 0.5, alpha: float = 1.0, inplace: bool = False
    ) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def __call__(self, inputs: Tensor, *, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param batch: Float tensor of size (B, C, H, W)
        :param target: Integer tensor of size (B, )
        :returns: Randomly transformed batch.
        """
        if inputs.ndim != 4:
            raise ValueError("Batch ndim should be 4. Got {}".format(inputs.ndim))
        elif targets.ndim != 1:
            raise ValueError("Target ndim should be 1. Got {}".format(targets.ndim))
        elif not inputs.is_floating_point():
            raise TypeError("Batch dtype should be a float tensor. Got {}.".format(inputs.dtype))
        elif targets.dtype != torch.int64:
            raise TypeError("Target dtype should be torch.int64. Got {}".format(targets.dtype))

        if not self.inplace:
            inputs = inputs.clone()
            targets = targets.clone()

        if targets.ndim == 1:
            targets = F.one_hot(targets, num_classes=self.num_classes).to(dtype=inputs.dtype)

        if torch.rand(1).item() >= self.p:
            return inputs, targets

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = inputs.roll(1, 0)
        target_rolled = targets.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = TF.get_image_size(inputs)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        inputs[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        targets.mul_(lambda_param).add_(target_rolled)

        return inputs, targets

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_classes={num_classes}"
        s += ", p={p}"
        s += ", alpha={alpha}"
        s += ", inplace={inplace}"
        s += ")"
        return s.format(**self.__dict__)
