from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, cast

from conduit.data.structures import BinarySample, NamedSample
from conduit.models.utils import aggregate_over_epoch, prefix_keys
from conduit.types import MetricDict, Stage
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from ranzen import implements
import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric
from typing_extensions import TypeGuard

from landscapes.transforms import BatchTransform

__all__ = ["Algorithm"]


class Algorithm(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        metrics: Dict[str, Metric],
        lr: float = 5.0e-5,
        batch_transforms: Optional[List[BatchTransform]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.metrics = metrics
        self.batch_transforms = batch_transforms

    def _apply_batch_transforms(self, batch: BinarySample[Tensor]) -> None:
        if self.batch_transforms is not None:
            for tform in self.batch_transforms:
                transformed_x, transformed_y = tform(inputs=batch.x, targets=batch.y)
                batch.x = transformed_x
                batch.y = transformed_y

    @implements(pl.LightningModule)
    def on_after_batch_transfer(self, batch: BinarySample[Tensor], dataloader_idx: Optional[int]):
        if self.training:
            self._apply_batch_transforms(batch)
        return batch

    @abstractmethod
    @implements(pl.LightningModule)
    def training_step(self, batch: Dict[str, BinarySample], batch_idx: int) -> STEP_OUTPUT:
        ...

    @abstractmethod
    def inference_step(self, batch: BinarySample, stage: Stage) -> STEP_OUTPUT:
        ...

    def aggregate_and_evaluate(self, outputs: EPOCH_OUTPUT) -> MetricDict:
        logits_all = aggregate_over_epoch(outputs=outputs, metric="logits")
        targets_all = aggregate_over_epoch(outputs=outputs, metric="targets")
        metrics = {
            key: metric(logits_all.detach().cpu(), targets_all.cpu())
            for key, metric in self.metrics.items()
        }
        return metrics

    def _is_epoch_output(self, output: List[Any]) -> TypeGuard[EPOCH_OUTPUT]:
        return isinstance(output[0], (Tensor, Dict))

    @torch.no_grad()
    def _epoch_end(self, outputs: Union[List[EPOCH_OUTPUT], EPOCH_OUTPUT]) -> MetricDict:
        # check whether outputs contains the results from multiple data-loaders
        if self._is_epoch_output(outputs):
            return self.aggregate_and_evaluate(outputs)
        # perform evaluation for multiple data-loaders
        outputs = cast(List[EPOCH_OUTPUT], outputs)
        results_dict: MetricDict = {}
        for dataloader_idx, outputs_i in enumerate(outputs):
            results_dict |= prefix_keys(
                self.aggregate_and_evaluate(outputs_i), prefix=str(dataloader_idx), sep="/"
            )
        return results_dict

    @implements(pl.LightningModule)
    @torch.no_grad()
    def validation_step(
        self, batch: BinarySample, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> STEP_OUTPUT:
        return self.inference_step(batch=batch, stage=Stage.validate)

    @implements(pl.LightningModule)
    @torch.no_grad()
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        results_dict = self._epoch_end(outputs=outputs)
        results_dict = prefix_keys(dict_=results_dict, prefix=str(Stage.validate), sep="/")
        self.log_dict(results_dict)

    @implements(pl.LightningModule)
    @torch.no_grad()
    def test_step(
        self, batch: BinarySample, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> STEP_OUTPUT:
        return self.inference_step(batch=batch, stage=Stage.test)

    @implements(pl.LightningModule)
    @torch.no_grad()
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        results_dict = self._epoch_end(outputs=outputs)
        results_dict = prefix_keys(dict_=results_dict, prefix=str(Stage.test), sep="/")
        self.log_dict(results_dict)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @implements(pl.LightningModule)
    @abstractmethod
    def predict_step(
        self, batch: NamedSample[Tensor], batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tensor:
        ...
