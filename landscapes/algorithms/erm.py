from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

from conduit.data.structures import BinarySample, NamedSample
from conduit.metrics import hard_prediction
from conduit.models.utils import prefix_keys
from conduit.types import LRScheduler, Stage
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from ranzen.decorators import implements
from ranzen.torch import CrossEntropyLoss
from ranzen.torch.data import TrainingMode
from ranzen.torch.loss import ReductionType
from ranzen.torch.optimizers import SAM
import torch
from torch import Tensor, optim
import torch.nn as nn
from torchmetrics import Metric

from landscapes.algorithms.base import Algorithm
from landscapes.transforms import BatchTransform

__all__ = ["ERM"]


class ERM(Algorithm):
    def __init__(
        self,
        model: nn.Module,
        *,
        metrics: Dict[str, Metric],
        lr: float = 5.0e-4,
        label_smoothing: float = 0.1,
        batch_transforms: Optional[List[BatchTransform]] = None,
        optimizer_cls: str = "torch.optim.AdamW",
        optimizer_kwargs: Optional[DictConfig] = None,
        use_sam: bool = False,
        sam_rho: float = 0.05,
        scheduler_cls: Optional[str] = None,
        scheduler_kwargs: Optional[DictConfig] = None,
        lr_sched_interval: TrainingMode = TrainingMode.epoch,
        lr_sched_freq: int = 1,
    ) -> None:
        super().__init__(model=model, lr=lr, metrics=metrics, batch_transforms=batch_transforms)
        self.loss_fn = CrossEntropyLoss(
            reduction=ReductionType.mean, label_smoothing=label_smoothing
        )
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs
        self.lr_sched_interval = lr_sched_interval
        self.lr_sched_freq = lr_sched_freq

        self.use_sam = use_sam
        self.sam_rho = sam_rho

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> Union[
        Tuple[
            Union[List[optim.Optimizer], optim.Optimizer],
            List[Mapping[str, Union[LRScheduler, int, TrainingMode]]],
        ],
        Union[List[optim.Optimizer], optim.Optimizer],
    ]:
        optimizer_config = DictConfig({"_target_": self.optimizer_cls})
        if self.optimizer_kwargs is not None:
            optimizer_config.update(self.optimizer_kwargs)

        base_optimizer = instantiate(optimizer_config, params=self.model.parameters(), lr=self.lr)
        if self.use_sam:
            optimizer = SAM(base_optimizer, rho=self.sam_rho)
        optimizer = SAM(base_optimizer, rho=self.sam_rho) if self.use_sam else base_optimizer

        if self.scheduler_cls is not None:
            scheduler_config = DictConfig({"_target_": self.scheduler_cls})
            if self.scheduler_kwargs is not None:
                scheduler_config.update(self.scheduler_kwargs)
            scheduler = instantiate(scheduler_config, optimizer=base_optimizer)
            scheduler_config = {
                "scheduler": scheduler,
                "interval": self.lr_sched_interval.name,
                "frequency": self.lr_sched_freq,
            }
            return [optimizer], [scheduler_config]
        return optimizer

    @implements(Algorithm)
    def training_step(self, batch: BinarySample[Tensor], batch_idx: int) -> STEP_OUTPUT:
        logits = self.model(batch.x)
        loss = self.loss_fn(input=logits, target=batch.y)

        results_dict = {
            "batch_loss": loss.item(),
        }
        results_dict = prefix_keys(dict_=results_dict, prefix=str(Stage.fit), sep="/")
        self.log_dict(results_dict)

        return {"loss": loss}

    @implements(Algorithm)
    @torch.no_grad()
    def inference_step(self, batch: BinarySample[Tensor], stage: Stage) -> STEP_OUTPUT:
        logits = self.forward(batch.x)
        return {"logits": logits.cpu(), "targets": batch.y.cpu()}

    @implements(pl.LightningModule)
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: optim.Optimizer,
        optimizer_idx: int,
        optimizer_closure: Optional[Callable],
        on_tpu: bool,
        using_native_amp: bool,
        using_lbfgs: bool,
    ) -> None:
        if (optimizer_closure is not None) and isinstance(optimizer, SAM):
            optimizer_closure()

            def _closure() -> Tensor:
                return optimizer_closure._step_fn(None).closure_loss

            optimizer.step(_closure)
        else:
            optimizer.step(optimizer_closure)

    @implements(Algorithm)
    def predict_step(
        self, batch: NamedSample, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tensor:
        return hard_prediction(self.model(batch.x)).cpu()
