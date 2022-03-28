"""NICO data-module."""
from typing import Any, Optional

import attr
from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.utils import CdtDataLoader, stratified_split
from conduit.data.structures import NamedSample, TrainValTestSplit
from conduit.types import Stage
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from ranzen import implements
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from landscapes.data.datasets.wakehurst import Wakehurst

__all__ = ["WakehurstDataModule"]


@attr.define(kw_only=True)
class WakehurstDataModule(CdtVisionDataModule[Wakehurst, Wakehurst.SampleType]):
    """Data-module for the NICO dataset."""

    image_size: int = 256
    imagery: Wakehurst.ImageryType = Wakehurst.ImageryType.AERIAL
    predict_data: Wakehurst = attr.field(init=False)

    @property  # type: ignore[misc]
    @implements(CdtVisionDataModule)
    def _default_train_transforms(self) -> T.Compose:
        base_transforms = T.Compose(
            [
                T.Resize(self.image_size, interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(self.image_size),
                T.TrivialAugmentWide,
                T.RandomErasing(),
            ]
        )
        normalization = super()._default_train_transforms
        return T.Compose([base_transforms, normalization])

    @property  # type: ignore[misc]
    @implements(CdtVisionDataModule)
    def _default_test_transforms(self) -> T.Compose:
        return self._default_train_transforms

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        Wakehurst(
            root=self.root,
            imagery=self.imagery,
            split=Wakehurst.Split.TRAIN,
        )

    @implements(CdtDataModule)
    def _get_splits(self) -> TrainValTestSplit[Wakehurst]:
        all_data = Wakehurst(
            root=self.root,
            imagery=self.imagery,
            transform=None,
            split=Wakehurst.Split.TRAIN,
        )
        train_val_prop = 1 - self.test_prop
        train_val_data, test_data = stratified_split(
            all_data,
            default_train_prop=train_val_prop,
            seed=self.seed,
        )
        train_data, val_data = stratified_split(
            all_data,
            default_train_prop=self.val_prop / train_val_prop,
            seed=self.seed,
        )

        return TrainValTestSplit(train=train_data, val=val_data, test=test_data)

    @implements(CdtVisionDataModule)
    def _setup(self, stage: Optional[Stage] = None) -> None:
        super()._setup(stage)
        self.predict_data = Wakehurst(
            root=self.root,
            imagery=self.imagery,
            transform=self.test_transforms,
            split=Wakehurst.Split.TRAIN,
        )

    @implements(pl.LightningDataModule)
    def predict_dataloader(self) -> CdtDataLoader[NamedSample]:
        return CdtDataLoader(
            self.predict_data,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persist_workers,
        )
