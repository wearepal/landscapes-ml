"""NICO data-module."""
from typing import Any, List, Optional

import attr
from conduit.data.datamodules.base import CdtDataModule
from conduit.data.datamodules.vision.base import CdtVisionDataModule
from conduit.data.datasets.utils import CdtDataLoader, PillowTform, stratified_split
from conduit.data.structures import NamedSample, TrainValTestSplit
from conduit.types import Stage
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from ranzen import implements
import torchvision.transforms as T

from landscapes.data.datasets.wakehurst import Wakehurst
from landscapes.transforms import rgba_to_rgb

__all__ = ["WakehurstDataModule"]


@attr.define(kw_only=True)
class WakehurstDataModule(CdtVisionDataModule[Wakehurst, Wakehurst.SampleType]):
    """PyTorch Lighgtning DataModule for the Wakehurst dataset."""

    imagery: Wakehurst.ImageryType = Wakehurst.ImageryType.AERIAL
    predict_data: Wakehurst = attr.field(init=False)

    @property  # type: ignore[misc]
    @implements(CdtVisionDataModule)
    def _default_train_transforms(self) -> T.Compose:
        transforms_ls: List[PillowTform] = [
            rgba_to_rgb,
            T.TrivialAugmentWide(),
            T.ToTensor(),
            T.RandomErasing(p=0.1, value=0),
        ]
        if self.norm_values is not None:
            transforms_ls.append(T.Normalize(mean=self.norm_values.mean, std=self.norm_values.std))

        return T.Compose(transforms_ls)

    @property  # type: ignore[misc]
    @implements(CdtVisionDataModule)
    def _default_test_transforms(self) -> T.Compose:
        transforms_ls: List[PillowTform] = [
            rgba_to_rgb,
            T.ToTensor(),
        ]
        if self.norm_values is not None:
            transforms_ls.append(T.Normalize(mean=self.norm_values.mean, std=self.norm_values.std))

        return T.Compose(transforms_ls)

    @implements(LightningDataModule)
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        Wakehurst(
            root=self.root,
            imagery=self.imagery,
            split=Wakehurst.Split.TEST,
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
        val_data, train_data = stratified_split(
            train_val_data,
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
            split=Wakehurst.Split.TEST,
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

    def __str__(self) -> str:
        ds_name = self._train_data_base.__class__.__name__
        size_info = (
            f"- Number of training samples: {len(self.train_data)}\n"
            f"- Number of validation samples: {len(self.val_data)}\n"
            f"- Number of test samples: {len(self.test_data)}\n"
            f"- Number of samples to be predicted: {len(self.predict_data)}"
        )
        return f"\nDataModule for dataset of type '{ds_name}'\n{size_info}"
