from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional

import attr
from fairscale.nn import auto_wrap  # type: ignore
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
import pandas as pd  # type: ignore
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from ranzen.decorators import implements
from ranzen.hydra import Option, Relay
import torch
import torch.nn as nn
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.calibration_error import CalibrationError
from torchmetrics.classification.f_beta import F1Score

from landscapes.algorithms.base import Algorithm
from landscapes.conf import WandbLoggerConf
from landscapes.data.datamodules.wakehurst import WakehurstDataModule

__all__ = ["LandscapesRelay"]


@attr.define(kw_only=True)
class LandscapesRelay(Relay):
    dm: DictConfig
    alg: DictConfig
    model: DictConfig
    meta_model: Optional[DictConfig] = None
    trainer: DictConfig
    logger: DictConfig
    seed: Optional[int] = 42
    save_dir: str = "results"

    @classmethod
    @implements(Relay)
    def with_hydra(
        cls,
        root: Path | str,
        *,
        dm: list[type[Any] | Option],
        alg: list[type[Any] | Option],
        model: list[type[Any] | Option],
        meta_model: list[type[Any] | Option],
        clear_cache: bool = False,
    ) -> None:

        configs = dict(
            dm=dm,
            alg=alg,
            model=model,
            meta_model=meta_model,
            trainer=[Option(class_=pl.Trainer, name="trainer")],
            logger=[Option(class_=WandbLoggerConf, name="logger")],
        )
        super().with_hydra(
            root=root,
            instantiate_recursively=False,
            clear_cache=clear_cache,
            **configs,
        )

    @implements(Relay)
    def run(self, raw_config: Dict[str, Any] | None = None) -> None:
        self.log(f"Current working directory: '{os.getcwd()}'")
        pl.seed_everything(self.seed)

        trainer: pl.Trainer = instantiate(self.trainer)

        dm: WakehurstDataModule = instantiate(self.dm)
        dm.prepare_data()
        dm.setup()
        model = instantiate(self.model, target_dim=dm.card_y)

        if self.meta_model is not None:
            model: nn.Module = instantiate(self.meta_model, model=model)
        # enable parameter sharding with fairscale.
        # Note: when fully-sharded training is not enabled this is a no-op
        model = auto_wrap(model)

        metrics = {
            "F1": F1Score(average="weighted", num_classes=dm.card_y, compute_on_step=True),
            "Calibration": CalibrationError(norm="l1", compute_on_step=True),
            "Aggregate Accuracy": Accuracy(average="micro", compute_on_step=True),
            "Balanced Accuracy": Accuracy(
                average="weighted", num_classes=dm.card_y, compute_on_step=True
            ),
        }
        alg: Algorithm = instantiate(self.alg, model=model, metrics=metrics)

        if self.logger.get("group", None) is None:
            default_group = f"{dm.__class__.__name__.removesuffix('DataModule')}"
            default_group += "_".join(obj.__class__.__name__ for obj in (model, alg))
            self.logger["group"] = default_group
        logger: WandbLogger = instantiate(self.logger, reinit=True)
        if raw_config is not None:
            logger.log_hyperparams(raw_config)  # type: ignore
        trainer.logger = logger
        # Runs routines to tune hyperparameters before training.
        trainer.tune(
            model=alg,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )
        # Train the model
        trainer.fit(
            model=alg,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )
        # Test the model
        trainer.test(
            model=alg,
            dataloaders=dm.test_dataloader(),
        )
        # Generate predictions for the unlabeled data
        predictions_ls = trainer.predict(
            model=alg,
            dataloaders=dm.predict_dataloader(),
        )
        assert predictions_ls is not None
        predictions_np = torch.cat(predictions_ls, dim=0).cpu().numpy()
        filenames = dm.predict_data.x[: len(predictions_np)]
        predictions_df = pd.DataFrame(
            np.stack([filenames, predictions_np], axis=-1), columns=["filename", "prediction"]
        )

        save_dir = Path(self.save_dir).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        predictions_save_path = save_dir / "predictions.csv"
        predictions_df.to_csv(predictions_save_path)
        self.log(f"Predictions saved to '{predictions_save_path.resolve()}'")

        model_save_path = save_dir / "model.pt"
        save_dict = {
            "config": {"model": dict(self.model)},
            "state": model.state_dict(),
        }
        if self.meta_model is not None:
            save_dict["config"]["meta_model"] = dict(self.meta_model)

        torch.save(save_dict, model_save_path)
        self.log(f"Model config and state saved to '{predictions_save_path.resolve()}'")
