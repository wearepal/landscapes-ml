from __future__ import annotations
from typing import Any

from ranzen.hydra import Option

from landscapes.algorithms import ERM
from landscapes.data.datamodules import WakehurstDataModule
from landscapes.models.meta import BitFit, LinearProbe
from landscapes.models.vision import CLIP, ConvNeXt, RegNet, ResNet
from landscapes.relay import LandscapesRelay

if __name__ == "__main__":
    dm_ops: list[type[Any] | Option] = [
        Option(WakehurstDataModule, name="wakehurst"),
    ]
    alg_ops: list[type[Any] | Option] = [Option(ERM, "erm")]
    model_ops: list[type[Any] | Option] = [
        Option(ResNet, "resnet"),
        Option(CLIP, "clip"),
        Option(RegNet, "regnet"),
        Option(ConvNeXt, "convnext"),
    ]
    mm_ops: list[type[Any] | Option] = [Option(LinearProbe, "lp"), Option(BitFit, "bitfit")]

    LandscapesRelay.with_hydra(
        root="conf",
        dm=dm_ops,
        alg=alg_ops,
        model=model_ops,
        meta_model=mm_ops,
        clear_cache=True,
    )
