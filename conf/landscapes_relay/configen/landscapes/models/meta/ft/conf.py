
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Any


@dataclass
class LinearProbeConf:
    _target_: str = "landscapes.models.meta.ft.LinearProbe"
    model: Any = MISSING  # ClassificationModel


@dataclass
class BitFitConf:
    _target_: str = "landscapes.models.meta.ft.BitFit"
    model: Any = MISSING  # Model
