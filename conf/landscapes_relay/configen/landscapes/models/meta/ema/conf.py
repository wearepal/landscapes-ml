
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Any


@dataclass
class EmaModelConf:
    _target_: str = "landscapes.models.meta.ema.EmaModel"
    model: Any = MISSING  # Module
    decay: float = MISSING
    update_frequency: int = 1
