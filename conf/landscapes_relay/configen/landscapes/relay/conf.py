
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Any
from typing import Optional


@dataclass
class LandscapesRelayConf:
    _target_: str = "landscapes.relay.LandscapesRelay"
    dm: Any = MISSING  # DictConfig
    alg: Any = MISSING  # DictConfig
    model: Any = MISSING  # DictConfig
    meta_model: Any = None  # Optional[DictConfig]
    trainer: Any = MISSING  # DictConfig
    logger: Any = MISSING  # DictConfig
    seed: Optional[int] = 42
    arftifact_dir: str = "."
