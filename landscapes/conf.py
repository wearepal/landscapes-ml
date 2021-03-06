from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass
class WandbLoggerConf:
    _target_: str = "pytorch_lightning.loggers.wandb.WandbLogger"
    name: Optional[str] = None
    save_dir: Optional[str] = None
    offline: Optional[bool] = False
    id: Optional[str] = None
    anonymous: Optional[bool] = None
    version: Optional[str] = None
    project: Optional[str] = None
    log_model: Optional[bool] = False
    experiment: Any = None
    prefix: Optional[str] = None
    group: Optional[str] = None
    entity: Optional[str] = None
    tags: Optional[Sequence] = (None,)
    reinit: bool = False
