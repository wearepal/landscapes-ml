
from dataclasses import dataclass, field
from omegaconf import MISSING
from ranzen.torch.data import TrainingMode
from typing import Any
from typing import Optional


@dataclass
class ERMConf:
    _target_: str = "landscapes.algorithms.erm.ERM"
    model: Any = MISSING  # Module
    metrics: Any = MISSING  # Dict[str, Metric]
    lr: float = 0.0005
    label_smoothing: float = 0.1
    batch_transforms: Any = None  # Optional[List[BatchTransform]]
    optimizer_cls: str = "torch.optim.AdamW"
    optimizer_kwargs: Any = None  # Optional[DictConfig]
    use_sam: bool = False
    sam_rho: float = 0.05
    scheduler_cls: Optional[str] = None
    scheduler_kwargs: Any = None  # Optional[DictConfig]
    lr_sched_interval: TrainingMode = TrainingMode.epoch
    lr_sched_freq: int = 1
