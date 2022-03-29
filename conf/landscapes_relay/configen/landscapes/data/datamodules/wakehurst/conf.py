
from dataclasses import dataclass, field
from landscapes.data.datasets.wakehurst import ImageryType
from omegaconf import MISSING
from ranzen.torch.data import TrainingMode
from typing import Any
from typing import Optional


@dataclass
class WakehurstDataModuleConf:
    _target_: str = "landscapes.data.datamodules.wakehurst.WakehurstDataModule"
    train_batch_size: int = 64
    eval_batch_size: Optional[int] = None
    val_prop: float = 0.2
    test_prop: float = 0.2
    num_workers: int = 0
    seed: int = 47
    persist_workers: bool = False
    pin_memory: bool = True
    stratified_sampling: bool = False
    instance_weighting: bool = False
    training_mode: TrainingMode = TrainingMode.epoch
    root: Any = MISSING  # Union[str, Path]
    train_transforms: Any = None  # Union[Compose, BasicTransform, Callable[[Image], Any], NoneType]
    test_transforms: Any = None  # Union[Compose, BasicTransform, Callable[[Image], Any], NoneType]
    imagery: ImageryType = ImageryType.AERIAL
