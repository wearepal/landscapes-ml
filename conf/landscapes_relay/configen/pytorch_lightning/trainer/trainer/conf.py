
from dataclasses import dataclass, field
from typing import Any
from typing import Optional


@dataclass
class TrainerConf:
    _target_: str = "pytorch_lightning.trainer.trainer.Trainer"
    logger: Any = True  # Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]
    checkpoint_callback: Optional[bool] = None
    enable_checkpointing: bool = True
    callbacks: Any = None  # Union[List[Callback], Callback, NoneType]
    default_root_dir: Optional[str] = None
    gradient_clip_val: Any = None  # Union[int, float, NoneType]
    gradient_clip_algorithm: Optional[str] = None
    process_position: int = 0
    num_nodes: int = 1
    num_processes: int = 1
    devices: Any = None  # Union[int, str, List[int], NoneType]
    gpus: Any = None  # Union[int, str, List[int], NoneType]
    auto_select_gpus: bool = False
    tpu_cores: Any = None  # Union[int, str, List[int], NoneType]
    ipus: Optional[int] = None
    log_gpu_memory: Optional[str] = None
    progress_bar_refresh_rate: Optional[int] = None
    enable_progress_bar: bool = True
    overfit_batches: Any = 0.0  # Union[int, float]
    track_grad_norm: Any = -1  # Union[int, float, str]
    check_val_every_n_epoch: int = 1
    fast_dev_run: Any = False  # Union[int, bool]
    accumulate_grad_batches: Any = None  # Union[int, Dict[int, int], NoneType]
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: int = -1
    min_steps: Optional[int] = None
    max_time: Any = None  # Union[str, timedelta, Dict[str, int], NoneType]
    limit_train_batches: Any = 1.0  # Union[int, float]
    limit_val_batches: Any = 1.0  # Union[int, float]
    limit_test_batches: Any = 1.0  # Union[int, float]
    limit_predict_batches: Any = 1.0  # Union[int, float]
    val_check_interval: Any = 1.0  # Union[int, float]
    flush_logs_every_n_steps: Optional[int] = None
    log_every_n_steps: int = 50
    accelerator: Any = None  # Union[str, Accelerator, NoneType]
    strategy: Any = None  # Union[str, TrainingTypePlugin, NoneType]
    sync_batchnorm: bool = False
    precision: Any = 32  # Union[int, str]
    enable_model_summary: bool = True
    weights_summary: Optional[str] = "top"
    weights_save_path: Optional[str] = None
    num_sanity_val_steps: int = 2
    resume_from_checkpoint: Any = None  # Union[str, Path, NoneType]
    profiler: Any = None  # Union[BaseProfiler, str, NoneType]
    benchmark: bool = False
    deterministic: bool = False
    reload_dataloaders_every_n_epochs: int = 0
    reload_dataloaders_every_epoch: bool = False
    auto_lr_find: Any = False  # Union[bool, str]
    replace_sampler_ddp: bool = True
    detect_anomaly: bool = False
    auto_scale_batch_size: Any = False  # Union[str, bool]
    prepare_data_per_node: Optional[bool] = None
    plugins: Any = None  # Union[TrainingTypePlugin, PrecisionPlugin, ClusterEnvironment, CheckpointIO, str, List[Union[TrainingTypePlugin, PrecisionPlugin, ClusterEnvironment, CheckpointIO, str]], NoneType]
    amp_backend: str = "native"
    amp_level: Optional[str] = None
    move_metrics_to_cpu: bool = False
    multiple_trainloader_mode: str = "max_size_cycle"
    stochastic_weight_avg: bool = False
    terminate_on_nan: Optional[bool] = None
