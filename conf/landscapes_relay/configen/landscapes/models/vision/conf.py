
from dataclasses import dataclass, field
from classy_vision.models.anynet import ActivationType
from classy_vision.models.anynet import BlockType
from classy_vision.models.anynet import StemType
from landscapes.models.vision import CLIPVersion
from landscapes.models.vision import ConvNeXtVersion
from landscapes.models.vision import ResNetVersion
from omegaconf import MISSING
from typing import Optional


@dataclass
class ResNetConf:
    _target_: str = "landscapes.models.vision.ResNet"
    target_dim: Optional[int] = MISSING
    features_only: bool = False
    pretrained: bool = False
    version: ResNetVersion = ResNetVersion.RN18


@dataclass
class CLIPConf:
    _target_: str = "landscapes.models.vision.CLIP"
    target_dim: Optional[int] = MISSING
    features_only: bool = False
    version: CLIPVersion = CLIPVersion.ViT_B32
    download_root: Optional[str] = None


@dataclass
class RegNetConf:
    _target_: str = "landscapes.models.vision.RegNet"
    target_dim: Optional[int] = MISSING
    depth: int = MISSING
    w_0: int = MISSING
    w_a: float = MISSING
    w_m: float = MISSING
    group_width: int = MISSING
    bottleneck_multiplier: float = 1.0
    stem_type: StemType = StemType.SIMPLE_STEM_IN
    stem_width: int = 32
    block_type: BlockType = BlockType.RES_BOTTLENECK_BLOCK
    activation: ActivationType = ActivationType.RELU
    use_se: bool = True
    se_ratio: float = 0.25
    bn_epsilon: float = 1e-05
    bn_momentum: float = 0.1
    features_only: bool = False
    checkpoint: Optional[str] = None


@dataclass
class ConvNeXtConf:
    _target_: str = "landscapes.models.vision.ConvNeXt"
    target_dim: Optional[int] = MISSING
    features_only: bool = False
    pretrained: bool = False
    version: ConvNeXtVersion = ConvNeXtVersion.BASE
