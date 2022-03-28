from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Optional, OrderedDict, Tuple, cast

from classy_vision.models import RegNet as ClassyRegNet
from classy_vision.models.anynet import ActivationType, BlockType, StemType
from classy_vision.models.regnet import RegNetParams
from conduit.data.datasets.utils import PillowTform
from conduit.data.datasets.vision.base import CdtVisionDataset
from hydra.utils import to_absolute_path
from ranzen.decorators import implements, parsable
import torch
import torch.nn as nn
import torchvision.models as tvm
from typing_extensions import TypeAlias

from landscapes.models.base import ClassificationModel, Model

__all__ = [
    "CLIP",
    "ConvNeXt",
    "RegNet",
    "ResNet",
]


class ResNetVersion(Enum):
    RN18 = "18"
    RN34 = "34"
    RN50 = "50"
    RN101 = "101"


class ResNet(ClassificationModel):
    Version: TypeAlias = ResNetVersion

    @parsable
    def __init__(
        self,
        dataset: CdtVisionDataset,
        *,
        features_only: bool = False,
        pretrained: bool = False,
        version: ResNetVersion = ResNetVersion.RN18,
    ) -> None:
        self.pretrained = pretrained
        self.version = version
        super().__init__(dataset=dataset, features_only=features_only)

    @implements(Model)
    def build(self, dataset: CdtVisionDataset) -> Tuple[nn.Module, int]:
        model: tvm.ResNet = getattr(tvm, f"resnet{self.version.value}")(pretrained=self.pretrained)
        out_dim = model.fc.in_features
        model.fc = nn.Identity()  # type: ignore
        return model, out_dim


class CLIPVersion(Enum):
    RN50 = "RN50"
    RN101 = "RN101"
    RN50x4 = "RN50x4"
    RN50x16 = "RN50x16"
    RN50x64 = "RN50x64"
    ViT_B32 = "ViT-B/32"
    ViT_B16 = "ViT-B/16"
    ViT_L14 = "ViT-L/14"


class CLIP(ClassificationModel):
    transforms: PillowTform
    Version: TypeAlias = CLIPVersion

    @parsable
    def __init__(
        self,
        dataset: CdtVisionDataset,
        *,
        features_only: bool = False,
        version: CLIPVersion = CLIPVersion.ViT_B32,
        download_root: Optional[str] = None,
    ) -> None:
        self.version = version
        self.download_root = download_root
        super().__init__(dataset=dataset, features_only=features_only)

    @implements(Model)
    def build(self, dataset: CdtVisionDataset) -> Tuple[nn.Module, int]:
        import clip  # type: ignore

        model, self.transforms = clip.load(
            name=self.version.value, device="cpu", download_root=self.download_root  # type: ignore
        )
        visual_model = model.visual
        out_dim = visual_model.output_dim
        return visual_model, out_dim


class RegNet(ClassificationModel):
    """
    Wrapper for ClassyVision RegNet model so we can map layers into feature
    blocks to facilitate feature extraction and benchmarking at several layers.
    This model is defined on the fly from a RegNet base class and a configuration file.
    We follow the feature naming convention defined in the ResNet vissl trunk.
    [ Adapted from VISSL ]
    """

    @parsable
    def __init__(
        self,
        dataset: CdtVisionDataset,
        *,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        stem_type: StemType = StemType.SIMPLE_STEM_IN,
        stem_width: int = 32,
        block_type: BlockType = BlockType.RES_BOTTLENECK_BLOCK,
        activation: ActivationType = ActivationType.RELU,
        use_se: bool = True,
        se_ratio: float = 0.25,
        bn_epsilon: float = 1e-05,
        bn_momentum: float = 0.1,
        features_only: bool = False,
        checkpoint: Optional[str] = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.regnet_params = RegNetParams(
            depth=depth,
            w_0=w_0,
            w_a=w_a,
            w_m=w_m,
            group_width=group_width,
            bottleneck_multiplier=bottleneck_multiplier,
            stem_type=stem_type,
            stem_width=stem_width,
            block_type=block_type,
            activation=activation,
            use_se=use_se,
            se_ratio=se_ratio,
            bn_epsilon=bn_epsilon,
            bn_momentum=bn_momentum,  # type: ignore
        )

        super().__init__(dataset=dataset, features_only=features_only)

    def _load_from_checkpoint(self, checkpoint: str | Path) -> None:
        checkpoint = Path(to_absolute_path(str(checkpoint)))
        if not checkpoint.exists():
            raise AttributeError(f"Checkpoint '{checkpoint}' does not exist.")
        self.logger.info(
            f"Attempting to load {self.__class__.__name__} model from path '{str(checkpoint)}'."
        )

        state_dict = torch.load(f=checkpoint, map_location=torch.device("cpu"))
        trunk_params = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
        self.load_state_dict(trunk_params)
        self.logger.info(
            f"Successfully loaded {self.__class__.__name__} model from path '{str(checkpoint)}'."
        )

    def build(self, dataset: CdtVisionDataset) -> Tuple[nn.Module, int]:
        regnet = ClassyRegNet(self.regnet_params)  # type: ignore
        # Now map the models to the structure we want to expose for SSL tasks
        # The upstream RegNet model is made of :
        # - `stem`
        # - n x blocks in trunk_output, named `block1, block2, ..`

        # We're only interested in the stem and successive blocks
        # everything else is not picked up on purpose
        stem = cast(nn.Sequential, regnet.stem)
        feature_blocks: dict[str, nn.Module] = OrderedDict({"conv1": stem})
        # - get all the feature blocks
        for name, module in regnet.trunk_output.named_children():  # type: ignore
            if not name.startswith("block"):
                raise AttributeError(f"Unexpected layer name {name}")
            block_index = len(feature_blocks) + 1

            feature_blocks[f"res{block_index}"] = module

        # - finally, add avgpool and flatten.
        feature_blocks["avgpool"] = nn.AdaptiveAvgPool2d((1, 1))
        feature_blocks["flatten"] = nn.Flatten()

        self._feature_blocks = nn.Sequential(feature_blocks)

        if self.checkpoint is not None:
            self._load_from_checkpoint(self.checkpoint)

        out_dim: int = cast(int, regnet.trunk_output[-1][0].proj.out_channels)  # type: ignore

        return self._feature_blocks, out_dim


class ConvNeXtVersion(Enum):
    TINY = "convnext_tiny"
    SMALL = "convnext_small"
    BASE = "convnext_base"
    LARGE = "convnext_large"


class ConvNeXt(ClassificationModel):
    Version: TypeAlias = ConvNeXtVersion

    @parsable
    def __init__(
        self,
        dataset: CdtVisionDataset,
        *,
        features_only: bool = False,
        pretrained: bool = False,
        version: ConvNeXtVersion = ConvNeXtVersion.BASE,
    ) -> None:
        tvm.ConvNeXt
        self.pretrained = pretrained
        self.version = version
        super().__init__(dataset=dataset, features_only=features_only)

    @implements(Model)
    def build(self, dataset: CdtVisionDataset) -> Tuple[nn.Module, int]:
        model: tvm.ConvNeXt = getattr(tvm, self.version.value)(pretrained=self.pretrained)
        out_dim = cast(int, model.classifier[-1].in_features)
        model.classifier = nn.Identity()  # type: ignore
        return model, out_dim
