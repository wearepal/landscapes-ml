"""Wakehurst Dataset."""
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, cast

from conduit.data.datasets.utils import ImageTform
from conduit.data.datasets.vision.base import CdtVisionDataset
from conduit.data.structures import TernarySample
import pandas as pd  # type: ignore
from ranzen import parsable, str_to_enum
import torch
from torch import Tensor
from typing_extensions import TypeAlias

__all__ = ["Wakehurst", "ImageryType"]


class WakehurstSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class ImageryType(Enum):
    AERIAL = "Aerial Photography (12.5cm)"


SampleType: TypeAlias = TernarySample


class Wakehurst(CdtVisionDataset[TernarySample, Tensor, None]):
    Split: TypeAlias = WakehurstSplit
    SampleType: TypeAlias = SampleType
    ImageryType: TypeAlias = ImageryType

    @parsable
    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: Union[Split, str] = Split.TRAIN,
        imagery: ImageryType = ImageryType.AERIAL,
        transform: Optional[ImageTform] = None,
    ) -> None:
        self.root = Path(root)
        self.split = str_to_enum(str_=split, enum=self.Split)
        self._base_dir = self.root / self.__class__.__name__.lower() / self.split.value
        if split is WakehurstSplit.TRAIN:
            self._base_dir /= imagery.value
        self._metadata_path = self._base_dir / "metadata.csv"
        self.imagery = str_to_enum(str_=imagery, enum=ImageryType)

        if not self._metadata_path.exists():
            self._extract_metadata()
        self.metadata = pd.DataFrame(pd.read_csv(self._base_dir / "metadata.csv"))

        x = self.metadata["filepath"].to_numpy()
        y = torch.as_tensor(self.metadata["label"].to_numpy(), dtype=torch.long)

        super().__init__(x=x, y=y, transform=transform, image_dir=self._base_dir)

    def _extract_metadata(self) -> None:
        """Extract the filenames and labels from the image filepaths and save them to csv."""
        self.logger.info("Extracting metadata.")
        image_paths: List[Path] = []
        for ext in ("jpg", "jpeg", "png"):
            # Glob images from child folders recusrively, excluding hidden files
            image_paths.extend(self._base_dir.glob(f"**/[!.]*.{ext}"))
        image_paths_str = [str(image.relative_to(self._base_dir)) for image in image_paths]
        filepaths = pd.Series(image_paths_str)
        metadata = cast(
            pd.DataFrame,
            filepaths.str.split("/", expand=True)  # type: ignore[attr-defined]
            .dropna(axis=1)
            .rename(columns={0: "label", 1: "filename"}),
        )
        metadata["filepath"] = filepaths
        metadata.sort_index(axis=1, inplace=True)
        metadata.sort_values(by=["filepath"], axis=0, inplace=True)
        metadata.to_csv(self._metadata_path)
