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
        split: Optional[Union[Split, str]] = Split.TRAIN,
        imagery: Optional[ImageryType] = ImageryType.AERIAL,
        transform: Optional[ImageTform] = None,
    ) -> None:

        self.split = None if split is None else str_to_enum(str_=split, enum=self.Split)
        self.imagery = None if imagery is None else str_to_enum(str_=imagery, enum=ImageryType)
        self.root = Path(root)
        self._base_dir = self.root / self.__class__.__name__.lower()  # / self.split.value
        self._metadata_path = self._base_dir / "metadata.csv"

        if not self._metadata_path.exists():
            self._extract_metadata()
        self.metadata = pd.DataFrame(pd.read_csv(self._metadata_path))
        # Extract the relevant subsets of the data.
        if self.split is not None:
            self.metadata = self.metadata[self.metadata["split"] == self.split.value]
        # Imagery-type is only applicable to the training data.
        if self.split is self.Split.TRAIN and self.imagery is not None:
            self.metadata = self.metadata[self.metadata["imagery"] == self.imagery.value]

        x = self.metadata["filepath"].to_numpy()
        y = (
            torch.as_tensor(self.metadata["label"].to_numpy(), dtype=torch.long)
            if "label" in self.metadata.columns
            else None
        )

        super().__init__(x=x, y=y, transform=transform, image_dir=self._base_dir)

    def _extract_metadata(self) -> None:
        """Extract the filenames and labels from the image filepaths and save them to csv."""
        self.logger.info("Extracting metadata.")
        image_paths: List[Path] = []
        for ext in ("jpg", "jpeg", "png"):
            # Glob images from child folders recusrively, excluding hidden files
            image_paths.extend(self._base_dir.glob(f"**/[!.]*.{ext}"))
        image_paths_str = [str(image.relative_to(self._base_dir)) for image in image_paths]
        filepaths = pd.Series(image_paths_str, name="filepath")
        metadata = cast(pd.DataFrame, filepaths.str.split("/", expand=True))
        metadata.rename(columns={0: "split", 1: "imagery", 2: "label", 3: "filename"}, inplace=True)
        # Since the filepaths of the test images only have two components (split and filename)
        # their comprised filenames need to be moved to the last column after the expansion
        # so as to be aligned with those of the training data.
        split_mask = metadata["split"] == self.Split.TEST.value
        metadata.loc[split_mask, "filename"] = metadata[split_mask]["imagery"]
        metadata.loc[split_mask, "imagery"] = None
        metadata["filepath"] = filepaths
        # sort the images according to 'split' in preparation for de-duplication
        # (order is 'test' then 'train')
        metadata.sort_values(by=["split"], axis=0, inplace=True)
        # The test data is a superset of the training data -- any samples that are
        # in both the training data and the test data are de-duplicated such that only
        # the labeled version in the training data is retained.
        metadata.drop_duplicates("filename", keep="last", inplace=True)
        metadata.sort_values(by=["filepath"], axis=0, inplace=True)  #
        metadata.reset_index(inplace=True)
        metadata.sort_index(axis=1, inplace=True)
        metadata.to_csv(self._metadata_path)
