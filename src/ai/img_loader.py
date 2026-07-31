"""PyTorch image loading helpers for the damaged-box package dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.logger import get_logger

logger = get_logger(__name__)

_IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageDatasetConfig:
    """Configuration for the damaged-box image dataset."""

    root_dir: Path
    image_size: int = 224
    batch_size: int = 32
    validation_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 42
    augment: bool = True


@dataclass(frozen=True)
class ImageDataBundle:
    """Container holding the train/validation/test dataloaders."""

    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    class_to_idx: dict[str, int]


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTENSIONS


def _list_class_directories(root_dir: Path) -> list[Path]:
    return sorted([path for path in root_dir.iterdir() if path.is_dir()])


def _to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


class PackageImageDataset(Dataset[tuple[Tensor, Tensor]]):
    """Folder-based package image dataset for binary intact/damaged labels."""

    def __init__(self, root_dir: Path, image_size: int = 224, augment: bool = False) -> None:
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Image dataset directory not found: {self.root_dir}"
            )

        class_dirs = _list_class_directories(self.root_dir)
        if not class_dirs:
            raise ValueError(
                f"No class subdirectories found under {self.root_dir}. "
                "Expected structure like root/intact/*.jpg and root/damaged/*.jpg."
            )

        self.class_names = [directory.name for directory in class_dirs]
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}

        self.samples: list[tuple[Path, int]] = []
        for class_dir in class_dirs:
            class_index = self.class_to_idx[class_dir.name]
            for file_path in sorted(class_dir.rglob("*")):
                if file_path.is_file() and _is_image_file(file_path):
                    self.samples.append((file_path, class_index))

        if not self.samples:
            raise ValueError(
                f"No supported image files found under {self.root_dir}."
            )

        logger.info(
            "Loaded image dataset: %s samples across %s classes",
            len(self.samples),
            len(self.class_names),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_image(self, image: Image.Image) -> Image.Image:
        if not self.augment:
            return image

        if random.random() < 0.5:
            image = ImageOps.mirror(image)

        if random.random() < 0.3:
            angle = random.uniform(-12.0, 12.0)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR)

        return image

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self._augment_image(image)
            image = ImageOps.fit(
                image,
                (self.image_size, self.image_size),
                method=Image.Resampling.BILINEAR,
            )
            tensor = _to_tensor(image)

        return tensor, torch.tensor(label, dtype=torch.long)


def _split_indices(
    dataset_size: int,
    validation_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1.")
    if not 0.0 <= test_split < 1.0:
        raise ValueError("test_split must be between 0 and 1.")
    if validation_split + test_split >= 1.0:
        raise ValueError("validation_split + test_split must be less than 1.")

    indices = list(range(dataset_size))
    random.Random(seed).shuffle(indices)

    test_count = max(1, int(round(dataset_size * test_split))) if dataset_size >= 3 else 0
    validation_count = (
        max(1, int(round(dataset_size * validation_split))) if dataset_size >= 3 else 0
    )

    if test_count + validation_count >= dataset_size:
        validation_count = max(0, dataset_size - test_count - 1)

    train_count = dataset_size - validation_count - test_count
    if train_count <= 0:
        raise ValueError(
            "Dataset is too small for the requested train/validation/test split."
        )

    train_indices = indices[:train_count]
    validation_indices = indices[train_count : train_count + validation_count]
    test_indices = indices[train_count + validation_count :]
    return train_indices, validation_indices, test_indices


def build_image_dataloaders(config: ImageDatasetConfig) -> ImageDataBundle:
    """Build train/validation/test dataloaders for the package image dataset."""

    train_dataset = PackageImageDataset(
        root_dir=config.root_dir,
        image_size=config.image_size,
        augment=config.augment,
    )
    evaluation_dataset = PackageImageDataset(
        root_dir=config.root_dir,
        image_size=config.image_size,
        augment=False,
    )

    train_indices, validation_indices, test_indices = _split_indices(
        len(train_dataset),
        validation_split=config.validation_split,
        test_split=config.test_split,
        seed=config.seed,
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    validation_loader = DataLoader(
        Subset(evaluation_dataset, validation_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        Subset(evaluation_dataset, test_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return ImageDataBundle(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        class_names=train_dataset.class_names,
        class_to_idx=train_dataset.class_to_idx,
    )


def describe_dataset(root_dir: Path) -> str:
    """Return a concise dataset summary for CLI usage."""

    dataset = PackageImageDataset(root_dir=root_dir, augment=False)
    per_class: dict[str, int] = {name: 0 for name in dataset.class_names}
    for _, label in dataset.samples:
        class_name = dataset.class_names[label]
        per_class[class_name] += 1

    parts = [f"{class_name}={count}" for class_name, count in per_class.items()]
    return f"{len(dataset)} images | " + ", ".join(parts)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", type=Path, help="Path to the damaged-box image root")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = ImageDatasetConfig(
        root_dir=args.root_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    bundle = build_image_dataloaders(config)
    logger.info("Dataset summary: %s", describe_dataset(args.root_dir))
    logger.info(
        "Dataloaders ready | train=%s | validation=%s | test=%s",
        len(bundle.train_loader),
        len(bundle.validation_loader),
        len(bundle.test_loader),
    )


if __name__ == "__main__":
    main()