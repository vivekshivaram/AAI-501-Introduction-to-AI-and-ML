"""Download the damaged-box image dataset into data/raw.

Usage (from project root):
    python -m src.ingestion.download_damaged_box
    python -m src.ingestion.download_damaged_box --target data/raw/damaged_box
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub

from src.config import DATA_RAW_DIRECTORY
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET_HANDLE = "teomingzhe/damaged-box"
DEFAULT_TARGET_DIRECTORY = DATA_RAW_DIRECTORY / "damaged_box"


def _copy_downloaded_tree(source_path: Path, target_path: Path) -> None:
    if source_path.is_file():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        return

    target_path.mkdir(parents=True, exist_ok=True)
    for item in source_path.iterdir():
        destination = target_path / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def download_damaged_box_dataset(
    dataset_handle: str = DEFAULT_DATASET_HANDLE,
    target_directory: Path = DEFAULT_TARGET_DIRECTORY,
    force_download: bool = False,
) -> Path:
    """Download the damaged-box dataset and place it under data/raw.

    Parameters
    ----------
    dataset_handle:
        KaggleHub dataset handle.
    target_directory:
        Destination folder under data/raw.
    force_download:
        Redownload the dataset even if it is already cached.
    """

    if target_directory.exists() and any(target_directory.iterdir()) and not force_download:
        logger.info("Dataset already present at %s; skipping download.", target_directory)
        return target_directory

    logger.info("Downloading %s ...", dataset_handle)
    downloaded_path = Path(
        kagglehub.dataset_download(dataset_handle, force_download=force_download)
    )

    if target_directory.exists() and force_download:
        shutil.rmtree(target_directory)

    _copy_downloaded_tree(downloaded_path, target_directory)
    logger.info("Damaged-box dataset saved to %s", target_directory)
    return target_directory


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the damaged-box Kaggle dataset into data/raw."
    )
    parser.add_argument(
        "--handle",
        default=DEFAULT_DATASET_HANDLE,
        help=f'Dataset handle (default: "{DEFAULT_DATASET_HANDLE}")',
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET_DIRECTORY,
        help=f"Destination directory (default: {DEFAULT_TARGET_DIRECTORY})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite the target directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    download_damaged_box_dataset(
        dataset_handle=args.handle,
        target_directory=args.target,
        force_download=args.force,
    )


if __name__ == "__main__":
    main()