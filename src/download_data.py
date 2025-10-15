"""
Helpers for downloading and extracting chess datasets from Kaggle.
"""

from __future__ import annotations

import bz2
import glob
import shutil
from pathlib import Path
from typing import Optional

import kagglehub


def _extract_bz2_archive(archive_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / archive_path.with_suffix("").name
    with bz2.open(archive_path, "rb") as src, output_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return output_path


def download_chess_data(
    dataset_name: str = "ironicninja/raw-chess-games-pgn",
    download_path: str | Path = "data/input",
) -> Optional[Path]:
    """
    Download the specified Kaggle dataset and extract any `.bz2` archives.

    Returns:
        The directory containing the extracted files, or None if the download failed.
    """
    download_path = Path(download_path)
    print(f"[download_data] downloading dataset '{dataset_name}' via kagglehub...")

    try:
        dataset_dir = Path(kagglehub.dataset_download(dataset_name))
        print(f"[download_data] dataset staged at {dataset_dir}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[download_data] failed to download dataset: {exc}")
        return None

    archives = glob.glob(str(dataset_dir / "*.bz2"))
    extracted_files = []

    for archive in archives:
        archive_path = Path(archive)
        print(f"[download_data] extracting {archive_path.name}")
        extracted = _extract_bz2_archive(archive_path, download_path)
        extracted_files.append(extracted)
        print(f"[download_data] extracted to {extracted}")

    if not extracted_files:
        print("[download_data] no .bz2 archives found; copying files directly.")
        download_path.mkdir(parents=True, exist_ok=True)
        for file_path in dataset_dir.iterdir():
            if file_path.is_file():
                destination = download_path / file_path.name
                shutil.copyfile(file_path, destination)
                extracted_files.append(destination)

    if extracted_files:
        print(f"[download_data] ready: {len(extracted_files)} files in {download_path}")
        return download_path

    print("[download_data] no files extracted.")
    return None


if __name__ == "__main__":
    result = download_chess_data()
    if result is None:
        print("Download failed.")
    else:
        print(f"Data available in {result}")
