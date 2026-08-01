"""Download and extract the ViOCRVQA dataset into data/.

Example:
    python -m viocrvqa.cli.download_data
"""

import argparse
import zipfile

from huggingface_hub import hf_hub_download

from ..config import DATA_ROOT, DEFAULT_DATA_DIR, IMG_DIR_NAME, SPLITS

REPO_ID = "huyhuy123/ViOCRVQA"
ARCHIVE = "data_ViOCRVQA.zip"


def build_parser():
    """Command-line interface of the download script."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=str(DATA_ROOT), help="where to download and extract")
    return ap


def download(dest):
    """Fetch the dataset archive from the Hub (~6.5 GB) and return its path."""
    print(f"Downloading {ARCHIVE} (6.5 GB) ...", flush=True)
    path = hf_hub_download(REPO_ID, ARCHIVE, repo_type="dataset", local_dir=dest)
    print("downloaded:", path, flush=True)
    return path


def extract(archive, dest):
    """Unzip the archive in place."""
    print("Extracting ...", flush=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    print("done", flush=True)


def verify(data_dir=DEFAULT_DATA_DIR):
    """Report the image count and which split files landed."""
    img_dir = data_dir / IMG_DIR_NAME
    print("images:", len(list(img_dir.iterdir())) if img_dir.is_dir() else "MISSING")
    for split in SPLITS:
        print(split, (data_dir / f"{split}.json").exists())


def main():
    """Download, extract and sanity-check the dataset."""
    args = build_parser().parse_args()
    extract(download(args.dest), args.dest)
    verify()


if __name__ == "__main__":
    main()
