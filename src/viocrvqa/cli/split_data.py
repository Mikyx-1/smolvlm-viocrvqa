"""Subsample the ViOCRVQA dataset into a smaller fraction for fast iteration.

Images are symlinked rather than copied, so a 1% sample costs nothing on disk.

Examples:
    python -m viocrvqa.cli.split_data --fraction 0.01
    python -m viocrvqa.cli.split_data --fraction 0.1 --splits train dev
"""

import argparse
import json
import random
from pathlib import Path

from ..config import DEFAULT_DATA_DIR, IMG_DIR_NAME, SPLITS


def build_parser():
    """Command-line interface of the subsampling script."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="dir containing {split}.json files and the image dir")
    ap.add_argument("--img-dir", default=IMG_DIR_NAME,
                    help="name of the shared image folder inside --data-dir")
    ap.add_argument("--output-dir", default=None,
                    help="where to write the sample (default: <data-dir>_sample_<fraction>)")
    ap.add_argument("--fraction", type=float, default=0.01,
                    help="fraction of each split to keep, e.g. 0.01 = 1%%")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--seed", type=int, default=42)
    return ap


def subsample_split(data_dir: Path, img_dir: Path, out_dir: Path,
                    split: str, fraction: float, seed: int):
    """Keep a fraction of one split's images plus all of their annotations."""
    data = json.loads((data_dir / f"{split}.json").read_text(encoding="utf-8"))
    images = data["images"]

    rng = random.Random(seed)
    sampled = rng.sample(images, max(1, round(len(images) * fraction)))
    kept_ids = {img["id"] for img in sampled}
    annotations = [a for a in data["annotations"] if a["image_id"] in kept_ids]

    (out_dir / f"{split}.json").write_text(
        json.dumps({"images": sampled, "annotations": annotations},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    link_images(sampled, img_dir, out_dir / img_dir.name)

    print(f"{split}: kept {len(sampled)}/{len(images)} images, "
          f"{len(annotations)}/{len(data['annotations'])} annotations")


def link_images(images, img_dir: Path, out_img_dir: Path):
    """Symlink each kept image into the sample's image folder."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        dst = out_img_dir / img["filename"]
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to((img_dir / img["filename"]).resolve())


def main():
    """Write a subsampled copy of every requested split."""
    ap = build_parser()
    args = ap.parse_args()
    if not 0 < args.fraction <= 1:
        ap.error("--fraction must be in (0, 1]")

    data_dir = Path(args.data_dir)
    out_dir = (Path(args.output_dir) if args.output_dir
               else data_dir.parent / f"{data_dir.name}_sample_{args.fraction:g}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        subsample_split(data_dir, data_dir / args.img_dir, out_dir,
                        split, args.fraction, args.seed)

    print(f"\nWrote sampled dataset to {out_dir}")


if __name__ == "__main__":
    main()
