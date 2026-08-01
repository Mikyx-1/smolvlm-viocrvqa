"""Reading one ViOCRVQA data directory.

A data dir holds {split}.json plus a shared image folder, and the same
layout is used by the full dataset and by the subsampled copies produced
by scripts/split_data.py -- so everything is parametrized by data_dir.
"""

import json
import os
from pathlib import Path

from PIL import Image

from .. import metrics as M
from ..config import DEFAULT_DATA_DIR, IMG_DIR_NAME

BLANK_IMAGE_SIZE = (512, 512)
BLANK_IMAGE_COLOR = (128, 128, 128)


class VQACorpus:
    """The {split}.json + bf_all_img/ pair living under one data directory."""

    def __init__(self, data_dir=DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self._splits = {}
        self._answer_vocab = None

    def __repr__(self):
        """Identify the corpus by its directory."""
        return f"VQACorpus({str(self.data_dir)!r})"

    @property
    def img_dir(self) -> Path:
        """Directory holding every image referenced by the splits."""
        return self.data_dir / IMG_DIR_NAME

    def require_images(self):
        """Raise a helpful error if the images were never downloaded."""
        if not self.img_dir.is_dir():
            raise SystemExit(
                f"missing images at {self.img_dir}; "
                f"run `python -m viocrvqa.cli.download_data` first")

    def load(self, split, limit=0, shard=0, num_shards=1):
        """Return the split's samples, optionally truncated and/or sharded."""
        samples = self._read_split(split)
        if limit:
            samples = samples[:limit]
        if num_shards > 1:
            samples = samples[shard::num_shards]
        return samples

    def answer_vocab(self):
        """Normalised set of train answers, used to flag unseen gold answers."""
        if self._answer_vocab is None:
            self._answer_vocab = {M.normalize(s["gold"]) for s in self._read_split("train")}
        return self._answer_vocab

    def open_image(self, sample, blank=False):
        """Load a sample's image as RGB, or a grey placeholder when blank."""
        if blank:
            return Image.new("RGB", BLANK_IMAGE_SIZE, BLANK_IMAGE_COLOR)
        return Image.open(os.path.join(self.img_dir, sample["image"])).convert("RGB")

    def group_by_image(self, samples):
        """Bucket samples by image filename, preserving their order."""
        by_img = {}
        for s in samples:
            by_img.setdefault(s["image"], []).append(s)
        return by_img

    def _read_split(self, split):
        """Parse {split}.json into flat {image, question, gold} dicts (cached)."""
        if split not in self._splits:
            with open(self.data_dir / f"{split}.json", encoding="utf-8") as f:
                d = json.load(f)
            id2file = {im["id"]: im["filename"] for im in d["images"]}
            self._splits[split] = [
                {
                    "image": id2file[a["image_id"]],
                    "question": a["question"],
                    "gold": a["answers"][0],
                }
                for a in d["annotations"]
            ]
        return self._splits[split]
