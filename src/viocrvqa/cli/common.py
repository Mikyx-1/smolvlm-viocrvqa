"""Argument groups shared by several entry points."""

from ..config import DEFAULT_DATA_DIR, DEFAULT_MODEL, PROMPT


def add_model_args(ap, device="cuda:0", dtype=None):
    """--model / --device / --no-image-splitting (and --dtype if offered)."""
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=device, help="auto | cpu | cuda:N")
    if dtype:
        ap.add_argument("--dtype", default=dtype,
                        choices=["auto", "bfloat16", "float16", "float32"],
                        help="auto follows the checkpoint's config; forcing bfloat16 "
                             "on a fine-tune rounds most of its weight delta away")
    ap.add_argument("--no-image-splitting", action="store_true",
                    help="1 tile instead of up to 17; much faster, less readable text")
    return ap


def add_data_args(ap, data_dir=DEFAULT_DATA_DIR, prompt=PROMPT):
    """--data-dir / --prompt, pointing at one {split}.json + bf_all_img/ dir."""
    ap.add_argument("--data-dir", default=str(data_dir),
                    help="dir containing {split}.json and bf_all_img/")
    ap.add_argument("--prompt", default=prompt,
                    help="prompt name (instruct, raw, vi) or a template containing {q}")
    return ap
