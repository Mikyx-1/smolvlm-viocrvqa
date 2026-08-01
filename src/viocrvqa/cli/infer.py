"""Run SmolVLM on arbitrary images for a quick look at what it does.

Examples:
    python -m viocrvqa.cli.infer                                 # demo image
    python -m viocrvqa.cli.infer -i cat.jpg -p "What breed is this?"
    python -m viocrvqa.cli.infer -i https://example.com/x.png -p "Read the text."
    python -m viocrvqa.cli.infer -i a.jpg -i b.jpg -p "Compare these two images."
    python -m viocrvqa.cli.infer --no-image -p "Who wrote Dune?"   # text only
"""

import argparse
import io

import requests
from PIL import Image

from ..config import DEFAULT_MODEL
from ..generation import BatchGenerator
from ..models import ModelBundle

DEMO_IMAGE = "https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg"
TIMEOUT = 30


def load_image(src: str) -> Image.Image:
    """Open an image from a local path or an http(s) URL, as RGB."""
    if src.startswith(("http://", "https://")):
        resp = requests.get(src, timeout=TIMEOUT, headers={"User-Agent": "smolvlm-demo"})
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
    else:
        img = Image.open(src)
    return img.convert("RGB")


def build_parser():
    """Command-line interface of the ad-hoc inference script."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--image", action="append", help="image path or URL (repeatable)")
    ap.add_argument("--no-image", action="store_true", help="run text-only, no image input")
    ap.add_argument("-p", "--prompt", default="Describe this image in detail.")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("-n", "--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:N")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding")
    return ap


def build_chat_text(processor, prompt, n_images):
    """One user turn holding `n_images` image placeholders and the prompt."""
    messages = [{"role": "user", "content":
                 [{"type": "image"} for _ in range(n_images)]
                 + [{"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def main():
    """Load the model, answer the prompt about the given images, print the reply."""
    args = build_parser().parse_args()

    sources = [] if args.no_image else (args.image or [DEMO_IMAGE])
    images = [load_image(s) for s in sources]

    print(f"Loading {args.model} on {args.device} ({args.dtype}) ...")
    bundle = ModelBundle.for_inference(args.model, device=args.device, dtype=args.dtype)
    print(f"Parameters: {bundle.n_params / 1e6:.1f}M")

    text = build_chat_text(bundle.processor, args.prompt, len(images))
    generator = BatchGenerator(bundle, args.max_new_tokens, args.temperature)
    reply = generator.generate([images] if images else [], [text])[0].strip()

    print("\n" + "=" * 60)
    print(f"Images: {', '.join(sources) if sources else 'none (text-only)'}")
    print(f"Prompt: {args.prompt}")
    print("-" * 60)
    print(reply)
    print("=" * 60)


if __name__ == "__main__":
    main()
