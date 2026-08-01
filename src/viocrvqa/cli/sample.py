"""Run SmolVLM on a small sample of ViOCRVQA and print predictions for reading.

Complements eval_baseline, which scores the whole split. This one is for
looking at what the model actually emits:

  * samples at random rather than taking the head of the split (the first N
    annotations all belong to the first handful of images)
  * can group by image, so the several questions asked about one cover appear
    together -- that is how you see whether the answer varies with the question
  * can run several prompt templates over the same sample and score each, so a
    prompt change is comparable rather than anecdotal

Examples:
    python -m viocrvqa.cli.sample                            # 12 random dev samples
    python -m viocrvqa.cli.sample -n 24 --seed 7
    python -m viocrvqa.cli.sample --by-image 4               # 4 covers, all their questions
    python -m viocrvqa.cli.sample --compare instruct raw vi  # score 3 prompts
    python -m viocrvqa.cli.sample --blank-image              # memorisation control
    python -m viocrvqa.cli.sample --device cpu --no-image-splitting
"""

import argparse
import random

from ..config import PROMPTS, resolve_prompt
from ..data.corpus import VQACorpus
from ..evaluation import EvalConfig, Evaluator, ProgressPrinter
from ..models import ModelBundle
from ..reporting import WRAP, Report
from .common import add_data_args, add_model_args


def build_parser():
    """Command-line interface of the sampling script."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, device="auto", dtype="auto")
    add_data_args(ap, prompt="instruct")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("-n", type=int, default=12, help="number of random annotations")
    ap.add_argument("--by-image", type=int, default=0, metavar="K",
                    help="instead sample K images and use every question about them")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", nargs="+", default=None, metavar="PROMPT",
                    help=f"score several prompts on the same sample; "
                         f"names ({', '.join(PROMPTS)}) or templates containing {{q}}")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--blank-image", action="store_true",
                    help="feed grey images; shows what is answerable without reading")
    return ap


def pick_samples(corpus, args):
    """Draw random annotations, or every question about K random images."""
    samples = corpus.load(args.split)
    rng = random.Random(args.seed)
    if args.by_image:
        by_img = corpus.group_by_image(samples)
        chosen = rng.sample(sorted(by_img), min(args.by_image, len(by_img)))
        return [s for img in chosen for s in by_img[img]]
    return rng.sample(samples, min(args.n, len(samples)))


def main():
    """Score the sample under one or more prompts and print the predictions."""
    args = build_parser().parse_args()

    corpus = VQACorpus(args.data_dir)
    corpus.require_images()
    templates = [(name, resolve_prompt(name)) for name in (args.compare or [args.prompt])]

    samples = pick_samples(corpus, args)
    n_images = len({s["image"] for s in samples})
    print(f"{len(samples)} samples over {n_images} image(s) "
          f"(split={args.split} seed={args.seed}"
          f"{' BLANK' if args.blank_image else ''})")

    bundle = ModelBundle.for_inference(
        args.model, device=args.device, dtype=args.dtype,
        image_splitting=not args.no_image_splitting)
    config = EvalConfig(batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                        blank_image=args.blank_image)

    reports = {}
    for name, template in templates:
        print(f"\nprompt {name!r}: {template}")
        evaluator = Evaluator(bundle, corpus, bundle.formatter(template), config=config)
        progress = ProgressPrinter(inline=True)
        reports[name] = Report(evaluator.evaluate(samples, on_batch=progress))
        progress.done()
        if len(templates) == 1:
            reports[name].print_predictions(grouped=bool(args.by_image))

    print("\n" + "=" * WRAP)
    for name, _ in templates:
        reports[name].print_line(name)
    print("=" * WRAP)

    if len(templates) > 1:
        best = max(reports, key=lambda k: reports[k].metrics["token_f1"])
        print(f"\nbest token F1: {best!r}; its predictions:")
        reports[best].print_predictions(grouped=bool(args.by_image))


if __name__ == "__main__":
    main()
