"""Score SmolVLM on a ViOCRVQA split (zero-shot, or a fine-tuned checkpoint).

The dataset's own test split is unlabelled (every answer is the literal string
"your answer"), so `dev` is used as the evaluation set.

Examples:
    python -m viocrvqa.cli.eval_baseline --limit 200
    python -m viocrvqa.cli.eval_baseline                            # full dev split
    python -m viocrvqa.cli.eval_baseline --blank-image              # memorisation floor
    python -m viocrvqa.cli.eval_baseline --shard 0 --num-shards 3   # one GPU per shard
    python -m viocrvqa.cli.eval_baseline --model checkpoints/SmolVLM-256M-Instruct/epoch3
"""

import argparse

from ..config import DEFAULT_MAX_NEW_TOKENS, RESULTS_DIR
from ..data.corpus import VQACorpus
from ..evaluation import EvalConfig, Evaluator, ProgressPrinter
from ..models import ModelBundle
from ..reporting import Report
from .common import add_data_args, add_model_args


def build_parser():
    """Command-line interface of the evaluation script."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, dtype="auto")
    add_data_args(ap)
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--blank-image", action="store_true",
                    help="feed a grey image; measures how much is answerable without reading")
    ap.add_argument("--out", default=None, help="where to write the records JSON")
    return ap


def output_path(args):
    """Default the records path to results/baseline_<split>_<tag>_shard<N>.json."""
    if args.out:
        return args.out
    tag = "blank" if args.blank_image else "image"
    return RESULTS_DIR / f"baseline_{args.split}_{tag}_shard{args.shard}.json"


def main():
    """Evaluate one split, save the records and print the report."""
    args = build_parser().parse_args()

    corpus = VQACorpus(args.data_dir)
    corpus.require_images()
    samples = corpus.load(args.split, limit=args.limit,
                          shard=args.shard, num_shards=args.num_shards)
    print(f"{len(samples)} samples (split={args.split} "
          f"shard={args.shard}/{args.num_shards})")

    bundle = ModelBundle.for_inference(
        args.model, device=args.device, dtype=args.dtype,
        image_splitting=not args.no_image_splitting)

    evaluator = Evaluator(
        bundle, corpus, bundle.formatter(args.prompt),
        config=EvalConfig(batch_size=args.batch_size,
                          max_new_tokens=args.max_new_tokens,
                          blank_image=args.blank_image),
        seen_vocab=corpus.answer_vocab(),
    )
    report = Report(evaluator.evaluate(samples, on_batch=ProgressPrinter()))

    path = report.save(output_path(args))
    report.print()
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
