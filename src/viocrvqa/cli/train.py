"""Fine-tune SmolVLM on a ViOCRVQA split (e.g. data/data_sample_0.01).

Full fine-tune (no LoRA/quantization) -- SmolVLM-256M/500M are small enough
to train outright on a single GPU. Loss is computed only on the assistant's
answer tokens; the image + question + template tokens are masked out.

Examples:
    python -m viocrvqa.cli.train
    python -m viocrvqa.cli.train --data-dir data/data_sample_0.01 --epochs 3
    python -m viocrvqa.cli.train --eval-every 1 --eval-limit 200
"""

import argparse

import torch

from ..config import CHECKPOINT_DIR, DEFAULT_MAX_NEW_TOKENS, SAMPLE_DATA_DIR
from ..data.corpus import VQACorpus
from ..evaluation import EvalConfig, Evaluator
from ..models import ModelBundle
from ..tracking import RunTracker
from ..training import Trainer, TrainingConfig
from .common import add_data_args, add_model_args


def build_parser():
    """Command-line interface of the training script."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap)
    add_data_args(ap, data_dir=SAMPLE_DATA_DIR)
    ap.add_argument("--output-dir", default=None,
                    help="where to save checkpoints (default: checkpoints/<model-name>)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-grad-checkpointing", action="store_true",
                    help="disable gradient checkpointing (uses much more VRAM: "
                         "each image splits into up to 17 tiles)")

    ap.add_argument("--eval-every", type=int, default=1,
                    help="evaluate on dev every N epochs; 0 disables")
    ap.add_argument("--eval-limit", type=int, default=200,
                    help="cap dev samples used for periodic eval; 0 = all")
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                    help="for periodic dev-set eval")

    ap.add_argument("--wandb-project", default="smolvlm-viocrvqa")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    return ap


def build_evaluator(args, bundle, corpus, formatter):
    """The periodic dev-set evaluator, or None when --eval-every is 0."""
    if not args.eval_every:
        return None
    return Evaluator(
        bundle, corpus, formatter,
        config=EvalConfig(batch_size=args.eval_batch_size,
                          max_new_tokens=args.max_new_tokens),
        seen_vocab=corpus.answer_vocab(),
    )


def main():
    """Load data and model, then fine-tune with per-epoch eval and checkpoints."""
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    corpus = VQACorpus(args.data_dir)
    train_samples = corpus.load("train")
    dev_samples = corpus.load("dev", limit=args.eval_limit)
    print(f"train={len(train_samples)} dev={len(dev_samples)} samples "
          f"(data_dir={args.data_dir})")

    bundle = ModelBundle.for_training(
        args.model, device=args.device,
        image_splitting=not args.no_image_splitting,
        gradient_checkpointing=not args.no_grad_checkpointing,
    )
    formatter = bundle.formatter(args.prompt)

    tracker = RunTracker.create(
        enabled=not args.no_wandb, project=args.wandb_project,
        name=args.wandb_run_name, config=vars(args))

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        eval_every=args.eval_every,
        output_dir=args.output_dir or CHECKPOINT_DIR / args.model.split("/")[-1],
    )
    Trainer(
        bundle, corpus, train_samples, formatter, config,
        evaluator=build_evaluator(args, bundle, corpus, formatter),
        dev_samples=dev_samples,
        tracker=tracker,
    ).train()

    tracker.finish()


if __name__ == "__main__":
    main()
