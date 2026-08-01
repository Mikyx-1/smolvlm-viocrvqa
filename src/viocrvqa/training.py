"""The supervised fine-tuning loop."""

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from .config import CHECKPOINT_DIR
from .data.dataset import VQACollator, VQADataset
from .reporting import Report
from .tracking import RunTracker


@dataclass
class TrainingConfig:
    """Hyper-parameters and schedule of one fine-tuning run."""

    epochs: int = 50
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 1e-5
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    eval_every: int = 1  # evaluate on dev every N epochs; 0 disables
    output_dir: Path = CHECKPOINT_DIR

    def __post_init__(self):
        """Accept a string output_dir from argparse."""
        self.output_dir = Path(self.output_dir)


class Trainer:
    """Full fine-tune with gradient accumulation, dev eval and checkpoints.

    Loss comes from the model itself: the collator masks the labels so only
    the assistant's answer tokens contribute.
    """

    def __init__(self, bundle, corpus, train_samples, formatter, config,
                 evaluator=None, dev_samples=(), tracker=None):
        self.bundle = bundle
        self.corpus = corpus
        self.formatter = formatter
        self.config = config
        self.evaluator = evaluator
        self.dev_samples = list(dev_samples)
        self.tracker = tracker or RunTracker()

        self.loader = self._build_loader(train_samples)
        self.steps_per_epoch = -(-len(self.loader) // config.grad_accum_steps)  # ceil
        self.total_steps = self.steps_per_epoch * config.epochs
        self.optimizer, self.scheduler = self._build_optimizer()
        self.step = 0
        self.t0 = time.time()

    def train(self):
        """Run every epoch: train, optionally evaluate, then checkpoint."""
        print(f"{self.steps_per_epoch} optimizer steps/epoch, {self.total_steps} total")
        self.bundle.model.train()
        self.step, self.t0 = 0, time.time()
        for epoch in range(1, self.config.epochs + 1):
            self.train_epoch(epoch)
            if self.should_eval(epoch):
                self.evaluate(epoch)
            self.save_checkpoint(epoch)

    def train_epoch(self, epoch):
        """One pass over the loader, stepping every grad_accum_steps batches."""
        running_loss = 0.0
        for i, batch in enumerate(self.loader):
            loss = self.bundle.model(**batch).loss / self.config.grad_accum_steps
            loss.backward()
            running_loss += loss.item()

            if (i + 1) % self.config.grad_accum_steps == 0 or i + 1 == len(self.loader):
                self.optimizer_step()
                self.log_step(epoch, running_loss)
                running_loss = 0.0

    def optimizer_step(self):
        """Clip gradients, step the optimizer and LR schedule, then zero grads."""
        torch.nn.utils.clip_grad_norm_(self.bundle.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.step += 1

    def log_step(self, epoch, loss):
        """Print and track the accumulated loss of one optimizer step."""
        print(f"  epoch {epoch} step {self.step}/{self.total_steps}  "
              f"loss={loss:.4f}  {time.time() - self.t0:.0f}s", flush=True)
        self.tracker.log({"train/loss": loss, "train/lr": self.scheduler.get_last_lr()[0],
                          "epoch": epoch}, step=self.step)

    def should_eval(self, epoch):
        """True when an evaluator exists and this epoch lands on eval_every."""
        return bool(self.evaluator and self.config.eval_every
                    and epoch % self.config.eval_every == 0)

    def evaluate(self, epoch):
        """Score the dev samples, print the breakdown and track the metrics."""
        print(f"\n--- dev eval after epoch {epoch} ---")
        report = Report(self.evaluator.evaluate(self.dev_samples))
        report.print()
        self.tracker.log(report.scalars(prefix="dev/"), step=self.step)
        return report

    def save_checkpoint(self, epoch):
        """Write the bundle to <output_dir>/epoch<N>."""
        path = self.bundle.save(self.config.output_dir / f"epoch{epoch}")
        print(f"saved checkpoint to {path}")
        return path

    def _build_loader(self, train_samples):
        """Shuffled DataLoader over VQADataset with the answer-masking collator."""
        return DataLoader(
            VQADataset(train_samples, self.corpus),
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=VQACollator(self.formatter, self.bundle.device),
        )

    def _build_optimizer(self):
        """AdamW plus a cosine schedule with warmup_ratio of the total steps."""
        optimizer = torch.optim.AdamW(self.bundle.model.parameters(), lr=self.config.lr)
        scheduler = get_scheduler(
            "cosine", optimizer=optimizer,
            num_warmup_steps=round(self.total_steps * self.config.warmup_ratio),
            num_training_steps=self.total_steps,
        )
        return optimizer, scheduler
