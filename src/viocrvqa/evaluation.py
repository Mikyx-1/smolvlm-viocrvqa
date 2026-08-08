"""Predicting a split and scoring every prediction."""

import contextlib
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from . import metrics as M
from .config import DEFAULT_MAX_NEW_TOKENS
from .data.dataset import VQACollator, VQADataset
from .data.fields import field
from .generation import BatchGenerator


@dataclass
class EvalConfig:
    """Knobs for one evaluation pass."""

    batch_size: int = 16
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    blank_image: bool = False  # grey images; measures what is answerable without reading


class Evaluator:
    """Generate answers for a list of samples and score them record by record.

    The same object serves the zero-shot baseline, the qualitative sampler
    and the periodic dev-set check during training -- they differ only in
    which records they then print.
    """

    def __init__(self, bundle, corpus, formatter, config=None, seen_vocab=frozenset()):
        self.bundle = bundle
        self.corpus = corpus
        self.formatter = formatter
        self.config = config or EvalConfig()
        self.seen_vocab = seen_vocab
        self.generator = BatchGenerator(bundle, self.config.max_new_tokens)

    @torch.no_grad()
    def evaluate(self, samples, on_batch=None):
        """Predict every sample and return its scored record.

        on_batch(records, total) is called after each batch for progress
        reporting; the caller decides what (if anything) to print.
        """
        records = []
        with self.bundle.generation_mode():
            for preds, batch in self._iter_predictions(samples):
                records.extend(self._score(s, p) for s, p in zip(batch, preds))
                if on_batch:
                    on_batch(records, len(samples))
        return records

    def predict(self, samples, on_batch=None):
        """Just the prediction strings, in sample order."""
        return [r["pred"] for r in self.evaluate(samples, on_batch)]

    def _iter_predictions(self, samples):
        """Yield (predictions, batch) for each batch of the split."""
        for start in range(0, len(samples), self.config.batch_size):
            batch = samples[start : start + self.config.batch_size]
            images, texts = self._prepare(batch)
            yield self.generator.generate(images, texts), batch

    def _prepare(self, batch):
        """Load each sample's image and render its generation prompt."""
        images, texts = [], []
        for s in batch:
            images.append([self.corpus.open_image(s, blank=self.config.blank_image)])
            texts.append(self.formatter.prompt_text(s["question"]))
        return images, texts

    def _score(self, sample, pred):
        """Build one record: the prediction plus every metric against gold."""
        p = pred.strip()
        gold = sample["gold"]
        return {
            "image": sample["image"], "question": sample["question"], "gold": gold, "pred": p,
            "field": field(sample["question"]),
            "seen": M.normalize(gold) in self.seen_vocab,
            "em": M.exact_match(p, gold),
            "nem": M.normalized_em(p, gold),
            "f1": M.token_f1(p, gold),
            "cer": M.cer(p, gold),
        }


class LossEvaluator:
    """Teacher-forced loss over a split, measured the way training measures it.

    The generation metrics say whether the model *emits* the gold answer;
    this says how much probability it puts on it. Reading them together is
    what makes either one trustworthy -- a near-zero loss beside a poor exact
    match means the eval path has drifted from the training path, not that
    the model failed to learn.

    Same collator and the same bf16 autocast as Trainer, so the number is
    directly comparable to the loss printed during training.
    """

    def __init__(self, bundle, corpus, formatter, batch_size=2, amp=True,
                 blank_image=False):
        self.bundle = bundle
        self.corpus = corpus
        self.formatter = formatter
        self.batch_size = batch_size
        self.amp = amp
        self.blank_image = blank_image

    def autocast(self):
        """bf16 compute, matching Trainer, so the loss is on the same scale."""
        if not self.amp or not str(self.bundle.device).startswith("cuda"):
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=torch.bfloat16)

    @torch.no_grad()
    def evaluate(self, samples, on_batch=None):
        """Return loss, perplexity and argmax accuracy over the answer tokens."""
        loader = DataLoader(
            VQADataset(samples, self.corpus, blank_image=self.blank_image),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=VQACollator(self.formatter, self.bundle.device),
        )
        total = dict(nll=0.0, tokens=0, correct=0, answers=0, exact=0)
        with self.bundle.teacher_forcing_mode():
            for batch in loader:
                self._accumulate(batch, total)
                if on_batch:
                    on_batch(total, len(samples))
        return self._summarise(total, len(samples))

    def _accumulate(self, batch, total):
        """Add one batch's answer-token statistics into `total`."""
        with self.autocast():
            logits = self.bundle.model(**batch).logits
        # standard causal shift: position t predicts token t+1
        logits, labels = logits[:, :-1], batch["labels"][:, 1:]
        mask = labels != -100

        # gather answer positions before upcasting -- the full logit tensor is
        # [batch, seq, 49k] and seq is dominated by image tokens
        answer_logits = logits[mask].float()
        answer_labels = labels[mask]
        total["nll"] += torch.nn.functional.cross_entropy(
            answer_logits, answer_labels, reduction="sum").item()
        total["tokens"] += int(mask.sum())

        correct = answer_logits.argmax(-1) == answer_labels
        total["correct"] += int(correct.sum())
        # split back per sample to ask whether a whole answer would survive
        for row in correct.split(mask.sum(dim=1).tolist()):
            if len(row):
                total["answers"] += 1
                total["exact"] += int(bool(row.all()))

    @staticmethod
    def _summarise(total, n_samples):
        """Turn the running sums into the reported scalars."""
        tokens = max(total["tokens"], 1)
        answers = max(total["answers"], 1)
        loss = total["nll"] / tokens
        return {
            "n": n_samples,
            "answer_tokens": total["tokens"],
            "loss": loss,
            "perplexity": float(torch.exp(torch.tensor(loss))),
            "token_accuracy": 100 * total["correct"] / tokens,
            "answer_accuracy": 100 * total["exact"] / answers,
        }


class ProgressPrinter:
    """An on_batch callback printing throughput and running accuracy."""

    def __init__(self, inline=False):
        self.inline = inline  # overwrite one line instead of scrolling
        self.t0 = time.time()

    def __call__(self, records, total):
        """Print progress for the records produced so far."""
        done = len(records)
        rate = done / max(time.time() - self.t0, 1e-9)
        nem = 100 * sum(r["nem"] for r in records) / done
        line = f"  {done}/{total}  {rate:.1f} sample/s  nEM={nem:.2f}%"
        print(line, end="\r" if self.inline else "\n", flush=True)

    def done(self):
        """Clear the inline progress line once the pass is over."""
        if self.inline:
            print(" " * 60, end="\r")


class LossProgressPrinter:
    """An on_batch callback for LossEvaluator's running totals."""

    def __init__(self):
        self.t0 = time.time()

    def __call__(self, total, n_samples):
        """Print the running loss over the answer tokens seen so far."""
        tokens = max(total["tokens"], 1)
        done = total["answers"]
        rate = done / max(time.time() - self.t0, 1e-9)
        print(f"  {done}/{n_samples}  {rate:.1f} sample/s  "
              f"loss={total['nll'] / tokens:.4f}", flush=True)
