"""Predicting a split and scoring every prediction."""

import time
from dataclasses import dataclass

import torch

from . import metrics as M
from .config import DEFAULT_MAX_NEW_TOKENS
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
            "intent": M.intent(sample["question"]),
            "seen": M.normalize(gold) in self.seen_vocab,
            "em": M.exact_match(p, gold),
            "nem": M.normalized_em(p, gold),
            "f1": M.token_f1(p, gold),
            "cer": M.cer(p, gold),
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
