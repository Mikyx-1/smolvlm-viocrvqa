"""Every way a list of scored records gets printed or saved."""

import collections
import json
import textwrap
from pathlib import Path

from . import metrics as M

WIDTH = 66
WRAP = 76
INTENTS = ["title", "author", "genre", "publisher", "other"]


class Report:
    """A scored record list, with the breakdowns used to read it."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        """Number of scored records."""
        return len(self.records)

    @classmethod
    def from_files(cls, paths):
        """Merge several saved shard files into one report."""
        records = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                records.extend(json.load(f))
        return cls(records)

    @property
    def metrics(self):
        """Aggregate metrics including the record count `n`."""
        return M.aggregate(self.records)

    def scalars(self, prefix=""):
        """Metrics without `n`, ready to hand to a tracker."""
        return {f"{prefix}{k}": v for k, v in self.metrics.items() if k != "n"}

    def save(self, path):
        """Write the raw records as JSON, creating the parent directory."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False)
        return path

    def subset(self, predicate):
        """A new Report over the records satisfying `predicate`."""
        return Report([r for r in self.records if predicate(r)])

    def print(self):
        """The standard block: overall metrics, then both breakdowns."""
        print("\n" + "=" * WIDTH)
        self.print_overall()
        self.print_by_familiarity()
        self.print_by_intent()
        print("=" * WIDTH)

    def print_overall(self):
        """Corpus-level exact match, normalized EM, token F1 and CER."""
        o = self.metrics
        print(f"OVERALL  n={o['n']}")
        print(f"  exact match     {o['exact_match']:6.2f}%")
        print(f"  normalized EM   {o['normalized_em']:6.2f}%")
        print(f"  token F1        {o['token_f1']:6.2f}%")
        print(f"  CER             {o['cer']:6.2f}%")

    def print_by_familiarity(self):
        """Split scores by whether the gold answer occurs in the train split."""
        print("\nBY ANSWER FAMILIARITY")
        for label, want in [("answer seen in train", True), ("answer unseen", False)]:
            self._print_group(label, self.subset(lambda r: r.get("seen") is want))

    def print_by_intent(self):
        """Split scores by what the question asks for (title, author, ...)."""
        print("\nBY QUESTION INTENT")
        for it in INTENTS:
            self._print_group(it, self.subset(lambda r, it=it: r["intent"] == it))

    def print_prediction_stats(self, top=8):
        """Degenerate-output check: empty rate, lengths, most frequent strings."""
        n = len(self.records) or 1
        empty = sum(1 for r in self.records if not r["pred"].strip())
        print(f"\nempty predictions: {empty} ({100 * empty / n:.1f}%)")
        lens = [len(r["pred"]) for r in self.records] or [0]
        print(f"prediction length chars: mean={sum(lens) / len(lens):.1f} max={max(lens)}")
        print("\nmost frequent predictions:")
        counts = collections.Counter(M.normalize(r["pred"]) for r in self.records)
        for text, count in counts.most_common(top):
            print(f"  {count:6d}  {text[:70]!r}")

    def print_examples(self, limit):
        """Terse question/gold/pred triples for the first `limit` records."""
        if not limit:
            return
        print("\nexamples:")
        for r in self.records[:limit]:
            print(f"  Q: {r['question']}")
            print(f"  gold: {r['gold']!r}")
            print(f"  pred: {r['pred'][:90]!r}\n")

    def print_predictions(self, grouped=False):
        """Wrapped gold/pred pairs for reading, optionally bucketed by image."""
        order = range(len(self.records))
        if grouped:
            order = sorted(order, key=lambda i: self.records[i]["image"])

        last_img = None
        for i in order:
            r = self.records[i]
            if grouped and r["image"] != last_img:
                print(f"\n=== {r['image']} " + "=" * max(0, WRAP - 5 - len(r["image"])))
                last_img = r["image"]
            print(f"\n[{r['intent']}] nEM={r['nem']:.0f} F1={r['f1']:.2f} CER={r['cer']:.2f}"
                  + ("" if grouped else f"  {r['image']}"))
            for label, text in (("Q   ", r["question"]), ("gold", r["gold"]), ("pred", r["pred"])):
                body = textwrap.fill(text or "<empty>", WRAP, subsequent_indent=" " * 6)
                print(f"  {label}: {body}")

    def print_line(self, name):
        """One-line summary, for comparing several runs side by side."""
        a = self.metrics
        print(f"  {name:10s} n={a['n']:4d}  EM={a['exact_match']:5.2f}%  "
              f"nEM={a['normalized_em']:5.2f}%  F1={a['token_f1']:5.2f}%  CER={a['cer']:5.2f}%")

    def _print_group(self, label, group):
        """One breakdown row, skipped when the group is empty."""
        if not len(group):
            return
        a = group.metrics
        print(f"  {label:22s} n={a['n']:6d}  "
              f"nEM={a['normalized_em']:6.2f}%  F1={a['token_f1']:6.2f}%")
