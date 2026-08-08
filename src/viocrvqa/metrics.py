"""Metrics for ViOCRVQA evaluation.

Vietnamese needs Unicode normalisation: the same word can be stored composed
(U+1EA5) or decomposed (a + U+0302 + U+0301), and those compare unequal as raw
strings even though they render identically.
"""

import re
import string
import unicodedata

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize(s: str) -> str:
    """NFC-normalise, lowercase, strip punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFC", s).lower().strip()
    s = _PUNCT.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(pred.strip() == gold.strip())


def normalized_em(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = 0
    gold_left = list(g)
    for t in p:
        if t in gold_left:
            gold_left.remove(t)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, gold: str) -> float:
    """Character error rate against the gold answer, capped at 1.0."""
    p, g = normalize(pred), normalize(gold)
    if not g:
        return float(bool(p))
    return min(edit_distance(p, g) / len(g), 1.0)


def aggregate(records):
    """records: list of dicts with keys pred, gold, question, field, seen."""
    if not records:
        return {}
    n = len(records)
    out = {
        "n": n,
        "exact_match": 100 * sum(r["em"] for r in records) / n,
        "normalized_em": 100 * sum(r["nem"] for r in records) / n,
        "token_f1": 100 * sum(r["f1"] for r in records) / n,
        "cer": 100 * sum(r["cer"] for r in records) / n,
    }
    return out
