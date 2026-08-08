"""Recover the paper's five question fields for every question template.

ViOCRVQA was built by writing ~240 question templates, each belonging to one
field (author, title, publisher, translator, genre), then asking each book
exactly one template per field it has. The field label is therefore a property
of the template, not something recoverable from the question's wording -- but
the construction leaves a signature that recovers it exactly:

  two templates of the SAME field never appear on the same image (a book gets
  one question per field), while templates of DIFFERENT fields co-occur all
  the time. So each field is a set of templates whose co-occurrence
  neighbourhoods are identical: "everything except my own field".

Grouping on that yields exactly five clusters, which are then named from their
answer distributions (publishers are a closed set prefixed "nhà xuất bản",
genres are a closed set of ~170, titles are near-unique, author and translator
are person names distinguished by how many books have one).

The result ships as data/field_labels.json; rerun this only to regenerate it.

Example:
    python -m viocrvqa.cli.label_fields
    python -m viocrvqa.cli.label_fields --check   # verify against Table 3
"""

import argparse
import collections
import itertools
import json

from ..config import DEFAULT_DATA_DIR
from ..data.fields import FIELDS, FIELD_LABELS_PATH

# Table 3 of the paper, over the whole dataset; used only to sanity-check.
PAPER_TABLE_3 = {"author": 27881, "title": 28283, "publisher": 28283,
                 "translator": 11051, "genre": 28283}

# a template pair may co-occur by chance in a handful of images, and two
# templates of one field share all but their own field's neighbours, so both
# comparisons need slack rather than exact equality
PROFILE_TOLERANCE = 0.15


def build_parser():
    """Command-line interface of the field-recovery script."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--splits", nargs="+", default=["train", "dev"],
                    help="labelled splits to learn from (test has no answers)")
    ap.add_argument("--out", default=str(FIELD_LABELS_PATH))
    ap.add_argument("--check", action="store_true",
                    help="compare field proportions against the paper's Table 3")
    return ap


def read_annotations(data_dir, splits):
    """Yield every (image key, question, answer) triple of the given splits."""
    for split in splits:
        with open(f"{data_dir}/{split}.json", encoding="utf-8") as f:
            data = json.load(f)
        for a in data["annotations"]:
            yield (split, a["image_id"]), a["question"], a["answers"][0]


def cluster_templates(triples):
    """Group templates into fields by their image co-occurrence profile."""
    by_image = collections.defaultdict(set)
    answers = collections.defaultdict(list)
    for key, question, answer in triples:
        by_image[key].add(question)
        answers[question].append(answer)

    co = collections.defaultdict(collections.Counter)
    for questions in by_image.values():
        for a, b in itertools.combinations(questions, 2):
            co[a][b] += 1
            co[b][a] += 1

    templates = sorted(answers)
    neighbours = {q: set(co[q]) for q in templates}
    parent = {q: q for q in templates}

    def root(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    for a, b in itertools.combinations(templates, 2):
        if co[a][b]:
            continue  # they share an image, so they cannot be the same field
        shared, differing = neighbours[a] | neighbours[b], neighbours[a] ^ neighbours[b]
        if len(differing) <= PROFILE_TOLERANCE * len(shared):
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[ra] = rb

    clusters = collections.defaultdict(list)
    for q in templates:
        clusters[root(q)].append(q)
    return list(clusters.values()), answers


def name_cluster(members, answers, total):
    """Identify which of the paper's five fields a cluster of templates is."""
    answered = [a for q in members for a in answers[q]]
    unique = set(answered)
    if sum(a.startswith("nhà xuất bản") for a in answered) / len(answered) > 0.9:
        return "publisher"          # closed set of 176 publishers
    if len(unique) < 500:
        return "genre"              # closed set of ~32 genres
    if len(unique) / len(answered) > 0.9:
        return "title"              # essentially one title per book
    # author and translator are both person names; only ~39% of books have a
    # translator, so the translator cluster is far smaller
    return "translator" if len(answered) < 0.6 * total else "author"


def label_templates(data_dir, splits):
    """Return {question template: field} for every template in the dataset."""
    clusters, answers = cluster_templates(read_annotations(data_dir, splits))
    if len(clusters) != len(FIELDS):
        raise SystemExit(f"expected {len(FIELDS)} fields, recovered {len(clusters)}; "
                         f"the data may differ from the released ViOCRVQA")
    biggest = max(sum(len(answers[q]) for q in c) for c in clusters)
    labels = {}
    for cluster in clusters:
        field = name_cluster(cluster, answers, biggest)
        for q in cluster:
            labels[q] = field
    if set(labels.values()) != set(FIELDS):
        raise SystemExit(f"clusters did not map onto the five fields: "
                         f"{sorted(set(labels.values()))}")
    return labels


def print_check(labels, data_dir, splits):
    """Print recovered field proportions beside the paper's Table 3."""
    counts = collections.Counter()
    for split in ["train", "dev", "test"]:
        with open(f"{data_dir}/{split}.json", encoding="utf-8") as f:
            for a in json.load(f)["annotations"]:
                counts[labels[a["question"]]] += 1
    ours, paper = sum(counts.values()), sum(PAPER_TABLE_3.values())
    print(f"\n{'field':12s} {'recovered':>10s} {'share':>7s}   "
          f"{'paper':>10s} {'share':>7s}")
    for field in FIELDS:
        print(f"{field:12s} {counts[field]:10d} {100 * counts[field] / ours:6.2f}%   "
              f"{PAPER_TABLE_3[field]:10d} {100 * PAPER_TABLE_3[field] / paper:6.2f}%")
    print(f"{'total':12s} {ours:10d}           {paper:10d}")


def main():
    """Recover the field of every template and save the mapping."""
    args = build_parser().parse_args()
    labels = label_templates(args.data_dir, args.splits)

    counts = collections.Counter(labels.values())
    print(f"{len(labels)} templates -> "
          + ", ".join(f"{counts[f]} {f}" for f in FIELDS))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"wrote {args.out}")

    if args.check:
        print_check(labels, args.data_dir, args.splits)


if __name__ == "__main__":
    main()
