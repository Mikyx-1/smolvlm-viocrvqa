"""Merge sharded eval outputs and print the report.

Example:
    python -m viocrvqa.cli.merge_reports 'results/baseline_dev_image_shard*.json'
"""

import argparse
import glob

from ..reporting import Report


def build_parser():
    """Command-line interface of the merge script."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pattern", help="glob, e.g. 'results/baseline_dev_image_shard*.json'")
    ap.add_argument("--examples", type=int, default=0, help="print the first N records")
    return ap


def main():
    """Load every matching shard, then print metrics and prediction stats."""
    args = build_parser().parse_args()

    files = sorted(glob.glob(args.pattern))
    report = Report.from_files(files)
    print(f"merged {len(files)} shards -> {len(report)} records")
    if not len(report):
        raise SystemExit(f"no records matched {args.pattern!r}")

    report.print()
    report.print_prediction_stats()
    report.print_examples(args.examples)


if __name__ == "__main__":
    main()
