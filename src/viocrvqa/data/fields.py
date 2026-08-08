"""The paper's five question fields, looked up per question template.

ViOCRVQA questions come from ~240 hand-written templates, each belonging to one
field. The field is a construction-time label, so it is looked up by exact
question string rather than guessed from the wording -- `viocrvqa.cli.label_fields`
recovers the mapping and writes field_labels.json beside this module.

Questions absent from the mapping (a different dataset, a hand-typed question)
fall back to `UNKNOWN` so they are visible in a report instead of being silently
folded into a real field.
"""

import json
from pathlib import Path

# ordered as the paper reports them, translator before title so a lookup miss
# never matters more than the order of printed rows
FIELDS = ["title", "author", "publisher", "translator", "genre"]

# Section 6.3: genres "must synthesize information from the remaining fields",
# so the paper's headline table excludes them and reports them separately
MAIN_FIELDS = ["title", "author", "publisher", "translator"]

UNKNOWN = "unknown"

FIELD_LABELS_PATH = Path(__file__).with_name("field_labels.json")

_labels = None


def field_labels():
    """The {question template: field} mapping, loaded once."""
    global _labels
    if _labels is None:
        with open(FIELD_LABELS_PATH, encoding="utf-8") as f:
            _labels = json.load(f)
    return _labels


def field(question):
    """The paper's field for one question, or UNKNOWN if it is not a template."""
    return field_labels().get(question.strip(), UNKNOWN)
