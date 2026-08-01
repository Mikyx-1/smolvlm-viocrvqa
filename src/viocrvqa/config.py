"""Paths, model defaults and named prompt templates.

Every other module imports its defaults from here so that a path or a
template lives in exactly one place.
"""

from pathlib import Path

# .../<repo>/src/viocrvqa/config.py -> <repo>
ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "data"
DEFAULT_DATA_DIR = DATA_ROOT / "data"
SAMPLE_DATA_DIR = DATA_ROOT / "data_sample_0.01"
IMG_DIR_NAME = "bf_all_img"

CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_MAX_NEW_TOKENS = 48

SPLITS = ("train", "dev", "test")

# Named templates so prompt variants can be referred to by name on the CLI.
# Every template must contain the {q} placeholder for the question.
PROMPTS = {
    "instruct": "Trả lời ngắn gọn câu hỏi về hình ảnh này. {q}",
    "raw": "{q}",
    "vi": "Đọc chữ trên ảnh. Trả lời bằng tiếng Việt, thật ngắn. {q}",
}
PROMPT = PROMPTS["instruct"]


def resolve_prompt(name_or_template: str) -> str:
    """Map a template name to its text; pass an inline template through."""
    return PROMPTS.get(name_or_template, name_or_template)
