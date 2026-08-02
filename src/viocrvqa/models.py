"""Loading, configuring and saving the SmolVLM model + processor pair."""

import contextlib
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, PretrainedConfig

from .config import DEFAULT_MODEL
from .data.prompts import PromptFormatter

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
MIN_FREE_MIB = 2048


def resolve_device(requested="auto"):
    """Resolve 'auto' to the emptiest GPU, or CPU when none has room.

    The GPUs here are usually busy with a full-split eval, and a run that
    OOMs a neighbouring job is worse than a slow one.
    """
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    mib, idx = max(
        (torch.cuda.mem_get_info(i)[0] / 2**20, i) for i in range(torch.cuda.device_count()))
    if mib < MIN_FREE_MIB:
        print(f"no GPU with >={MIN_FREE_MIB // 1024} GiB free "
              f"(best: cuda:{idx}, {mib:.0f} MiB); using cpu")
        return "cpu"
    print(f"using cuda:{idx} ({mib:.0f} MiB free)")
    return f"cuda:{idx}"


def checkpoint_dtype(model_id):
    """The dtype a checkpoint records in its config, or None if it records none."""
    try:
        config, _ = PretrainedConfig.get_config_dict(model_id)
    except OSError:
        return None
    # transformers renamed the field; older checkpoints still carry torch_dtype
    return DTYPES.get(config.get("dtype") or config.get("torch_dtype"))


def resolve_dtype(name="auto", device="cuda:0", model_id=None):
    """Map a dtype name to a torch dtype; 'auto' follows the checkpoint.

    A fine-tune saved in fp32 must be *loaded* in fp32. At lr=1e-5 the learned
    per-weight delta is 2.5e-4, the same order as one bf16 step at the median
    weight magnitude (2.4e-4 at |w| = 0.037), so casting to bf16 rounds the
    delta away entirely for 41% of the model -- the checkpoint reproduces 55%
    of its own training answers instead of 98%. The pretrained model records
    bfloat16 and is unaffected, so honouring the config gets both right.

    Falls back to bf16 on GPU and fp32 on CPU when nothing is recorded.
    """
    if name != "auto":
        return DTYPES[name]
    recorded = checkpoint_dtype(model_id) if model_id else None
    if recorded is not None:
        return recorded
    return torch.float32 if str(device).startswith("cpu") else torch.bfloat16


class ModelBundle:
    """A model and its processor, kept together because they must agree.

    Padding side, image splitting and KV caching are properties of the pair,
    not of either half, so they are set and restored here.
    """

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    @classmethod
    def load(cls, model_id=DEFAULT_MODEL, device="cuda:0", dtype="auto",
             padding_side="left", image_splitting=True, gradient_checkpointing=False,
             training=False):
        """Load a checkpoint onto `device` and configure it for train or eval."""
        device = resolve_device(device)
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = padding_side
        if not image_splitting:
            processor.image_processor.do_image_splitting = False

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=resolve_dtype(dtype, device, model_id),
            # flash_attention_2 is faster but not installed here; sdpa works everywhere
            attn_implementation="sdpa",
        ).to(device)

        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        model.train(training)
        return cls(model, processor, device)

    @classmethod
    def for_training(cls, model_id=DEFAULT_MODEL, device="cuda:0", dtype="float32",
                     image_splitting=True, gradient_checkpointing=True):
        """Load right-padded (so labels line up), in fp32, and in train mode.

        The master weights must be fp32. AdamW's per-parameter step is ~lr
        regardless of gradient scale, and at lr=1e-5 that is well below the
        bf16 resolution at typical weight magnitudes (median |w| = 0.037,
        where one bf16 step is 2.4e-4), so `w + update` rounds straight back
        to `w` for 92% of the model and training silently stalls. Trainer
        wraps the forward pass in bf16 autocast, so the speed and activation
        memory of bf16 are kept -- only the accumulation is fp32.
        """
        return cls.load(model_id, device=device, dtype=dtype, padding_side="right",
                        image_splitting=image_splitting,
                        gradient_checkpointing=gradient_checkpointing, training=True)

    @classmethod
    def for_inference(cls, model_id=DEFAULT_MODEL, device="cuda:0", dtype="auto",
                      image_splitting=True):
        """Load left-padded (required for batched generation) and in eval mode.

        The default dtype follows the checkpoint: bf16 for the pretrained
        model, fp32 for a fine-tune, because bf16 rounds most of a fine-tune's
        weight delta away. See resolve_dtype.
        """
        return cls.load(model_id, device=device, dtype=dtype, padding_side="left",
                        image_splitting=image_splitting, training=False)

    @property
    def n_params(self):
        """Total parameter count."""
        return sum(p.numel() for p in self.model.parameters())

    @property
    def pad_token_id(self):
        """Pad token used to fill batched generation inputs."""
        return self.processor.tokenizer.pad_token_id

    def formatter(self, template):
        """A PromptFormatter bound to this bundle's processor."""
        return PromptFormatter(self.processor, template)

    def save(self, path):
        """Write model + processor to `path` so it can be re-loaded by id."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        return path

    @contextlib.contextmanager
    def teacher_forcing_mode(self):
        """Switch to the layout the label mask assumes, then restore it.

        VQACollator lines labels up against right-padded sequences, so scoring
        likelihood needs the training layout even on a bundle that was loaded
        for generation. The mirror image of generation_mode().
        """
        padding_side_before = self.processor.tokenizer.padding_side
        was_training = self.model.training
        self.model.eval()
        self.processor.tokenizer.padding_side = "right"
        try:
            yield self
        finally:
            self.processor.tokenizer.padding_side = padding_side_before
            self.model.train(was_training)

    @contextlib.contextmanager
    def generation_mode(self):
        """Switch to inference settings for the block, then restore them.

        Training turns KV caching off and pads on the right; generation needs
        the opposite of both.
        """
        was_training = self.model.training
        use_cache_before = self.model.config.use_cache
        padding_side_before = self.processor.tokenizer.padding_side
        self.model.eval()
        self.model.config.use_cache = True
        self.processor.tokenizer.padding_side = "left"
        try:
            yield self
        finally:
            self.model.config.use_cache = use_cache_before
            self.processor.tokenizer.padding_side = padding_side_before
            self.model.train(was_training)
