"""Batched decoding that survives the memory spikes of image splitting."""

import torch

from .config import DEFAULT_MAX_NEW_TOKENS


class BatchGenerator:
    """Decode a batch of (images, prompt) pairs, halving the batch on OOM.

    Image splitting turns one image into up to 17 tiles depending on its
    resolution, so peak memory varies wildly between batches of the same
    size. Rather than tune the batch size for the worst case, back off when
    it happens.
    """

    def __init__(self, bundle, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, temperature=0.0):
        self.bundle = bundle
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, images, texts):
        """Return one decoded string per prompt, recursively splitting on OOM."""
        try:
            return self._generate_once(images, texts)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(texts) == 1:
                print("  OOM on a single sample; emitting empty prediction", flush=True)
                return [""]
            mid = len(texts) // 2
            print(f"  OOM at batch {len(texts)}, splitting", flush=True)
            return (self.generate(images[:mid], texts[:mid])
                    + self.generate(images[mid:], texts[mid:]))

    def _generate_once(self, images, texts):
        """One forward pass: encode, generate, and decode only the new tokens."""
        processor = self.bundle.processor
        inputs = processor(text=texts, images=images or None,
                           return_tensors="pt", padding=True).to(self.bundle.device)
        with torch.inference_mode():
            out = self.bundle.model.generate(
                **inputs,
                pad_token_id=self.bundle.pad_token_id,
                **self._gen_kwargs(),
            )
        return processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _gen_kwargs(self):
        """Greedy by default; sampling only when a temperature was asked for."""
        kwargs = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            kwargs.update(do_sample=False)
        return kwargs
