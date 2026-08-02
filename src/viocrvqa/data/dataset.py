"""torch Dataset and collator for supervised fine-tuning."""

from torch.utils.data import Dataset


class VQADataset(Dataset):
    """(image, question, answer) triples, decoded lazily from disk."""

    def __init__(self, samples, corpus, blank_image=False):
        self.samples = samples
        self.corpus = corpus
        self.blank_image = blank_image  # grey images; the control for scoring

    def __len__(self):
        """Number of samples in the split."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Decode one sample's image and pair it with its question/answer."""
        s = self.samples[idx]
        return {
            "image": self.corpus.open_image(s, blank=self.blank_image),
            "question": s["question"],
            "answer": s["gold"],
        }


class VQACollator:
    """Batch samples and mask the loss down to the assistant's answer tokens.

    Requires a right-padding tokenizer so that the real tokens of every
    sequence start at index 0 and the label mask lines up with them.
    """

    def __init__(self, formatter, device):
        self.formatter = formatter
        self.processor = formatter.processor
        self.device = device

    def __call__(self, batch):
        """Collate into processor inputs plus answer-only `labels`, on device."""
        images, prompt_texts, full_texts = [], [], []
        for ex in batch:
            p_text, f_text = self.formatter.texts(ex["question"], ex["answer"])
            images.append([ex["image"]])
            prompt_texts.append(p_text)
            full_texts.append(f_text)

        inputs = self.processor(
            text=full_texts, images=images, return_tensors="pt", padding=True)
        inputs["labels"] = self._build_labels(inputs, prompt_texts, full_texts)
        return inputs.to(self.device)

    def _build_labels(self, inputs, prompt_texts, full_texts):
        """Copy input_ids, then mask padding and everything before the answer."""
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        for i, (p_text, f_text) in enumerate(zip(prompt_texts, full_texts)):
            answer_len = self.formatter.answer_token_len(p_text, f_text)
            seq_len = int(inputs["attention_mask"][i].sum())
            labels[i, : seq_len - answer_len] = -100
        return labels
