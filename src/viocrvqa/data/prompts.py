"""Turning (question, answer) pairs into SmolVLM chat strings."""

from ..config import PROMPT, resolve_prompt


class PromptFormatter:
    """Render one prompt template through a processor's chat template."""

    def __init__(self, processor, template=PROMPT):
        self.processor = processor
        self.template = resolve_prompt(template)
        if "{q}" not in self.template:
            raise ValueError(f"prompt template has no {{q}} placeholder: {self.template!r}")

    def __repr__(self):
        """Identify the formatter by its template."""
        return f"PromptFormatter({self.template!r})"

    def user_message(self, question):
        """The user turn: one image placeholder plus the formatted question."""
        return [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": self.template.format(q=question)},
        ]}]

    def prompt_text(self, question):
        """Chat string ending at the assistant header, ready for generation."""
        return self.processor.apply_chat_template(
            self.user_message(question), add_generation_prompt=True)

    def full_text(self, question, answer):
        """Chat string including the assistant's answer, used as a train target."""
        messages = self.user_message(question) + [
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=False)

    def texts(self, question, answer):
        """Both renderings of one sample: (prompt_text, full_text)."""
        return self.prompt_text(question), self.full_text(question, answer)

    def answer_token_len(self, prompt_text, full_text):
        """Token count of the answer = len(full) - len(prompt).

        The prompt text is always a literal prefix of the full text, so both
        can be tokenized *without* images (the shared image placeholder
        cancels out in the difference) instead of running the image
        processor a second time per sample.
        """
        tok = self.processor.tokenizer
        n_prompt = len(tok(prompt_text, add_special_tokens=False)["input_ids"])
        n_full = len(tok(full_text, add_special_tokens=False)["input_ids"])
        return n_full - n_prompt
