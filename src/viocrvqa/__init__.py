"""SmolVLM fine-tuning and evaluation on the ViOCRVQA dataset.

Layout:
    config      paths, model/prompt defaults
    metrics     string metrics (EM, nEM, token F1, CER)
    data/       corpus loading, prompt formatting, torch Dataset + collator
    models      ModelBundle: loading, device/dtype, save, generation mode
    generation  BatchGenerator: batched greedy decoding with OOM back-off
    evaluation  Evaluator: predict a split and score every sample
    reporting   Report: every printed breakdown of a record list
    training    Trainer: gradient-accumulating fine-tune loop
    tracking    RunTracker: wandb logging, or a no-op
    cli/        command-line entry points (python -m viocrvqa.cli.<name>)
"""
