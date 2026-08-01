# SmolVLM on ViOCRVQA

Full fine-tuning (no LoRA/quantization) and evaluation of SmolVLM-256M/500M on
[ViOCRVQA](https://huggingface.co/datasets/huyhuy123/ViOCRVQA), a Vietnamese
book-cover VQA dataset. Loss is computed only on the assistant's answer tokens;
the image, question and template tokens are masked out.

## Layout

```
src/viocrvqa/
  config.py        paths, model defaults, named prompt templates
  metrics.py       EM / normalized EM / token F1 / CER, question intents
  data/
    corpus.py      VQACorpus       — splits, images, train answer vocabulary
    prompts.py     PromptFormatter — chat templating, answer token length
    dataset.py     VQADataset, VQACollator — batching + answer-only labels
  models.py        ModelBundle     — load/save, device & dtype, generation mode
  generation.py    BatchGenerator  — batched decoding, halves the batch on OOM
  evaluation.py    Evaluator, EvalConfig, ProgressPrinter
  reporting.py     Report          — every printed breakdown, JSON save/merge
  training.py      Trainer, TrainingConfig
  tracking.py      RunTracker      — wandb logging, or a no-op
  cli/             one module per command (see below)
data/              datasets: data/data (full), data/data_sample_<f> (subsampled)
checkpoints/       fine-tuned models, one dir per epoch
results/           saved eval records
scratch/           throwaway experiments; legacy/ holds the pre-refactor scripts
```

## Setup

```bash
pip install -e .            # or: export PYTHONPATH=src
```

## Commands

Run from the repository root; every command supports `--help`.

```bash
# dataset (6.5 GB) and a 1% sample for fast iteration
python -m viocrvqa.cli.download_data
python -m viocrvqa.cli.split_data --fraction 0.01

# fine-tune, with a dev-set check and a checkpoint after each epoch
python -m viocrvqa.cli.train --data-dir data/data_sample_0.01 --epochs 3
python -m viocrvqa.cli.train --eval-every 1 --eval-limit 200 --no-wandb

# score a split (zero-shot baseline, or any checkpoint)
python -m viocrvqa.cli.eval_baseline --limit 200
python -m viocrvqa.cli.eval_baseline --blank-image             # memorisation floor
python -m viocrvqa.cli.eval_baseline --shard 0 --num-shards 3  # one GPU per shard
python -m viocrvqa.cli.eval_baseline --model checkpoints/SmolVLM-256M-Instruct/epoch3
python -m viocrvqa.cli.merge_reports 'results/baseline_dev_image_shard*.json'

# read predictions, compare prompt templates
python -m viocrvqa.cli.sample --by-image 4
python -m viocrvqa.cli.sample --compare instruct raw vi

# one-off inference on any image or URL
python -m viocrvqa.cli.infer -i cover.jpg -p "Tên cuốn sách này là gì?"
```

After `pip install -e .` the same commands exist as `viocrvqa-train`,
`viocrvqa-eval`, `viocrvqa-sample`, `viocrvqa-infer`, `viocrvqa-merge-reports`,
`viocrvqa-download-data`, `viocrvqa-split-data`.

## Library use

```python
from viocrvqa.data.corpus import VQACorpus
from viocrvqa.evaluation import EvalConfig, Evaluator
from viocrvqa.models import ModelBundle
from viocrvqa.reporting import Report

corpus = VQACorpus("data/data_sample_0.01")
bundle = ModelBundle.for_inference("HuggingFaceTB/SmolVLM-256M-Instruct")
evaluator = Evaluator(bundle, corpus, bundle.formatter("instruct"),
                      EvalConfig(batch_size=16), seen_vocab=corpus.answer_vocab())
Report(evaluator.evaluate(corpus.load("dev", limit=200))).print()
```

## Notes

- The dataset's `test` split is unlabelled (every answer is the literal string
  "your answer"), so `dev` is the evaluation set.
- Training pads on the right so labels line up with real tokens; generation
  pads on the left and needs KV caching. `ModelBundle.generation_mode()` flips
  between the two and restores the previous state afterwards, so the same
  `Evaluator` can be used mid-training.
- Image splitting turns one image into up to 17 tiles, so memory varies wildly
  between equally sized batches; `BatchGenerator` halves the batch on OOM
  instead of forcing a worst-case batch size.
- Vietnamese needs Unicode normalisation before comparison — the same word can
  be stored composed or decomposed and compares unequal as a raw string.
