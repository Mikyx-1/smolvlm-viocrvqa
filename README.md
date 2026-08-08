# SmolVLM on ViOCRVQA

Full fine-tuning (no LoRA/quantization) and evaluation of SmolVLM-256M/500M on
[ViOCRVQA](https://huggingface.co/datasets/huyhuy123/ViOCRVQA), a Vietnamese
book-cover VQA dataset. Loss is computed only on the assistant's answer tokens;
the image, question and template tokens are masked out.

## Layout

```
src/viocrvqa/
  config.py        paths, model defaults, named prompt templates
  metrics.py       EM / normalized EM / token F1 / CER
  data/
    corpus.py      VQACorpus       — splits, images, train answer vocabulary
    fields.py      the paper's five question fields, per question template
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

# recover the paper's five question fields (already checked in; --check vs Table 3)
python -m viocrvqa.cli.label_fields --check

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
- Precision matters at both ends. At `lr=1e-5` the per-weight update, and the
  total learned delta, are the same order as one bf16 step at typical weight
  magnitudes. So training keeps fp32 master weights (bf16 rounds ~92% of the
  updates away and the run silently stalls), and `--dtype auto` loads a
  checkpoint in whatever dtype it was saved in (casting a fp32 fine-tune to
  bf16 erases the delta for 41% of the model). Only the compute is bf16, via
  autocast.
- Image splitting turns one image into up to 17 tiles, so memory varies wildly
  between equally sized batches; `BatchGenerator` halves the batch on OOM
  instead of forcing a worst-case batch size.
- Vietnamese needs Unicode normalisation before comparison — the same word can
  be stored composed or decomposed and compares unequal as a raw string.
- The paper reports EM and F1 per question field (title, author, publisher,
  translator, genre), but the release carries no field labels. The field is a
  property of the question template, not of its wording, so it is recovered
  rather than guessed: a book is asked exactly one template per field, so two
  templates of the same field never share an image while templates of different
  fields co-occur constantly. Grouping the 240 templates on that signature
  yields exactly five clusters matching Table 3's proportions to within 0.1
  points. `cli/label_fields.py --check` reproduces the mapping and the
  comparison; `data/field_labels.json` is the result the report reads.
- The paper's headline table (8) omits genre and reports it separately (9), so
  the report prints the average over the four main fields as well as over all
  five. Its numbers are on `test`, which is unlabelled here, so `dev` is the
  closest comparable split — same size and construction, different books.
