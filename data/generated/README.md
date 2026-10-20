# data/generated/

Fresh customer questions made with a separate small model, so a test run does not keep drawing from the same fixed pool of 16,000
questions (`data/synthetic/`).

## The three kinds of file here

| File | Made by | What happens to it |
|---|---|---|
| `current_batch.csv` | `python -m src.pipeline N --fresh` | **Deleted and replaced by every fresh run.** The pipeline only ever deletes this file |
| `seen_questions.txt` | The same fresh runs | A list of one-way fingerprints (not the questions) of every question used, so a deleted question cannot come back. Never deleted by a run |
| `generated_questions_<date-time>.csv` (or any name you give with `--out`) | `python scripts/generate_questions.py` | **Never touched by a pipeline run.** Kept until you delete it |

`current_batch.csv` and `seen_questions.txt` are local working files and are not committed. A batch you make by hand can be committed if you want a
result to be repeatable.

## A batch can be a little short

The request types for the whole batch are drawn first (the pool's own mix), and each is retried up to six times. If the model cannot produce a type within that many tries, the slot is left empty and named in the summary, so a batch of 100 may hold 97. It is never padded with easier types, which would quietly shift the mix away from the pool's.

## What a batch file holds

The columns `instruction`, `category` and `intent` (the three the pipeline needs) plus `is_exception`, `generator_model`, `prompt_version`,
`seed`, `style`, `label_scope` (`unit`, `category` or `skipped`), `label_method` (how the judge was asked), `label_model` (the model that did the label check), `label_check` and `nearest_similarity`, so every question can be traced to the model, prompt and seed that made it.

## What happens to the questions of a deleted batch

The pipeline logs a filtered copy of every question it asks, so `results/logs/results_history.csv` remains the record of what was asked. The batch
seed is in `question_source` (for example `generated:qwen2.5:7b:seed48213`), so passing `--seed 48213` with the same model and prompt repeats the batch.
That repeat can differ slightly if the fingerprint list had blocked a candidate the first time. To keep the original text of a batch, make it with
`scripts/generate_questions.py --out <a name you choose>` instead.

The questions are synthetic, a small model checks their labels, and the files hold no customer data.
