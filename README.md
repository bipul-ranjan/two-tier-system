# scripts/

One-off setup and data-preparation scripts -- things you run once (or
occasionally), as opposed to `src/`, which is the actual pipeline that
runs every time you do research work.

## Why these are separate from src/

`src/` is a *package* -- its files import from each other (`from .tier1
import ask_tier1`), which is why they need `python -m src.pipeline` to
run correctly. The scripts in this folder don't import from each other
or from `src/` -- they only use external libraries (`datasets`, `git`
via `subprocess`, `pandas`). Because of that, they're run directly, the
simpler way, with no `-m` and no `__init__.py` needed:

```
python scripts/fetch_data.py
```

If you're unsure why that distinction matters, see `src/README.md` --
same underlying concept, just the version that happens to need the
extra step.

## The three scripts, in the order you'd normally run them

### 1. fetch_data.py

Downloads the datasets used to test the two-tier cascade itself:

- **Banking77** (via the `mteb/banking77` mirror on Hugging Face) → `data/raw/banking77_train.csv`, `data/raw/banking77_test.csv`
- **FinQA** (via `git clone`) → `FinQA/dataset/train.json`, `dev.json`, `test.json`

```
python scripts/fetch_data.py
```

Note: uses the `mteb/banking77` mirror rather than the original
`PolyAI/banking77` repo -- the original currently fails with `Dataset
scripts are no longer supported`, since its repo uses a legacy loading
script format Hugging Face's `datasets` library (v4.0+) no longer runs
for security reasons. `mteb/banking77` is the same underlying data
(confirmed: same 3,080-example test split, same 77 labels), hosted
natively in modern Parquet format.

### 2. convert_finqa_to_csv.py

FinQA's raw files are deeply nested JSON (`pre_text`, `post_text`,
`table`, and a `qa` sub-object). This flattens each record into one CSV
row, so it can be loaded and filtered the same way as everything else
in `data/raw/`:

```
python scripts/convert_finqa_to_csv.py
```

Produces `data/raw/finqa_train.csv`, `finqa_dev.csv`, `finqa_test.csv`.
Requires `FinQA/dataset/` to already exist -- run `fetch_data.py` first.
The `table` and `gold_inds` columns are stored as JSON strings inside
the CSV (so nested structure survives), and deserialize back cleanly
with `json.loads()` if you need the original structure later.

### 3. fetch_bitext_banking.py

Downloads the Bitext retail-banking chatbot dataset (25,545 real
instruction/response pairs) and splits it by category into your two
business units -- this is the training data for `training/train_payments.py`
and `training/train_retail.py`, not for the main cascade pipeline:

```
python scripts/fetch_bitext_banking.py
```

Produces, in both JSONL and CSV:
- `data/raw/bitext_payments.jsonl` / `.csv` (CARD, TRANSFER, ATM, FEES categories)
- `data/raw/bitext_retail_bank.jsonl` / `.csv` (ACCOUNT, LOAN, PASSWORD, CONTACT, FIND categories)

License: CDLA-Sharing 1.0 -- free to use, requires attribution, and any
derivative data you share must stay under the same license. Cite:
Bitext Innovations, "Bitext-retail-banking-llm-chatbot-training-dataset", 2024.

## All three are safe to re-run

Each script checks whether its output already exists before doing any
work, and skips with a message rather than re-downloading or
overwriting. Running any of them twice by mistake costs nothing.

## Adding your own utility scripts later

If you find yourself typing the same multi-step terminal commands
repeatedly for something else, that's the signal to write a script and
drop it in here rather than continuing to type it by hand -- same
reasoning as all three scripts above: reproducible, reviewable in Git,
and doesn't rely on you remembering the exact steps next time.
