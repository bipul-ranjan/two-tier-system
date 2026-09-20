# data/

Downloaded datasets used to test the cascade. Nothing in this folder is
handwritten — everything here comes from an external source and can be
re-downloaded if deleted (see the commands below), which is exactly why
`*.csv` in this folder is excluded from Git in `.gitignore` — there's no
need to version-control a copy of a public dataset.

## raw/

Sits exactly as downloaded, with no cleaning or modification.

| File | Source | How to get it |
|---|---|---|
| `banking77_test.csv` | Banking77 (Hugging Face) | `python -c "from datasets import load_dataset; ds = load_dataset('PolyAI/banking77'); ds['test'].to_pandas().to_csv('data/raw/banking77_test.csv', index=False)"` |

`src/pipeline.py` reads directly from this folder — if you delete or
move this file, re-run the command above before running the pipeline
again.

## If you add the FinQA dataset later (Tier 2 escalation testing)

FinQA isn't a single CSV — it comes as a cloned GitHub repo. Keep it
outside this `raw/` folder (e.g. a top-level `FinQA/` folder) since its
structure doesn't match the flat-CSV pattern the rest of this folder
uses, and reference its path directly from whichever script needs it.
