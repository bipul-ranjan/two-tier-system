# scripts/

One-off setup and utility scripts — things you run once (or occasionally),
as opposed to `src/`, which is the actual pipeline that runs every time
you do research work.

## Why this is separate from src/

`src/` is a *package* — its files import from each other (`from .tier1
import ask_tier1`), which is why they need `python -m src.pipeline` to
run correctly. The scripts in this folder don't import from each other
or from `src/` — they only use external libraries (`datasets`, `git`
via `subprocess`). Because of that, they're run directly, the simpler
way, with no `-m` and no `__init__.py` needed:

```
python scripts/fetch_data.py
```

If you're unsure why that distinction matters, see `src/README.md` —
same underlying concept, just the version of it that happens to need
the extra step.

## fetch_data.py

Downloads everything this project needs as raw data, into `data/raw/`
and a top-level `FinQA/` folder:

- **Banking77** (via Hugging Face `datasets`) → `data/raw/banking77_train.csv`, `data/raw/banking77_test.csv`
- **FinQA** (via `git clone`) → `FinQA/dataset/train.json`, `dev.json`, `test.json`

Run it once, from the project root:
```
python scripts/fetch_data.py
```

Safe to run again later — it checks whether each dataset already
exists first, and skips anything already downloaded rather than
re-fetching or overwriting it. You'll see a "skipping" message instead
of a re-download if you run it twice.

## Adding your own utility scripts later

If you find yourself typing the same multi-step terminal commands
repeatedly for something else (cleaning a dataset, generating a report,
etc.), that's the signal to write a script and drop it in here rather
than continuing to type it by hand — same reasoning as `fetch_data.py`:
a script is reproducible, reviewable in Git, and doesn't rely on you
remembering the exact steps next time.
