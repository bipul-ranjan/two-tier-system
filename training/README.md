# training/

Fine-tuning scripts for the Payment Assistant (Phi-3-mini) and Retail
Bank Assistant (Qwen2.5-1.5B). These live in this repo like any other
code, but they **require a GPU and are meant to run on Google Colab**,
not on your local machine — that's the whole reason this is a separate
folder rather than mixed into `src/`.

## Files

| File | Purpose |
|---|---|
| `train_config.py` | Shared LoRA/training settings and the dataset-formatting function both scripts use |
| `train_payments.py` | Fine-tunes Phi-3-mini on `data/raw/bitext_payments.jsonl` |
| `train_retail.py` | Fine-tunes Qwen2.5-1.5B on `data/raw/bitext_retail_bank.jsonl` + `data/raw/retail_exception_synthetic.jsonl` |
| `requirements.txt` | Training-only dependencies (unsloth, trl, etc.) — separate from the main project's `requirements.txt` since these are GPU-specific and not needed for day-to-day pipeline runs |

## Full Colab workflow

### 1. Open a new Colab notebook with a GPU

**Runtime → Change runtime type → T4 GPU**

### 2. Store your GitHub token as a Colab secret (not in a cell)

Click the key icon in the left sidebar → **Add new secret** → name it `GITHUB_TOKEN`, paste your GitHub Personal Access Token as the value, toggle notebook access on.

This keeps the token out of the notebook's actual cell contents — important since notebooks get saved with their output, and a token pasted directly into a cell would end up sitting in your Colab/Drive history in plain text.

### 3. Clone this repo

```python
from google.colab import userdata
token = userdata.get('GITHUB_TOKEN')

!git clone https://{token}@github.com/YOUR_USERNAME/two-tier-system.git
%cd two-tier-system
```

### 4. Install dependencies

```python
!pip install -r training/requirements.txt
```

### 5. Regenerate the training data

The actual data files aren't stored in Git (see the main `.gitignore` — same reasoning as `data/raw/` throughout this project: reproducible from code, not worth version-controlling). Regenerate them fresh in Colab:

```python
!python scripts/fetch_bitext_banking.py
```

### 6. Train the Payment Assistant

```python
%cd training
!python train_payments.py
```

### 7. Restart the runtime before training the second model

**Runtime → Restart runtime** — clears the T4's memory from the first model. Skipping this risks an out-of-memory error on the second training run.

After restarting, repeat steps 3–5 (clone, install, regenerate data) since a runtime restart clears the whole environment, then generate the retail exception training data (see below), then:

```python
!python scripts/generate_exception_training_data.py
%cd training
!python train_retail.py
```

#### Why that extra step: retail_bank's exception-scenario data

The real Bitext dataset has no fraud/hardship/vulnerable-customer category at all — only the "normal" categories (ACCOUNT/LOAN/PASSWORD/CONTACT/FIND). Without this step, retail-bank-assistant never sees anything resembling those scenarios during training, which was the main reason RETAIL_EXCEPTION scored the lowest confidence of any category in pipeline runs. `scripts/generate_exception_training_data.py` reuses the same synthetic templates the pipeline already uses for testing, reformatted as training data, and writes `data/raw/retail_exception_synthetic.jsonl` — deduplicated against every pipeline test query, so training data can never leak into a test run. `train_retail.py` now trains on this file plus the real data, combined. This file is also gitignored (`data/raw/*.jsonl`) like the rest of `data/raw/` — it's regenerated fresh here, same as everything else in this section, not committed. It needs `data/synthetic/synthetic_bitext_*.csv` to exist for that dedup check, which it does right after cloning (unlike `data/raw/`, `data/synthetic/` **is** committed to this repo).

### 8. Download both GGUF files

Both scripts save their output to `training/outputs/*.gguf`. Use the Colab file browser (left sidebar, folder icon) to download:
- `training/outputs/payment_assistant.gguf`
- `training/outputs/retail_bank_assistant.gguf`

## What goes back into Git, and what doesn't

**Committed to this repo:** `train_config.py`, `train_payments.py`, `train_retail.py`, `requirements.txt`, this README — the actual training *logic*, which is small and worth version-controlling.

**Never committed** (see `.gitignore`): anything in `training/outputs/` — GGUF model files are large binaries (often 1-4GB even quantized), which Git and GitHub are not built to store efficiently, and GitHub actively rejects files over 100MB. The trained models live on your local machine (via Ollama) and, optionally, wherever you choose to archive them for your dissertation submission — not in Git history.

## After downloading: updating local Ollama

See the main project README, or ask directly — the short version:
```
ollama create payment-assistant -f Modelfile       # Modelfile contains: FROM ./payment_assistant.gguf
ollama create retail-bank-assistant -f Modelfile   # same pattern, other file
```
Then update `src/config.py`'s `BUSINESS_UNITS` model names to `payment-assistant` / `retail-bank-assistant`.
