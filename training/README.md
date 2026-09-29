# training/

Fine-tuning scripts for the Payment Assistant (Phi-3-mini) and Retail
Bank Assistant (Qwen2.5-1.5B). These live in this repo like any other
code, but they **require a GPU and are meant to run on Google Colab**,
not on your local machine — that's the whole reason this is a separate
folder rather than mixed into `src/`.

## Files

| File | Purpose |
|---|---|
| `train_config.py` | Shared LoRA/training settings and the dataset-formatting function both scripts use -- currently `r=32, lora_alpha=32, num_train_epochs=5, learning_rate=1.5e-4` for both models (raised from `r=16, lora_alpha=16, num_train_epochs=3, learning_rate=2e-4`; see note below) |
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

#### Note on hyperparameters and what they confound

`train_config.py`'s `LORA_CONFIG`/`TRAINING_ARGS` were raised (rank/alpha 16→32, epochs 3→5, learning rate 2e-4→1.5e-4) for both models in the same change that added retail's synthetic exception data. That means: **payments' retrain result isolates the hyperparameter effect alone** (its training data did not change), while **retail's retrain result reflects the hyperparameter change and the new data together** (both changed at once). Worth stating plainly in your methodology section, since retail's confidence shift after this retrain cannot be attributed to one cause or the other from this run alone -- a follow-up run holding one of the two fixed would be needed to separate them.

Also worth noting for planning: 5 epochs at rank 32 will take longer per model than the original 3-epoch, rank-16 setup -- budget extra Colab session time accordingly, and keep an eye on the T4's session limits if you're on the free tier.

#### Resuming after a dropped session

Both scripts now checkpoint to Google Drive every 250 steps during training (`payments_checkpoints/` / `retail_checkpoints/` alongside the GGUF output), keeping the 3 most recent. If a session disconnects mid-run, just re-run the same script in a fresh session (same repo clone, same steps) -- it detects the last checkpoint on Drive automatically and resumes from there instead of starting over from step 0. You'll see `Found an existing checkpoint at ... -- resuming from there instead of starting over.` printed if this happens. This is specifically why the checkpoint directory is on Drive and not local Colab disk: a checkpoint on local disk would be wiped by the same disconnect it's meant to protect against.

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

### 8. Retrieve both GGUF files from Google Drive

Both scripts save directly to Google Drive during the run (not to `training/outputs/` or Colab's local disk), so there's nothing to download from the Colab file browser. Find them at:
- `LJMU_Research/two-tier-system-models/payment_assistant.gguf`
- `LJMU_Research/two-tier-system-models/retail_bank_assistant.gguf`

in your Drive. If you have Drive for Desktop set up, right-click each and choose **Mirror files** (or wait for normal sync) to get them onto your local machine.

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
