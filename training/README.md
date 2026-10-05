# training/

Fine-tuning scripts for the two local models: the **Payment Assistant** (Phi-3-mini) and the **Retail Bank Assistant**
(Qwen2.5-1.5B). They need a GPU, so they are meant to run on **Google Colab**, not on your own computer. Training is
**optional**: the system also runs on the base models (see the main README, Step 5, Path A), but the fine-tuned models
are the ones the project was measured with.

## Files

| File | Purpose |
|---|---|
| `train_config.py` | The settings and helpers both scripts share: LoRA `r=32, lora_alpha=32`, `num_train_epochs=5`, `learning_rate=1.5e-4`, batches of 2 collected 4 times, checkpoints every 250 steps; and the text format each training example is turned into |
| `train_payments.py` | Fine-tunes Phi-3-mini on `data/raw/bitext_payments.jsonl` (10,902 examples) |
| `train_retail.py` | Fine-tunes Qwen2.5-1.5B on `data/raw/bitext_retail_bank.jsonl` (14,643 examples) plus `data/raw/retail_exception_synthetic.jsonl` (4,200 synthetic exception examples), 18,843 in all |
| `requirements.txt` | Training-only packages (unsloth, trl, transformers, ...). Kept apart from the main `requirements.txt` because they are GPU-specific and not needed to run the system |

## What training does, in plain words

The base model is loaded in a compressed 4-bit form and a small set of extra adjustable weights (a "LoRA adapter", about 1.5% of
the model for Phi-3-mini and 2.3% for Qwen) is trained on the example questions and answers. Each example is turned into
`### Instruction: You are the <assistant name> for a retail bank. ### Query: <question> ### Response: <answer>`, which is why
Tier 1's prompt names the assistant the same way. When training ends, the model is exported as one `.gguf` file (4-bit,
`q4_k_m`) that Ollama can load.

## How long it takes

| Run | GPU | Time | Notes |
|---|---|---|---|
| Version 2, payments (10,902 examples, 5 epochs, 6,815 steps) | Colab L4 | about 3 hours 44 minutes | measured |
| Version 2, retail bank (18,843 examples, 5 epochs, 11,780 steps) | Colab L4 | about 5.5 hours | measured; resumed from a checkpoint after one disconnect |
| Version 1 (rank 16, 3 epochs), either model | Colab L4 | about 2.5 hours | estimated from the run's pace |
| Version 1, payments | Colab T4 (free tier) | projected 5.5 to 6 hours | a first attempt was stopped at 36% after two hours, so the L4 is recommended |

Colab bills its paid tiers in "compute units" per hour. Rates change, so read them from Colab's own screen. On the author's
account the L4 used about 1.5 units an hour, which put the cost of both version 2 runs at roughly 14 units.

## The full Colab workflow

**Run every command from the project root** (the folder that contains `training/`, `scripts/` and `data/`). The scripts read
`data/raw/...` relative to that folder, and they will stop with a message if the data is missing. Each block is one Colab cell.

### 1. Open a notebook with a GPU

Go to colab.research.google.com, click **New notebook**, then **Runtime, Change runtime type, L4 GPU** (T4 works but is about
twice as slow). Keep the browser tab open and your computer awake while training.

### 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Approve the pop-up. Colab's own disk is wiped whenever the session ends **or you change the GPU type**; your Drive is not. That is
why the models and checkpoints are saved straight to Drive, and why the repository is cloned onto it too.

### 3. Clone the project onto Drive

```python
%cd /content/drive/MyDrive
!git clone https://github.com/bipul-ranjan/two-tier-system.git
%cd /content/drive/MyDrive/two-tier-system
```

If the folder already exists from an earlier session, run `!git pull` inside it instead of cloning. For a **private** copy of the
repository, store a GitHub personal access token as a Colab Secret (the key icon in the left bar, add a secret named `GITHUB_TOKEN`,
switch notebook access on) and clone with it, so the token never appears in a cell, which Colab saves with its output:

```python
from google.colab import userdata
token = userdata.get('GITHUB_TOKEN')
!git clone https://{token}@github.com/YOUR_USERNAME/two-tier-system.git
```

### 4. Install the training packages

```python
!pip install -r training/requirements.txt
```

This takes a few minutes and prints a lot. If a later step complains that a package cannot be imported, use
**Runtime, Restart runtime** once and run steps 2, 3 and 5 again (not the install).

### 5. Fetch the training data

```python
!python scripts/fetch_bitext_banking.py
```

The data files are not stored in Git (the main `.gitignore` excludes `data/raw/`, because they can be regenerated). This
downloads the Bitext dataset and splits it into the payments and retail-bank files.

### 6. Train the Payment Assistant

```python
!python training/train_payments.py
```

You will see a stream of numbers (the loss falls as the model learns). It saves checkpoints to Drive every 250 steps, then exports
the finished model to `LJMU_Research/two-tier-system-models/` on your Drive.

### 7. Restart the runtime, then train the Retail Bank Assistant

**Runtime, Restart runtime** clears the first model from the GPU's memory. Skipping this risks an out-of-memory error. A restart wipes
the whole environment, so repeat steps 2 and 3 (steps 4 and 5 again if the installs were lost), then add the exception data and train:

```python
!python scripts/generate_exception_training_data.py
!python training/train_retail.py
```

**Why the extra data step.** The real Bitext dataset has no fraud, hardship or vulnerable-customer category, only the normal
ones (ACCOUNT, LOAN, PASSWORD, CONTACT, FIND). Without it, the retail model never sees anything like those scenarios, which was the
main reason exception questions scored the lowest confidence in early runs. `generate_exception_training_data.py` reuses the
synthetic templates the pipeline tests with, as training rows, and checks every one against `data/synthetic/` so that **no
training question can also appear as a test question**. It prints how many test instructions it is avoiding; **if that number is
0, stop**: the check is checking nothing. (`data/synthetic/` is committed, so the number is non-zero straight after cloning.)

## If the session drops: resuming

Both scripts save a checkpoint to Drive every 250 steps (in `payments_checkpoints/` and `retail_checkpoints/`, keeping the 3 most
recent). If the session disconnects, open a new one, repeat steps 2 to 5 (and 7's data step for retail), and **run the same training
command again**. The script finds the last checkpoint on Drive and prints "Found an existing checkpoint ... resuming from there"
instead of starting at step 0. This only works because the checkpoints are on Drive: a checkpoint on Colab's own disk would be wiped by
the same disconnect it was meant to protect against.

## What you get, and what to do with it

On your Drive, in `LJMU_Research/two-tier-system-models/` (the path is `MODEL_OUTPUT_DIR` near the top of each script): the payments
model as `payment_assistant` and the retail model as `retail_bank_assistant`, each exported to a 4-bit `.gguf`. Unsloth may place
the `.gguf` in a folder of that name (for example `payment_assistant_gguf`) together with a `Modelfile`; use whichever `.gguf` you find.
Get the files onto your computer by installing Google Drive for desktop and choosing "Mirror files", or by right-clicking each file in
Colab's file browser and choosing Download.

Then, on your own computer, in the folder that holds each `.gguf`, create the Ollama models under the names `src/config.py` expects:

```powershell
# the Modelfile is a text file with no extension and ONE line: FROM ./<the .gguf file's name>
ollama create payment-assistant-v2 -f Modelfile
ollama create retail-bank-assistant-v2 -f Modelfile
ollama list
ollama run payment-assistant-v2 "Why was my card payment declined?"
```

If Unsloth did not write a `Modelfile`, create one with Notepad: a single line such as `FROM ./payment_assistant.gguf`, saved as
exactly `Modelfile` (on Windows, set "Save as type" to All Files so no `.txt` is added). The names `payment-assistant-v2` and
`retail-bank-assistant-v2` already match `src/config.py`. If you choose other names, change them there too. Finally run `python -m
src.tier1` to check a real answer, and retrain the router (`python scripts/train_text_router.py`) because it was trained on a
particular model's answers.

## A note on what the version 2 change can and cannot show

`train_config.py` was changed from version 1 (rank 16, 3 epochs, learning rate 2e-4) to version 2 (rank 32, 5 epochs, learning rate
1.5e-4) for **both** models, in the same change that added the retail exception data. So **payments' result isolates the effect of the
settings alone** (its data did not change), while **retail's result reflects the settings and the new data together**. State that plainly
in any write-up: retail's confidence shift after this retrain cannot be credited to one cause from this run alone, and separating them
would need a run that changes only one. In the project's runs, retail's average confidence stayed at about 0.53 after retraining, so
confidence did not improve even though the answers changed.

## What goes into Git and what does not

**Committed:** `train_config.py`, `train_payments.py`, `train_retail.py`, `requirements.txt` and this README: the training logic,
which is small. **Never committed** (see `.gitignore`): `training/outputs/` and any `.gguf` file. A quantised model is 1 to 4 GB, Git
is not built to store that, and GitHub rejects files over 100 MB. The trained models live in Ollama on your computer and on your Drive.

## Problems

| You see | Fix |
|---|---|
| "Google Drive is not mounted" | Run the mount cell (step 2) first, in its own cell |
| `data/raw/bitext_...jsonl` not found | Run step 5 (and the exception-data step for retail) from the project root |
| Out of memory on the second model | **Runtime, Restart runtime**, then repeat the setup steps before `train_retail.py` |
| The session timed out or disconnected | Resume as described above |
| The progress is slow (hours for few steps) | Check the GPU type under Runtime, Change runtime type. A T4 is about twice as slow as an L4 |
| Hugging Face asks for a token, or warns about unauthenticated downloads | Add a free read token as a Colab Secret named `HF_TOKEN` and switch notebook access on |
| A leftover file from an earlier run confuses the export | Delete the old `.gguf` and checkpoint folders in your Drive model folder, then run again |
