"""
Fine-tunes Qwen2.5-1.5B into the Retail Bank Assistant, using the
retail_bank split of the Bitext retail-banking dataset (see
scripts/fetch_bitext_banking.py) PLUS synthetic exception-scenario examples
(see scripts/generate_exception_training_data.py) -- the real Bitext data
has no fraud/hardship/vulnerable-customer category at all, so without this
second file the model never saw anything like those scenarios during
training. That gap was the main driver behind RETAIL_EXCEPTION scoring the
lowest confidence of any category in pipeline runs.

Run this on a Colab GPU runtime, after cloning this repo, in a FRESH
runtime (Runtime -> Restart runtime, or after switching GPU type) if
you already trained the Payment Assistant in the same session -- this
clears the GPU memory from the first model before loading the second.
See training/README.md for the full sequence. Do NOT run this on your
local machine -- it requires a GPU.

IMPORTANT: the final GGUF file is saved directly to Google Drive (see
MODEL_OUTPUT_DIR below), not to Colab's local /content disk -- same
reasoning as train_payments.py. Local disk is wiped on disconnect,
timeout, AND on switching GPU type, so saving straight to Drive means
the trained model survives all of these.

Mount Drive before running this script:
    from google.colab import drive
    drive.mount('/content/drive')

    python training/train_retail.py
"""
import os
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

from train_config import LORA_CONFIG, TRAINING_ARGS, load_and_format_dataset, resume_checkpoint

DATA_FILES = [
    "data/raw/bitext_retail_bank.jsonl",              # real data: ACCOUNT/LOAN/PASSWORD/CONTACT/FIND
    "data/raw/retail_exception_synthetic.jsonl",       # synthetic: RETAIL_EXCEPTION (not in the real dataset)
]

MODEL_OUTPUT_DIR = "/content/drive/MyDrive/LJMU_Research/two-tier-system-models"
GGUF_NAME = f"{MODEL_OUTPUT_DIR}/retail_bank_assistant"

# On Drive, not local disk: see the matching comment in train_payments.py -- a local
# checkpoint would be wiped by the same disconnect it's meant to protect against.
CHECKPOINT_DIR = f"{MODEL_OUTPUT_DIR}/retail_checkpoints"


def main():
    if not os.path.exists("/content/drive/MyDrive"):
        raise RuntimeError(
            "Google Drive is not mounted. Run this first, in its own cell:\n"
            "  from google.colab import drive\n"
            "  drive.mount('/content/drive')\n"
            "Then re-run this script -- without this, your trained model "
            "would only exist on Colab's temporary local disk."
        )

    missing = [f for f in DATA_FILES if not os.path.exists(f)]
    if missing:
        hint = {
            "data/raw/bitext_retail_bank.jsonl": "Run scripts/fetch_bitext_banking.py first (from the project root).",
            "data/raw/retail_exception_synthetic.jsonl": "Run scripts/generate_exception_training_data.py first (from the project root).",
        }
        details = "\n".join(f"  - {f}: {hint.get(f, 'file not found')}" for f in missing)
        raise FileNotFoundError(f"Missing training data file(s):\n{details}")

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    print("Loading Qwen2.5-1.5B (4-bit) and attaching LoRA adapters...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    print(f"Loading training data from {DATA_FILES}...")
    dataset = load_and_format_dataset(DATA_FILES, assistant_name="Retail Bank Assistant")
    print(f"Loaded {len(dataset)} training examples")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        args=TrainingArguments(output_dir=CHECKPOINT_DIR, **TRAINING_ARGS),
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint(CHECKPOINT_DIR))

    print(f"Exporting to GGUF at {GGUF_NAME}.gguf (on Google Drive)...")
    model.save_pretrained_gguf(GGUF_NAME, tokenizer, quantization_method="q4_k_m")

    print(f"\nDone. {GGUF_NAME}.gguf is saved directly to your Google Drive --")
    print("no manual download needed, and it will survive this Colab session ending.")
    print("Find it in Drive under: LJMU_Research/two-tier-system-models/retail_bank_assistant.gguf")


if __name__ == "__main__":
    main()
