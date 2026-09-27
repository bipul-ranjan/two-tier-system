"""
Fine-tunes Qwen2.5-1.5B into the Retail Bank Assistant, using the
retail_bank split of the Bitext retail-banking dataset (see
scripts/fetch_bitext_banking.py).

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

from train_config import LORA_CONFIG, TRAINING_ARGS, load_and_format_dataset

DATA_FILE = "data/raw/bitext_retail_bank.jsonl"

MODEL_OUTPUT_DIR = "/content/drive/MyDrive/LJMU_Research/two-tier-system-models"
GGUF_NAME = f"{MODEL_OUTPUT_DIR}/retail_bank_assistant"

LOCAL_CHECKPOINT_DIR = "training/outputs/retail_checkpoints"


def main():
    if not os.path.exists("/content/drive/MyDrive"):
        raise RuntimeError(
            "Google Drive is not mounted. Run this first, in its own cell:\n"
            "  from google.colab import drive\n"
            "  drive.mount('/content/drive')\n"
            "Then re-run this script -- without this, your trained model "
            "would only exist on Colab's temporary local disk."
        )

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run scripts/fetch_bitext_banking.py first "
            f"(from the project root) to regenerate the training data."
        )

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    print("Loading Qwen2.5-1.5B (4-bit) and attaching LoRA adapters...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    print(f"Loading training data from {DATA_FILE}...")
    dataset = load_and_format_dataset(DATA_FILE, assistant_name="Retail Bank Assistant")
    print(f"Loaded {len(dataset)} training examples")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        args=TrainingArguments(output_dir=LOCAL_CHECKPOINT_DIR, **TRAINING_ARGS),
    )

    print("Starting training...")
    trainer.train()

    print(f"Exporting to GGUF at {GGUF_NAME}.gguf (on Google Drive)...")
    model.save_pretrained_gguf(GGUF_NAME, tokenizer, quantization_method="q4_k_m")

    print(f"\nDone. {GGUF_NAME}.gguf is saved directly to your Google Drive --")
    print("no manual download needed, and it will survive this Colab session ending.")
    print("Find it in Drive under: LJMU_Research/two-tier-system-models/retail_bank_assistant.gguf")


if __name__ == "__main__":
    main()
