"""
Fine-tunes Phi-3-mini into the Payment Assistant, using the payments
split of the Bitext retail-banking dataset (see scripts/fetch_bitext_banking.py).

Run this on a Colab GPU runtime (T4/L4 is sufficient for this model
size), after cloning this repo -- see training/README.md for the full
sequence. Do NOT run this on your local machine -- it requires a GPU.

IMPORTANT: the final GGUF file is saved directly to Google Drive (see
MODEL_OUTPUT_DIR below), not to Colab's local /content disk. Local disk
is wiped every time the VM is recycled -- on a full disconnect, a
session timeout, AND on switching GPU type (e.g. T4 -> L4), which
allocates an entirely new machine. Saving straight to Drive means the
trained model survives all of these, without a manual download step
racing against a disconnect.

Mount Drive before running this script:
    from google.colab import drive
    drive.mount('/content/drive')

    python training/train_payments.py

Settings for version 2 (shared with the retail script through train_config.py): LoRA rank 32 and alpha 32, 5 epochs,
learning rate 1.5e-4, batches of 2 collected 4 times, 1,024-token examples, base model loaded in 4-bit. Training on
10,902 examples took about 3 hours 44 minutes on a Colab L4 (6,815 steps; 1.54% of the parameters trained).

Outputs, on Google Drive: payment_assistant.gguf (4-bit q4_k_m) in LJMU_Research/two-tier-system-models, and
checkpoints every 250 steps in payments_checkpoints/. If the session drops, run the same command again: the script
finds the latest checkpoint and resumes instead of starting over.

Afterwards, in the folder that holds the .gguf (details in training/README.md):
    ollama create payment-assistant-v2 -f Modelfile      # Modelfile contains: FROM ./payment_assistant.gguf
"""
import os
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

from train_config import LORA_CONFIG, TRAINING_ARGS, load_and_format_dataset, resume_checkpoint

DATA_FILE = "data/raw/bitext_payments.jsonl"

# Saved to Drive, not local /content -- survives disconnects, timeouts,
# and GPU-type switches. Change this path if your Drive layout differs.
MODEL_OUTPUT_DIR = "/content/drive/MyDrive/LJMU_Research/two-tier-system-models"
GGUF_NAME = f"{MODEL_OUTPUT_DIR}/payment_assistant"

# On Drive, not local disk: a run at 5 epochs takes several hours, long enough that a
# dropped session is a real risk, not a theoretical one. A checkpoint on local /content
# disk would be wiped by the same disconnect it's supposed to protect against -- it would
# only survive an in-process crash with the VM itself still alive, which isn't the failure
# mode that actually matters for a multi-hour run. The Drive I/O cost of saving here is
# accepted deliberately in exchange for that protection.
CHECKPOINT_DIR = f"{MODEL_OUTPUT_DIR}/payments_checkpoints"


def main():
    """Check the data file exists, load the 4-bit Phi-3-mini with LoRA adapters, train with checkpoints
    (resuming from one if it exists), and export the finished model as a GGUF file straight to
    Google Drive.
    """
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

    print("Loading Phi-3-mini (4-bit) and attaching LoRA adapters...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    print(f"Loading training data from {DATA_FILE}...")
    dataset = load_and_format_dataset(DATA_FILE, assistant_name="Payment Assistant")
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
    print("Find it in Drive under: LJMU_Research/two-tier-system-models/payment_assistant.gguf")


if __name__ == "__main__":
    main()
