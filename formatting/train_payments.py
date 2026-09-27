"""
Fine-tunes Phi-3-mini into the Payment Assistant, using the payments
split of the Bitext retail-banking dataset (see scripts/fetch_bitext_banking.py).

Run this on a Colab GPU runtime (T4 is sufficient for this model size),
after cloning this repo -- see training/README.md for the full sequence.
Do NOT run this on your local machine -- it requires a GPU.

    python training/train_payments.py
"""
import os
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

from train_config import LORA_CONFIG, TRAINING_ARGS, load_and_format_dataset

DATA_FILE = "data/raw/bitext_payments.jsonl"
OUTPUT_DIR = "training/outputs/payments_checkpoints"
GGUF_NAME = "training/outputs/payment_assistant"


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run scripts/fetch_bitext_banking.py first "
            f"(from the project root) to regenerate the training data."
        )

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
        args=TrainingArguments(output_dir=OUTPUT_DIR, **TRAINING_ARGS),
    )

    print("Starting training...")
    trainer.train()

    print(f"Exporting to GGUF at {GGUF_NAME}.gguf ...")
    os.makedirs("training/outputs", exist_ok=True)
    model.save_pretrained_gguf(GGUF_NAME, tokenizer, quantization_method="q4_k_m")

    print(f"\nDone. Download {GGUF_NAME}.gguf from the Colab file browser.")


if __name__ == "__main__":
    main()
