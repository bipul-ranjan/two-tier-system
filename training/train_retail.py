"""
Fine-tunes Qwen2.5-1.5B into the Retail Bank Assistant, using the
retail_bank split of the Bitext retail-banking dataset (see
scripts/fetch_bitext_banking.py).

Run this on a Colab GPU runtime, after cloning this repo, in a FRESH
runtime (Runtime -> Restart runtime) if you already trained the Payment
Assistant in the same session -- this clears the GPU memory from the
first model before loading the second one. See training/README.md for
the full sequence. Do NOT run this on your local machine -- it requires
a GPU.

    python training/train_retail.py
"""
import os
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

from train_config import LORA_CONFIG, TRAINING_ARGS, load_and_format_dataset

DATA_FILE = "data/raw/bitext_retail_bank.jsonl"
OUTPUT_DIR = "training/outputs/retail_checkpoints"
GGUF_NAME = "training/outputs/retail_bank_assistant"


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run scripts/fetch_bitext_banking.py first "
            f"(from the project root) to regenerate the training data."
        )

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
