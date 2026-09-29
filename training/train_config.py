"""
Shared configuration and helpers for both training scripts. Keeping this
separate means train_payments.py and train_retail.py only differ in the
three lines that are genuinely different (base model, data file, output
name) -- everything else is defined once, here.

Part of the training/ package -- these scripts are meant to be run on a
Colab GPU runtime after cloning this repo, not on your local machine.
"""
import torch
from datasets import load_dataset

LORA_CONFIG = dict(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

TRAINING_ARGS = dict(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
)


def load_and_format_dataset(jsonl_path, assistant_name: str):
    """Load one Bitext-format JSONL file, or a list of them (concatenated), and format
    into the instruction-tuning text format SFTTrainer expects.
    """
    dataset = load_dataset("json", data_files=jsonl_path, split="train")

    def formatting_func(example):
        text = (
            f"### Instruction:\nYou are the {assistant_name} for a retail bank.\n\n"
            f"### Query:\n{example['instruction']}\n\n"
            f"### Response:\n{example['response']}"
        )
        return {"text": text}

    return dataset.map(formatting_func)
