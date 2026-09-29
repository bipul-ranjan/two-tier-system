"""
Shared configuration and helpers for both training scripts. Keeping this
separate means train_payments.py and train_retail.py only differ in the
three lines that are genuinely different (base model, data file, output
name) -- everything else is defined once, here, and applies to both models.

Part of the training/ package -- these scripts are meant to be run on a
Colab GPU runtime after cloning this repo, not on your local machine.
"""
import os
import torch
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

# r/lora_alpha raised from 16 to 32 (more adapter capacity) and num_train_epochs
# raised from 3 to 5 (more passes over the data), with learning_rate lowered from
# 2e-4 to 1.5e-4 to keep convergence stable with the extra epochs and capacity.
# Applies to BOTH models -- payments' training data is unchanged, so if its
# confidence shifts after retraining, that isolates the effect of this change
# specifically (retail's data changed too in the same retrain, so its result
# reflects both changes together, not this one alone).
LORA_CONFIG = dict(
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# save_strategy/save_steps/save_total_limit added so training can resume after a
# disconnect instead of restarting from step 0 -- see resume_checkpoint() below.
# At 5 epochs this run is long enough (several hours per model) that losing all
# progress to a dropped Colab session is a real risk, not a theoretical one.
TRAINING_ARGS = dict(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    num_train_epochs=5,
    learning_rate=1.5e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=3,
)


def resume_checkpoint(output_dir: str):
    """Return the path to resume from if output_dir already has a checkpoint in it (e.g.
    from a session that disconnected mid-run), else None. output_dir MUST be on Google
    Drive, not Colab's local /content disk -- local disk is wiped on disconnect, which
    would make this check pointless (nothing would ever be there to resume from)."""
    if os.path.isdir(output_dir):
        found = get_last_checkpoint(output_dir)
        if found:
            print(f"Found an existing checkpoint at {found} -- resuming from there instead of starting over.")
            return found
    return None


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
