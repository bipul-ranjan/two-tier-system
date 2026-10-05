"""
Central configuration: where Ollama is, which local model each business unit uses, and the assistant
persona each model answers as.

Every other file in this package reads its settings from here instead of keeping its own copy, so changing
a model name here changes it everywhere: the pipeline, the Tier 1 calls, the model preloading, and the
dashboard's "new models" marker.

What is configured
    OLLAMA_URL                 the local Ollama server's generate endpoint (http://localhost:11434/api/generate)
    BUSINESS_UNITS             one entry per unit, with
                                 "model"           the model's name in Ollama
                                 "assistant_name"  the persona the model is told it is
                                 "description"     free text, printed by  python -m src.config
    DEFAULT_BUSINESS_UNIT and DEFAULT_MODEL   fallbacks used only when src.tier1 is run on its own for a quick test

Which models must exist in Ollama
    As shipped, the two units use the FINE-TUNED version 2 models: payment-assistant-v2 (Phi-3-mini) and
    retail-bank-assistant-v2 (Qwen2.5-1.5B). They cannot be fetched with `ollama pull`. You create them from
    the .gguf files that training/train_payments.py and training/train_retail.py produce on Google Colab
    (see training/README.md, and section 15 of the user manual):  ollama create <name> -f Modelfile
    To try the system without training anything, point the two "model" entries at the base models instead
    ("phi3:mini" and "qwen2.5:1.5b", after `ollama pull` for each). The answers will differ from the fine-tuned
    models', and so will their confidence, so retrain the learned router afterwards:
    python scripts/train_text_router.py.

Adding a business unit
    1. Add an entry to BUSINESS_UNITS.
    2. Map its data categories to it in unit_for_category() in pipeline.py, and, if you want instructions
       specific to the unit, add a block for it to prompts/business_unit_prompts.json.
    3. Make sure its model exists in Ollama.
    4. Retrain the learned router. A unit it has never seen only gets the generic part of its prediction.

Check your setup (no model is called):    python -m src.config
"""

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "payment-assistant"
DEFAULT_BUSINESS_UNIT = "payments"

BUSINESS_UNITS = {
    "payments": {
        "model": "payment-assistant-v2",
        "assistant_name": "Payment Assistant v2",
        "description": "Fine-tuned Phi-3-mini on Bitext payments data",
        },
    "retail_bank": {
        "model": "retail-bank-assistant-v2",
        "assistant_name": "Retail Bank Assistant v2",
        "description": "Fine-tuned Qwen2.5-1.5B on Bitext retail banking data",
    },
}


def get_unit_config(business_unit: str) -> dict:
    """Return the full config (model, assistant_name, description) for a business unit."""
    if business_unit not in BUSINESS_UNITS:
        raise ValueError(
            f"Unknown business unit '{business_unit}'. "
            f"Valid options: {list(BUSINESS_UNITS.keys())}"
        )
    return BUSINESS_UNITS[business_unit]


def get_model_for_unit(business_unit: str) -> str:
    """Look up which model a given business unit should use."""
    return get_unit_config(business_unit)["model"]


if __name__ == "__main__":
    print(f"OLLAMA_URL: {OLLAMA_URL}")
    print(f"DEFAULT_MODEL: {DEFAULT_MODEL}")
    for unit, cfg in BUSINESS_UNITS.items():
        print(f"{unit}: {cfg['assistant_name']} -> {cfg['model']} ({cfg['description']})")
