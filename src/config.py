"""
Central configuration: Ollama connection details, default model, and
which local SLM (plus assistant persona) each business unit's Tier 1
instance runs.

This is what makes "distributed, per-business-unit SLM" a real,
testable claim rather than a diagram -- each unit's traffic actually
goes to a different model, under a different assistant persona, and
every other file in this package reads its settings from here rather
than defining its own copy.

Run `ollama pull <model>` for every model listed here before use.

Part of the src/ package -- run from the project root with:
    python -m src.config
"""

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "payment-assistant"
DEFAULT_BUSINESS_UNIT = "payments"

BUSINESS_UNITS = {
    "payments": {
        "model": "payment-assistant",
        "assistant_name": "Payment Assistant",
        "description": "Fine-tuned Phi-3-mini on Bitext payments data",
        },
    "retail_bank": {
        "model": "retail-bank-assistant",
        "assistant_name": "Retail Bank Assistant",
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
