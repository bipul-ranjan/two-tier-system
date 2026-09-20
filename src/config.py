"""
Business-unit configuration: which local SLM each business unit's
Tier 1 instance runs. This is what makes "distributed, per-business-unit
SLM" a real, testable claim rather than a diagram — each unit's traffic
actually goes to a different model.

Run `ollama pull <model>` for every model listed here before use.

Part of the src/ package — run from the project root with:
    python -m src.config
"""

BUSINESS_UNITS = {
    "payments": {
        "model": "phi3:mini",
        "description": "3.8B params, MIT license",
    },
    "retail_bank": {
        "model": "qwen2.5:1.5b",
        "description": "1.5B params, Apache 2.0 license",
    },
}


def get_model_for_unit(business_unit: str) -> str:
    """Look up which model a given business unit should use."""
    if business_unit not in BUSINESS_UNITS:
        raise ValueError(
            f"Unknown business unit '{business_unit}'. "
            f"Valid options: {list(BUSINESS_UNITS.keys())}"
        )
    return BUSINESS_UNITS[business_unit]["model"]


if __name__ == "__main__":
    for unit, cfg in BUSINESS_UNITS.items():
        print(f"{unit}: {cfg['model']} ({cfg['description']})")
