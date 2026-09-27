"""
Tier 1: Local SLM classification with confidence scoring via Ollama.
Each business unit's SLM answers under its own assistant persona (e.g.
"Payment Assistant" vs "Retail Bank Assistant"), not a generic identity.

Run `ollama pull phi3:mini` and `ollama pull qwen2.5:1.5b` once before
using this module -- see config.py for which business unit uses which.

Part of the src/ package -- run from the project root with:
    python -m src.tier1
"""
import requests
from datetime import datetime

from .config import OLLAMA_URL, DEFAULT_BUSINESS_UNIT, get_unit_config

PROMPT_TEMPLATE = """You are the {assistant_name}, a banking assistant. Classify the customer query into one intent.
Query: {query}
Respond in this exact format:
Intent: <intent name>
Confidence: <Very Low / Low / Medium / High / Very High>
"""

CONFIDENCE_MAP = {
    "very low": 0.1,
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7,
    "very high": 0.9,
}


def ask_tier1(query: str, business_unit: str = DEFAULT_BUSINESS_UNIT) -> dict:
    """Send a query to the local Tier 1 SLM for the given business unit,
    and parse intent + confidence.

    Looks up both the model and the assistant persona from config.py, so
    a "payments" query is answered by the Payment Assistant on phi3:mini,
    and a "retail_bank" query by the Retail Bank Assistant on
    qwen2.5:1.5b -- different model, different identity in the prompt.
    """
    unit_cfg = get_unit_config(business_unit)
    model = unit_cfg["model"]
    assistant_name = unit_cfg["assistant_name"]

    prompt = PROMPT_TEMPLATE.format(query=query, assistant_name=assistant_name)
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["response"]

    result = parse_response(text)
    result["business_unit"] = business_unit
    result["assistant_name"] = assistant_name
    result["model"] = model
    return result


def parse_response(text: str) -> dict:
    """Extract intent and confidence score from the model's raw text output."""
    intent = "unknown"
    confidence_label = "low"
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("intent:"):
            intent = line.split(":", 1)[1].strip()
        elif line.lower().startswith("confidence:"):
            confidence_label = line.split(":", 1)[1].strip().lower()
    confidence_score = CONFIDENCE_MAP.get(confidence_label, 0.3)
    return {
        "intent": intent,
        "confidence_label": confidence_label,
        "confidence_score": confidence_score,
    }


if __name__ == "__main__":
    test_query = "Why was my card payment declined?"
    test_business_unit = "payments"

    start_time = datetime.now()
    result = ask_tier1(test_query, business_unit=test_business_unit)
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()

    print(f"Start Time     : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Business Unit  : {result['business_unit']} ({result['assistant_name']})")
    print(f"Query          : {test_query}")
    print(f"Result         : {result['intent']}")
    print(f"Confidence     : {result['confidence_score']} ({result['confidence_label']})")
    print(f"Elapsed        : {elapsed_time:.2f} seconds")
    print("-" * 60)
