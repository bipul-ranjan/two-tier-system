"""
Tier 1: Local SLM classification with confidence scoring via Ollama.
Run `ollama pull phi3:mini` once before using this module.

Part of the src/ package — run from the project root with:
    python -m src.tier1
"""
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"

PROMPT_TEMPLATE = """You are a banking assistant. Classify the customer query into one intent.
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


def ask_tier1(query: str, model: str = MODEL) -> dict:
    """Send a query to the local Tier 1 SLM and parse intent + confidence.

    `model` lets each business unit call a different local SLM — see
    config.py for the BUSINESS_UNITS mapping.
    """
    prompt = PROMPT_TEMPLATE.format(query=query)
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["response"]
    return parse_response(text)


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
    result = ask_tier1(test_query)
    print(result)
