"""
Tier 1: Local SLM generates a REAL answer (not just an intent label),
with confidence computed from actual token log-probabilities returned
by Ollama -- not a verbalized self-reported label. This is the token-
logit confidence method described in the Coco paper (Li et al., CIKM
2025) that this project cites as its closest prior work.

Requires Ollama v0.12.11 or newer (for logprobs support in /api/generate).
Check your version with: ollama --version

Run `ollama pull phi3:mini` and `ollama pull qwen2.5:1.5b` once before
using this module -- or, if you've fine-tuned, `payment-assistant` and
`retail-bank-assistant` -- see config.py for which business unit uses
which model.

Part of the src/ package -- run from the project root with:
    python -m src.tier1
"""
import json
import math
import os
import requests
from datetime import datetime

from .config import OLLAMA_URL, DEFAULT_BUSINESS_UNIT, get_unit_config

PROMPTS_FILE = "prompts/business_unit_prompts.json"

PROMPT_TEMPLATE = """You are the {assistant_name} for a retail bank. Answer the customer's query directly, clearly, and completely.
{additional_instructions}

Query: {query}
"""

_PROMPTS_CACHE = None


def load_business_unit_prompts() -> dict:
    """Load the editable per-unit instructions file. Cached after first
    load. Missing file/entries degrade gracefully -- this file is meant
    for optional tuning, not a hard requirement.
    """
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE
    if not os.path.exists(PROMPTS_FILE):
        _PROMPTS_CACHE = {}
        return _PROMPTS_CACHE
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        _PROMPTS_CACHE = json.load(f)
    return _PROMPTS_CACHE


def ask_tier1(query: str, business_unit: str = DEFAULT_BUSINESS_UNIT) -> dict:
    """Send a query to the local Tier 1 SLM for the given business unit,
    get a real answer, and compute confidence from actual token
    log-probabilities (not a self-reported label).
    """
    unit_cfg = get_unit_config(business_unit)
    model = unit_cfg["model"]
    assistant_name = unit_cfg["assistant_name"]

    unit_prompts = load_business_unit_prompts()
    additional_instructions = unit_prompts.get(business_unit, {}).get("additional_instructions", "")

    prompt = PROMPT_TEMPLATE.format(
        query=query,
        assistant_name=assistant_name,
        additional_instructions=additional_instructions,
    )

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "logprobs": True,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    answer = data["response"].strip()
    confidence_score = compute_confidence(data)

    return {
        "business_unit": business_unit,
        "assistant_name": assistant_name,
        "model": model,
        "answer": answer,
        "confidence_score": confidence_score,
        "confidence_label": confidence_to_label(confidence_score),
    }


def compute_confidence(ollama_response: dict) -> float:
    """Compute confidence as the geometric mean of per-token probabilities
    across the actually-generated response -- i.e. exp(mean(logprob)).
    This is equivalent to 1/perplexity: a response made of consistently
    high-probability tokens scores close to 1.0; a response the model
    was guessing its way through scores much lower.

    Falls back to a neutral 0.5 if the Ollama server didn't return
    logprobs (e.g. an older Ollama version, or a model/backend that
    doesn't support it) -- this is a real limitation worth stating
    explicitly in your dissertation's methodology section, not silently
    hiding.
    """
    logprobs_data = ollama_response.get("logprobs")
    if not logprobs_data:
        return 0.5

    token_logprobs = [entry["logprob"] for entry in logprobs_data if "logprob" in entry]
    if not token_logprobs:
        return 0.5

    avg_logprob = sum(token_logprobs) / len(token_logprobs)
    return math.exp(avg_logprob)


def confidence_to_label(score: float) -> str:
    """Human-readable bucket, purely for logging/printing -- the actual
    routing decision in router.py uses the numeric score directly, not
    this label.
    """
    if score >= 0.8:
        return "very high"
    elif score >= 0.6:
        return "high"
    elif score >= 0.4:
        return "medium"
    elif score >= 0.2:
        return "low"
    else:
        return "very low"


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
    print(f"Answer         : {result['answer']}")
    print(f"Confidence     : {result['confidence_score']:.4f} ({result['confidence_label']})")
    print(f"Elapsed        : {elapsed_time:.2f} seconds")
    print("-" * 60)
