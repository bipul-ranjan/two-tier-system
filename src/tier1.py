"""
Tier 1: Local SLM generates a REAL answer (not just an intent label),
with confidence computed from actual token log-probabilities returned
by Ollama -- not a verbalized self-reported label.

Two confidence methods are computed and returned side by side:
- confidence_score_avg: geometric mean across all tokens (the Coco-paper
  style method). Smooth, but diluted by common filler words ("I'm",
  "sorry", "to", "hear") that the model is always confident about
  regardless of whether it understood your actual question.
- confidence_score_min: the single least-confident token in the whole
  response. Sharper, more sensitive to one genuine moment of real
  uncertainty, since it isn't averaged away by easy surrounding words.

`confidence_score` (used by router.py for the actual routing decision)
is currently set to the avg method, for consistency with what's been
evaluated so far -- see scripts/compare_confidence_methods.py for how
to test whether switching to min changes routing behaviour.

Requires Ollama v0.12.11 or newer (for logprobs support in /api/generate).

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
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE
    if not os.path.exists(PROMPTS_FILE):
        _PROMPTS_CACHE = {}
        return _PROMPTS_CACHE
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        _PROMPTS_CACHE = json.load(f)
    return _PROMPTS_CACHE


def ask_tier1(query: str, business_unit: str = DEFAULT_BUSINESS_UNIT, keep_alive=None) -> dict:
    """keep_alive (optional) is passed to Ollama so the model stays loaded after this
    call -- e.g. "30m". Left out by default, so standalone calls behave as before.
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

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "logprobs": True,
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    answer = data["response"].strip()
    confidence_avg = compute_confidence_avg(data)
    confidence_min = compute_confidence_min(data)

    return {
        "business_unit": business_unit,
        "assistant_name": assistant_name,
        "model": model,
        "answer": answer,
        "confidence_score": confidence_avg,          # the one router.py actually uses
        "confidence_score_avg": confidence_avg,
        "confidence_score_min": confidence_min,
        "confidence_label": confidence_to_label(confidence_avg),
        # Ollama reports durations in nanoseconds. load_duration is the time
        # spent loading the model into memory for this call -- near zero when
        # the model is already loaded, several seconds on a cold start.
        "ollama_load_ms": round(data.get("load_duration", 0) / 1e6, 1),
        "ollama_output_tokens": data.get("eval_count"),
    }


def compute_confidence_avg(ollama_response: dict) -> float:
    """Geometric mean of per-token probabilities across the whole
    response -- exp(mean(logprob)). Equivalent to 1/perplexity.
    """
    token_logprobs = _extract_token_logprobs(ollama_response)
    if not token_logprobs:
        return 0.5
    avg_logprob = sum(token_logprobs) / len(token_logprobs)
    return math.exp(avg_logprob)


def compute_confidence_min(ollama_response: dict) -> float:
    """Probability of the single least-confident token in the response --
    exp(min(logprob)). Sharper signal, less diluted by filler words.
    """
    token_logprobs = _extract_token_logprobs(ollama_response)
    if not token_logprobs:
        return 0.5
    return math.exp(min(token_logprobs))


def _extract_token_logprobs(ollama_response: dict):
    logprobs_data = ollama_response.get("logprobs")
    if not logprobs_data:
        return None
    token_logprobs = [entry["logprob"] for entry in logprobs_data if "logprob" in entry]
    return token_logprobs if token_logprobs else None


def confidence_to_label(score: float) -> str:
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
    print(f"Confidence AVG : {result['confidence_score_avg']:.4f}")
    print(f"Confidence MIN : {result['confidence_score_min']:.4f}")
    print(f"Elapsed        : {elapsed_time:.2f} seconds")
    print("-" * 60)
