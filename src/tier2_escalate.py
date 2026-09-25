"""
Tier 2: escalate a query to a shared Claude model (the "single escalation
LLM" in the architecture). Requires ANTHROPIC_API_KEY to be set as an
environment variable. Install with: pip install anthropic
"""
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# claude-haiku-4-5 is the cheapest current Claude model -- the right
# choice for a cost-cascade's escalation tier, same role GPT-4o-mini
# played in the original design.
MODEL = "claude-haiku-4-5-20251001"

# USD per 1,000 tokens. Current as of the rates confirmed when this file
# was written -- verify against docs.claude.com/en/docs/about-claude/pricing
# before reporting final cost figures, since pricing can change.
PRICE_PER_1K_INPUT = 0.001
PRICE_PER_1K_OUTPUT = 0.005


def ask_tier2(query: str, context: str = "") -> dict:
    """Send an escalated query to the Tier 2 LLM and return the answer + token usage."""
    user_content = f"{context}\n\nQuestion: {query}" if context else query

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system="You are a financial reasoning assistant. Answer precisely.",
        messages=[{"role": "user", "content": user_content}],
    )

    answer = response.content[0].text
    return {
        "answer": answer,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of one Tier 2 call from its token counts."""
    return (input_tokens / 1000) * PRICE_PER_1K_INPUT + (output_tokens / 1000) * PRICE_PER_1K_OUTPUT


if __name__ == "__main__":
    result = ask_tier2("What was the year-over-year change in operating margin?")
    cost = estimate_cost(result["input_tokens"], result["output_tokens"])
    print(result["answer"])
    print(f"Estimated cost: ${cost:.6f}")
