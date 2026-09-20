"""
Tier 2: escalate a query to a shared commercial LLM (e.g. GPT-4o-mini).
Requires OPENAI_API_KEY to be set as an environment variable.
Install with: pip install openai
"""
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# Approximate GPT-4o-mini pricing in USD per 1,000 tokens.
# Verify current rates on the provider's pricing page before reporting final costs.
PRICE_PER_1K_INPUT = 0.00015
PRICE_PER_1K_OUTPUT = 0.0006


def ask_tier2(query: str, context: str = "") -> dict:
    """Send an escalated query to the Tier 2 LLM and return the answer + token usage."""
    messages = [
        {"role": "system", "content": "You are a financial reasoning assistant. Answer precisely."},
        {"role": "user", "content": f"{context}\n\nQuestion: {query}" if context else query},
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
    answer = response.choices[0].message.content
    usage = response.usage
    return {
        "answer": answer,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of one Tier 2 call from its token counts."""
    return (input_tokens / 1000) * PRICE_PER_1K_INPUT + (output_tokens / 1000) * PRICE_PER_1K_OUTPUT


if __name__ == "__main__":
    result = ask_tier2("What was the year-over-year change in operating margin?")
    cost = estimate_cost(result["input_tokens"], result["output_tokens"])
    print(result["answer"])
    print(f"Estimated cost: ${cost:.6f}")
