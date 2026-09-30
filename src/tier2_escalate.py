"""
Tier 2: escalate a query to a shared Claude model (the "single escalation
LLM" in the architecture). Requires ANTHROPIC_API_KEY to be set as an
environment variable. Install with: pip install anthropic
"""
import os
import re
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _extract_text(response) -> str:
    """response.content can include a ThinkingBlock (the model's internal reasoning) before
    the actual text answer, when extended thinking is active for this model/account --
    content[0] is NOT reliably the answer. Concatenate every block whose type is "text"
    instead of assuming position 0 (which has no .text attribute on a ThinkingBlock and
    crashes if one is ever returned here)."""
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")

# claude-haiku-4-5 is the cheapest current Claude model -- the right
# choice for a cost-cascade's escalation tier, same role GPT-4o-mini
# played in the original design.
MODEL = "claude-haiku-4-5-20251001"

# USD per 1,000 tokens. Current as of the rates confirmed when this file
# was written -- verify against docs.claude.com/en/docs/about-claude/pricing
# before reporting final cost figures, since pricing can change.
PRICE_PER_1K_INPUT = 0.001
PRICE_PER_1K_OUTPUT = 0.005

# The Anthropic API does not return token-level log-probabilities the way
# Ollama does for Tier 1, so there is no way to compute a Claude confidence
# the same way. Instead, Claude is asked to self-report one on a final line,
# which is parsed out here and never shown to the end user. This is a
# self-assessment, not a measured probability -- treat it as a weaker,
# differently-biased signal than the Tier 1 log-prob confidence, not a
# like-for-like replacement.
CONFIDENCE_INSTRUCTION = (
    "\n\nAfter your answer, on its own final line, write exactly:\n"
    "CONFIDENCE: <number>\n"
    "where <number> is between 0.00 and 1.00 and reflects how confident you are "
    "that your answer is accurate and complete. Write nothing after that line."
)
_CONFIDENCE_RE = re.compile(r"\n*CONFIDENCE:\s*([01](?:\.\d+)?)\s*$", re.IGNORECASE)


def _split_confidence(text: str):
    """Pull the trailing 'CONFIDENCE: 0.xx' line off Claude's answer.
    Returns (answer_without_the_line, confidence_or_None)."""
    m = _CONFIDENCE_RE.search(text)
    if not m:
        print("WARNING: tier2 response had no parseable CONFIDENCE line -- logging confidence as blank")
        return text, None
    confidence = max(0.0, min(1.0, float(m.group(1))))
    return text[: m.start()].rstrip(), confidence


def ask_tier2(query: str, context: str = "") -> dict:
    """Send an escalated query to the Tier 2 LLM and return the answer, its
    self-reported confidence, and token usage."""
    user_content = f"{context}\n\nQuestion: {query}" if context else query

    # thinking explicitly disabled: see the matching comment in src/quality.py -- avoids the
    # same truncated/empty-response failure mode if extended thinking is ever on by default
    # for this model/account, which would eat into max_tokens before the real answer starts.
    response = client.messages.create(
        model=MODEL,
        max_tokens=520,
        thinking={"type": "disabled"},
        system="You are a financial reasoning assistant. Answer precisely." + CONFIDENCE_INSTRUCTION,
        messages=[{"role": "user", "content": user_content}],
    )

    answer, confidence = _split_confidence(_extract_text(response))
    return {
        "answer": answer,
        "confidence": confidence,
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
    print(f"Claude self-reported confidence: {result['confidence']}")
    print(f"Estimated cost: ${cost:.6f}")

