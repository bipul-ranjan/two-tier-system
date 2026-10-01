"""
Tier 2: escalate a query to a shared Claude model (the "single escalation
LLM" in the architecture). Requires ANTHROPIC_API_KEY to be set as an
environment variable. Install with: pip install anthropic
"""
import os
import re
from anthropic import Anthropic

from .config import get_unit_config

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

# Bump this string every time _system_prompt's wording changes meaningfully. Logged per
# escalated row as tier2_prompt_version, the same way tier1_model is logged per row -- so a
# prompt change shows up on the dashboard's trend charts automatically, exactly like a model
# or threshold change does, without needing a hand-maintained list of dates. "v2-persona" is
# this change: giving Claude the same first-person bank-employee framing the Tier 1 SLMs were
# trained on (see _system_prompt below) -- "v1-generic" is the original, role-less prompt.
PROMPT_VERSION = "v2-persona"

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

# PREVIOUS prompt, kept here (not deleted) so it's clear exactly what changed and why, for the
# dissertation writeup. This generic, role-less framing is the likely cause of a pattern found
# in the real escalated answers when reviewing quality scores: Claude referring to "your bank"
# in the THIRD person (as if it were an outside advisor, not part of the bank), hedging with
# "I'm an AI and cannot..." on routine requests, and giving generic textbook advice instead of
# bank-specific guidance. None of the fine-tuned Tier 1 SLMs show this pattern, because every
# one of their training examples explicitly states "You are the {assistant_name} for a retail
# bank" (see training/train_config.py) -- Claude was simply never told who it's supposed to be.
#
#   _OLD_SYSTEM_PROMPT = "You are a financial reasoning assistant. Answer precisely."

def _system_prompt(assistant_name: str) -> str:
    """Give Claude the same first-person, bank-employee framing the Tier 1 SLM for this
    business unit was fine-tuned on (see training/train_config.py), instead of the generic
    prompt above. This is a change to VOICE and HELPFULNESS calibration, not identity: it does
    NOT instruct Claude to deny being an AI or to deceive the customer in any way. It simply
    establishes that Claude is speaking AS the bank's own team -- the same way a human bank
    employee would -- with caution reserved for situations that genuinely call for it, rather
    than applied reflexively to every routine request.
    """
    return (
        f"You are {assistant_name}, part of a retail bank's customer service team. Answer as "
        f"the bank, in first person (\"we\", \"your account\", \"our policy\") -- never refer "
        f"to the bank in the third person, as if you were an outside advisor (for example, "
        f"never say \"contact your bank\"; you ARE the bank). Give specific, actionable next "
        f"steps rather than generic external financial advice. Answer routine questions (fees, "
        f"password resets, account information, branch or ATM locations, standard procedures) "
        f"directly and helpfully, without unnecessary disclaimers. Reserve caution and a "
        f"hand-off to a human colleague for situations that genuinely need it -- identity "
        f"verification, legal matters, or an action you cannot perform yourself -- and in "
        f"those cases, say clearly what the customer should do next."
    )


def _split_confidence(text: str):
    """Pull the trailing 'CONFIDENCE: 0.xx' line off Claude's answer.
    Returns (answer_without_the_line, confidence_or_None)."""
    m = _CONFIDENCE_RE.search(text)
    if not m:
        print("WARNING: tier2 response had no parseable CONFIDENCE line -- logging confidence as blank")
        return text, None
    confidence = max(0.0, min(1.0, float(m.group(1))))
    return text[: m.start()].rstrip(), confidence


def ask_tier2(query: str, business_unit: str = None, context: str = "") -> dict:
    """Send an escalated query to the Tier 2 LLM and return the answer, its
    self-reported confidence, and token usage.

    business_unit selects which assistant persona Claude answers as (see _system_prompt
    above) -- matching the Tier 1 SLM for that unit. Optional only for backward compatibility
    (e.g. the __main__ demo below, which has no business_unit of its own); every real pipeline
    call should pass it.
    """
    user_content = f"{context}\n\nQuestion: {query}" if context else query
    assistant_name = get_unit_config(business_unit)["assistant_name"] if business_unit else "the Customer Assistant"

    # thinking explicitly disabled: see the matching comment in src/quality.py -- avoids the
    # same truncated/empty-response failure mode if extended thinking is ever on by default
    # for this model/account, which would eat into max_tokens before the real answer starts.
    response = client.messages.create(
        model=MODEL,
        max_tokens=520,
        thinking={"type": "disabled"},
        system=_system_prompt(assistant_name) + CONFIDENCE_INSTRUCTION,
        messages=[{"role": "user", "content": user_content}],
    )

    answer, confidence = _split_confidence(_extract_text(response))
    return {
        "answer": answer,
        "confidence": confidence,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "prompt_version": PROMPT_VERSION,
    }


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of one Tier 2 call from its token counts."""
    return (input_tokens / 1000) * PRICE_PER_1K_INPUT + (output_tokens / 1000) * PRICE_PER_1K_OUTPUT


if __name__ == "__main__":
    result = ask_tier2("What was the year-over-year change in operating margin?", business_unit="payments")
    cost = estimate_cost(result["input_tokens"], result["output_tokens"])
    print(result["answer"])
    print(f"Claude self-reported confidence: {result['confidence']}")
    print(f"Estimated cost: ${cost:.6f}")

