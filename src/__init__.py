"""
The pipeline package: everything that happens to a customer question after it arrives.

Modules, in the order a question passes through them:
    config           which local model each business unit uses, and Ollama's address
    tier1            the local model answers and reports how confident it was
    learned_router   predicts how good Tier 1's answer is and decides: keep it or escalate (the default router)
    router           the older rule: escalate when confidence is below a pass mark (--threshold-router)
    semantic_cache   re-serves a past Claude answer when a question is a near-exact repeat
    tier2_escalate   Claude Haiku answers the questions that were escalated
    quality          Claude as examiner: marks answers 1-5 on five dimensions
    pipeline         wires all of the above together, logs every decision, scores quality
    evaluate         headline summary tables from a results log

Run any module from the project root with  python -m src.<name>  and never with  python src/<name>.py:
the modules import each other with relative imports (from .tier1 import ask_tier1), which only work
when src is run as a package. This file is what makes src a package.
"""
# Makes src/ a proper Python package so files inside it can import from
# each other with relative imports (from .tier1 import ask_tier1, etc.)
# and so the whole project can be run as `python -m src.pipeline` from
# the project root.
