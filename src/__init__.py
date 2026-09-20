# Makes src/ a proper Python package so files inside it can import from
# each other with relative imports (from .tier1 import ask_tier1, etc.)
# and so the whole project can be run as `python -m src.pipeline` from
# the project root.
