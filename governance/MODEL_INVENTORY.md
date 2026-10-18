# Model inventory

Every model or learned component in the system. Risk tiers are **proposed** for the model-risk function to confirm. Owners are to be
assigned: no owner has been named in this repository.

| Component | Type | Provider, licence | Version or label | Where it runs | What it sees | Purpose | Proposed tier | Validation status |
|---|---|---|---|---|---|---|---|---|
| Payment Assistant | Fine-tuned small language model (Phi-3-mini, LoRA, version 2) | Microsoft base (MIT licence); our adapter | `payment-assistant-v2` | Your server, in Ollama | The original question | Answers payments questions first | To be set | Tested on synthetic questions; marks are Claude's; no independent validation |
| Retail Bank Assistant | Fine-tuned small language model (Qwen2.5-1.5B, LoRA, version 2) | Alibaba base (Apache 2.0); our adapter | `retail-bank-assistant-v2` | Your server, in Ollama | The original question | Answers retail-banking questions first | To be set | As above |
| Learned router | Text regression (TF-IDF with ridge) | Our own, scikit-learn | version label in `models/router/text_probe_router_info.json` | Your server | Filtered question and answer, confidence, unit | Decides keep or escalate | To be set | Held-out question types; correlation +0.477 with Claude's marks; not independently validated |
| Embedding model | Pretrained sentence encoder (all-MiniLM-L6-v2), frozen | Open model, used unchanged | `all-MiniLM-L6-v2` | Your server | The filtered question | Cache matching | To be set | Not tested against alternatives; no measured false-hit rate |
| Personal-data filter | Rule-based (regular expressions with Luhn and IBAN checks) | Our own | `regex-v2` | Your server | All outbound and logged text | Removes patterned identifiers | To be set | 45 automated tests; recall measured on generated cases only; free-text names and addresses not covered |
| Claude Haiku 4.5 | External large language model | Anthropic (vendor model) | `claude-haiku-4-5-20251001` (a dated snapshot) | Anthropic's servers | The filtered question | Writes escalated answers | To be set | Vendor model: the firm remains accountable; exit plan not written |
| Claude Sonnet 5 | External large language model | Anthropic (vendor model) | `claude-sonnet-5` (set in `src/quality.py`) | Anthropic's servers | Filtered question and answers | Marks answer quality (the examiner) | To be set | Marks not checked against human markers |

## Notes for the reviewer

- **Intended use:** informational customer questions for two banking units. It takes no actions and holds no account data.
- **Out of scope:** personalised account queries, regulated advice, credit decisions, and any use on real customer data.
- **Known limitations of each model** are in the project record (Sections 7, 10, 11 and 12) and in the router and cache documents.
