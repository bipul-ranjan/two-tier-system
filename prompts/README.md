# prompts/

## business_unit_prompts.json

Extra instructions added to Tier 1's prompt, one block per business unit. `src/tier1.py` reads it once, finds the block whose key is
the question's business unit, and places its `additional_instructions` text under the line "You are the <assistant name> for a retail
bank. Answer the customer's query directly, clearly, and completely."

```json
{
  "payments": { "additional_instructions": "Keep answers focused on card payments, transfers, declined transactions, and fees. ..." },
  "retail_bank": { "additional_instructions": "Keep answers focused on account opening, savings products, balances, and branch services. ..." }
}
```

* The keys must match the business-unit names in `src/config.py`.
* A missing file, or a unit with no block, simply adds nothing; nothing breaks.
* The fine-tuned models were trained with the persona line only. These extra instructions are applied on top at run time.
* Changing this file changes Tier 1's answers, and so its confidence and the router's predictions. Retrain the router afterwards
  (`python scripts/train_text_router.py`) if the change is more than a small wording tweak.

Claude's own prompt (for escalated questions) is not here: it is built in `src/tier2_escalate.py` (`_system_prompt`, versioned by `PROMPT_VERSION`).
