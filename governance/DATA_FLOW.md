# Data flow: what leaves your computer, and what protects it

This describes the pipeline as built (`src/pipeline.py`). It is the document that answers "where does customer text go?".

## Step by step

| # | Step | What is sent | Where it goes | Filter applied | Notes |
|---|---|---|---|---|---|
| 1 | Tier 1 answers | The question as typed | Ollama on your own server | No (inside your boundary) | The small model sees the original |
| 2 | Router predicts quality | Filtered question and filtered Tier 1 answer | In memory, on your server | Yes | The router reads the same filtered text it was trained on |
| 3 | Cache lookup | The filtered question | MiniLM runs locally, so nothing leaves | Yes | MiniLM downloads once from Hugging Face; that download carries no customer data |
| 4 | Claude writes an escalated answer | The filtered question | **External: Anthropic API** | Yes | Only rows with decision `ESCALATE` |
| 5 | Claude marks the answers (the examiner) | The filtered question and filtered answer(s) | **External: Anthropic API** | Yes | **Every row, including local ones, in this research harness** (see below) |
| 6 | Logs | Filtered question, filtered answers, audit columns | Local disk, `results/logs/` | Yes | Rows logged before the filter existed hold raw text until `scripts/redact_history.py --apply` is run |
| 7 | Dashboard | Reads the logs | Local | n/a | Shows filtered text |
| 8 | Model training on Colab | Bitext training data and synthetic examples | Google Colab and Google Drive | n/a | No customer data is used |

## How much reaches the external provider

- **With quality marking on (the default):** the provider receives a filtered copy of every question and answer, so exposure is
  **100% of rows**. This is a research setting. In production, marking would be sampled, run on a judge inside your boundary, or done
  on redacted text only.
- **With marking off (`--noquality`):** the provider receives only escalated questions. In the 5 October run of 100 questions that
  was 8 (69 local, 23 from the cache, 8 to Claude).
- Cache hits do not call Claude.

## What the filter does (`src/privacy.py`, version `regex-v2`)

- **Questions** are filtered in full: emails, phone numbers, card numbers (Luhn-checked), IBANs (mod-97-checked), account numbers,
  sort codes, national IDs, a card's last four digits, dates of birth after a cue, UK postcodes, and names after a cue.
- **Answers** are filtered only for customer-identifier types, plus any value the customer typed that the answer repeats. Public
  helplines, ombudsman addresses and support emails in answers are kept, because redacting them protects nobody and hides useful
  content.
- **Amounts, plain dates, rates and reference numbers** are left alone.

## What was measured

Run `python scripts/evaluate_privacy_filter.py` to reproduce these (seed 7, 200 generated cases per type):

| Measurement | Result |
|---|---|
| Recall, 12 covered types, on generated cases | 100% each. **This is a ceiling:** the generated values are in the formats the filter knows |
| Recall for free-text names ("I'm Priya") | 0% (not designed to be caught) |
| Recall for free-text street addresses | 0% (not designed to be caught) |
| Look-alikes left unchanged (amounts, dates, rates, references) | 12 of 12 |
| Real synthetic questions changed (3,000 sampled) | 144 (4.8%), all because of a card's last four digits |
| Your 6,310-row history, dry run of `redact_history.py` | 968 identifiers: 957 card last-four, 11 names; no public contact details were removed |

## What this does not show

The filter has not been tested on real customer text, which is messier than generated text. It does not catch names or addresses in
free text, employer names, or identifiers in formats it does not know. A real deployment needs a trained entity-recognition model
for those and a recall target agreed with the privacy team. Do not use real customer data until `GAP_REGISTER.md` items marked
"before real data" are closed.
