"""
Personal-data filter: finds personal identifiers in a piece of text and replaces each with a placeholder
such as [CARD_NUMBER], so the text can leave the organisation's boundary (to Claude, to the examiner) or be
written to a log without carrying the identifier.

Where it is used (src/pipeline.py). Tier 1 runs on the organisation's own servers, so it sees the original
question. Everything that leaves that boundary or is stored gets the redacted copy: the question sent to Claude,
the question used for the cache lookup, the question and answers sent to Claude as examiner, and the question
and answers written to the log. The router also reads the redacted copy, so it sees the same text it was
trained on. Turn the filter off with --no-redact (only to reproduce older runs).

What it catches: structured identifiers that follow a recognisable pattern.
    EMAIL, CARD_NUMBER (checked with the Luhn test), IBAN (checked with the mod-97 test), PHONE, SORT_CODE,
    NATIONAL_ID (US SSN, UK National Insurance number, Indian PAN, Aadhaar-style 4-4-4 numbers),
    ACCOUNT_NUMBER (8 to 18 digits in a row, or a short number after the word "account"),
    CARD_LAST4 (four to six digits after "ending" or "last four digits", or straight after the word card, card number, Mastercard,
    Amex or RuPay: "card 0577", "card number 0361", "card ending 123456"), DATE_OF_BIRTH (a date after "born" or
    "date of birth"), POSTCODE (UK format) and NAME (only after "my name is" or a title such as Mr or Dr).

Known limits of the card-digit rule: it needs one of those cue words, so "Visa 4421" is not caught (Visa is also a travel document, so it
is deliberately left out), and a year written straight after the word card ("card 2026") would be replaced by mistake.

What it does NOT catch, and the project says so plainly: names and addresses written in free text without
those cues ("I'm Rahul"), street addresses, employer names, and any identifier in a format not listed above.
A real deployment needs a trained entity-recognition model for those, and a recall target agreed with the
privacy team. Use scripts/evaluate_privacy_filter.py to measure what this filter catches on a labelled set.

Questions and answers are treated differently. A customer's question is filtered in full. An answer is filtered
only for the identifier types that identify a customer (cards, accounts, national IDs, IBANs, sort codes, dates
of birth, names), plus any value the customer typed that the answer repeats back. Phone numbers, emails and
addresses that appear in an answer are normally public contact details (a helpline, an ombudsman's address), so
they are left alone: removing them would not protect anyone and would hide useful content from the examiner.

Amounts of money are deliberately left alone: they are what the customer is asking about, they carry little
identifying value on their own, and removing them would change the question.

The filter is deterministic and fast (a few regular expressions), makes no network call, and is idempotent:
filtering already-filtered text changes nothing. FILTER_VERSION is logged on every row.
"""
import re
from dataclasses import dataclass, field

FILTER_VERSION = "regex-v3"

# The identifier types that identify a customer. These are the ones applied to an ANSWER (see the docstring).
ANSWER_TYPES = frozenset({"CARD_NUMBER", "CARD_LAST4", "ACCOUNT_NUMBER", "NATIONAL_ID", "IBAN", "SORT_CODE",
                          "DATE_OF_BIRTH", "NAME"})


@dataclass
class RedactionResult:
    """The filtered text, and how many identifiers of each type were replaced."""
    text: str
    findings: dict = field(default_factory=dict)  # type -> number replaced
    originals: list = field(default_factory=list)  # (type, original value) pairs; kept in memory only, never logged

    @property
    def count(self) -> int:
        """Total identifiers replaced."""
        return sum(self.findings.values())

    @property
    def types(self) -> str:
        """The types found, sorted and comma-separated (empty string if none), for the log."""
        return ",".join(sorted(self.findings))


def luhn_ok(digits: str) -> bool:
    """True if a string of digits passes the Luhn check that payment card numbers satisfy."""
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_ok(candidate: str) -> bool:
    """True if a string passes the IBAN length and mod-97 check."""
    s = re.sub(r"\s+", "", candidate).upper()
    if not 15 <= len(s) <= 34 or not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", s):
        return False
    moved = s[4:] + s[:4]
    number = "".join(str(int(ch, 36)) for ch in moved)
    return int(number) % 97 == 1


_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
_CARD = re.compile(r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_NINO = re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b", re.IGNORECASE)
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_AADHAAR = re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)")
_SORT = re.compile(r"(?<!\d)\d{2}-\d{2}-\d{2}(?!\d)")
_DATE = r"(?:0?[1-9]|[12]\d|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:19|20)\d{2}"
_DOB = re.compile(r"(?i)\b((?:born(?:\s+on)?|d\.?o\.?b\.?|date\s+of\s+birth)\s*[:\-]?\s*)" + _DATE + r"\b")
_PHONE_PLUS = re.compile(r"(?<![\w.])\+\d{1,3}[ \-]?\(?\d{1,5}\)?(?:[ \-]?\d{2,5}){2,4}(?!\d)")
_PHONE_US = re.compile(r"(?<!\d)\(?\d{3}\)?[ \-.]\d{3}[ \-.]\d{4}(?!\d)")
_PHONE_NAT = re.compile(r"(?<!\d)(?:0\d{9,10}|[6-9]\d{9})(?!\d)")
_ACCT_LONG = re.compile(r"(?<!\d)\d{8,18}(?!\d)")
_ACCT_CTX = re.compile(r"(?i)\b((?:account|acct|a/c)(?:\s+(?:number|no\.?|#))?\s*(?:is|:|#)?\s*)\d{4,7}\b")
_LAST4 = re.compile(r"(?i)\b((?:ending(?:\s+(?:in|with))?|last\s+(?:4|four)(?:\s+digits)?(?:\s+(?:are|of))?)\s*[:#]?\s*)\d{4,6}\b")
_CARD_BARE = re.compile(r"(?i)\b((?:card|mastercard|amex|rupay)(?:\s+(?:number|no\.?|num)(?:\s+is)?)?\s*[:#]?\s*)\d{4,6}\b")
_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b")
_NAME_CUE = re.compile(r"\b((?:[Mm]y name is|[Nn]ame is|[Nn]ame:|[Ii] am called)\s+)[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}")
_NAME_TITLE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")


def _bump(findings: dict, kind: str) -> None:
    findings[kind] = findings.get(kind, 0) + 1


def _sub(pattern, text, kind, findings, originals, only, replacement=None, keep_group=False, validate=None):
    """Replace every match of a pattern with [KIND], counting each one and remembering the original value.
    Skipped entirely if only is given and does not include this kind. validate(match_text) may reject a match
    (it is then left alone). keep_group keeps the matched group 1 (the cue words) in front of the placeholder."""
    if only is not None and kind not in only:
        return text

    def fn(m):
        if validate is not None and not validate(m.group(0)):
            return m.group(0)
        _bump(findings, kind)
        tag = replacement or f"[{kind}]"
        if keep_group:
            originals.append((kind, m.group(0)[len(m.group(1)):]))
            return m.group(1) + tag
        originals.append((kind, m.group(0)))
        return tag
    return pattern.sub(fn, text)


def redact(text, only=None, known=None) -> RedactionResult:
    """Replace personal identifiers in text with placeholders, and report what was replaced.

    only  : restrict the filter to these identifier types (None = all). Pass ANSWER_TYPES for an answer.
    known : (type, value) pairs the customer typed in the question (RedactionResult.originals). Any occurrence of
            one of these exact values is replaced too, even if it is not in a pattern the filter recognises.
    Non-string input (None, NaN, a number) comes back as text with no findings."""
    if not isinstance(text, str):
        return RedactionResult(text="" if text is None or text != text else str(text))
    if not text:
        return RedactionResult(text=text)
    findings: dict = {}
    originals: list = []
    t = text
    for kind, value in (known or []):
        if len(value) >= 4:
            pat = re.compile(re.escape(value), re.IGNORECASE)
            t, n = pat.subn(f"[{kind}]", t)
            if n:
                findings[kind] = findings.get(kind, 0) + n
    kw = dict(findings=findings, originals=originals, only=only)
    t = _sub(_EMAIL, t, "EMAIL", **kw)
    t = _sub(_IBAN, t, "IBAN", validate=iban_ok, **kw)
    t = _sub(_CARD, t, "CARD_NUMBER", validate=lambda s_: luhn_ok(re.sub(r"[ \-]", "", s_)), **kw)
    t = _sub(_SSN, t, "NATIONAL_ID", **kw)
    t = _sub(_NINO, t, "NATIONAL_ID", **kw)
    t = _sub(_PAN, t, "NATIONAL_ID", **kw)
    t = _sub(_AADHAAR, t, "NATIONAL_ID", **kw)
    t = _sub(_SORT, t, "SORT_CODE", **kw)
    t = _sub(_DOB, t, "DATE_OF_BIRTH", keep_group=True, **kw)
    t = _sub(_PHONE_PLUS, t, "PHONE", validate=lambda s_: 8 <= len(re.sub(r"\D", "", s_)) <= 15, **kw)
    t = _sub(_PHONE_US, t, "PHONE", **kw)
    t = _sub(_PHONE_NAT, t, "PHONE", **kw)
    t = _sub(_ACCT_LONG, t, "ACCOUNT_NUMBER", **kw)
    t = _sub(_ACCT_CTX, t, "ACCOUNT_NUMBER", keep_group=True, **kw)
    t = _sub(_LAST4, t, "CARD_LAST4", keep_group=True, **kw)
    t = _sub(_CARD_BARE, t, "CARD_LAST4", keep_group=True, **kw)
    t = _sub(_POSTCODE, t, "POSTCODE", **kw)
    t = _sub(_NAME_CUE, t, "NAME", keep_group=True, **kw)
    t = _sub(_NAME_TITLE, t, "NAME", **kw)
    return RedactionResult(text=t, findings=findings, originals=originals)


def merge_findings(*results: RedactionResult) -> dict:
    """Add up the findings of several RedactionResults into one type -> count dictionary."""
    total: dict = {}
    for r in results:
        for k, v in r.findings.items():
            total[k] = total.get(k, 0) + v
    return total
