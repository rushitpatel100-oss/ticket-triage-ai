"""Stage 0: clean up a ticket before anything reads it.

Tickets that arrive by email carry signatures, legal disclaimers and quoted reply chains. Retrieval and
classification should see the user's problem, not the footer. `clean_ticket` removes the common patterns;
`redact` masks personal data before text reaches an LLM prompt or a log (OWASP LLM02).

Both are deliberately simple rules: they are measured in `src.robustness` (E5) and are one layer of
protection, not a guarantee. A production system would add a trained PII detector such as Microsoft
Presidio, which itself warns that no tool finds all personal data.
"""

import re

# Everything from a reply or forward marker onwards is an older message, not this ticket.
THREAD_MARKERS = [
    re.compile(r"^\s*-{2,}\s*(original message|forwarded message)\s*-{2,}\s*$", re.I),
    re.compile(r"^\s*on .{3,120} wrote:\s*$", re.I),
    re.compile(r"^\s*begin forwarded message:?\s*$", re.I),
]
HEADER = re.compile(r"^\s*(from|sent|date|to|cc|subject)\s*:", re.I)
SIGN_OFF = re.compile(r"^\s*((kind|best|warm|many thanks and)\s+regards|regards|thanks( you| again)?|thank you|cheers|"
                      r"sincerely|best|br)\s*[,.!]?\s*$", re.I)
SIGNATURE_LINE = re.compile(r"^\s*(sent from my \w+|get outlook for \w+)", re.I)
DISCLAIMER = re.compile(r"(confidential and (may be )?(legally )?privileged|intended (solely|only) for the (use of the )?"
                        r"(addressee|recipient|individual)|received this (e-?mail|message) in error|"
                        r"consider the environment before printing|^\s*disclaimer\s*:)", re.I)
MAX_SIGNATURE_LINES = 8  # a sign-off this close to the end starts the signature; further up it is just text


def strip_thread(lines: list[str]) -> list[str]:
    for i, line in enumerate(lines):
        if any(m.match(line) for m in THREAD_MARKERS):
            return lines[:i]
        # an Outlook-style header block: "From: ..." followed by "Sent:/Date:/To:" within a few lines
        if i and HEADER.match(line) and line.lower().lstrip().startswith("from") and \
                sum(bool(HEADER.match(x)) for x in lines[i + 1:i + 5]) >= 2:
            return lines[:i]
    return [line for line in lines if not line.lstrip().startswith(">")]


def strip_signature(lines: list[str]) -> list[str]:
    lines = [line for line in lines if not DISCLAIMER.search(line) and not SIGNATURE_LINE.match(line)]
    content = [i for i, line in enumerate(lines) if line.strip()]
    for i in reversed(content):
        if SIGN_OFF.match(lines[i]):
            after = sum(1 for j in content if j > i)
            if after <= MAX_SIGNATURE_LINES and i > content[0]:
                return lines[:i]
            break
    return lines


def clean_ticket(text: str) -> str:
    """The ticket without quoted replies, forwarded threads, sign-offs, signatures or disclaimers."""
    lines = str(text or "").replace("\r\n", "\n").split("\n")
    lines = strip_signature(strip_thread(lines))
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return cleaned or str(text or "").strip()  # never return an empty ticket


EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b")
PHONE = re.compile(r"(?<!\w)(\+\d{1,3}[\s-]?)?(\(?\d{2,5}\)?[\s-]?)\d{3,4}[\s-]?\d{3,4}(?!\w)")
LONG_NUMBER = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")  # 13-19 digits: card or account-like numbers


def redact(text: str) -> str:
    """Mask e-mail addresses, phone numbers and long card- or account-like numbers."""
    text = EMAIL.sub("[EMAIL]", str(text or ""))
    text = LONG_NUMBER.sub("[NUMBER]", text)
    return PHONE.sub("[PHONE]", text)
